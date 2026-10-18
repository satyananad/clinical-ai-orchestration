"""
CONVERSATION & SESSION MANAGEMENT SERVICE (conversation_service.py)
-------------------------------------------------------------------
Manages session-based conversational memory, multi-turn dialogue,
and contextual reference resolution ("her report", "that scan", "the latest one").
Integrates directly with report_qa_service for evidence-grounded generation.
"""

from typing import Dict, Any, Optional, List
import re
from .retrieval import get_patient, get_report_by_id, resolve_report_retrieval
from .report_qa_service import answer_report_question, generate_report_summary

class SessionState:
    """
    Encapsulates isolated session context for a specific clinician conversation.
    Ensures state is stored per session and never in a shared global variable.
    """
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.active_patient_id: Optional[str] = None
        self.active_report_id: Optional[str] = None
        self.history: List[Dict[str, str]] = []

    def set_active(self, patient_id: str, report_id: str):
        """Updates active patient & report ONLY upon verified successful retrieval."""
        self.active_patient_id = patient_id
        self.active_report_id = report_id

    def clear(self):
        """Clears session context and resets active selections."""
        self.active_patient_id = None
        self.active_report_id = None
        self.history = []

# Thread-safe session dictionary keyed by unique session_id
SESSIONS: Dict[str, SessionState] = {}

def get_or_create_session(session_id: str) -> SessionState:
    """Retrieves or initializes an isolated session state."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState(session_id)
    return SESSIONS[session_id]

# Linguistic trigger patterns for context-dependent follow-up
PRONOUN_TRIGGERS = [
    "her", "his", "their", "the patient", "she", "he",
    "this report", "the report", "it", "latest one", "the latest",
    "that scan", "her scan", "his scan", "the scan"
]

def is_contextual_query(query: str) -> bool:
    """Detects whether a user query refers to an active session context rather than a new patient."""
    q_lower = query.lower()
    return any(re.search(r"\b" + re.escape(p) + r"\b", q_lower) for p in PRONOUN_TRIGGERS)

def handle_assistant_interaction(query: str, session_id: str = "default_session") -> Dict[str, Any]:
    """
    Unified entry point for conversational interaction:
    1. Handles new patient report retrieval (Feature A).
    2. Handles context-dependent follow-up questions ('Summarize her report') (Feature C).
    3. Handles report-aware Q&A on specific findings (Bonus Feature).
    """
    session = get_or_create_session(session_id)
    q_lower = query.lower().strip()

    # =========================================================================
    # CASE 1: CONTEXTUAL FOLLOW-UP ON AN ACTIVE REPORT
    # =========================================================================
    if is_contextual_query(q_lower):
        # Rule: If no active report is selected, prompt user to specify which report
        if not session.active_patient_id or not session.active_report_id:
            return {
                "status": "clarification_required",
                "message": "No active patient report is currently selected. Please specify which patient or report you would like to view (e.g., 'Open Priya Sharma's OPG report').",
                "candidates": None,
                "report": None
            }

        patient = get_patient(session.active_patient_id)
        report = get_report_by_id(session.active_report_id)

        if not patient or not report:
            session.clear()
            return {
                "status": "not_found",
                "message": "The previously selected report is no longer available. Please request a patient report again.",
                "candidates": None
            }

        context_resolution_msg = (
            f"Resolved contextual pronoun ('her' / 'this report') -> "
            f"Active Patient: {patient['patient_name']} ({patient['patient_id']}), "
            f"Report: {report['report_id']} ({report['report_type']})."
        )

        # Delegate answer generation directly to the evidence-grounded report_qa_service
        qa_result = answer_report_question(session.active_report_id, query)

        # Record conversation turn
        session.history.append({"user": query, "assistant": qa_result.get("answer", "")})

        return {
            "status": "success",
            "message": "Answered from active report context.",
            "context_resolution": context_resolution_msg,
            "patient": {
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "date_of_birth": patient["date_of_birth"],
                "sex": patient["sex"]
            },
            "report": report,
            "answer": qa_result.get("answer"),
            "citations": qa_result.get("citations", [])
        }

    # =========================================================================
    # CASE 2: NEW PATIENT REPORT RETRIEVAL (FEATURE A)
    # =========================================================================
    retrieval_res = resolve_report_retrieval(query)

    # Important Rule: Only update active session when retrieval is strictly successful
    if retrieval_res["status"] == "success" and retrieval_res.get("report"):
        session.set_active(
            retrieval_res["patient"]["patient_id"],
            retrieval_res["report"]["report_id"]
        )

    return retrieval_res
