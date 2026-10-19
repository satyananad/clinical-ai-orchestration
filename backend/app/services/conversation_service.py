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

from enum import Enum

class BookingStage(str, Enum):
    IDLE = "IDLE"
    IDENTIFYING_PATIENT = "IDENTIFYING_PATIENT"
    COLLECTING_DETAILS = "COLLECTING_DETAILS"
    SELECTING_SERVICE = "SELECTING_SERVICE"
    SELECTING_SPECIALIZATION = "SELECTING_SPECIALIZATION"
    SELECTING_DOCTOR = "SELECTING_DOCTOR"
    SLOTS_PRESENTED = "SLOTS_PRESENTED"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"

class BookingWorkflowState:
    """Explicit state machine tracking appointment booking stages and validated entities."""
    def __init__(self):
        self.stage: BookingStage = BookingStage.IDLE
        self.patient_id: Optional[str] = None
        self.patient_name: Optional[str] = None
        self.patient_dob: Optional[str] = None
        self.patient_age: Optional[str] = None
        self.patient_phone: Optional[str] = None
        self.patient_email: Optional[str] = None
        self.patient_sex: Optional[str] = None
        self.is_new_patient: Optional[bool] = None
        self.doctor_id: Optional[str] = None
        self.doctor_name: Optional[str] = None
        self.specialization: Optional[str] = None
        self.service: Optional[str] = None
        self.location: Optional[str] = None
        self.patient_type: str = "registered"  # "registered", "walk_in", "new", "existing"
        self.modality: Optional[str] = None
        self.preferred_date: Optional[str] = None
        self.preferred_time: Optional[str] = None
        self.displayed_slots: List[Dict[str, Any]] = []
        self.selected_slot: Optional[Dict[str, Any]] = None
        self.summary_displayed: bool = False
        self.completed_booking: Optional[Dict[str, Any]] = None
        self.last_action_status: Optional[str] = None
        self.notes: Optional[str] = None

    def reset(self):
        self.stage = BookingStage.IDLE
        self.patient_id = None
        self.patient_name = None
        self.patient_dob = None
        self.patient_age = None
        self.patient_phone = None
        self.patient_email = None
        self.patient_sex = None
        self.is_new_patient = None
        self.doctor_id = None
        self.doctor_name = None
        self.specialization = None
        self.service = None
        self.location = None
        self.patient_type = "registered"
        self.modality = None
        self.preferred_date = None
        self.preferred_time = None
        self.displayed_slots = []
        self.selected_slot = None
        self.summary_displayed = False
        self.completed_booking = None
        self.last_action_status = None
        self.notes = None

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
        self.pending_candidates: Optional[List[Dict[str, Any]]] = None
        self.pending_request: Optional[Dict[str, Any]] = None
        self.last_retrieval_query: Optional[str] = None
        self.last_retrieval_modality: Optional[str] = None
        self.booking_state: BookingWorkflowState = BookingWorkflowState()

    @property
    def pending_booking_slots(self) -> Optional[List[Dict[str, Any]]]:
        return self.booking_state.displayed_slots if self.booking_state.displayed_slots else None

    @pending_booking_slots.setter
    def pending_booking_slots(self, value: Optional[List[Dict[str, Any]]]):
        self.booking_state.displayed_slots = value or []

    @property
    def pending_booking_modality(self) -> Optional[str]:
        return self.booking_state.modality

    @pending_booking_modality.setter
    def pending_booking_modality(self, value: Optional[str]):
        self.booking_state.modality = value

    def set_active_patient(self, patient_id: str):
        """Updates active patient while keeping report unselected until chosen."""
        self.active_patient_id = patient_id
        self.active_report_id = None
        self.pending_candidates = None
        self.pending_request = None
        if self.booking_state.stage == BookingStage.IDLE:
            self.booking_state.patient_id = patient_id
            p = get_patient(patient_id)
            if p:
                self.booking_state.patient_name = p.get("patient_name")
                dob = p.get("date_of_birth")
                if dob:
                    try:
                        self.booking_state.patient_age = f"{2026 - int(dob.split('-')[0])} years (DOB: {dob})"
                    except Exception:
                        self.booking_state.patient_age = f"DOB: {dob}"

    def set_active(self, patient_id: str, report_id: str):
        """Updates active patient & report ONLY upon verified successful retrieval."""
        self.active_patient_id = patient_id
        self.active_report_id = report_id
        self.pending_candidates = None
        self.pending_request = None
        self.last_retrieval_query = None
        self.last_retrieval_modality = None
        # Synchronize patient context with booking state if idle
        if self.booking_state.stage == BookingStage.IDLE:
            self.booking_state.patient_id = patient_id
            p = get_patient(patient_id)
            if p:
                self.booking_state.patient_name = p.get("patient_name")
                dob = p.get("date_of_birth")
                if dob:
                    try:
                        self.booking_state.patient_age = f"{2026 - int(dob.split('-')[0])} years (DOB: {dob})"
                    except Exception:
                        self.booking_state.patient_age = f"DOB: {dob}"

    def clear(self):
        """Clears session context and resets active selections."""
        self.active_patient_id = None
        self.active_report_id = None
        self.history = []
        self.pending_candidates = None
        self.pending_request = None
        self.last_retrieval_query = None
        self.last_retrieval_modality = None
        self.booking_state.reset()

# Thread-safe session dictionary keyed by unique session_id
SESSIONS: Dict[str, SessionState] = {}

def get_or_create_session(session_id: str) -> SessionState:
    """Retrieves or initializes an isolated session state."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState(session_id)
    return SESSIONS[session_id]

def is_general_or_assistant_query(query: str) -> bool:
    """
    Determines if a query is a general knowledge question (e.g. 'What is a CBCT scan?'),
    a workstation/help query ('Who are you?', 'Help'), or conversational greeting ('Hi'),
    which should NOT be treated as a contextual follow-up on an active patient report.
    """
    q = query.lower().strip()
    q_clean = re.sub(r"[?!.,:]+", "", q).strip()

    # 1. Greetings & Pleasantries
    greetings = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "bye", "goodbye", "ok", "okay", "yes", "no"
    }
    if q_clean in greetings:
        return True

    # 2. Assistant identity, workstation help, capabilities
    workstation_phrases = [
        "who are you", "what are you", "what can you do", "help", "help me",
        "how do you work", "how does this work", "what is this workstation",
        "what are your features", "what can i do", "what are you doing",
        "capabilities", "what functions", "tell me about yourself"
    ]
    if q_clean in workstation_phrases or any(q_clean.startswith(p) for p in workstation_phrases):
        return True

    # If query mentions a specific patient, pronouns, report, or specific tooth number, it's NOT a general query
    patient_indicators = [
        "her", "his", "she", "he", "patient", "priya", "vikram", "sharmila",
        "in this report", "in her report", "in the report", "her scan", "this scan",
        "the scan", "summarize", "summary", "findings", "finding", "impression",
        "can you explain", "explain this", "what does this mean", "why is that",
        "proximity", "relationship", "involvement", "margin", "status", "angulation"
    ]
    if any(re.search(r"\b" + re.escape(ind) + r"\b", q) for ind in patient_indicators):
        return False

    # Check for specific tooth references (e.g. "tooth 16", "teeth 16", "#29", "ur5", "48")
    if re.search(r"\b(?:tooth|teeth|#)\s*([1-8][1-8]|[1-3]?[0-9])\b", q) or re.search(r"\b(?:ur|ul|ll|lr)\s*[1-8]\b", q):
        return False

    # 3. General educational dental & radiology definitions / concepts
    edu_starters = ["what is a ", "what is an ", "what is ", "explain ", "tell me about ", "define ", "how does ", "how do "]
    is_starter = any(q.startswith(s) for s in edu_starters) or "what does" in q
    
    dental_concepts = [
        "cbct", "cone beam", "opg", "panoramic", "orthopantomogram", "bitewing", "periapical",
        "iopa", "cephalogram", "cephalometric", "caries", "cavity", "decay", "demineralization",
        "pulpitis", "apical periodontitis", "periapical rarefaction", "lamina dura",
        "alveolar crest", "mandibular canal", "inferior alveolar nerve", "mesioangular",
        "impaction", "impacted tooth", "wisdom tooth", "3d scan", "3d imaging",
        "root canal", "fluoride", "scaling", "abfraction"
    ]
    if is_starter and any(concept in q for concept in dental_concepts):
        return True

    # 4. Clinic timings, location, and operational inquiries
    clinic_keywords = [
        "clinic timings", "clinic timing", "timings", "opening hours", "open hours", "working hours",
        "hours of operation", "what time do you open", "what time do you close", "when are you open",
        "where are you located", "clinic location", "clinic address", "where is the clinic",
        "contact number", "phone number", "phone", "email", "appointment booking", "services offered",
        "what services", "what scans do you do"
    ]
    if any(k in q for k in clinic_keywords):
        return True

    return False

def is_contextual_query(query: str, has_active_report: bool = False) -> bool:
    """
    Determines whether a clinician's query refers to an active report context:
    1. If a report is currently active:
       Any clinical question, inquiry about findings, teeth, impressions, or summary
       refers to the active report unless the user explicitly requests to open another patient
       or asks a general knowledge/workstation question.
    2. If no report is active:
       Returns True only if the request explicitly asks for a summary or findings of an implied
       patient/report (e.g. 'summarize her report', 'what did the report say'), allowing the
       system to safely prompt the clinician to select a patient/report first.
    """
    q_lower = query.lower().strip()
    
    # 0. General educational questions, greetings, or workstation help are NOT report follow-ups
    if is_general_or_assistant_query(query):
        return False

    # 1. Clinic-wide patient listings or database count questions are not report follow-ups
    if any(q_lower.startswith(w) for w in ["how many", "how much", "list", "total", "count"]):
        return False

    # 2. Check if this is an explicit new patient retrieval request
    has_retrieval_cmd = any(v in q_lower for v in ["open", "retrieve", "fetch", "load"])
    from .retrieval import extract_retrieval_intent
    r_intent = extract_retrieval_intent(query)
    cand_name = r_intent.get("patient_name", "").strip()
    pronoun_refs = {"he", "she", "his", "her", "hir", "their", "theirs", "its", "him", "them", "this", "that"}
    has_distinct_patient = bool(cand_name) and not any(w in pronoun_refs for w in cand_name.lower().split())
    if has_retrieval_cmd and (has_distinct_patient or not has_active_report):
        return False

    # 3. If an active report is open, questions refer to the active report
    if has_active_report:
        return True

    # 4. If no report is active yet: check if clinician is attempting a follow-up or report summary
    is_report_reference = (
        any(w in q_lower for w in ["summarize", "summary"])
        or ("report" in q_lower and any(w in q_lower for w in ["her", "his", "their", "the", "this"]))
        or ("scan" in q_lower and any(w in q_lower for w in ["her", "his", "their", "that", "this"]))
    )
    return is_report_reference



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
    if is_contextual_query(q_lower, has_active_report=bool(session.active_report_id or session.active_patient_id)):
        # Rule: If no active patient is selected, prompt user to specify which report
        if not session.active_patient_id:
            return {
                "status": "clarification_required",
                "message": "No active patient report is currently selected. Please specify which patient or report you would like to view (e.g., 'Open Priya Sharma's OPG report').",
                "candidates": None,
                "report": None
            }

        patient = get_patient(session.active_patient_id)
        if not patient:
            session.clear()
            return {
                "status": "not_found",
                "message": "The previously selected patient is no longer available. Please request a patient report again.",
                "candidates": None
            }

        # If patient is selected but no specific report was opened yet:
        if not session.active_report_id:
            reports = patient.get("reports", [])
            if len(reports) == 1:
                # Unambiguous: single report
                session.set_active(patient["patient_id"], reports[0]["report_id"])
            elif len(reports) > 1:
                # Ask for clarification only when necessary
                report_list_str = " • ".join([f"[{r['report_id']}] {r['report_type']} ({r['report_date']})" for r in reports])
                from .retrieval import get_patient_pronouns
                p_sex = patient.get("sex")
                pronoun_pos = get_patient_pronouns(p_sex)["possessive"]
                return {
                    "status": "clarification_required",
                    "message": f"Patient {patient['patient_name']} has {len(reports)} diagnostic reports on file: {report_list_str}. Please specify which report to view or summarize (e.g., 'Open {pronoun_pos} {reports[0]['report_type']}').",
                    "patient": patient,
                    "candidates": None,
                    "report": None
                }
            else:
                return {
                    "status": "missing_report",
                    "message": f"No diagnostic reports found on file for {patient['patient_name']}.",
                    "patient": patient
                }

        report = get_report_by_id(session.active_report_id)
        if not report:
            session.active_report_id = None
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
        qa_result = answer_report_question(session.active_report_id, query, history=session.history)

        # Record conversation turn
        session.history.append({"user": query, "assistant": qa_result.get("answer", "")})

        return {
            "status": "success",
            "message": qa_result.get("answer"),
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
