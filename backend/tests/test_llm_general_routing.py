"""
TEST SUITE FOR LLM-DRIVEN DYNAMIC ROUTING & GENERAL AGENT TOOL DISPATCHING
--------------------------------------------------------------------------
Verifies:
1. LLM-driven tool selection: The orchestrator parses structured tool selections
   from the LLM based on tool schemas, conversation history, and application state.
2. Varied natural-language requests and unseen phrasings:
   Handles diverse conversational requests without requiring brittle keyword matches.
3. Contextual follow-up resolution:
   Follow-ups resolve dynamically against active patient/report state without hardcoded pronoun arrays.
4. Deterministic non-conversational safeguards:
   - Patient disambiguation (safety halt on ambiguous names, candidate presentation with DOBs & IDs).
   - Equipment limitation checks (e.g. MRI not supported).
   - Past date rejection (scheduling before current clock rejected).
   - Confirmation requirements before consequential booking mutations.
"""

import json
from unittest.mock import patch
import pytest
from datetime import datetime

from backend.app.services.orchestrator_agent import (
    run_multi_agent_pipeline,
    triage_clinician_request,
    select_tool_with_llm
)
from backend.app.services.conversation_service import (
    get_or_create_session,
    SESSIONS,
    BookingStage
)
from backend.app.services.clock import set_frozen_clock
from backend.app.services.booking_service import reset_slots


@pytest.fixture(autouse=True)
def clean_environment():
    """Reset session memory, booking slots, and clock before each test."""
    SESSIONS.clear()
    reset_slots()
    set_frozen_clock(datetime(2026, 10, 19, 9, 0, 0))
    yield
    set_frozen_clock(None)
    reset_slots()
    SESSIONS.clear()


# ==============================================================================
# 1. DYNAMIC LLM TOOL SELECTION VIA TOOL SCHEMAS
# ==============================================================================

def test_llm_tool_selection_for_unseen_retrieval():
    """Verify orchestrator respects LLM tool decision for arbitrary, unseen retrieval requests."""
    session_id = "test_llm_retrieval_unseen"
    session = get_or_create_session(session_id)

    mock_llm_response = json.dumps({
        "selected_agent": "RetrievalAgent",
        "tool_name": "search_patient_reports",
        "arguments": {
            "patient_name": "Vikram Malhotra",
            "modality": "CBCT"
        },
        "rationale": "Clinician wants to view Vikram Malhotra's 3D volumetric CBCT scan."
    })

    with patch("backend.app.services.orchestrator_agent.query_llm", return_value=mock_llm_response):
        res = triage_clinician_request("Could you pull up the 3D scan volume for Vikram Malhotra?", session_id=session_id)
        assert res["selected_agent"] == "RetrievalAgent"
        assert res["tool_name"] == "search_patient_reports"
        assert res["arguments"]["patient_name"] == "Vikram Malhotra"
        assert res["arguments"]["modality"] == "CBCT"


def test_llm_tool_selection_for_clinical_dictation():
    """Verify orchestrator respects LLM tool decision for spoken clinical dictation."""
    session_id = "test_llm_dictation"
    dictation = "Pre-operative examination reveals vertical bone loss distal to tooth 47 with probing depth 7 mm"

    mock_llm_response = json.dumps({
        "selected_agent": "ClinicalScribeAgent",
        "tool_name": "structure_clinical_dictation",
        "arguments": {
            "dictation_text": dictation
        },
        "rationale": "Input describes periodontal probing measurements and anatomical tooth observations."
    })

    with patch("backend.app.services.orchestrator_agent.query_llm", return_value=mock_llm_response):
        res = triage_clinician_request(dictation, session_id=session_id)
        assert res["selected_agent"] == "ClinicalScribeAgent"
        assert res["tool_name"] == "structure_clinical_dictation"
        assert res["arguments"]["dictation_text"] == dictation


def test_llm_tool_selection_for_active_report_followup():
    """Verify orchestrator respects LLM tool decision for report follow-up using session state."""
    session_id = "test_llm_active_qa"
    session = get_or_create_session(session_id)
    session.set_active("PAT-1009", "RPT-2026")

    mock_llm_response = json.dumps({
        "selected_agent": "ConsultationAgent",
        "tool_name": "summarize_active_report",
        "arguments": {
            "report_id": "RPT-2026",
            "query": "Is there proximity between the root apex and the mandibular canal?"
        },
        "rationale": "Follow-up inquiry refers to the active CBCT report RPT-2026."
    })

    with patch("backend.app.services.orchestrator_agent.query_llm", return_value=mock_llm_response):
        res = triage_clinician_request("Is there proximity between the root apex and the mandibular canal?", session_id=session_id)
        assert res["selected_agent"] == "ConsultationAgent"
        assert res["tool_name"] == "summarize_active_report"
        assert res["arguments"]["report_id"] == "RPT-2026"


# ==============================================================================
# 2. CONTEXTUAL FOLLOW-UPS WITHOUT PRONOUN LISTS OR KEYWORD TRIGGERS
# ==============================================================================

def test_contextual_followups_arbitrary_clinical_questions():
    """
    Clinicians ask diverse questions after opening a report without using
    pronouns like 'her' or 'his' (e.g. 'Status of tooth 48', 'Any decay?', 'What does it say about bone?').
    All must ground in the active report and return citations.
    """
    session_id = "test_general_followup_session"
    
    # 1. Open Vikram Malhotra's CBCT report
    r1 = run_multi_agent_pipeline("Open Vikram Malhotra's CBCT report", session_id=session_id)
    assert r1["status"] == "success"
    assert r1.get("report", {}).get("report_id") == "RPT-2026"

    # 2. Question with NO pronouns: "What are the findings for tooth 48?"
    r2 = run_multi_agent_pipeline("What are the findings for tooth 48?", session_id=session_id)
    assert r2["status"] == "success"
    assert r2.get("agent_trace", {}).get("active_agent") == "ConsultationAgent"
    assert len(r2.get("citations", [])) > 0
    assert any("48" in c or "molar" in c.lower() or "impaction" in c.lower() for c in r2["citations"])

    # 3. Question with NO pronouns: "Explain the mandibular canal proximity."
    r3 = run_multi_agent_pipeline("Explain the mandibular canal proximity.", session_id=session_id)
    assert r3["status"] == "success"
    assert r3.get("agent_trace", {}).get("active_agent") == "ConsultationAgent"
    assert len(r3.get("citations", [])) > 0

    # 4. Question with NO pronouns: "What treatment is advised?"
    r4 = run_multi_agent_pipeline("What treatment is advised?", session_id=session_id)
    assert r4["status"] == "success"
    assert r4.get("agent_trace", {}).get("active_agent") == "ConsultationAgent"


def test_followup_when_unselected_prompts_clarification():
    """Asking a follow-up when NO report is open prompts the clinician to specify a report."""
    session_id = "test_no_report_open"
    res = run_multi_agent_pipeline("Summarize her report", session_id=session_id)
    assert res["status"] == "clarification_required"
    assert "No active patient report is currently selected" in res["message"]


# ==============================================================================
# 3. AMBIGUOUS PATIENT REFERENCES (DETERMINISTIC SAFETY GATES)
# ==============================================================================

def test_ambiguous_patient_search_halts_and_returns_candidates():
    """Searching for 'Priya Sharma' matches two patients; system must halt and return candidates."""
    session_id = "test_ambig_safety"
    res = run_multi_agent_pipeline("Open Priya Sharma's report", session_id=session_id)
    assert res["status"] == "clarification_required"
    candidates = res.get("candidates", [])
    assert len(candidates) >= 2
    # Verify candidate records contain distinct identifiers
    pids = [c["patient_id"] for c in candidates]
    assert "PAT-1001" in pids
    assert "PAT-1012" in pids

    # Step 2: Clinician disambiguates by specifying date of birth or ID
    res2 = run_multi_agent_pipeline("1985-02-14", session_id=session_id)
    assert res2["status"] == "success"
    assert res2["patient"]["patient_id"] == "PAT-1012"



# ==============================================================================
# 4. DETERMINISTIC SAFETY CONSTRAINTS (EQUIPMENT, PAST DATES, CONFIRMATION)
# ==============================================================================

def test_unsupported_modality_equipment_limitation():
    """Requesting an unsupported scan modality (e.g. MRI) returns equipment limitation without guessing rooms."""
    session_id = "test_equip_limit"
    res = run_multi_agent_pipeline("Book an MRI scan tomorrow at 10 AM", session_id=session_id)
    assert res["status"] == "equipment_limitation"
    assert "does not possess equipment for MRI" in res["message"]


def test_past_date_appointment_rejected():
    """Requesting an appointment for a past date is rejected by clinic scheduling policy."""
    session_id = "test_past_date"
    res = run_multi_agent_pipeline("Schedule a CBCT scan on 2020-01-01", session_id=session_id)
    assert res["status"] == "date_rejected"
    assert "Cannot schedule an appointment for a past date" in res["message"]


def test_consequential_booking_requires_explicit_confirmation():
    """Selecting an appointment slot presents a summary and requires confirmation before DB write."""
    session_id = "test_booking_confirmation_flow"
    session = get_or_create_session(session_id)

    # 1. Inquire slots
    r1 = run_multi_agent_pipeline("What slots are available for CBCT tomorrow?", session_id=session_id)
    assert r1["status"] == "confirmation_required"
    assert len(r1.get("available_slots", [])) > 0
    slot_id = r1["available_slots"][0]["slot_id"]

    # 2. Select slot -> enters AWAITING_CONFIRMATION
    r2 = run_multi_agent_pipeline(f"Let's select {slot_id}", session_id=session_id)
    assert session.booking_state.stage == BookingStage.AWAITING_CONFIRMATION

    # 3. Explicit confirmation
    r3 = run_multi_agent_pipeline("Yes, confirm the booking", session_id=session_id)
    assert r3["status"] == "success"
    assert r3.get("booking_reference") is not None
    assert session.booking_state.stage == BookingStage.CONFIRMED
