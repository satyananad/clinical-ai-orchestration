"""
TESTS FOR CONVERSATION SERVICE & REPORT-GROUNDED QA SERVICE
-----------------------------------------------------------
Verifies:
1. Isolated session state management (no shared globals).
2. Reference resolution ("her report", "that scan").
3. Direct evidence-grounded generation over selected report.
4. Absence detection (stating when information is not in report).
5. Unselected state safety check (asking user which report they mean).
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.conversation_service import get_or_create_session, handle_assistant_interaction
from backend.app.services.report_qa_service import answer_report_question

client = TestClient(app)

def test_unselected_context_prompts_user_for_report():
    """Rule 3: If no report is active, ask the user which report they mean."""
    session_id = "test_empty_session_99"
    res = handle_assistant_interaction("Summarize her report", session_id=session_id)
    assert res["status"] == "clarification_required"
    assert "No active patient report is currently selected" in res["message"]


def test_session_isolation_between_clinicians():
    """Rule 1: Store context by session, not in a single shared global variable."""
    session_a = "session_doctor_a"
    session_b = "session_doctor_b"

    # Doctor A opens Priya Sharma's OPG
    handle_assistant_interaction("Open report PAT-1001", session_id=session_a)
    s_a = get_or_create_session(session_a)
    assert s_a.active_patient_id == "PAT-1001"

    # Doctor B has not opened anything
    s_b = get_or_create_session(session_b)
    assert s_b.active_patient_id is None

    # Doctor B asking for a summary is told no report is active
    res_b = handle_assistant_interaction("Summarize her report", session_id=session_b)
    assert res_b["status"] == "clarification_required"


def test_conversational_follow_up_and_report_qa_workflow():
    """Complete workflow: Open report -> Summarize her report -> Specific QA questions."""
    session_id = "test_full_qa_session"

    # 1. Open report
    r1 = handle_assistant_interaction("Open PAT-1001's OPG report", session_id=session_id)
    assert r1["status"] == "success"
    assert r1["report"]["report_id"] == "RPT-2001"

    # 2. Mandatory: Summarize her report
    r2 = handle_assistant_interaction("Summarize her report", session_id=session_id)
    assert r2["status"] == "success"
    assert "Resolved contextual pronoun" in r2["context_resolution"]
    assert "Priya Sharma" in r2["answer"]
    assert len(r2["citations"]) > 0

    # 3. Bonus: "What does the impression say?"
    r3 = handle_assistant_interaction("What does the impression say about this report?", session_id=session_id)
    assert r3["status"] == "success"
    assert "Mesioangular impaction of tooth 48" in r3["answer"] or "impaction" in r3["answer"].lower()
    assert len(r3["citations"]) > 0

    # 4. Bonus: "Which tooth is mentioned?"
    r4 = handle_assistant_interaction("Which tooth is mentioned in the report?", session_id=session_id)
    assert r4["status"] == "success"
    assert any(tooth in r4["answer"] for tooth in ["48", "38", "16", "third molar"])

    # 5. Bonus: "Does the report mention an earlier scan?"
    r5 = handle_assistant_interaction("Does this report mention an earlier scan from 2020?", session_id=session_id)
    assert r5["status"] == "success"
    # Anti-hallucination check: Explains that earlier scan is not documented
    assert "not documented" in r5["answer"].lower() or "impression is" in r5["answer"].lower()


def test_full_name_with_speech_artifact_and_contextual_follow_up():
    """Verifies that queries containing full names with extraneous or misrecognized words open directly and support follow-up."""
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
    session_id = "test_artifact_session_42"
    session = get_or_create_session(session_id)
    session.clear()

    # 1. Spoken input with speech artifact: 'can you open report PAT-1001 OPG pineapple report'
    res1 = run_multi_agent_pipeline("can you open report PAT-1001 OPG pineapple report", session_id=session_id)
    assert res1["status"] == "success"
    assert res1["patient"]["patient_name"] == "Priya Sharma"
    assert res1["report"]["report_id"] == "RPT-2001"

    # 2. Conversational follow-up: 'Summarize her report'
    res2 = run_multi_agent_pipeline("Summarize her report", session_id=session_id)
    assert res2["status"] == "success"
    assert "Priya Sharma" in res2["answer"]
    assert len(res2["citations"]) > 0

    # 3. Grounded question answering: 'What did the radiologist say about tooth 48?'
    res3 = run_multi_agent_pipeline("What did the radiologist say about tooth 48?", session_id=session_id)
    assert res3["status"] == "success"
    assert len(res3["citations"]) > 0


def test_speech_correction_repair_turn():
    """Verifies multi-turn speech repair when clinician corrects a misrecognized term."""
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
    session_id = "test_repair_session_99"
    session = get_or_create_session(session_id)
    session.clear()

    # Ambiguous search with speech error
    res1 = run_multi_agent_pipeline("open Priya pineapple report", session_id=session_id)
    assert res1["status"] == "clarification_required"

    # Speech repair: 'not pineapple I am saying panoramic'
    res2 = run_multi_agent_pipeline("not pineapple I am saying panoramic", session_id=session_id)
    assert res2["status"] == "clarification_required"

    # Clinician clarifies candidate with specific Patient ID
    res3 = run_multi_agent_pipeline("PAT-1001", session_id=session_id)
    assert res3["status"] == "success"
    assert res3["patient"]["patient_name"] == "Priya Sharma"
    assert res3["patient"]["patient_id"] == "PAT-1001"


def test_patient_listing_and_menu_retrieval_queries():
    """
    Verifies that requests for patient names or reports correctly query the database
    instead of wrongly attempting contextual report QA or treating menu commands as patient names.
    """
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
    session_id = "test_listing_session_1"
    session = get_or_create_session(session_id)
    session.clear()

    # 1. User asks for list of patients and reports: "how many report you show me the list od patient and report"
    r1 = run_multi_agent_pipeline("how many report you show me the list od patient and report", session_id=session_id)
    assert r1["status"] in ("success", "clarification_required")
    ans1 = (r1.get("answer") or "") + " " + (r1.get("message") or "")
    # Must list patients or registered database statistics
    assert any(name in ans1 for name in ["Priya Sharma", "Vikram Malhotra", "registered patients", "radiological reports"])

    # 2. User asks: "first tell me the patient names"
    r2 = run_multi_agent_pipeline("first tell me the patient names", session_id=session_id)
    assert r2["status"] == "success"
    ans2 = (r2.get("answer") or "") + " " + (r2.get("message") or "")
    assert "registered patients in the clinic database" in ans2
    assert "Priya Sharma" in ans2
    assert "Vikram Malhotra" in ans2

    # 3. User types or clicks the menu item: "Retrieving a patient report" or "Retrieve a report"
    r3 = run_multi_agent_pipeline("Retrieving a patient report", session_id=session_id)
    ans3 = (r3.get("answer") or "") + " " + (r3.get("message") or "")
    # Must NOT search for a patient named 'retrieving'
    assert "No patient records found matching 'retrieving'" not in ans3
    assert any(w in ans3.lower() for w in ["retrieve patient imaging", "open", "patient", "explore"])

