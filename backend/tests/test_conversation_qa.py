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


def test_dynamic_report_qa_intent_coverage():
    """
    Evaluates dynamic question understanding across clinical report JSON:
    1. 'Which tooth has caries?' -> identifies Tooth 16; excludes non-carious Tooth 25.
    2. 'What treatment is advised?' -> extracts treatment/recommendations.
    3. 'What does it say about the gums?' -> targets periodontium/gingiva findings.
    4. 'Is there any measurement for tooth 25?' -> extracts thermal test response.
    5. 'Does this report mention tooth 48?' -> explicitly states not documented; no hallucinated normalcy.
    6. "What is Priya's age?" -> checks authorized Master Patient Index demographics.
    """
    # 1. Which tooth has caries?
    res_caries = answer_report_question("RPT-2003", "Which tooth has caries?")
    assert res_caries["status"] == "success"
    ans_caries = res_caries["answer"]
    assert "Tooth 16" in ans_caries
    assert "Tooth 25" not in ans_caries

    # 2. What treatment is advised?
    res_treat = answer_report_question("RPT-2003", "What treatment is advised?")
    assert res_treat["status"] == "success"
    assert "topical varnish" in res_treat["answer"] or "composite resin restoration" in res_treat["answer"]

    # 3. What does it say about the gums?
    res_gums = answer_report_question("RPT-2003", "What does it say about the gums?")
    assert res_gums["status"] == "success"
    assert "Periodontium" in res_gums["answer"]
    assert "gingival inflammation" in res_gums["answer"].lower()

    # 4. Is there any measurement for tooth 25?
    res_meas = answer_report_question("RPT-2003", "Is there any measurement for tooth 25?")
    assert res_meas["status"] == "success"
    assert "Tooth 25" in res_meas["answer"]
    assert "2 seconds" in res_meas["answer"]

    # 5. Does this report mention tooth 48?
    res_48 = answer_report_question("RPT-2003", "Does this report mention tooth 48?")
    assert res_48["status"] == "success"
    assert "not documented" in res_48["answer"].lower()
    assert "must not be assumed that the tooth is clinically normal" in res_48["answer"]

    # 6. What is Priya's age?
    res_age = answer_report_question("RPT-2003", "What is Priya's age?")
    assert res_age["status"] == "success"
    assert "1998-04-12" in res_age["answer"]
    assert "Master Patient Index" in res_age["answer"]


def test_multiturn_explanation_of_specific_tooth_finding():
    """
    Verifies that when a clinician asks about a specific finding (e.g. Teeth 16 and 46)
    and follows up with 'can you explain', the assistant provides a focused clinical
    explanation of that exact finding (depth into mid-dentin, pulp vitality, caries)
    instead of reciting the entire report from third molar impactions.
    """
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline

    session_id = "test_explain_finding_session_101"
    # 1. Open Priya's OPG
    r1 = run_multi_agent_pipeline("i want to open the OPG report of priya sharma pat 1001", session_id=session_id)
    assert r1["status"] == "success"
    assert r1.get("report", {}).get("report_id") == "RPT-2001"

    # 2. Ask about specific teeth
    r2 = run_multi_agent_pipeline("what is this : Teeth 16 and 46", session_id=session_id)
    assert r2["status"] == "success"
    ans2 = r2.get("answer") or r2.get("message")
    assert "16" in ans2
    assert "radiolucency" in ans2.lower()

    # 3. Follow up: "can you explain"
    r3 = run_multi_agent_pipeline("can you explain", session_id=session_id)
    assert r3["status"] == "success"
    ans3 = r3.get("answer") or r3.get("message")
    assert ans3 is not None
    # Must explain the specific finding: tooth 16 / coronal radiolucency / caries / dentin
    assert any(w in ans3.lower() for w in ["tooth 16", "teeth 16", "radiolucency", "caries", "dentin"])
    # Must explain clinical significance of "without periapical rarefaction" or pulp status
    assert any(w in ans3.lower() for w in ["pulp", "vital", "periapical", "restoration", "filling"])
    # Must NOT dump or focus on tooth 48 third molar impaction
    assert "tooth 48 is partially erupted" not in ans3.lower()


def test_booking_slot_name_extraction_and_post_booking_transition():
    """
    Verifies that 'Confirm slot SLOT-106' does not extract 'Confirm -106' as the patient name,
    and after confirmation, natural clinician replies ('ok', 'yes') do not trap the session
    in repeat 'already confirmed' messages.
    """
    from datetime import datetime
    from backend.app.services.clock import set_frozen_clock
    from backend.app.services.booking_service import reset_slots
    from backend.app.services.conversation_service import SESSIONS
    from backend.app.services.retrieval import extract_retrieval_intent
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline, triage_clinician_request, is_valid_patient_name

    reset_slots()
    SESSIONS.clear()
    set_frozen_clock(datetime(2026, 10, 18, 9, 0, 0))

    try:
        # 1. Verification of name extraction & validation
        intent = extract_retrieval_intent("Confirm slot SLOT-106")
        extracted_name = intent.get("patient_name", "")
        assert not is_valid_patient_name(extracted_name)
        assert extracted_name.lower() not in ["confirm -106", "confirm 106"]

        # 2. Session execution
        session_id = "test_booking_name_session_202"
        r1 = run_multi_agent_pipeline("Schedule a periapical scan for tomorrow", session_id=session_id)
        assert r1["status"] in ["success", "awaiting_confirmation", "slots_presented", "confirmation_required"]

        r2 = run_multi_agent_pipeline("Confirm slot SLOT-106", session_id=session_id)
        assert r2["status"] == "success"
        # Patient must NOT be 'Confirm -106'
        assert r2.get("slot", {}).get("patient_name") != "Confirm -106"

        # 3. Clinician says 'ok' - must transition out of confirmed stage
        r3 = run_multi_agent_pipeline("ok", session_id=session_id)
        # Should not re-trigger duplicate booking
        assert r3.get("tool_called") != "booking_already_confirmed"
        msg3 = r3.get("answer") or r3.get("message") or ""
        assert "already confirmed" not in msg3.lower()
    finally:
        set_frozen_clock(None)
        reset_slots()
        SESSIONS.clear()


def test_personal_information_query_after_opening_report():
    """
    Verifies that asking 'what personal information have in there report'
    returns the patient's personal and demographic details with proper gender pronouns,
    rather than dumping a clinical diagnostic findings summary.
    """
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
    session_id = "test_arjun_personal_info_session"

    # Step 1: Open report using fuzzy speech/spelling 'open report arjun mejta'
    r1 = run_multi_agent_pipeline("open report arjun mejta", session_id=session_id)
    assert r1["status"] == "success"
    assert r1["patient"]["patient_name"] == "Arjun Mehta"
    assert r1["patient"]["patient_id"] == "PAT-1005"

    # Step 2: Inquire about personal information in the report
    r2 = run_multi_agent_pipeline("what personal information have in there report", session_id=session_id)
    assert r2["status"] == "success"
    ans2 = r2.get("answer") or r2.get("message") or ""

    # Must contain personal and demographic details
    assert "Arjun Mehta" in ans2
    assert "PAT-1005" in ans2
    assert "1992-06-30" in ans2
    assert "34 years" in ans2
    assert "Male" in ans2
    assert "Bitewing Radiographs" in ans2

    # Must NOT dump diagnostic findings summary as personal info
    assert "Verification of crown margin seating" not in ans2

    # Must provide proper source evidence citations
    assert any("Master Patient Index" in c for c in r2.get("citations", []))
    assert any("Source Report Header" in c for c in r2.get("citations", []))

    # Step 3: Specific age inquiry with pronoun check
    r3 = run_multi_agent_pipeline("how old is he", session_id=session_id)
    assert r3["status"] == "success"
    ans3 = r3.get("answer") or r3.get("message") or ""
    assert "his date of birth is 1992-06-30" in ans3
    assert "34 years" in ans3

    # Step 4: Contextual follow-up summary works seamlessly after personal info query
    r4 = run_multi_agent_pipeline("Summarize his report", session_id=session_id)
    assert r4["status"] == "success"
    ans4 = r4.get("answer") or r4.get("message") or ""
    assert "Arjun Mehta" in ans4
    assert any(w in ans4.lower() for w in ["bitewing", "crown margin", "tooth 25", "zirconia"])


def test_female_demographics_query():
    """
    Verifies female patient demographic pronoun and personal info resolution.
    """
    from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
    session_id = "test_priya_personal_info_session"

    r1 = run_multi_agent_pipeline("Open Priya Sharma PAT-1001 OPG", session_id=session_id)
    assert r1["status"] == "success"
    assert r1["patient"]["patient_name"] == "Priya Sharma"

    r2 = run_multi_agent_pipeline("how old is she", session_id=session_id)
    assert r2["status"] == "success"
    ans2 = r2.get("answer") or r2.get("message") or ""
    assert "her date of birth is 1998-04-12" in ans2
    assert "28 years" in ans2

    r3 = run_multi_agent_pipeline("what is her personal info", session_id=session_id)
    assert r3["status"] == "success"
    ans3 = r3.get("answer") or r3.get("message") or ""
    assert "Priya Sharma" in ans3
    assert "PAT-1001" in ans3
    assert "Female" in ans3


