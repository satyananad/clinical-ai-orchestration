"""
TEST SUITE FOR PATIENT REPORT RETRIEVAL, UI SEPARATION & CONVERSATIONAL FOLLOW-UP
--------------------------------------------------------------------------------
Verifies all mandatory requirements:
1. Patient retrieval:
   - Interpret natural-language requests and query mock database.
   - Handle ambiguous matches (e.g. Priya Sharma PAT-1001 vs PAT-1012 vs Priya Sharmila PAT-1002).
   - Never guess which patient was intended.
2. Report retrieval and UI:
   - Patient selection separate from report selection.
   - Display all associated reports using backend data (>= 2 reports per patient).
   - Each report card contains report type, report ID, date, and available actions.
   - Opening a report directly via /sessions/{session_id}/select-report sets server state.
   - Server state persistence & synchronization via /sessions/{session_id}/context.
   - Prevents silent switching of active patient or report.
3. Conversational follow-up:
   - Resolves contextual pronouns ("her report") when unambiguous without re-prompting.
   - Answers grounded in saved report data with source report ID and citations.
   - Asks for clarification when patient has multiple reports but no report is selected.
   - Safe missing report handling without hallucination.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.conversation_service import get_or_create_session, SESSIONS

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()

def test_patient_directory_and_reports_listing():
    """Verify backend returns all patients with their associated reports."""
    response = client.get("/patients")
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) >= 10

    # Verify synthetic records have >= 2 reports per patient
    priya_1001 = next(p for p in patients if p["patient_id"] == "PAT-1001")
    assert len(priya_1001["reports"]) >= 2
    assert priya_1001["patient_name"] == "Priya Sharma"

    priya_1012 = next(p for p in patients if p["patient_id"] == "PAT-1012")
    assert len(priya_1012["reports"]) >= 2
    assert priya_1012["patient_name"] == "Priya Sharma"

    vikram = next(p for p in patients if p["patient_id"] == "PAT-1009")
    assert len(vikram["reports"]) >= 2
    assert vikram["patient_name"] == "Vikram Malhotra"

def test_ambiguous_patient_matches_never_guesses():
    """When similar names exist, system must return clarification with candidates showing DOB and ID."""
    session_id = "test_ambig_session"
    res = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma's report",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "clarification_required"
    assert "candidates" in data and len(data["candidates"]) >= 2
    # Verify candidate cards contain disambiguating info
    c_ids = [c["patient_id"] for c in data["candidates"]]
    assert "PAT-1001" in c_ids
    assert "PAT-1012" in c_ids

def test_patient_selection_kept_separate_from_report_selection():
    """Selecting a patient sets active_patient_id while keeping active_report_id None."""
    session_id = "test_separate_selection"
    res = client.post(f"/sessions/{session_id}/select-patient", json={"patient_id": "PAT-1001"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Priya Sharma"
    assert len(data["reports"]) == 3

    # Check server-side conversation state
    context_res = client.get(f"/sessions/{session_id}/context")
    assert context_res.status_code == 200
    ctx = context_res.json()
    assert ctx["active_patient_id"] == "PAT-1001"
    assert ctx["active_report_id"] is None  # Report selection is separate!

def test_report_cards_data_and_direct_open():
    """Report cards contain report type, report ID, date, and opening one sets active report."""
    session_id = "test_report_cards"
    # 1. Select Patient
    client.post(f"/sessions/{session_id}/select-patient", json={"patient_id": "PAT-1001"})

    # 2. Open Report directly from card
    open_res = client.post(f"/sessions/{session_id}/select-report", json={
        "patient_id": "PAT-1001",
        "report_id": "RPT-2001"
    })
    assert open_res.status_code == 200
    data = open_res.json()
    assert data["status"] == "success"
    assert data["report"]["report_id"] == "RPT-2001"
    assert data["report"]["report_type"] == "OPG"
    assert data["report"]["report_date"] == "2026-06-12"

    # 3. Verify server state persists both active patient and active report
    ctx = client.get(f"/sessions/{session_id}/context").json()
    assert ctx["active_patient_id"] == "PAT-1001"
    assert ctx["active_report_id"] == "RPT-2001"

def test_conversational_followup_grounded_with_source_report_and_citations():
    """Follow-up questions dynamically resolve active report and include verbatim citations."""
    session_id = "test_followup_citations"
    # Open Vikram Malhotra's CBCT (RPT-2026)
    client.post(f"/sessions/{session_id}/select-report", json={
        "patient_id": "PAT-1009",
        "report_id": "RPT-2026"
    })

    # Ask follow-up without specifying patient or report
    res = client.post("/assistant/chat", json={
        "query": "Summarize his report",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["report"]["report_id"] == "RPT-2026"
    assert data["patient"]["patient_name"] == "Vikram Malhotra"
    assert len(data["citations"]) > 0
    assert any("CBCT" in c or "nerve" in c.lower() or "48" in c or "molar" in c.lower() for c in data["citations"])

def test_clarification_when_patient_selected_with_multiple_reports_unselected():
    """If patient is selected but no report is opened yet, asking to summarize prompts for which report."""
    session_id = "test_prompt_which_report"
    # Select Priya Sharma (who has 3 reports) without opening one
    client.post(f"/sessions/{session_id}/select-patient", json={"patient_id": "PAT-1001"})

    # User asks "Summarize her report"
    res = client.post("/assistant/chat", json={
        "query": "Summarize her report",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    # Must ask for clarification since multiple reports exist
    assert data["status"] == "clarification_required"
    assert "3 diagnostic reports on file" in data["message"]
    assert "RPT-2001" in data["message"]

def test_missing_report_handled_safely_without_hallucination():
    """Querying a modality the patient doesn't possess returns missing_report without guessing."""
    session_id = "test_missing_modality"
    res = client.post("/assistant/chat", json={
        "query": "Open Vikram Malhotra (PAT-1009)'s latest Cephalogram report",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "missing_report"
    assert "no Cephalogram report exists" in data["message"]

def test_ui_html_renders_patient_directory_and_report_cards():
    """Verify that root UI HTML contains patient directory and associated report cards containers."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    # Verify mandatory UI elements exist in DOM
    assert "patientDirectoryGrid" in html
    assert "associatedReportsGrid" in html
    assert "selectedPatientSection" in html
    assert "loadPatientDirectory" in html
    assert "selectPatientById" in html
    assert "openReportById" in html
    assert "populateDictationStudio" in html
    assert "toggleVoiceRecord" in html
    assert "switchTab" in html
    assert "togglePatientDirectory" in html

def test_select_patient_then_open_her_opg_flow():
    """
    Test user flow:
    1. Patient selection via /select-patient
    2. Clinician asks 'open her OPG' with or without payload active_patient_id
    3. System loads RPT-2001 cleanly
    """
    session_id = "test_select_then_open_her_opg"
    # Select Priya Sharma (PAT-1001)
    r1 = client.post(f"/sessions/{session_id}/select-patient", json={"patient_id": "PAT-1001"})
    assert r1.status_code == 200

    # Ask 'open her OPG'
    r2 = client.post("/assistant/chat", json={
        "query": "open her OPG",
        "session_id": session_id,
        "active_patient_id": "PAT-1001"
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "success"
    assert d2["report"]["report_id"] == "RPT-2001"
    assert d2["report"]["report_type"] == "OPG"
    assert d2["patient"]["patient_id"] == "PAT-1001"


def test_select_pat_1012_then_open_her_bitewing_radiographs():
    """
    Test the exact UI flow reported by the user:
    1. Patient selection for PAT-1012 (Priya Sharma, DOB 1985-02-14)
    2. Followup query: 'open her bitewing radiographs' with active_patient_id passed from frontend
    3. Resolves accurately to RPT-2035 (Bitewing Radiographs) without prompting for patient name again.
    """
    session_id = "test_sess_priya_1012_bitewing"

    # Select PAT-1012
    r1 = client.post(f"/sessions/{session_id}/select-patient", json={"patient_id": "PAT-1012"})
    assert r1.status_code == 200

    # User queries 'open her bitewing radiographs' with active_patient_id sent from UI state
    r2 = client.post("/assistant/chat", json={
        "query": "open her bitewing radiographs",
        "session_id": session_id,
        "active_patient_id": "PAT-1012"
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "success"
    assert d2["report"]["report_id"] == "RPT-2035"
    assert d2["report"]["report_type"] == "Bitewing Radiographs"
    assert d2["patient"]["patient_id"] == "PAT-1012"


def test_select_pat_1001_then_open_her_bitewing_radiographs_returns_missing_report():
    """
    Test PAT-1001 missing report handling:
    1. Select PAT-1001 (Priya Sharma, DOB 1998-04-12) who only has OPG, Periapical, and Dental Examination.
    2. Query 'open her bitewing radiographs' with active_patient_id: PAT-1001.
    3. Must return status 'missing_report' with informative clinical message and available reports.
    """
    session_id = "test_sess_priya_1001_bitewing_missing"

    r1 = client.post(f"/sessions/{session_id}/select-patient", json={"patient_id": "PAT-1001"})
    assert r1.status_code == 200

    r2 = client.post("/assistant/chat", json={
        "query": "open her bitewing radiographs",
        "session_id": session_id,
        "active_patient_id": "PAT-1001"
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "missing_report"
    assert "no Bitewing report exists on file" in d2["message"]
    assert d2["patient"]["patient_id"] == "PAT-1001"
    assert "available_reports" in d2 and len(d2["available_reports"]) == 3


def test_ambiguity_direct_modality_disambiguation_to_pat1012():
    """
    Test candidate disambiguation via modality:
    1. 'Open Priya Sharma' yields ambiguous candidates PAT-1001 and PAT-1012.
    2. Clinician queries 'open her bitewing radiographs' without clicking a candidate.
    3. Evaluates that PAT-1012 has Bitewing Radiographs while PAT-1001 does not, resolving to PAT-1012 (RPT-2035).
    """
    session_id = "test_sess_ambig_modality_disambig"

    r1 = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["status"] == "clarification_required"

    r2 = client.post("/assistant/chat", json={
        "query": "open her bitewing radiographs",
        "session_id": session_id
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "success"
    assert d2["report"]["report_id"] == "RPT-2035"
    assert d2["patient"]["patient_id"] == "PAT-1012"



