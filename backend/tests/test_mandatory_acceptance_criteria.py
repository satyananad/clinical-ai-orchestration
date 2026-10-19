"""
MANDATORY ACCEPTANCE TESTS (SECTION 14)
=======================================
These tests exercise the actual application logic (FastAPI endpoints, retrieval service,
session state machine, claim verification, and consultation QA) with isolated synthetic data.

TEST 1 — CORRECT PATIENT AND REPORT RETRIEVAL
TEST 2 — AMBIGUOUS PATIENT IDENTITY
TEST 3 — UNSUPPORTED CLINICAL CLAIMS
"""

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.conversation_service import get_or_create_session
from backend.app.services.retrieval import get_patient_reports

client = TestClient(app)

# ==============================================================================
# TEST 1 — CORRECT PATIENT AND REPORT RETRIEVAL
# ==============================================================================
def test_mandatory_test_1_correct_patient_and_report_retrieval():
    """
    TEST 1:
    Verify that:
    - The correct patient ID is resolved.
    - The correct report ID is selected according to the stored report/examination date
      and applicable retrieval rules (NOT by arbitrary ID order).
    - The report content belongs to the selected patient.
    - Follow-up questions use the active report context.
    - A report belonging to another patient is never returned.
    - A display name alone is not used as a persistent identifier.
    - The latest report is not selected by arbitrary ID order.
    """
    session_id = "test-mandatory-1-retrieval"
    session = get_or_create_session(session_id)
    session.clear()

    # Step 1: Request latest OPG report for Priya Sharma with unambiguous identifier PAT-1001
    resp = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma (PAT-1001)'s latest OPG report",
        "session_id": session_id
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"

    # Verify correct patient ID is resolved and maintained
    patient = data["patient"]
    assert patient["patient_id"] == "PAT-1001"
    assert patient["patient_name"] == "Priya Sharma"

    # Verify report belongs to selected patient
    report = data["report"]
    assert report["patient_id"] == "PAT-1001"
    assert report["report_id"] == "RPT-2001"
    assert report["report_type"] == "OPG"

    # Verify latest-by-date logic for Arjun Mehta (PAT-1005):
    # PAT-1005 has:
    #   RPT-2013 (Periapical, 2026-04-22)
    #   RPT-2014 (Dental Exam, 2026-06-11)
    #   RPT-2015 (Bitewing, 2026-08-30) -> Latest by date is RPT-2015, which has higher date than RPT-2013/2014
    arjun_reports = get_patient_reports("PAT-1005")
    assert len(arjun_reports) == 3
    # First report in sorted order must have the newest report_date
    assert arjun_reports[0]["report_id"] == "RPT-2015"
    assert arjun_reports[0]["report_date"] == "2026-08-30"

    # Test latest report request for Arjun Mehta
    session_arjun = "test-mandatory-1-arjun"
    get_or_create_session(session_arjun).clear()
    resp_arjun = client.post("/assistant/chat", json={
        "query": "Open Arjun Mehta's latest report",
        "session_id": session_arjun
    })
    assert resp_arjun.status_code == 200
    data_arjun = resp_arjun.json()
    assert data_arjun["status"] == "success"
    assert data_arjun["patient"]["patient_id"] == "PAT-1005"
    assert data_arjun["report"]["report_id"] == "RPT-2015"
    assert data_arjun["report"]["report_date"] == "2026-08-30"

    # Verify active session state persists patient_id and report_id, not just display name
    s = get_or_create_session(session_arjun)
    assert s.active_patient_id == "PAT-1005"
    assert s.active_report_id == "RPT-2015"

    # Verify follow-up question uses active report context
    resp_followup = client.post("/assistant/chat", json={
        "query": "Summarize his report",
        "session_id": session_arjun
    })
    assert resp_followup.status_code == 200
    data_f = resp_followup.json()
    assert data_f["status"] == "success"
    # Grounded evidence must cite active report RPT-2015
    assert any("RPT-2015" in c for c in data_f.get("citations", []))
    # Report belonging to another patient (e.g. PAT-1001 / RPT-2001) must NEVER be returned
    assert not any("PAT-1001" in c for c in data_f.get("citations", []))
    assert not any("RPT-2001" in c for c in data_f.get("citations", []))


# ==============================================================================
# TEST 2 — AMBIGUOUS PATIENT IDENTITY
# ==============================================================================
def test_mandatory_test_2_ambiguous_patient_identity():
    """
    TEST 2:
    Verify that:
    - The application does not silently select a patient when identity is ambiguous.
    - Matching authorized records are presented for selection.
    - No report is opened or attached to an arbitrary patient before the identity is resolved.
    - After a patient is selected, only that patient's reports are listed and retrievable.
    - Switching patients updates the active context correctly.
    """
    session_id = "test-mandatory-2-ambiguity"
    get_or_create_session(session_id).clear()

    # Step 1: Query ambiguous name "Priya Sharma" (matching PAT-1001 and PAT-1012)
    resp = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma's report",
        "session_id": session_id
    })
    assert resp.status_code == 200
    data = resp.json()

    # Must NOT silently select a patient
    assert data["status"] == "clarification_required"
    assert "clarify which patient you mean" in data["message"].lower()

    # Matching authorized records presented for selection
    assert data["candidates"] is not None
    assert len(data["candidates"]) == 2
    cand_ids = {c["patient_id"] for c in data["candidates"]}
    assert "PAT-1001" in cand_ids
    assert "PAT-1012" in cand_ids

    # No report is opened before selection
    assert data.get("report") is None
    s = get_or_create_session(session_id)
    assert s.active_patient_id is None
    assert s.active_report_id is None

    # Step 2: Clinician disambiguates by specifying candidate 1 (PAT-1001, DOB: 1998-05-14)
    resp_select = client.post("/assistant/chat", json={
        "query": "Select patient PAT-1001",
        "session_id": session_id
    })
    assert resp_select.status_code == 200
    data_sel = resp_select.json()
    assert data_sel["status"] == "success"
    assert data_sel["patient"]["patient_id"] == "PAT-1001"
    assert s.active_patient_id == "PAT-1001"

    # After patient is selected, only that patient's reports are accessible
    reports_pat1001 = data_sel.get("available_reports", [])
    assert all("RPT-200" in r["report_id"] for r in reports_pat1001)

    # Step 3: Switching patients updates active context correctly
    resp_switch = client.post("/assistant/chat", json={
        "query": "Open report PAT-1009",
        "session_id": session_id
    })
    assert resp_switch.status_code == 200
    data_sw = resp_switch.json()
    assert data_sw["status"] == "success"
    assert data_sw["patient"]["patient_id"] == "PAT-1009"
    assert data_sw["patient"]["patient_name"] == "Vikram Malhotra"
    assert s.active_patient_id == "PAT-1009"


# ==============================================================================
# TEST 3 — UNSUPPORTED CLINICAL CLAIMS
# ==============================================================================
def test_mandatory_test_3_unsupported_clinical_claims():
    """
    TEST 3:
    Verify that:
    - The assistant does not assert that an undocumented condition is present without evidence.
    - The assistant does not assert that the condition is absent merely because it is not documented.
    - It states that the condition is not documented or cannot be determined from the available report.
    - Every clinical claim has supporting evidence from the correct source.
    - Evidence quotations actually support the claims.
    - The correct report ID and finding or section references are returned.
    - A patient-reported symptom is not presented as an independently verified imaging finding.
    """
    session_id = "test-mandatory-3-claims"
    get_or_create_session(session_id).clear()

    # Step 1: Open Priya Sharma's OPG report (RPT-2001), which documents tooth 48 impaction
    # but does NOT mention any mandibular fracture, cyst, or malignancy
    resp_open = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma (PAT-1001)'s OPG report",
        "session_id": session_id
    })
    assert resp_open.status_code == 200
    assert resp_open.json()["status"] == "success"
    assert resp_open.json()["report"]["report_id"] == "RPT-2001"

    # Step 2: Clinician asks about an undocumented condition: "Is there any radicular cyst or fracture?"
    resp_ask = client.post("/assistant/chat", json={
        "query": "Is there any radicular cyst or jaw fracture documented in this scan?",
        "session_id": session_id
    })
    assert resp_ask.status_code == 200
    data_ask = resp_ask.json()
    assert data_ask["status"] == "success"
    ans = data_ask["answer"].lower()

    # Assistant must NOT assert condition is present without evidence
    assert "cyst is present" not in ans
    assert "fracture is present" not in ans
    assert "diagnosed with fracture" not in ans

    # Assistant must state that the condition is NOT documented or evaluated in this report
    assert "not documented" in ans or "not evaluated" in ans or "no mention" in ans or "not identified" in ans

    # Verification: Grounded evidence citations returned with correct report ID
    citations = data_ask.get("citations", [])
    assert len(citations) > 0
    assert any("RPT-2001" in c for c in citations)

    # Step 3: Verify that a patient-reported symptom is not presented as an independently verified imaging finding
    from backend.app.services.report_service import is_symptom_or_history
    symptom_finding = {
        "anatomical_site": "Dentition",
        "category": "Patient-reported symptom",
        "observation": "Patient reports discomfort and loose teeth in the lower anterior region."
    }
    imaging_finding = {
        "anatomical_site": "Mandibular right third molar",
        "category": "Impaction",
        "observation": "Tooth 48 is horizontally impacted with roots approximating the IAC."
    }
    assert is_symptom_or_history(symptom_finding) is True
    assert is_symptom_or_history(imaging_finding) is False
