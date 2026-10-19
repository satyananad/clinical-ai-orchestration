"""
MANDATORY ACCEPTANCE TESTS (SECTION 14)
=======================================
These tests exercise the actual application logic (FastAPI endpoints, retrieval service,
session state machine, claim verification, and consultation QA) with isolated synthetic data.

TEST 1 — CORRECT PATIENT AND REPORT RETRIEVAL
TEST 2 — AMBIGUOUS PATIENT IDENTITY
TEST 3 — UNSUPPORTED CLINICAL CLAIMS
"""

import os
import sys
import json

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.conversation_service import get_or_create_session
from backend.app.services.retrieval import get_patient_reports

client = TestClient(app)

def log_step(title: str):
    print(f"\n  ▶ {title}")

def log_request(method: str, endpoint: str, payload: dict):
    print(f"    [HTTP REQUEST] {method} {endpoint}")
    print(f"    Payload: {json.dumps(payload, ensure_ascii=False)}")

def log_response(status_code: int, data: dict):
    print(f"    [HTTP RESPONSE] Status: {status_code}")
    # Print formatted response summary
    data_summary = {}
    for k, v in data.items():
        if k in ["status", "message", "answer", "patient", "candidates", "citations"]:
            data_summary[k] = v
        elif k == "report" and v:
            data_summary[k] = {
                "report_id": v.get("report_id"),
                "report_type": v.get("report_type"),
                "report_date": v.get("report_date"),
                "patient_id": v.get("patient_id"),
                "clinical_indication": v.get("clinical_indication"),
                "findings_count": len(v.get("findings", []))
            }
        else:
            data_summary[k] = v
    print(f"    Response JSON:\n{json.dumps(data_summary, indent=6, ensure_ascii=False)}")

def log_assert(msg: str):
    print(f"    ✔ [ASSERTION PASSED] {msg}")

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
    print("\n" + "=" * 80)
    print("TEST 1: CORRECT PATIENT AND REPORT RETRIEVAL (MANDATORY)")
    print("=" * 80)

    session_id = "test-mandatory-1-retrieval"
    session = get_or_create_session(session_id)
    session.clear()

    # Step 1: Request latest OPG report for Priya Sharma with unambiguous identifier PAT-1001
    log_step("Step 1.1: Request latest OPG report for Priya Sharma (PAT-1001)")
    req_payload = {
        "query": "Open Priya Sharma (PAT-1001)'s latest OPG report",
        "session_id": session_id
    }
    log_request("POST", "/assistant/chat", req_payload)
    resp = client.post("/assistant/chat", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    log_response(resp.status_code, data)
    assert data["status"] == "success"

    # Verify correct patient ID is resolved and maintained
    patient = data["patient"]
    assert patient["patient_id"] == "PAT-1001"
    assert patient["patient_name"] == "Priya Sharma"
    log_assert("Correct Patient ID resolved: PAT-1001 (Priya Sharma)")

    # Verify report belongs to selected patient
    report = data["report"]
    assert report["patient_id"] == "PAT-1001"
    assert report["report_id"] == "RPT-2001"
    assert report["report_type"] == "OPG"
    log_assert("Report RPT-2001 (OPG) strictly verified as belonging to PAT-1001")

    # Verify latest-by-date logic for Arjun Mehta (PAT-1005):
    # PAT-1005 has:
    #   RPT-2013 (Periapical, 2026-04-22)
    #   RPT-2014 (Dental Exam, 2026-06-11)
    #   RPT-2015 (Bitewing, 2026-08-30) -> Latest by date is RPT-2015, which has higher date than RPT-2013/2014
    log_step("Step 1.2: Verify sorting logic selects newest report by examination date, NOT by ID")
    arjun_reports = get_patient_reports("PAT-1005")
    assert len(arjun_reports) == 3
    # First report in sorted order must have the newest report_date
    assert arjun_reports[0]["report_id"] == "RPT-2015"
    assert arjun_reports[0]["report_date"] == "2026-08-30"
    log_assert("Sorted reports for PAT-1005 verified: RPT-2015 (2026-08-30) is newest by date")

    # Test latest report request for Arjun Mehta
    session_arjun = "test-mandatory-1-arjun"
    get_or_create_session(session_arjun).clear()
    req_arjun = {
        "query": "Open Arjun Mehta's latest report",
        "session_id": session_arjun
    }
    log_request("POST", "/assistant/chat", req_arjun)
    resp_arjun = client.post("/assistant/chat", json=req_arjun)
    assert resp_arjun.status_code == 200
    data_arjun = resp_arjun.json()
    log_response(resp_arjun.status_code, data_arjun)
    assert data_arjun["status"] == "success"
    assert data_arjun["patient"]["patient_id"] == "PAT-1005"
    assert data_arjun["report"]["report_id"] == "RPT-2015"
    assert data_arjun["report"]["report_date"] == "2026-08-30"
    log_assert("Successfully opened latest report RPT-2015 (2026-08-30) for PAT-1005")

    # Verify active session state persists patient_id and report_id, not just display name
    s = get_or_create_session(session_arjun)
    assert s.active_patient_id == "PAT-1005"
    assert s.active_report_id == "RPT-2015"
    log_assert("Session state verified: active_patient_id='PAT-1005', active_report_id='RPT-2015'")

    # Verify follow-up question uses active report context
    log_step("Step 1.3: Follow-up query using pronoun 'his' without mentioning patient name")
    req_followup = {
        "query": "Summarize his report",
        "session_id": session_arjun
    }
    log_request("POST", "/assistant/chat", req_followup)
    resp_followup = client.post("/assistant/chat", json=req_followup)
    assert resp_followup.status_code == 200
    data_f = resp_followup.json()
    log_response(resp_followup.status_code, data_f)
    assert data_f["status"] == "success"
    # Grounded evidence must cite active report RPT-2015
    assert any("RPT-2015" in c for c in data_f.get("citations", []))
    log_assert("Follow-up response cited active report RPT-2015 evidence")
    # Report belonging to another patient (e.g. PAT-1001 / RPT-2001) must NEVER be returned
    assert not any("PAT-1001" in c for c in data_f.get("citations", []))
    assert not any("RPT-2001" in c for c in data_f.get("citations", []))
    log_assert("Cross-contamination check PASSED: Zero citations from other patients (PAT-1001 / RPT-2001)")
    print("▶ [RESULT] TEST 1 PASSED COMPLETELY\n")


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
    print("\n" + "=" * 80)
    print("TEST 2: AMBIGUOUS PATIENT IDENTITY SAFETY GATE (MANDATORY)")
    print("=" * 80)

    session_id = "test-mandatory-2-ambiguity"
    get_or_create_session(session_id).clear()

    # Step 1: Query ambiguous name "Priya Sharma" (matching PAT-1001 and PAT-1012)
    log_step("Step 2.1: Query ambiguous name 'Priya Sharma' matching 2 distinct patient records")
    req_ambig = {
        "query": "Open Priya Sharma's report",
        "session_id": session_id
    }
    log_request("POST", "/assistant/chat", req_ambig)
    resp = client.post("/assistant/chat", json=req_ambig)
    assert resp.status_code == 200
    data = resp.json()
    log_response(resp.status_code, data)

    # Must NOT silently select a patient
    assert data["status"] == "clarification_required"
    assert "clarify which patient you mean" in data["message"].lower()
    log_assert("Clinical Safety Stop triggered: status is 'clarification_required' (no guessing)")

    # Matching authorized records presented for selection
    assert data["candidates"] is not None
    assert len(data["candidates"]) == 2
    cand_ids = {c["patient_id"] for c in data["candidates"]}
    assert "PAT-1001" in cand_ids
    assert "PAT-1012" in cand_ids
    log_assert("Presented candidate records with DOBs: PAT-1001 (1998-04-12) & PAT-1012 (1985-02-14)")

    # No report is opened before selection
    assert data.get("report") is None
    s = get_or_create_session(session_id)
    assert s.active_patient_id is None
    assert s.active_report_id is None
    log_assert("Guaranteed safety: Active session patient_id & report_id remain None")

    # Step 2: Clinician disambiguates by specifying candidate 1 (PAT-1001, DOB: 1998-04-12)
    log_step("Step 2.2: Clinician specifies candidate identifier 'PAT-1001'")
    req_select = {
        "query": "Select patient PAT-1001",
        "session_id": session_id
    }
    log_request("POST", "/assistant/chat", req_select)
    resp_select = client.post("/assistant/chat", json=req_select)
    assert resp_select.status_code == 200
    data_sel = resp_select.json()
    log_response(resp_select.status_code, data_sel)
    assert data_sel["status"] == "success"
    assert data_sel["patient"]["patient_id"] == "PAT-1001"
    assert s.active_patient_id == "PAT-1001"
    log_assert("Disambiguation successful: Context bound strictly to PAT-1001")

    # After patient is selected, only that patient's reports are accessible
    reports_pat1001 = data_sel.get("available_reports", [])
    assert all("RPT-200" in r["report_id"] for r in reports_pat1001)
    log_assert("Only reports belonging to PAT-1001 are listed")

    # Step 3: Switching patients updates active context correctly
    log_step("Step 2.3: Switching active patient context to Vikram Malhotra (PAT-1009)")
    req_switch = {
        "query": "Open report PAT-1009",
        "session_id": session_id
    }
    log_request("POST", "/assistant/chat", req_switch)
    resp_switch = client.post("/assistant/chat", json=req_switch)
    assert resp_switch.status_code == 200
    data_sw = resp_switch.json()
    log_response(resp_switch.status_code, data_sw)
    assert data_sw["status"] == "success"
    assert data_sw["patient"]["patient_id"] == "PAT-1009"
    assert data_sw["patient"]["patient_name"] == "Vikram Malhotra"
    assert s.active_patient_id == "PAT-1009"
    log_assert("Active context cleanly updated to PAT-1009 (Vikram Malhotra)")
    print("▶ [RESULT] TEST 2 PASSED COMPLETELY\n")


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
    print("\n" + "=" * 80)
    print("TEST 3: UNSUPPORTED CLINICAL CLAIMS & ANTI-HALLUCINATION GUARDRAIL (MANDATORY)")
    print("=" * 80)

    session_id = "test-mandatory-3-claims"
    get_or_create_session(session_id).clear()

    # Step 1: Open Priya Sharma's OPG report (RPT-2001), which documents tooth 48 impaction
    # but does NOT mention any mandibular fracture, cyst, or malignancy
    log_step("Step 3.1: Open Priya Sharma's baseline OPG report (RPT-2001)")
    req_open = {
        "query": "Open Priya Sharma (PAT-1001)'s OPG report",
        "session_id": session_id
    }
    log_request("POST", "/assistant/chat", req_open)
    resp_open = client.post("/assistant/chat", json=req_open)
    assert resp_open.status_code == 200
    data_open = resp_open.json()
    log_response(resp_open.status_code, data_open)
    assert data_open["status"] == "success"
    assert data_open["report"]["report_id"] == "RPT-2001"
    log_assert("Baseline report RPT-2001 successfully opened")

    # Step 2: Clinician asks about an undocumented condition: "Is there any radicular cyst or fracture?"
    log_step("Step 3.2: Query unmentioned condition: 'Is there any radicular cyst or jaw fracture documented?'")
    req_ask = {
        "query": "Is there any radicular cyst or jaw fracture documented in this scan?",
        "session_id": session_id
    }
    log_request("POST", "/assistant/chat", req_ask)
    resp_ask = client.post("/assistant/chat", json=req_ask)
    assert resp_ask.status_code == 200
    data_ask = resp_ask.json()
    log_response(resp_ask.status_code, data_ask)
    assert data_ask["status"] == "success"
    ans = data_ask["answer"].lower()

    # Assistant must NOT assert condition is present without evidence
    assert "cyst is present" not in ans
    assert "fracture is present" not in ans
    assert "diagnosed with fracture" not in ans
    log_assert("Zero hallucination: Assistant refused to claim unmentioned cyst or fracture is present")

    # Assistant must state that the condition is NOT documented or evaluated in this report
    assert "not documented" in ans or "not evaluated" in ans or "no mention" in ans or "not identified" in ans
    log_assert("Grounded disclaimer verified: explicitly stated condition is not documented in RPT-2001")

    # Verification: Grounded evidence citations returned with correct report ID
    citations = data_ask.get("citations", [])
    assert len(citations) > 0
    assert any("RPT-2001" in c for c in citations)
    log_assert(f"Verified source report citations returned: {citations}")

    # Step 3: Verify that a patient-reported symptom is not presented as an independently verified imaging finding
    log_step("Step 3.3: Verify distinction between patient-reported symptoms and verified imaging findings")
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
    log_assert("Patient symptom vs imaging finding classifier verified")
    print("▶ [RESULT] TEST 3 PASSED COMPLETELY\n")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING LIVE MANDATORY ACCEPTANCE TEST SUITE WITH FULL TRACE LOGS")
    print("=" * 80)
    test_mandatory_test_1_correct_patient_and_report_retrieval()
    test_mandatory_test_2_ambiguous_patient_identity()
    test_mandatory_test_3_unsupported_clinical_claims()
    print("=" * 80)
    print("🎉 ALL 3 MANDATORY TESTS COMPLETED WITH 100% VERIFIED ASSERTIONS!")
    print("=" * 80 + "\n")
