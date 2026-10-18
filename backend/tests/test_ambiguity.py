from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_ambiguous_patient_identity_triggers_clarification():
    """
    Safety Critical Test:
    When a clinician queries 'Priya' (matching Priya Sharma, Priya Sharmila, and Priya Nair),
    the system must NOT guess or open a random file. It must return status 'clarification_required'
    and return candidate identities with dates of birth and IDs.
    """
    response = client.post("/assistant/chat", json={"query": "Open Priya's latest OPG report"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "clarification_required"
    assert "clarify which patient you mean" in data["message"]
    assert data["candidates"] is not None
    assert len(data["candidates"]) == 3

    candidate_names = {c["patient_name"] for c in data["candidates"]}
    assert "Priya Sharma" in candidate_names
    assert "Priya Sharmila" in candidate_names
    assert "Priya Nair" in candidate_names

def test_disambiguation_with_full_name_succeeds():
    """
    Subsequent query providing full name resolves unambiguously.
    """
    response = client.post("/assistant/chat", json={"query": "Open Priya Sharmila's latest OPG report"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Priya Sharmila"
    assert data["patient"]["patient_id"] == "PAT-1002"
    assert data["report"]["report_id"] == "RPT-2006"

def test_surname_ambiguity_triggers_clarification():
    """
    Safety Critical Test:
    When a clinician queries 'Sharma' (matching Priya Sharma and Rohan Sharma),
    the system must NOT guess. It must return status 'clarification_required'
    and return candidate identities for both Priya Sharma and Rohan Sharma.
    """
    response = client.post("/assistant/chat", json={"query": "Open Sharma's latest OPG report"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "clarification_required"
    assert data["candidates"] is not None
    assert len(data["candidates"]) == 2

    candidate_names = {c["patient_name"] for c in data["candidates"]}
    assert "Priya Sharma" in candidate_names
    assert "Rohan Sharma" in candidate_names

def test_patient_id_direct_resolution_succeeds():
    """
    Clinician specifies patient ID directly (e.g. 'Open report PAT-1001').
    The system must resolve that exact patient without unnecessary clarification.
    """
    response = client.post("/assistant/chat", json={
        "query": "Open report PAT-1001",
        "session_id": "session_id_direct"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["patient"]["patient_id"] == "PAT-1001"
    assert data["patient"]["patient_name"] == "Priya Sharma"

def test_disambiguation_select_the_second_patient():
    """
    Multi-Turn Safety Flow:
    1. Ambiguous query ('Open Priya's latest OPG report') returns candidate list.
    2. Clinician replies 'Select the second patient'.
    3. System validates selection against pending candidates and opens report for candidate #2 (Priya Sharmila).
    """
    session_id = "session_ordinal_test"
    # Turn 1: Ambiguous query
    res1 = client.post("/assistant/chat", json={
        "query": "Open Priya's latest OPG report",
        "session_id": session_id
    })
    assert res1.json()["status"] == "clarification_required"
    assert len(res1.json()["candidates"]) == 3

    # Turn 2: Ordinal selection
    res2 = client.post("/assistant/chat", json={
        "query": "Select the second patient",
        "session_id": session_id
    })
    data = res2.json()
    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Priya Sharmila"
    assert data["patient"]["patient_id"] == "PAT-1002"
    assert data["report"]["report_id"] == "RPT-2006"

def test_disambiguation_select_invalid_candidate_rejected():
    """
    Safety Gate:
    When clarification is pending and clinician requests an out-of-range or invalid candidate
    (e.g. 'Select the fifth patient' when only 3 exist), the system must reject the selection
    and maintain the clarification gate.
    """
    session_id = "session_invalid_test"
    # Turn 1: Ambiguous query
    res1 = client.post("/assistant/chat", json={
        "query": "Open Priya's latest OPG report",
        "session_id": session_id
    })
    assert res1.json()["status"] == "clarification_required"

    # Turn 2: Invalid candidate request
    res2 = client.post("/assistant/chat", json={
        "query": "Select the fifth patient",
        "session_id": session_id
    })
    data = res2.json()
    assert data["status"] == "clarification_required"
    assert "Invalid candidate selection" in data["message"]
    assert data["candidates"] is not None
    assert len(data["candidates"]) == 3

def test_identical_full_names_safety_gate(monkeypatch):
    """
    Safety Critical Test:
    Two different patients have exactly the same first and last name:
    - Priya Sharma (PAT-1001, DOB: 1998-04-12)
    - Priya Sharma (PAT-1012, DOB: 1985-02-14)
    When clinician asks 'Open Priya Sharma's report', the system MUST NOT guess based on name alone.
    It must require clarification and present both candidates with DOBs and IDs.
    """
    import backend.app.services.retrieval as ret_mod

    # Create synthetic dataset with identical full names
    synthetic_patients = [
        {
            "patient_id": "PAT-1001",
            "patient_name": "Priya Sharma",
            "date_of_birth": "1998-04-12",
            "sex": "Female",
            "reports": [{"report_id": "RPT-2001", "report_type": "OPG", "report_date": "2026-06-12"}]
        },
        {
            "patient_id": "PAT-1012",
            "patient_name": "Priya Sharma",
            "date_of_birth": "1985-02-14",
            "sex": "Female",
            "reports": [{"report_id": "RPT-2035", "report_type": "OPG", "report_date": "2026-09-10"}]
        }
    ]
    monkeypatch.setattr(ret_mod, "PATIENTS_DATA", synthetic_patients)

    res = ret_mod.resolve_report_retrieval("Open Priya Sharma's latest OPG report")
    assert res["status"] == "clarification_required"
    assert len(res["candidates"]) == 2
    assert res["candidates"][0]["patient_id"] == "PAT-1001"
    assert res["candidates"][1]["patient_id"] == "PAT-1012"
    assert res["candidates"][0]["date_of_birth"] == "1998-04-12"
    assert res["candidates"][1]["date_of_birth"] == "1985-02-14"


