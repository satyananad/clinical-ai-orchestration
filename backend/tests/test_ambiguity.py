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

