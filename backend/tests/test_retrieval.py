from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_successful_patient_search():
    response = client.get("/patients?query=Vikram")
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) >= 1
    assert patients[0]["patient_id"] == "PAT-1009"
    assert "Vikram Malhotra" in patients[0]["patient_name"]

def test_successful_report_retrieval():
    # Retrieve report for Priya Sharma via assistant endpoint
    response = client.post("/assistant/chat", json={"query": "Open Priya Sharma's latest OPG report"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Priya Sharma"
    assert data["patient"]["patient_id"] == "PAT-1001"
    assert data["report"]["report_id"] == "RPT-2001"
    assert data["report"]["report_type"] == "OPG"

def test_missing_report_handling():
    # Request a Cephalogram for Priya Sharma who only has OPG, Periapical, and Dental Exam
    response = client.post("/assistant/chat", json={"query": "Open Priya Sharma's latest Cephalogram report"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "missing_report"
    assert "no Cephalogram report exists" in data["message"]

def test_nonexistent_patient_handling():
    response = client.post("/assistant/chat", json={"query": "Open Nonexistent Person's latest OPG report"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_found"
    assert "No patient records found" in data["message"]
