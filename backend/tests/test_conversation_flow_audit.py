"""
Tests for Conversation Flow & Grounded Report Consultation (test_conversation_flow_audit.py)
Verifies:
1. "can you list out only patient name" returns registered patient directory.
2. "how many report have you" returns database report statistics.
3. Multi-turn disambiguation (Priya -> Priya Sharma -> DOB: 1998) opens RPT-2003.
4. "what problem it have" returns actual clinical report findings & impressions, NEVER assistant software bugs.
5. Multi-report switching for active patient (e.g. "Open her OPG report").
"""

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.conversation_service import get_or_create_session

client = TestClient(app)

def test_patient_list_and_report_count_queries():
    session_id = "test-flow-query-session"
    get_or_create_session(session_id).clear()

    # 1. User asks "can you list out only patient name"
    resp = client.post("/assistant/chat", json={"query": "can you list out only patient name", "session_id": session_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "registered patients in the clinic database" in data["answer"].lower()
    assert "Priya Sharma" in data["answer"]
    assert "Vikram Malhotra" in data["answer"]
    assert len(data.get("citations", [])) > 0

    # 2. User asks "how manu patient report have you"
    resp2 = client.post("/assistant/chat", json={"query": "how manu patient report have you", "session_id": session_id})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "success"
    assert "reports across" in data2["answer"].lower()
    assert "12 registered patients" in data2["answer"]


def test_full_user_conversation_turn_by_turn():
    """
    Tests the exact sequence from the clinician's session:
    1. priya -> 4 candidates
    2. priya sharma -> 2 candidates
    3. DOB : 1998 -> opens Dental Examination report RPT-2003 with available_reports
    4. what problem it have -> returns clinical summary from RPT-2003, NOT AI system problems!
    5. Open her OPG -> switches to RPT-2001
    """
    session_id = "test-clinician-flow-turn-by-turn"
    get_or_create_session(session_id).clear()

    # Turn 1: "priya"
    r1 = client.post("/assistant/chat", json={"query": "priya", "session_id": session_id}).json()
    assert r1["status"] == "clarification_required"
    assert len(r1["candidates"]) == 4

    # Turn 2: "priya sharma"
    r2 = client.post("/assistant/chat", json={"query": "priya sharma", "session_id": session_id}).json()
    assert r2["status"] == "clarification_required"
    assert len(r2["candidates"]) == 2

    # Turn 3: "DOB : 1998"
    r3 = client.post("/assistant/chat", json={"query": "DOB : 1998", "session_id": session_id}).json()
    assert r3["status"] == "success"
    assert r3["report"]["report_id"] == "RPT-2003"
    assert r3["patient"]["patient_name"] == "Priya Sharma"
    assert len(r3.get("available_reports", [])) == 3
    assert any(ar["report_id"] == "RPT-2001" for ar in r3["available_reports"])

    # Turn 4: "what problem it have"
    r4 = client.post("/assistant/chat", json={"query": "what problem it have", "session_id": session_id}).json()
    assert r4["status"] == "success"
    ans4 = r4["answer"].lower()
    # MUST contain actual clinical impressions/findings
    assert "caries" in ans4 or "hypersensitivity" in ans4 or "gingivitis" in ans4 or "tooth 16" in ans4
    # MUST NOT contain AI software bugs or hallucinated assistant limitations
    assert "security vulnerabilities" not in ans4
    assert "training data" not in ans4
    assert "integration issues" not in ans4
    assert len(r4["citations"]) > 0
    assert any("RPT-2003" in c for c in r4["citations"])

    # Turn 5: "Open her OPG" (seamless report switching for active patient)
    r5 = client.post("/assistant/chat", json={"query": "Open her OPG", "session_id": session_id}).json()
    assert r5["status"] == "success"
    assert r5["report"]["report_id"] == "RPT-2001"
    assert r5["report"]["report_type"] == "OPG"

    # Turn 6: "what does the scan show"
    r6 = client.post("/assistant/chat", json={"query": "what does the scan show", "session_id": session_id}).json()
    assert r6["status"] == "success"
    assert any("RPT-2001" in c for c in r6["citations"])
