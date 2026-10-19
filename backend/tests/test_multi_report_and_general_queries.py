"""
TEST SUITE FOR MULTI-REPORT CONSULTATION & GENERAL QUERY HANDLING
-----------------------------------------------------------------
Verifies:
1. General Query Independence:
   - General dental radiology concept inquiries ("What is a CBCT scan?", "What is an OPG?", "Explain caries")
     are answered clearly and educational concepts explained, WITHOUT being hijacked by an active report or returning
     "Not documented in report RPT-xxxx".
   - Workstation capability questions ("Who are you?", "What can you do?", "Help") describe workstation functions.
   - Polite conversational greetings ("Hello", "Good morning") are answered courteously.
2. Multi-Report Answering across a Patient's Records:
   - When a patient has multiple diagnostic reports (e.g. Priya Sharma PAT-1001 with RPT-2001, RPT-2002, RPT-2003):
     - The agent can switch between reports via natural language ("Open her periapical report", "Open her latest report").
     - The agent can answer clinical questions targeting another modality on file (e.g. "What did her periapical scan show?").
     - The agent can answer questions about teeth evaluated in other reports (e.g. Tooth 25 evaluated in RPT-2002 and RPT-2003)
       even when OPG (RPT-2001) is currently active.
     - The agent can synthesize findings across all reports ("Summarize all her reports", "Does she have caries across her reports?"),
       providing source citations for each report ID.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.conversation_service import get_or_create_session, SESSIONS
from backend.app.services.orchestrator_agent import triage_clinician_request, run_multi_agent_pipeline

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_general_queries_not_hijacked_by_active_report():
    """
    When an active patient report is loaded, general queries must NOT be evaluated
    against the patient's report or return 'Not documented in report RPT-2001'.
    """
    session_id = "test_general_active_report"
    # 1. Open Priya Sharma's OPG report
    open_res = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma PAT-1001 OPG report",
        "session_id": session_id
    })
    assert open_res.status_code == 200
    assert open_res.json()["status"] == "success"

    # Verify session has active report
    session = get_or_create_session(session_id)
    assert session.active_report_id == "RPT-2001"

    # 2. General Query: "What is a CBCT scan?"
    cbct_res = client.post("/assistant/chat", json={
        "query": "What is a CBCT scan?",
        "session_id": session_id
    })
    assert cbct_res.status_code == 200
    data_cbct = cbct_res.json()
    assert data_cbct["status"] == "success"
    # Must NOT claim that CBCT is not documented in Priya's OPG report
    assert "not documented in" not in data_cbct["answer"].lower()
    assert any(term in data_cbct["answer"].lower() for term in ["cone beam", "3d", "volumetric", "tomography"])

    # 3. General Query: "What is an OPG?"
    opg_res = client.post("/assistant/chat", json={
        "query": "What is an OPG?",
        "session_id": session_id
    })
    assert opg_res.status_code == 200
    data_opg = opg_res.json()
    assert data_opg["status"] == "success"
    assert "not documented in" not in data_opg["answer"].lower()
    assert any(term in data_opg["answer"].lower() for term in ["panoramic", "orthopantomogram", "rotational", "radiograph"])

    # 4. General Query: "Who are you?"
    who_res = client.post("/assistant/chat", json={
        "query": "Who are you?",
        "session_id": session_id
    })
    assert who_res.status_code == 200
    data_who = who_res.json()
    assert data_who["status"] == "success"
    assert any(term in data_who["answer"].lower() for term in ["workstation", "assistant", "viscan", "radiology"])

    # 5. General Greeting: "Hello"
    hello_res = client.post("/assistant/chat", json={
        "query": "Hello",
        "session_id": session_id
    })
    assert hello_res.status_code == 200
    assert hello_res.json()["status"] == "success"


def test_multi_report_switching_for_active_patient():
    """
    Clinician can switch between a patient's multiple reports using natural language.
    """
    session_id = "test_multi_switch"
    # 1. Open Priya Sharma's OPG
    client.post("/assistant/chat", json={
        "query": "Open Priya Sharma PAT-1001 OPG report",
        "session_id": session_id
    })
    session = get_or_create_session(session_id)
    assert session.active_report_id == "RPT-2001"

    # 2. Switch to her periapical report
    switch_res = client.post("/assistant/chat", json={
        "query": "Open her periapical report",
        "session_id": session_id
    })
    assert switch_res.status_code == 200
    data_sw = switch_res.json()
    assert data_sw["status"] == "success"
    assert "RPT-2002" in data_sw["message"] or "Periapical" in data_sw["message"]
    assert session.active_report_id == "RPT-2002"

    # 3. Switch to her latest report
    latest_res = client.post("/assistant/chat", json={
        "query": "Open her latest report",
        "session_id": session_id
    })
    assert latest_res.status_code == 200
    data_latest = latest_res.json()
    assert data_latest["status"] == "success"
    assert session.active_report_id == "RPT-2003"


def test_cross_report_tooth_answering_when_different_report_active():
    """
    When OPG (RPT-2001) is active, querying Tooth 25 (evaluated in RPT-2002 & RPT-2003)
    must find and present the findings from her other reports rather than merely failing.
    """
    session_id = "test_cross_tooth"
    client.post("/assistant/chat", json={
        "query": "Open Priya Sharma PAT-1001 OPG report",
        "session_id": session_id
    })
    session = get_or_create_session(session_id)
    assert session.active_report_id == "RPT-2001"

    # Ask about Tooth 25 (which is in RPT-2002 and RPT-2003, not RPT-2001)
    res = client.post("/assistant/chat", json={
        "query": "What about tooth 25?",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    ans = data["answer"]
    # Notes that Tooth 25 is documented in other records (Periapical / Dental Examination)
    assert "Tooth 25" in ans
    assert any(w in ans for w in ["RPT-2002", "RPT-2003", "Periapical", "Dental Examination"])
    assert len(data["citations"]) >= 1


def test_cross_report_modality_specific_question():
    """
    When OPG is active, asking 'What did her periapical scan show?' must retrieve
    and answer from RPT-2002 with source citations.
    """
    session_id = "test_cross_modality"
    client.post("/assistant/chat", json={
        "query": "Open Priya Sharma PAT-1001 OPG report",
        "session_id": session_id
    })

    # Ask specifically about her periapical scan
    res = client.post("/assistant/chat", json={
        "query": "What did her periapical radiograph show?",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    ans = data["answer"]
    assert any(term in ans for term in ["Tooth 25", "lamina dura", "periapical", "RPT-2002"])
    assert any("RPT-2002" in c for c in data["citations"])


def test_summarize_all_reports_for_patient():
    """
    Clinician asks 'Summarize all her reports' -> synthesizes across all 3 reports with citations.
    """
    session_id = "test_summarize_all"
    client.post("/assistant/chat", json={
        "query": "Open Priya Sharma PAT-1001 OPG report",
        "session_id": session_id
    })

    res = client.post("/assistant/chat", json={
        "query": "Summarize all her reports",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    ans = data["answer"]
    # Must synthesize findings across RPT-2001, RPT-2002, and RPT-2003
    assert "RPT-2001" in ans
    assert "RPT-2002" in ans
    assert "RPT-2003" in ans
    assert len(data["citations"]) >= 3


def test_caries_assessment_across_all_reports():
    """
    Clinician asks 'Does she have caries across her reports?' -> identifies Tooth 16 caries
    across RPT-2001 and RPT-2003.
    """
    session_id = "test_caries_across"
    client.post("/assistant/chat", json={
        "query": "Open Priya Sharma PAT-1001 OPG report",
        "session_id": session_id
    })

    res = client.post("/assistant/chat", json={
        "query": "Does she have caries across her reports?",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    ans = data["answer"]
    assert "16" in ans
    assert "tooth 16" in ans.lower() or "teeth 16" in ans.lower()
    assert any(w in ans for w in ["RPT-2001", "RPT-2003", "caries", "radiolucency"])


def test_greeting_prefaced_retrieval_arjun_mehta():
    """
    Clinician uses natural greeting preface: 'hey i want to open the report of arjun mehta'.
    Must directly retrieve Arjun Mehta (PAT-1005), open his report, and list associated reports.
    Must NEVER reject with security verification or ask for DOB/ID when patient is unambiguous.
    """
    session_id = "test_greeting_arjun"
    res = client.post("/assistant/chat", json={
        "query": "hey i want to open the report of arjun mehta",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Arjun Mehta"
    assert data["patient"]["patient_id"] == "PAT-1005"
    assert data["report"] is not None
    # Verify no fake security refusal text
    assert "verify his details" not in data["answer"].lower()
    assert "safety and security purposes" not in data["answer"].lower()
    # Verify reports are loaded
    assert len(data.get("available_reports", [])) == 3


def test_multi_report_what_problem_they_have():
    """
    When patient has multiple reports on file and clinician asks 'what problem they have',
    the assistant synthesizes the clinical findings and impressions across their reports with citations.
    """
    session_id = "test_multi_problems_arjun"
    # 1. Open Arjun Mehta
    open_res = client.post("/assistant/chat", json={
        "query": "hey i want to open the report of arjun mehta",
        "session_id": session_id
    })
    assert open_res.status_code == 200
    assert open_res.json()["status"] == "success"

    # 2. Ask what problem they have
    res = client.post("/assistant/chat", json={
        "query": "what problem they have",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    ans = data["answer"].lower()
    # Should explain clinical condition of tooth 25 (obturation, crown preparation, or crown)
    assert any(term in ans for term in ["tooth 25", "25", "endodontic", "obturation", "crown", "root canal"])
    # Citations must be present
    assert len(data.get("citations", [])) >= 1


def test_greeting_prefaced_retrieval_variations():
    """
    Various natural conversational greetings must not obstruct report retrieval.
    """
    session_id = "test_greeting_vikram"
    res = client.post("/assistant/chat", json={
        "query": "good morning please open vikram malhotra's cbct",
        "session_id": session_id
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Vikram Malhotra"
    assert data["patient"]["patient_id"] == "PAT-1009"
    assert "CBCT" in data["report"]["report_type"]

