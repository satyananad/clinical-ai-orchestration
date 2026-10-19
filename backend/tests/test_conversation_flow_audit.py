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
    assert "registered patients" in data2["answer"].lower()


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


def test_what_they_have_pronoun_resolution():
    """
    Tests exact clinician sequence:
    1. open report priya sharma -> 2 candidates
    2. DOB: 1998 -> opens RPT-2003
    3. what they have -> summarizes RPT-2003 findings and impressions with citations, NEVER clinic suites!
    """
    session_id = "test-what-they-have-session"
    get_or_create_session(session_id).clear()

    # 1. open report priya sharma
    r1 = client.post("/assistant/chat", json={"query": "open report priya sharma", "session_id": session_id}).json()
    assert r1["status"] == "clarification_required"
    assert len(r1["candidates"]) == 2

    # 2. DOB: 1998
    r2 = client.post("/assistant/chat", json={"query": "DOB: 1998", "session_id": session_id}).json()
    assert r2["status"] == "success"
    assert r2["report"]["report_id"] == "RPT-2003"

    # 3. what they have
    r3 = client.post("/assistant/chat", json={"query": "what they have", "session_id": session_id}).json()
    assert r3["status"] == "success"
    ans = (r3.get("answer") or r3.get("message") or "").lower()
    # MUST contain actual clinical impressions/findings from RPT-2003
    assert "caries" in ans or "hypersensitivity" in ans or "tooth 16" in ans or "tooth 25" in ans
    # MUST NOT contain clinic facility equipment advertisement
    assert "specialized suites" not in ans
    assert "suite 1" not in ans
    assert "intraoral bay 3" not in ans
    # MUST contain verifiable report evidence citations
    assert len(r3.get("citations", [])) > 0
    assert any("RPT-2003" in c for c in r3["citations"])


def test_multi_report_switching_and_typo_query():
    """
    Tests:
    1. Disambiguation for Priya Sharma via DOB: 1998.
    2. Multi-report switching via 'Open OPG report RPT-2001' (verifying RPT-2001 does not match Sneha Reddy DOB 2001).
    3. Typo-tolerant consultation 'what problem they hsve' answering from active RPT-2001 report with citations.
    """
    session_id = "test-multi-report-typo-session"
    get_or_create_session(session_id).clear()

    # Step 1: Disambiguate Priya Sharma
    r1 = client.post("/assistant/chat", json={"query": "open report priya sharma", "session_id": session_id}).json()
    assert r1["status"] == "clarification_required"

    r2 = client.post("/assistant/chat", json={"query": "DOB: 1998", "session_id": session_id}).json()
    assert r2["status"] == "success"
    assert r2["report"]["report_id"] == "RPT-2003"

    # Step 2: Switch to OPG report RPT-2001
    r3 = client.post("/assistant/chat", json={"query": "Open OPG report RPT-2001", "session_id": session_id}).json()
    assert r3["status"] == "success"
    assert r3["report"]["report_id"] == "RPT-2001"
    assert r3["patient"]["patient_name"] == "Priya Sharma"

    # Step 3: Typo query "what problem they hsve"
    r4 = client.post("/assistant/chat", json={"query": "what problem they hsve", "session_id": session_id}).json()
    assert r4["status"] == "success"
    ans4 = (r4.get("answer") or r4.get("message") or "").lower()
    # Tooth 48 mesioangular impaction is the key finding in RPT-2001
    assert "tooth 48" in ans4 or "impaction" in ans4 or "periapical" in ans4
    assert len(r4.get("citations", [])) > 0
    assert any("RPT-2001" in c for c in r4["citations"])


def test_male_patient_pronoun_followup_and_typo_hir():
    """
    Verifies:
    1. 'open Amit kumar' (Male) opens latest report and prompts with 'his' (e.g. 'Open his OPG'), NEVER 'her'.
    2. 'Open hir OPG' (with typo 'hir') resolves to active patient Amit Kumar, opening RPT-2016 (OPG).
       NEVER returns 'No patient records found matching 'hir''.
    3. 'open his opg' also resolves smoothly.
    """
    session_id = "test-amit-kumar-pronoun-flow"
    get_or_create_session(session_id).clear()

    # Step 1: Open Amit Kumar
    r1 = client.post("/assistant/chat", json={"query": "open Amit kumar", "session_id": session_id}).json()
    assert r1["status"] == "success"
    assert r1["patient"]["patient_name"] == "Amit Kumar"
    assert r1["patient"]["sex"] == "Male"
    assert r1["report"]["report_id"] == "RPT-2018"
    assert len(r1.get("available_reports", [])) == 3
    # Follow-up message MUST use male pronoun 'his', NEVER 'her'
    msg1 = r1["message"]
    assert "Open his" in msg1
    assert "Open her" not in msg1

    # Step 2: Clinician sends 'Open hir OPG' (common typo for his/her)
    r2 = client.post("/assistant/chat", json={"query": "Open hir OPG", "session_id": session_id}).json()
    assert r2["status"] == "success"
    assert r2["report"]["report_id"] == "RPT-2016"
    assert r2["report"]["report_type"] == "OPG"
    assert r2["patient"]["patient_name"] == "Amit Kumar"
    assert "No patient records found" not in r2.get("message", "")

    # Step 3: Clinician sends 'open his opg'
    r3 = client.post("/assistant/chat", json={"query": "open his opg", "session_id": session_id}).json()
    assert r3["status"] == "success"
    assert r3["report"]["report_id"] == "RPT-2016"
    assert r3["patient"]["patient_name"] == "Amit Kumar"


def test_standalone_pronouns_without_active_patient():
    """
    Verifies that querying pronouns alone without active patient context safely returns
    missing_patient_name rather than treating 'hir'/'his'/'her' as patient surnames.
    """
    session_id = "test-standalone-pronouns"
    get_or_create_session(session_id).clear()

    r1 = client.post("/assistant/chat", json={"query": "Open hir OPG", "session_id": session_id}).json()
    assert r1["status"] == "missing_patient_name"
    assert "Please specify the patient name or ID" in r1["message"]
    assert "matching 'hir'" not in r1["message"]

    r2 = client.post("/assistant/chat", json={"query": "Open his report", "session_id": session_id}).json()
    assert r2["status"] == "missing_patient_name"


def test_get_patient_pronouns_mentions_both_if_unspecified():
    """
    Verifies that if patient gender is not specified or clear, both 'his/her' are mentioned.
    """
    from backend.app.services.retrieval import get_patient_pronouns

    male_res = get_patient_pronouns("Male")
    assert male_res["possessive"] == "his"
    assert male_res["subject"] == "he"

    female_res = get_patient_pronouns("Female")
    assert female_res["possessive"] == "her"
    assert female_res["subject"] == "she"

    # When gender is not clear/unspecified, mentions both
    unspec_res = get_patient_pronouns(None)
    assert unspec_res["possessive"] == "his/her"
    assert unspec_res["subject"] == "he/she"

    empty_res = get_patient_pronouns("")
    assert empty_res["possessive"] == "his/her"


def test_arjun_mehta_conversation_flow_report_aware_and_booking():
    """
    Verifies the exact scenario reported by the clinician:
    1. 'open arjun mehta report' opens Arjun Mehta's Bitewing report (RPT-2015).
    2. 'how many report have they' recognizes that Arjun Mehta is the active patient,
       answering with his 3 diagnostic reports (RPT-2015, RPT-2014, RPT-2013),
       and NEVER queries the entire clinic database (35 reports across 12 patients).
    3. Confirms that subsequent booking actions attribute to the active patient Arjun Mehta (PAT-1005).
    """
    from backend.app.services.booking_service import reset_slots
    reset_slots()
    session_id = "test-arjun-mehta-flow"
    get_or_create_session(session_id).clear()

    # Step 1: Open Arjun Mehta report
    r1 = client.post("/assistant/chat", json={"query": "open arjun mehta report", "session_id": session_id}).json()
    assert r1["status"] == "success"
    assert r1["patient"]["patient_name"] == "Arjun Mehta"
    assert r1["report"]["report_id"] == "RPT-2015"

    # Step 2: Clinician asks "how many report have they"
    r2 = client.post("/assistant/chat", json={"query": "how many report have they", "session_id": session_id}).json()
    assert r2["status"] == "success"
    ans2 = r2.get("answer") or r2.get("message")
    # MUST report Arjun Mehta's 3 reports, NOT the 35 reports across 12 patients
    assert "3 diagnostic reports" in ans2 or "3 diagnostic report" in ans2 or "3 reports" in ans2
    assert "35" not in ans2
    assert "across 12 registered patients" not in ans2
    assert "RPT-2015" in ans2
    assert "RPT-2014" in ans2
    assert "RPT-2013" in ans2

    # Step 3: Clinician books a slot for the active patient
    r3 = client.post("/assistant/chat", json={"query": "Confirm slot SLOT-105", "session_id": session_id}).json()
    assert r3["status"] == "success"
    assert r3["slot"]["patient_name"] == "Arjun Mehta"
    assert r3["slot"]["patient_id"] == "PAT-1005"
    assert r3["slot"]["patient_status"] == "REGISTERED"


