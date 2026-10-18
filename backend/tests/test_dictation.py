from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_dictation_preserves_uncertainty():
    """
    Evaluates that hedged observations ('possible radiolucency', 'suggestive of')
    remain labeled as suspected / differential diagnosis, not converted into false definitive findings.
    """
    dictation_text = "Tooth 25 shows possible radiolucency at the apical region, suggestive of chronic periapical lesion."
    response = client.post("/dictation/structure", json={"text": dictation_text})
    assert response.status_code == 200
    data = response.json()

    assert len(data["findings"]) >= 1
    finding = data["findings"][0]
    assert finding["certainty"] == "suspected / differential diagnosis"
    assert "possible" in finding["observation"].lower()
    assert data["report_metadata"]["clinician_review_required"] is True

def test_dictation_anatomical_tooth_mapping():
    """
    Evaluates mapping across FDI, Palmer, and Universal tooth notations.
    """
    dictation_text = "Lower right third molar is horizontally impacted. Also evaluate UR5 and tooth #29."
    response = client.post("/dictation/structure", json={"text": dictation_text})
    assert response.status_code == 200
    data = response.json()

    fdi_numbers = {site["fdi_number"] for site in data["anatomical_sites"] if site["fdi_number"]}
    assert 48 in fdi_numbers  # Lower right 3rd molar -> FDI 48
    assert 15 in fdi_numbers  # UR5 -> FDI 15
    assert 45 in fdi_numbers  # Tooth #29 (Universal) -> FDI 45

def test_conversational_follow_up_and_grounding():
    """
    Tests Feature C: First open a report, then ask 'Summarize her report'
    and verify it resolves patient context and provides citations.
    """
    session_id = "test_conv_session"

    # Step 1: Open Priya Sharma's OPG report
    r1 = client.post("/assistant/chat", json={
        "query": "Open Priya Sharma's latest OPG report",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["status"] == "success"

    # Step 2: Contextual follow-up using pronoun 'her'
    r2 = client.post("/assistant/chat", json={
        "query": "Summarize her report",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data = r2.json()

    assert data["status"] == "success"
    assert data["patient"]["patient_name"] == "Priya Sharma"
    assert data["report"]["report_id"] == "RPT-2001"
    assert len(data["citations"]) > 0
    assert "Tooth 48" in data["answer"]


def test_unsupported_clinical_claims():
    """
    Mandatory Safety Test from Assessment:
    Verifies that the anti-hallucination guardrail strictly flags and rejects
    any clinical claims not grounded in the dictated audio/transcript.
    """
    from backend.app.services.validation import verify_anti_hallucination

    dictation = "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal."

    # 1. Authentic grounded findings should pass
    grounded_findings = [
        {
            "anatomical_site": "Tooth 48",
            "observation": "Lower right third molar is horizontally impacted with roots approximating the canal"
        }
    ]
    check_pass = verify_anti_hallucination(dictation, grounded_findings)
    assert check_pass["status"] == "PASSED"
    assert len(check_pass["unsupported_findings"]) == 0

    # 2. Fabricated / ungrounded findings MUST be caught and flagged
    hallucinated_findings = [
        {
            "anatomical_site": "Tooth 48",
            "observation": "Lower right third molar is horizontally impacted with roots approximating the canal"
        },
        {
            "anatomical_site": "Mandible",
            "observation": "Displaced fracture through the left mandibular angle"
        }
    ]
    check_fail = verify_anti_hallucination(dictation, hallucinated_findings)
    assert check_fail["status"] == "FLAGGED_UNSUPPORTED"
    assert len(check_fail["unsupported_findings"]) == 1
    assert "Displaced fracture" in check_fail["unsupported_findings"][0]
