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
        "query": "Open PAT-1001's OPG report",
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

def test_spoken_dictation_assessment_requirements():
    """
    Validates all 7 clinical assessment requirements from clinical dictation:
    1. Patient context unassigned (no hardcoding of Priya Sharma / PAT-1001)
    2. Speech transcription error 'root Bridge option' flagged for clinician review
    3. Palmer quadrant notation uses standard bracket symbol '8 ┐'
    4. Zero hallucinated recommendations ([] strictly empty)
    5. Zero invented root bridging or root resorption claims
    6. Safety status FLAGGED_FOR_REVIEW due to unclear speech artifact
    7. Overgeneralization disclaimer attached to negative statement
    """
    spoken_text = (
        "hey can you generate the report for the lower right third molar is horizontally impacted "
        "it is close to the adjacent second molar with no evidence of root Bridge option "
        "the surrounding bone appear normal no other abnormal finding are noted in the supplied dictation"
    )

    response = client.post("/dictation/structure", json={"text": spoken_text})
    assert response.status_code == 200
    data = response.json()

    # 1. Patient Context Integrity
    assert data["patient_context"]["patient_name"] == "Unassigned / Pending Patient Selection"
    assert data["patient_context"]["patient_id"] == "None"
    assert data["patient_context"]["patient_assigned"] is False

    # 2. Palmer quadrant bracket notation
    sites = data["anatomical_sites"]
    assert len(sites) >= 1
    lr8_site = sites[0]
    assert lr8_site["palmer_notation"] == "8 ┐"
    assert lr8_site["fdi_number"] == 48

    # 3. Speech Transcription Error Detection
    findings = data["findings"]
    artifact_finding = next((f for f in findings if "root Bridge option" in f["observation"]), None)
    assert artifact_finding is not None
    assert artifact_finding["certainty"] == "unclear / clinician review required"

    # 4. Zero Hallucinated Recommendations
    assert data["recommendations"] == []

    # 5. Overgeneralization statement is qualified
    general_neg = next((f for f in findings if "no other abnormal finding" in f["observation"].lower()), None)
    assert general_neg is not None
    assert "not independently verified" in general_neg["observation"]

    # 6. Safety Guardrail Status
    guardrails = data["safety_guardrails"]
    assert guardrails["status"] == "FLAGGED_FOR_REVIEW"
    assert len(guardrails["unsupported_findings"]) >= 1
    assert any("root Bridge option" in u for u in guardrails["unsupported_findings"])
    assert "Unclear speech transcription artifact flagged" in guardrails["anti_hallucination_guarantee"]


def test_audio_upload_and_editable_confirmation():
    """
    Tests Feature B mandatory requirement:
    1. Accept audio or speech-to-text transcript.
    2. Produce editable structured JSON with findings, anatomical site, and readable draft report.
    3. Label uncertainty and require clinician review.
    4. Confirm and sign-off workflow.
    """
    import io

    # 1. Test Audio Upload Endpoint
    fake_audio = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    files = {"audio": ("test_voice_dictation.wav", fake_audio, "audio/wav")}
    data = {
        "transcript": "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal.",
        "patient_name": "Priya Sharma",
        "patient_id": "PAT-1004"
    }
    resp = client.post("/dictation/audio", files=files, data=data)
    assert resp.status_code == 200
    res_json = resp.json()

    assert res_json["patient_context"]["patient_name"] == "Priya Sharma"
    assert res_json["patient_context"]["patient_id"] == "PAT-1004"
    assert res_json["report_metadata"]["status"] == "DRAFT_PENDING_REVIEW"
    assert res_json["report_metadata"]["clinician_review_required"] is True

    # Verify FDI tooth numbering and findings
    findings = res_json["findings"]
    assert len(findings) >= 1
    f_impaction = findings[0]
    assert f_impaction["tooth_number"] == "48"
    assert "horizontally impacted" in f_impaction["observation"].lower()
    assert f_impaction["certainty"] == "definite observation"

    # Verify readable draft report is generated
    assert "DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF" in res_json["narrative_draft"]
    assert "Priya Sharma" in res_json["narrative_draft"]

    # 2. Test Chat Interface Clinical Dictation Draft Generation
    chat_resp = client.post("/assistant/chat", json={
        "query": "Lower right third molar is horizontally impacted",
        "session_id": "test-scribe-turn"
    })
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["status"] == "success"
    assert "Clinical Dictation Draft Report Generated" in chat_data["answer"]
    assert "Tooth 48" in chat_data["answer"]
    assert "horizontally impacted" in chat_data["answer"].lower()
    assert "DRAFT_PENDING_REVIEW" in chat_data["answer"]

    # 3. Test Editable Modification & Clinician Sign-Off
    # Simulate clinician editing observation and signing off
    edited_findings = [dict(f_impaction, observation="Confirmed mesioangular / horizontal impaction of tooth 48.")]
    confirm_payload = {
        "dictation_json_path": res_json["dictation_json_path"],
        "patient_name": "Priya Sharma",
        "patient_id": "PAT-1004",
        "findings": edited_findings,
        "impressions": ["Confirmed mesioangular / horizontal impaction of tooth 48."],
        "raw_dictation": res_json["raw_dictation"]
    }
    confirm_resp = client.post("/dictation/confirm", json=confirm_payload)
    assert confirm_resp.status_code == 200
    conf_data = confirm_resp.json()
    assert conf_data["report_metadata"]["status"] == "CONFIRMED_BY_CLINICIAN"
    assert conf_data["report_metadata"]["clinician_review_required"] is False
    assert conf_data["findings"][0]["observation"] == "Confirmed mesioangular / horizontal impaction of tooth 48."

