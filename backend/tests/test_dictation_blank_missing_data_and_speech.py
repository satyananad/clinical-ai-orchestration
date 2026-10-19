import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.llm_service import parse_dictation_with_llm
from backend.app.services.report_service import compose_narrative
from backend.app.services.pdf_service import generate_dictation_pdf

client = TestClient(app)

def test_audio_dictation_empty_transcript_rejects_hardcoded_fallback():
    """
    Verifies that audio upload without speech-to-text transcript does NOT
    fall back to the hardcoded 'Lower right third molar...' report, but rejects with 400 error.
    """
    fake_audio = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    files = {"audio": ("empty_speech.wav", fake_audio, "audio/wav")}
    data = {"transcript": ""}  # Empty transcript

    resp = client.post("/dictation/audio", files=files, data=data)
    assert resp.status_code == 400
    assert "No speech or transcript detected" in resp.json()["detail"]


def test_dictation_blanks_missing_patient_data():
    """
    User requirement: 'if report generated on patient data if any thing missing then blank them'
    Verifies that missing patient demographics (name, id, age, gender, dob) are left blank
    without injecting synthetic placeholders like 'PAT-RECORD', 'Pending demographics verification',
    or 'None (Requires patient selection...)'.
    """
    dictation_text = "Upper left second premolar shows occlusal caries approaching the dentinoenamel junction."
    res = client.post("/dictation/structure", json={"text": dictation_text})
    assert res.status_code == 200
    data = res.json()

    narrative = data["narrative_draft"]
    # Check headers exist
    assert "Patient Name:" in narrative
    assert "Patient ID:" in narrative
    assert "Demographics:" in narrative
    assert "Examination:" in narrative

    # Check synthetic placeholders are NOT present in the narrative
    assert "PAT-RECORD" not in narrative
    assert "Not specified (Pending demographics verification)" not in narrative
    assert "None (Requires patient selection before permanent record entry)" not in narrative
    assert "Pending Patient Linkage (Unassigned)" not in narrative


def test_dictation_with_registered_patient_pulls_real_data_and_blanks_missing():
    """
    When dictation is provided with an existing patient ID (e.g. PAT-1008 Vikram Malhotra),
    verified demographics are pulled and any missing fields are left blank.
    """
    dictation_text = "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal."
    res = client.post("/dictation/structure", json={
        "text": dictation_text,
        "patient_id": "PAT-1008",
        "patient_name": "Vikram Malhotra"
    })
    assert res.status_code == 200
    data = res.json()

    assert data["patient_context"]["patient_name"] == "Vikram Malhotra"
    assert data["patient_context"]["patient_id"] == "PAT-1008"
    assert data["patient_context"]["patient_assigned"] is True

    narrative = data["narrative_draft"]
    assert "Patient Name:    Vikram Malhotra" in narrative
    assert "Patient ID:      PAT-1008" in narrative
    assert "PAT-RECORD" not in narrative


def test_chat_scribe_does_not_hijack_booking_state_patient():
    """
    Verifies that a clinical dictation submitted via chat does NOT inherit
    a patient from booking_state (e.g. when an appointment was being scheduled).
    """
    session_id = "test-no-hijack-session"

    # 1. Start booking inquiry for Vikram Malhotra
    r1 = client.post("/assistant/chat", json={
        "query": "Book a CBCT scan tomorrow for Vikram Malhotra",
        "session_id": session_id
    })
    assert r1.status_code == 200

    # 2. Reset / clear context or switch to a new anonymous dictation
    client.post(f"/sessions/{session_id}/reset")

    # 3. Dictate clinical observations without patient name
    r2 = client.post("/assistant/chat", json={
        "query": "Tooth 16 has deep occlusal caries approaching the pulp chamber.",
        "session_id": session_id
    })
    assert r2.status_code == 200
    chat_data = r2.json()

    # Draft narrative should NOT claim it is Vikram Malhotra
    assert "Clinical Dictation Draft Report Generated" in chat_data["answer"]
    assert "Vikram Malhotra" not in chat_data["answer"]
    assert "Tooth 16" in chat_data["answer"]


def test_compound_3d_measurements_preserved():
    """
    Verifies that compound 3D measurements (e.g. '8 by 7 by 6 mm', '1 to 2 mm')
    are extracted as single coherent observations without splitting on 'with' or 'by'.
    """
    text = "UR5 is root treated with a unilocular well defined apical radiolucency 8 by 7 by 6 mm."
    parsed = parse_dictation_with_llm(text)
    findings = parsed.get("findings", [])

    assert len(findings) == 1
    f = findings[0]
    assert f["tooth_number"] == "15"
    assert "8 by 7 by 6 mm" in f["observation"]
    assert "8 by 7 by 6 mm" in f["measurements"]
    assert f["certainty"] == "definite observation"
