import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.claim_verification import verify_dictation_against_source, ClaimStatus

client = TestClient(app)

TARGET_TRANSCRIPT = (
    "Patient ID: SYN-014. Patient name: Priya Mehta. Study: Panoramic radiograph. "
    "The maxillary right first molar (FDI 16; Universal 3) shows a large restoration. "
    "No obvious displacement of adjacent teeth is identified."
)

def test_target_transcript_metadata_separation():
    """
    Error 1 & 5 Verification:
    - Administrative metadata (Patient ID: SYN-014, Patient name: Priya Mehta, Study: Panoramic radiograph)
      must be extracted into patient_context/report_metadata.
    - Findings must ONLY contain genuine clinical observations, NOT administrative metadata.
    - Patient assignment remains unassigned pending clinician selection, but supplied identity is preserved.
    """
    response = client.post("/dictation/structure", json={"text": TARGET_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    # 1. Patient Context separation
    ctx = data["patient_context"]
    assert ctx["patient_name"] == "Unassigned / Pending Patient Selection"
    assert ctx["patient_id"] == "None"
    assert ctx["patient_assigned"] is False
    assert ctx["supplied_patient_name"] == "Priya Mehta"
    assert ctx["supplied_patient_id"] == "SYN-014"
    assert "panoramic" in ctx["study_type"].lower()

    # 2. Study metadata in report_metadata
    meta = data["report_metadata"]
    assert "panoramic" in meta["report_type"].lower() or "panoramic" in str(meta.get("study_type", "")).lower()

    # 3. Clinical findings must not contain metadata sentences
    findings = data["findings"]
    for f in findings:
        obs_lower = f["observation"].lower()
        assert not obs_lower.startswith("patient id")
        assert not obs_lower.startswith("patient name")
        assert not obs_lower.startswith("study:")
        assert "syn-014" not in obs_lower
        assert "priya mehta" not in obs_lower

    # Only 2 clinical findings should exist: restoration & adjacent teeth
    assert len(findings) == 2


def test_target_transcript_anatomical_attribution():
    """
    Error 2 Verification:
    - The restoration finding refers to tooth 16.
    - The adjacent teeth statement does NOT identify a single tooth and must NOT be forced to tooth 16.
    """
    response = client.post("/dictation/structure", json={"text": TARGET_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    findings = data["findings"]
    restoration_finding = next((f for f in findings if "restoration" in f["observation"].lower()), None)
    adjacent_finding = next((f for f in findings if "adjacent" in f["observation"].lower()), None)

    assert restoration_finding is not None
    assert restoration_finding["tooth_number"] == "16"

    assert adjacent_finding is not None
    # Adjacent teeth finding must NOT be forced to tooth 16
    assert adjacent_finding["tooth_number"] is None
    assert "adjacent teeth" in adjacent_finding["anatomical_site"].lower()


def test_target_transcript_impression_purity():
    """
    Error 3 Verification:
    - Impressions summarize clinical findings cleanly without reproducing administrative metadata.
    """
    response = client.post("/dictation/structure", json={"text": TARGET_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    impressions = data["impressions"]
    assert len(impressions) >= 1

    joined_imp = " ".join(impressions).lower()
    assert "syn-014" not in joined_imp
    assert "priya mehta" not in joined_imp
    assert "patient id" not in joined_imp
    assert "patient name" not in joined_imp
    assert "study:" not in joined_imp
    assert "restoration" in joined_imp


def test_target_transcript_safety_guardrails_pass_cleanly():
    """
    Error 4 Verification:
    - When parsed correctly without metadata or forced attribution, safety verification passes.
    """
    response = client.post("/dictation/structure", json={"text": TARGET_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    guardrails = data["safety_guardrails"]
    assert guardrails["status"] == "PASSED"
    assert guardrails["grounded_ratio"] == 1.0
    assert len(guardrails["unsupported_findings"]) == 0


def test_safety_check_detects_metadata_misclassification():
    """
    Error 4 Guardrail Safety:
    - If administrative metadata IS erroneously classified as a clinical finding,
      the safety check must detect and flag it as a contradiction / metadata misclassification.
    """
    corrupted_findings = [
        {
            "anatomical_site": "Tooth 16",
            "tooth_number": "16",
            "observation": "Patient ID: SYN-014. Patient name: Priya Mehta",
            "certainty": "definite observation"
        },
        {
            "anatomical_site": "Tooth 16",
            "tooth_number": "16",
            "observation": "The maxillary right first molar shows a large restoration",
            "certainty": "definite observation"
        }
    ]

    verif = verify_dictation_against_source(TARGET_TRANSCRIPT, corrupted_findings)
    assert verif["status"] == "CONTRADICTED_CLAIMS_DETECTED"
    assert any(c["dimension"] == "metadata_misclassification" for c in verif["contradicted_claims"])


def test_safety_check_detects_unsupported_anatomical_attribution():
    """
    Error 4 Guardrail Safety:
    - If tooth 16 is forced onto an adjacent teeth statement,
      the safety check must detect and flag it as unsupported anatomical attribution.
    """
    corrupted_findings = [
        {
            "anatomical_site": "Tooth 16",
            "tooth_number": "16",
            "observation": "The maxillary right first molar shows a large restoration",
            "certainty": "definite observation"
        },
        {
            "anatomical_site": "Tooth 16",
            "tooth_number": "16",  # Erroneously forced
            "observation": "No obvious displacement of adjacent teeth is identified",
            "certainty": "negative / normal finding"
        }
    ]

    verif = verify_dictation_against_source(TARGET_TRANSCRIPT, corrupted_findings)
    assert verif["status"] in ("FLAGGED_UNSUPPORTED", "CONTRADICTED_CLAIMS_DETECTED")
    assert any(c["dimension"] == "unsupported_anatomical_attribution" for c in verif["detailed_results"])
