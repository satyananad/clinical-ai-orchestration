import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.claim_verification import (
    verify_dictation_against_source,
    classify_clinical_finding_status,
    extract_patient_demographics,
    ClaimStatus
)

client = TestClient(app)

BENCHMARK_DICTATION = (
    "Patient is a 46-year-old female. CBCT of the maxillary left quadrant was performed "
    "for implant assessment at the missing first molar site. The alveolar ridge measures "
    "5.2 mm in width and 10.8 mm in height. The buccal cortical plate is intact but thin. "
    "At the second molar site, a well-defined 7.5 mm radiolucent lesion is present, with "
    "cortical thinning and close proximity to the maxillary sinus floor; etiology is uncertain. "
    "The adjacent second premolar demonstrates mild widening of the periodontal ligament. "
    "No definite sinus communication is observed. No displaced root fragments are identified."
)

def test_rubric_patient_demographics():
    """Criteria 1: Age 46 and female must be extracted into patient_context."""
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    assert response.status_code == 200
    data = response.json()
    ctx = data["patient_context"]
    assert ctx["age"] == 46
    assert ctx["gender"] == "Female"
    assert "46-year-old" in ctx.get("demographics_summary", "").lower()

def test_rubric_tooth_mapping_quadrant_concordance():
    """Criteria 2: Teeth must map to maxillary left (FDI 26, 27, 25), NEVER mandibular right (46, 47)."""
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    fdi_numbers = [s.get("fdi_number") for s in data["anatomical_sites"] if s.get("fdi_number")]
    # FDI 26 (UL6), 27 (UL7), 25 (UL5) must be present
    assert 26 in fdi_numbers
    assert 27 in fdi_numbers
    assert 25 in fdi_numbers
    # Mandibular teeth 46 and 47 must NOT be present
    assert 46 not in fdi_numbers
    assert 47 not in fdi_numbers

def test_rubric_lesion_certainty_and_presence():
    """Criteria 3: 7.5 mm lesion certainty must reflect confirmed presence with uncertain etiology."""
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    lesion_finding = next((f for f in data["findings"] if "7.5 mm" in f["observation"]), None)
    assert lesion_finding is not None
    assert "etiology uncertain" in lesion_finding["certainty"].lower() or "definite observation" in lesion_finding["certainty"].lower()
    assert "7.5 mm" in lesion_finding.get("measurements", [])

def test_rubric_second_premolar_pdl_widening():
    """Criteria 4: Second premolar PDL widening is an observed finding (positive), NOT negative/normal."""
    status = classify_clinical_finding_status("The adjacent second premolar demonstrates mild widening of the periodontal ligament.")
    assert status["finding_status"] == "positive"
    assert status["certainty"] == "confirmed"
    assert status["legacy_label"] == "definite observation"

    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    pdl_finding = next((f for f in data["findings"] if "periodontal ligament" in f["observation"].lower()), None)
    assert pdl_finding is not None
    assert pdl_finding["certainty"] == "definite observation"

def test_rubric_implant_measurements_and_cortical_plate():
    """Criteria 5: Alveolar ridge dimensions (5.2 mm, 10.8 mm) and thin buccal plate must be extracted."""
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    ridge_finding = next((f for f in data["findings"] if "alveolar ridge" in f["observation"].lower()), None)
    assert ridge_finding is not None
    assert "5.2 mm" in ridge_finding.get("measurements", [])
    assert "10.8 mm" in ridge_finding.get("measurements", [])

    plate_finding = next((f for f in data["findings"] if "buccal cortical plate" in f["observation"].lower()), None)
    assert plate_finding is not None

def test_rubric_negative_findings_preserved():
    """Criteria 6: 'No definite sinus communication' and 'No displaced root fragments' must be extracted."""
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    sinus_finding = next((f for f in data["findings"] if "sinus communication" in f["observation"].lower()), None)
    assert sinus_finding is not None
    assert sinus_finding["certainty"] == "negative / normal finding"

    frag_finding = next((f for f in data["findings"] if "displaced root" in f["observation"].lower()), None)
    assert frag_finding is not None
    assert frag_finding["certainty"] == "negative / normal finding"

def test_rubric_comprehensive_impression():
    """Criteria 7: Impression must summarize lesion size (7.5 mm), cortical thinning, sinus proximity, PDL widening, and negative findings."""
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    impressions = " ".join(data["impressions"])
    assert "7.5 mm" in impressions
    assert "cortical thinning" in impressions.lower()
    assert "sinus" in impressions.lower()
    assert "periodontal ligament" in impressions.lower()
    assert "sinus communication" in impressions.lower() or "displaced root" in impressions.lower()

def test_rubric_safety_validation_strict_guardrails():
    """Criteria 8: Guardrails must reject quadrant discordance (FDI 46/47) and omitted measurements."""
    # 1. Correct findings must PASS
    response = client.post("/dictation/structure", json={"text": BENCHMARK_DICTATION})
    data = response.json()
    assert data["safety_guardrails"]["status"] == "PASSED"

    # 2. Quadrant discordance (tooth 46/47 claimed on maxillary left scan) must FAIL with CONTRADICTED_CLAIMS_DETECTED
    adversarial_findings_teeth = [
        {"anatomical_site": "Tooth 46", "observation": "Tooth 46 shows 7.5 mm lesion"}
    ]
    res_teeth = verify_dictation_against_source(BENCHMARK_DICTATION, adversarial_findings_teeth)
    assert res_teeth["status"] == "CONTRADICTED_CLAIMS_DETECTED"
    assert any(c["dimension"] == "quadrant_concordance" for c in res_teeth["contradicted_claims"])

    # 3. Omitted key measurements must FAIL with FLAGGED_UNSUPPORTED
    adversarial_findings_meas = [
        {"anatomical_site": "Maxillary left second molar", "observation": "At the second molar site, a well-defined 7.5 mm radiolucent lesion is present", "measurements": ["7.5 mm"]}
    ]
    res_meas = verify_dictation_against_source(BENCHMARK_DICTATION, adversarial_findings_meas)
    assert res_meas["status"] == "FLAGGED_UNSUPPORTED"
    assert any("5.2 mm" in u for u in res_meas["unsupported_findings"])
