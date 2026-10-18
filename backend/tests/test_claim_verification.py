"""
AUTOMATED TEST SUITE FOR REVISED RULE 2 & RULE 3 (CLAIM VERIFICATION)
---------------------------------------------------------------------
Verifies the 8 dimensions of clinical claim verification:
1. Tooth number mismatch detection (Tooth 16 vs 46)
2. Negation contradiction detection ("no caries" vs "caries present")
3. Uncertainty escalation detection ("possible lesion" vs "confirmed lesion")
4. Laterality contradiction detection (Left vs Right)
5. Measurement contradiction detection (4 mm vs 8 mm)
6. Missing information / Insufficient evidence detection
7. Patient identity mismatch detection (PAT-1001 vs PAT-1002)
8. Multi-dimensional clinical status classification
"""

import pytest
from backend.app.services.claim_verification import (
    verify_single_claim_deterministic,
    verify_dictation_against_source,
    verify_report_answer_claim,
    classify_clinical_finding_status,
    ClaimStatus
)

def test_tooth_number_mismatch_detected():
    """Verify that changing tooth 16 to tooth 46 is detected as a CONTRADICTED claim."""
    source = "Coronal radiolucency noted on the occlusal surface of tooth 16."
    claim = "Caries observed on tooth 46."
    res = verify_single_claim_deterministic(source, claim)
    assert res["status"] == ClaimStatus.CONTRADICTED
    assert res["dimension"] == "tooth_number"
    assert "tooth [46]" in res["details"]

def test_negation_contradiction_detected():
    """Verify that source 'no caries' vs claim 'caries is present' is flagged as CONTRADICTED."""
    source = "Tooth 25 shows no caries and intact crown margin."
    claim = "Tooth 25 has dental caries present on the distal margin."
    res = verify_single_claim_deterministic(source, claim)
    assert res["status"] == ClaimStatus.CONTRADICTED
    assert res["dimension"] == "negation"
    assert "absence of 'caries'" in res["details"]

def test_uncertainty_escalation_detected():
    """Verify that escalating a suspected finding to a confirmed diagnosis is flagged."""
    source = "Distal to UR8 is an unerupted tooth, possibly a dentigerous cyst."
    claim = "Distal to UR8 is a confirmed dentigerous cyst requiring enucleation."
    res = verify_single_claim_deterministic(source, claim)
    assert res["status"] == ClaimStatus.CONTRADICTED
    assert res["dimension"] == "uncertainty"

def test_laterality_mismatch_detected():
    """Verify that changing left side to right side is flagged as CONTRADICTED."""
    source = "Mandibular left third molar shows vertical impaction."
    claim = "Mandibular right third molar shows vertical impaction."
    res = verify_single_claim_deterministic(source, claim)
    assert res["status"] == ClaimStatus.CONTRADICTED
    assert res["dimension"] == "laterality"

def test_measurement_mismatch_detected():
    """Verify that changing 4 mm to 8 mm is flagged as CONTRADICTED."""
    source = "Periapical radiolucency measuring 4 mm at the apex."
    claim = "Apical lesion measuring 8 mm."
    res = verify_single_claim_deterministic(source, claim)
    assert res["status"] == ClaimStatus.CONTRADICTED
    assert res["dimension"] == "measurement"

def test_missing_information_flagged_as_insufficient_evidence():
    """Verify that asserting a cyst when none exists in source returns INSUFFICIENT_EVIDENCE."""
    source = "Generalized physiological bone height maintained within normal limits."
    claim = "Patient presents with extensive ameloblastoma and cyst."
    res = verify_single_claim_deterministic(source, claim)
    assert res["status"] == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert res["dimension"] == "missing_information"

def test_patient_identity_mismatch_blocked():
    """Verify that citing report from PAT-1002 when active patient is PAT-1001 blocks response."""
    report = {
        "report_id": "RPT-2004",
        "patient_id": "PAT-1002",
        "report_type": "OPG",
        "findings": [{"region": "Tooth 48", "observation": "Impacted"}]
    }
    answer = "The patient has an impacted tooth 48."
    res = verify_report_answer_claim(report, answer, expected_patient_id="PAT-1001")
    assert res["status"] == ClaimStatus.IDENTITY_MISMATCH
    assert res["gate_decision"] == "BLOCK"
    assert "Critical identity mismatch" in res["message"]

def test_multi_dimensional_clinical_classification():
    """Verify revised Rule 3: separates Certainty, Finding Status, and Image Limitation."""
    # Case A: Indeterminate negative
    r1 = classify_clinical_finding_status("Tooth 25 shows no definite periapical radiolucency.")
    assert r1["certainty"] == "indeterminate"
    assert r1["finding_status"] == "negative"
    assert r1["image_limitation"] == "none"

    # Case B: Confirmed negative (normal physiological)
    r2 = classify_clinical_finding_status("Marginal alveolar bone height within normal limits.")
    assert r2["certainty"] == "confirmed"
    assert r2["finding_status"] == "negative"

    # Case C: Suspected positive
    r3 = classify_clinical_finding_status("UR5 root treated with suspected radicular cyst.")
    assert r3["certainty"] == "suspected"
    assert r3["finding_status"] == "positive"

    # Case D: Image limitation (superimposition)
    r4 = classify_clinical_finding_status("Slight superimposition of distal ramus cortical margin.")
    assert r4["image_limitation"] == "superimposition"
