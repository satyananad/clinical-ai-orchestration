import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_case_1_root_fracture_with_measured_lesion():
    """Test Case 1: Root fracture with a measured lesion - preserves fracture, measurement, and correct site."""
    text = (
        "Tooth 12 has mid-root horizontal fracture with 2 mm separation of fragments. "
        "Center of fracture exhibits a 6.5 mm radiolucency with buccal bone expansion."
    )
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    findings = data["findings"]
    # Check tooth site
    assert any("12" in str(f.get("anatomical_site", "")) or f.get("tooth_number") == "12" for f in findings)
    # Check fracture finding
    fracture_f = next((f for f in findings if "fracture" in f["observation"].lower()), None)
    assert fracture_f is not None
    assert "2 mm" in fracture_f.get("measurements", []) or "2 mm" in fracture_f["observation"]
    
    # Check lesion measurement
    lesion_f = next((f for f in findings if "6.5 mm" in f["observation"] or "6.5 mm" in f.get("measurements", [])), None)
    assert lesion_f is not None
    assert "6.5 mm" in lesion_f.get("measurements", [])

def test_case_2_possible_resorption_near_impacted_tooth():
    """Test Case 2: Possible resorption near impacted tooth - uncertainty must remain possible / suspected."""
    text = "Upper right third molar is vertically impacted with possible external root resorption of adjacent second molar."
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    resorption_f = next((f for f in data["findings"] if "resorption" in f["observation"].lower()), None)
    assert resorption_f is not None
    assert resorption_f["certainty"] == "suspected / differential diagnosis"
    assert resorption_f.get("polarity") in ("present", "uncertain")

def test_case_3_multiple_findings_different_quadrants():
    """Test Case 3: Multiple findings in different quadrants - each retains its correct quadrant and tooth."""
    text = (
        "Tooth 16 in the upper right quadrant has coronal caries extending into pulp. "
        "Tooth 36 in the lower left quadrant presents with widening of the periodontal ligament."
    )
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    # Both teeth must be extracted
    fdis = [s.get("fdi_number") for s in data["anatomical_sites"]]
    assert 16 in fdis
    assert 36 in fdis
    
    caries_f = next((f for f in data["findings"] if "caries" in f["observation"].lower()), None)
    pdl_f = next((f for f in data["findings"] if "periodontal ligament" in f["observation"].lower()), None)
    assert caries_f is not None
    assert pdl_f is not None
    assert "16" in caries_f["anatomical_site"] or caries_f.get("tooth_number") == "16"
    assert "36" in pdl_f["anatomical_site"] or pdl_f.get("tooth_number") == "36"

def test_case_4_explicitly_normal_anatomical_structures():
    """Test Case 4: Explicitly normal anatomical structures - stated negative finding preserved."""
    text = "The maxillary sinuses are clear bilaterally with no mucosal thickening or sinus floor perforation."
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    sinus_f = next((f for f in data["findings"] if "sinus" in f["observation"].lower()), None)
    assert sinus_f is not None
    assert sinus_f["certainty"] == "negative / normal finding"
    assert sinus_f.get("polarity") == "absent"

def test_case_5_missing_tooth_number():
    """Test Case 5: Missing tooth number - unknown tooth number, not a guessed number."""
    text = "Periapical radiolucency observed in an unnumbered anterior root apex."
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    f = data["findings"][0]
    # Must NOT guess a random tooth number like 48
    assert f.get("tooth_number") is None or f.get("tooth_number") == ""
    assert "48" not in f["anatomical_site"]

def test_case_6_differential_diagnosis_and_recommendation():
    """Test Case 6: Differential diagnosis and recommendation - retained and correctly categorized."""
    text = (
        "Well-defined radiolucency suggestive of an ameloblastoma or odontogenic keratocyst. "
        "Histopathological biopsy is recommended for definitive diagnosis."
    )
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    # Differential certainty
    diff_f = next((f for f in data["findings"] if "ameloblastoma" in f["observation"].lower() or "keratocyst" in f["observation"].lower()), None)
    assert diff_f is not None
    assert diff_f["certainty"] == "suspected / differential diagnosis"
    
    # Recommendation
    assert len(data["recommendations"]) >= 1
    assert any("biopsy" in r.lower() or "recommended" in r.lower() for r in data["recommendations"])

def test_case_7_speech_with_unclear_tooth_identifier():
    """Test Case 7: Speech with unclear identifier - uncertainty is flagged for clinician review."""
    text = "Tooth was examined with no evidence of root Bridge option noted."
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    assert data["safety_guardrails"]["status"] == "FLAGGED_FOR_REVIEW"
    art_f = next((f for f in data["findings"] if "root Bridge option" in f["observation"]), None)
    assert art_f is not None
    assert art_f["requires_clarification"] is True

def test_case_8_unfamiliar_dental_terminology():
    """Test Case 8: Unfamiliar dental terminology - no invented findings when wording is unusual."""
    text = "Incidental finding of localized dense bone island adjacent to mandibular premolar."
    res = client.post("/dictation/structure", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    
    # Preserves the observation verbatim in findings without inventing false disease
    assert any("dense bone island" in f["observation"].lower() for f in data["findings"])
    assert data["safety_guardrails"]["status"] == "PASSED"
