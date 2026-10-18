import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.claim_verification import (
    verify_dictation_against_source,
    classify_clinical_finding_status
)

client = TestClient(app)

CBCT_TRANSCRIPT = (
    "DENTALSCAN OFFICE: Dr Name Surname REPORT TYPE: CBCT evaluation for dental implants "
    "on the LL5 and LL6 regions, noting medical history of thyroid disease. "
    "PATIENT: Name Surname DOB: 3/7/1938 AGE: 83 GENDER: Female SERIAL #: 6640 "
    "STUDY DATE: 8/20/2021 REPORT DATE: 3/9/2021 "
    "CLINICAL NOTES: Pre-surgical assessment for dental implants on teeth LL5 and LL6. "
    "DENTAL FINDINGS: Dentate adult in the focused field of view except for missing teeth. "
    "The posterior teeth are heavily restored. Two teeth are endodontically restored with "
    "compromised apical periodontal ligament (PDL) spaces, and evidence of periapical pathosis "
    "is observed on three teeth. Temporomandibular joints, paranasal sinuses, airway, and cervical "
    "spine are outside or not within the field of view. No abnormal calcifications are identified. "
    "IMPRESSIONS & SPECIFIC FINDINGS: Edentulous LL7 region exhibits a well-defined but incomplete "
    "unilocular radiolucency measuring 7 mm in diameter, with an incomplete anterior cortical border "
    "extending posteriorly beyond the scan volume. The lesion is located superior to and does not "
    "involve the inferior alveolar nerve canal (IANC), extending superiorly to the alveolar "
    "crest/marrow junction with no expansion or frank erosion of the cortical plates. "
    "Differential diagnosis suggests a probable retained radicular cyst and less likely an "
    "odontogenic keratocyst; further evaluation is recommended prior to implant placement. "
    "LL5 and LL6 crowns are missing. The LL6 mesial root is endodontically treated and underfilled, "
    "showing a 4 mm wide apical radiolucency. The LL5 root is also endodontically treated and "
    "underfilled, demonstrating a 0.5 to 1 mm widening of the apical periodontal ligament space. "
    "LR1, located at the anterior edge of the scan field, exhibits an apical radiolucency 2 mm wide "
    "accompanied by severe loss of bone. Localized severe periodontal bone loss is noted at teeth "
    "LL1 and LR1. Third molar LL8 is partially captured within the scan. Apical pathology "
    "identified at LL5, LL6, and LR1 is likely indicative of apical granulomas or radicular cysts "
    "secondary to chronic apical periodontitis. "
    "RADIOLOGY REPORT DETAILS: Conducted by Dr. Douglas K Benn, BDS, DDS, PhD, Dipl. Dental "
    "Radiology (Royal College of Radiologists), Oral and Maxillofacial Radiologist (GDC: 297167). "
    "Dental Scan Ltd., Suite 17, 75 Harley St, London W1G 8QL."
)


def test_cbct_dictation_metadata_separation():
    """
    Defect 1 & Defect 6 Verification:
    - Administrative metadata, dates, serial number, clinical notes, radiologist credentials,
      and address are extracted into structured metadata fields, NOT as clinical findings.
    - Serial # 6640 is preserved in report_metadata, NOT confused with Patient ID.
    - Conflicting dates (study date 8/20/2021 vs report date 3/9/2021) are preserved without distortion.
    """
    response = client.post("/dictation/structure", json={"text": CBCT_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    meta = data["report_metadata"]
    ctx = data["patient_context"]

    # Demographic & Contextual verification
    assert ctx["age"] == 83
    assert ctx["gender"] == "Female"
    assert ctx["date_of_birth"] == "3/7/1938"
    assert ctx["supplied_patient_name"] == "Name Surname"
    # Serial number is NOT patient ID
    assert ctx["patient_id"] == "None"
    assert meta.get("serial_number") == "6640"

    # Conflicting study and report dates preserved
    assert meta.get("study_date") == "8/20/2021"
    assert meta.get("report_date") == "3/9/2021"

    # Provider & clinic details preserved in metadata
    prov = meta.get("provider_details") or {}
    assert "Douglas K Benn" in prov.get("provider_name", "")
    assert prov.get("gdc_number") == "297167"
    assert "Harley St" in prov.get("address", "")

    # Findings must contain zero metadata items
    findings = data["findings"]
    for f in findings:
        obs_l = f["observation"].lower()
        assert "serial #:" not in obs_l
        assert "6640" not in obs_l
        assert "gdc:" not in obs_l
        assert "harley st" not in obs_l
        assert "conducted by dr" not in obs_l
        assert "radiology report details" not in obs_l
        assert "study date:" not in obs_l
        assert "report date:" not in obs_l


def test_cbct_dictation_anatomical_attribution():
    """
    Defect 2 Verification:
    - Only observations with explicit single-tooth evidence receive a tooth_number.
    - Multi-tooth findings (LL5 & LL6 crowns missing, teeth LL1 & LR1 bone loss,
      LL5/LL6/LR1 apical pathology) do NOT inherit a single tooth number.
    - Edentulous LL7 region does NOT inherit tooth 37.
    """
    response = client.post("/dictation/structure", json={"text": CBCT_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()
    findings = data["findings"]

    # 1. Single tooth findings
    ll6_finding = next((f for f in findings if "ll6 mesial root" in f["observation"].lower()), None)
    assert ll6_finding is not None
    assert ll6_finding["tooth_number"] == "36"

    ll5_finding = next((f for f in findings if "ll5 root is also" in f["observation"].lower()), None)
    assert ll5_finding is not None
    assert ll5_finding["tooth_number"] == "35"

    lr1_finding = next((f for f in findings if "lr1, located at the anterior edge" in f["observation"].lower()), None)
    assert lr1_finding is not None
    assert lr1_finding["tooth_number"] == "41"

    ll8_finding = next((f for f in findings if "third molar ll8" in f["observation"].lower()), None)
    assert ll8_finding is not None
    assert ll8_finding["tooth_number"] == "38"

    # 2. Multi-tooth observations must NOT have single tooth numbers
    crowns_finding = next((f for f in findings if "ll5 and ll6 crowns are missing" in f["observation"].lower()), None)
    assert crowns_finding is not None
    assert crowns_finding["tooth_number"] is None

    periodontal_finding = next((f for f in findings if "teeth ll1 and lr1" in f["observation"].lower()), None)
    assert periodontal_finding is not None
    assert periodontal_finding["tooth_number"] is None

    multi_apical = next((f for f in findings if "apical pathology identified at ll5, ll6, and lr1" in f["observation"].lower()), None)
    assert multi_apical is not None
    assert multi_apical["tooth_number"] is None

    # 3. Edentulous region must NOT have a tooth number
    edentulous_finding = next((f for f in findings if "edentulous ll7" in f["observation"].lower()), None)
    assert edentulous_finding is not None
    assert edentulous_finding["tooth_number"] is None


def test_cbct_dictation_lesion_extent_polarity():
    """
    Defect 4 Verification:
    - The lesion extent description ('lesion is located superior to and does not involve IANC...
      with no expansion or frank erosion') must be classified as positive/present observation,
      NOT negative or absent.
    """
    response = client.post("/dictation/structure", json={"text": CBCT_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()
    findings = data["findings"]

    ianc_finding = next((f for f in findings if "inferior alveolar nerve canal" in f["observation"].lower()), None)
    assert ianc_finding is not None
    assert ianc_finding["polarity"] == "present"
    assert ianc_finding["certainty"] == "definite observation"


def test_cbct_dictation_recommendations_and_impressions_purity():
    """
    Defect 3 & Defect 5 Verification:
    - Recommendations ('further evaluation is recommended...') are cleanly extracted into recommendations.
    - Impressions summarize clinical observations without administrative metadata, headings, or recommendations.
    """
    response = client.post("/dictation/structure", json={"text": CBCT_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    # Recommendations extracted
    recs = data["recommendations"]
    assert len(recs) >= 1
    assert any("further evaluation is recommended" in r.lower() for r in recs)

    # Impressions purity
    impressions = data["impressions"]
    assert len(impressions) >= 3
    joined_imp = " ".join(impressions).lower()
    assert "serial #:" not in joined_imp
    assert "6640" not in joined_imp
    assert "dr. douglas" not in joined_imp
    assert "harley st" not in joined_imp
    assert "radiology report details" not in joined_imp
    assert "further evaluation is recommended" not in joined_imp


def test_cbct_dictation_safety_guardrails_pass():
    """
    Defect 7 Verification:
    - Comprehensive multi-dimensional verification passes cleanly on the properly parsed transcript.
    """
    response = client.post("/dictation/structure", json={"text": CBCT_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()

    guardrails = data["safety_guardrails"]
    assert guardrails["status"] == "PASSED"
    assert guardrails["grounded_ratio"] == 1.0
    assert guardrails["claim_support_ratio"] == 1.0
    assert guardrails["measurement_preservation"] == 1.0
    assert guardrails["unsupported_findings"] == []


def test_negative_metadata_in_finding_is_rejected():
    """
    Negative Test 1: Administrative metadata masquerading as a finding must be rejected.
    """
    bad_finding = [{
        "anatomical_site": "Dentition",
        "tooth_number": None,
        "category": "General Observation",
        "observation": "SERIAL #: 6640 STUDY DATE: 8/20/2021",
        "polarity": "present",
        "certainty": "definite observation",
        "measurements": []
    }]
    v = verify_dictation_against_source(CBCT_TRANSCRIPT, bad_finding)
    assert v["status"] != "PASSED"
    assert any("SERIAL #: 6640" in item for item in v["unsupported_findings"])


def test_negative_multi_tooth_forced_to_single_is_rejected():
    """
    Negative Test 2: Forcing a multi-tooth finding to a single tooth must be rejected.
    """
    bad_finding = [{
        "anatomical_site": "Mandibular left second premolar (┌ 5)",
        "tooth_number": "35",
        "category": "Tooth Condition & Missing Teeth",
        "observation": "LL5 and LL6 crowns are missing.",
        "polarity": "present",
        "certainty": "definite observation",
        "measurements": []
    }]
    v = verify_dictation_against_source(CBCT_TRANSCRIPT, bad_finding)
    assert v["status"] != "PASSED"
    assert any("LL5 and LL6 crowns are missing" in item for item in v["unsupported_findings"])


def test_negative_lesion_extent_marked_negative_is_rejected():
    """
    Negative Test 3: Marking active lesion extent as negative/absent must be rejected.
    """
    bad_finding = [{
        "anatomical_site": "Inferior Alveolar Canal & Alveolar Crest",
        "tooth_number": None,
        "category": "Bone & Osseous Architecture",
        "observation": "The lesion is located superior to and does not involve the inferior alveolar nerve canal (IANC), extending superiorly to the alveolar crest/marrow junction with no expansion or frank erosion of the cortical plates.",
        "polarity": "absent",
        "certainty": "negative / normal finding",
        "measurements": []
    }]
    v = verify_dictation_against_source(CBCT_TRANSCRIPT, bad_finding)
    assert v["status"] != "PASSED"
    assert any("inferior alveolar nerve canal" in item.lower() for item in v["unsupported_findings"])


def test_negative_impression_with_metadata_is_rejected():
    """
    Negative Test 4: Contaminating impression with metadata must be rejected.
    """
    good_finding = [{
        "anatomical_site": "Mandibular left first molar (┌ 6)",
        "tooth_number": "36",
        "category": "Endodontic & Periapical Status",
        "observation": "The LL6 mesial root is endodontically treated and underfilled, showing a 4 mm wide apical radiolucency.",
        "polarity": "present",
        "certainty": "definite observation",
        "measurements": ["4 mm"]
    }]
    bad_impressions = ["SERIAL #: 6640 - Patient Name Surname conducted by Dr. Douglas K Benn"]
    v = verify_dictation_against_source(CBCT_TRANSCRIPT, good_finding, impressions=bad_impressions)
    assert v["status"] != "PASSED"
    assert any("SERIAL #: 6640" in item for item in v["unsupported_findings"])


def test_confirm_and_edit_endpoint_persists_json_and_rerenders_pdf():
    """
    Confirmation & Edit Pipeline Verification:
    - User can edit draft, assign registered patient, and re-confirm.
    - Updated JSON and re-rendered PDF are persisted and verified.
    """
    # 1. Structure initial draft
    init_res = client.post("/dictation/structure", json={"text": CBCT_TRANSCRIPT})
    assert init_res.status_code == 200
    draft_data = init_res.json()
    json_path = draft_data["dictation_json_path"]
    pdf_path = draft_data["dictation_pdf_path"]

    assert os.path.exists(json_path)
    assert os.path.exists(pdf_path)
    assert draft_data["report_metadata"]["status"] == "DRAFT_PENDING_REVIEW"
    assert draft_data["report_metadata"]["clinician_review_required"] is True

    # 2. Clinician performs edits and confirms report
    confirm_payload = {
        "dictation_json_path": json_path,
        "patient_name": "Evelyn Wright",
        "patient_id": "PAT-9002",
        "findings": draft_data["findings"],
        "impressions": draft_data["impressions"],
        "recommendations": draft_data["recommendations"]
    }
    confirm_res = client.post("/dictation/confirm", json=confirm_payload)
    assert confirm_res.status_code == 200
    confirmed = confirm_res.json()

    # 3. Status is now confirmed and review is no longer required
    assert confirmed["report_metadata"]["status"] == "CONFIRMED_BY_CLINICIAN"
    assert confirmed["report_metadata"]["clinician_review_required"] is False
    assert confirmed["patient_context"]["patient_name"] == "Evelyn Wright"
    assert confirmed["patient_context"]["patient_id"] == "PAT-9002"
    assert confirmed["patient_context"]["patient_assigned"] is True

    # 4. JSON file was updated
    with open(confirmed["dictation_json_path"], "r") as f:
        saved_json = json.load(f)
    assert saved_json["report_metadata"]["status"] == "CONFIRMED_BY_CLINICIAN"
    assert saved_json["patient_context"]["patient_name"] == "Evelyn Wright"

    # 5. PDF file was generated
    assert os.path.exists(confirmed["dictation_pdf_path"])
    assert os.path.getsize(confirmed["dictation_pdf_path"]) > 1000
