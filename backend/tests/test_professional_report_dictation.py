import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.report_service import compose_narrative, is_symptom_or_history
from backend.app.services.pdf_service import generate_dictation_pdf

client = TestClient(app)

def test_professional_report_dictation_workflow():
    """
    Comprehensive verification for Feature B professional dental radiology report generator:
    1. Given sample transcript with symptoms, impaction, and indeterminate canal relationship.
    2. Generates clean structured JSON as source of truth.
    3. Separates patient-reported symptoms from objective imaging findings.
    4. Groups findings by anatomical site into concise clinical paragraphs (not a numbered list).
    5. Preserves laterality, FDI/Universal tooth numbering, and labels uncertainty.
    6. Generates synchronized readable on-screen report and professional PDF export.
    7. Enforces clinician review and attestation before finalization.
    """
    dictation_text = (
        "patient reports discomfort in the lower right posterior region. "
        "The lower right third molar is horizontally impacted. "
        "The relationship between the tooth and the inferior alveolar canal cannot be determined from the information provided."
    )

    # 1. Structure dictation via API endpoint
    response = client.post("/dictation/structure", json={"text": dictation_text})
    assert response.status_code == 200
    data = response.json()

    # Verify structured JSON is source of truth
    assert data["raw_dictation"] == dictation_text
    assert data["report_metadata"]["clinician_review_required"] is True
    assert data["report_metadata"]["status"] == "DRAFT_PENDING_REVIEW"

    # Verify patient context
    assert data["patient_context"]["patient_assigned"] is False
    assert "Pending" in data["patient_context"]["patient_name"]

    # Verify findings separation
    findings = data["findings"]
    assert len(findings) >= 2

    # Check symptom is categorized as clinical symptoms/history
    symptom_finding = next((f for f in findings if is_symptom_or_history(f)), None)
    assert symptom_finding is not None
    assert "discomfort in the lower right posterior region" in symptom_finding["observation"]

    # Check impaction finding
    impaction_finding = next((f for f in findings if "impacted" in f["observation"].lower()), None)
    assert impaction_finding is not None
    assert impaction_finding["certainty"] == "definite observation"
    assert impaction_finding["tooth_number"] == "48"

    # Check indeterminate canal relationship
    canal_finding = next((f for f in findings if "canal" in f["observation"].lower()), None)
    assert canal_finding is not None
    assert canal_finding["certainty"] == "unclear / clinician review required"

    # Verify readable on-screen narrative draft
    narrative = data["narrative_draft"]

    # Clinic heading and report title
    assert "VISCAN MAXILLOFACIAL & DENTAL RADIOLOGY CLINIC" in narrative
    assert "Department of Diagnostic Radiology & Oral Imaging" in narrative
    assert "RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF" in narrative

    # Compact patient & exam details
    assert "PATIENT & EXAMINATION DETAILS:" in narrative
    assert "Patient Name:" in narrative
    assert "Patient ID:" in narrative
    assert "Demographics:" in narrative
    assert "Examination:" in narrative

    # Separate Clinical History section
    assert "CLINICAL HISTORY & PATIENT-REPORTED SYMPTOMS:" in narrative
    assert "discomfort in the lower right posterior region" in narrative

    # Detailed radiological findings grouped by anatomical site (NOT a numbered list like "1. [Dentition] • General Observation")
    assert "DETAILED RADIOLOGICAL FINDINGS:" in narrative
    assert "1. [Dentition]" not in narrative
    assert "1. [Tooth 48]" not in narrative
    assert "Tooth 48" in narrative
    assert "horizontally impacted" in narrative.lower()
    assert "indeterminate" in narrative.lower() or "cannot be determined" in narrative.lower()

    # Source disclaimer
    assert "[Source Note]: Findings are synthesized from supplied speech-to-text dictation" in narrative

    # Concise impression without repeating raw symptoms
    assert "IMPRESSION:" in narrative
    for imp in data["impressions"]:
        assert "discomfort in the lower right posterior region" not in imp.lower()

    # Recommendations & Clinician review block
    assert "RECOMMENDATIONS & CLINICAL CORRELATION:" in narrative
    assert "CLINICIAN REVIEW & ATTESTATION STATEMENT:" in narrative
    assert "[ ] I confirm that I have reviewed" in narrative

    # 2. Verify PDF generation
    pdf_path = data["dictation_pdf_path"]
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1500
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    assert pdf_bytes.startswith(b"%PDF-")

    # 3. Clinician Sign-Off and Confirmation Pipeline
    confirm_payload = {
        "dictation_json_path": data["dictation_json_path"],
        "patient_name": "Priya Sharma",
        "patient_id": "PAT-1001",
        "findings": data["findings"],
        "impressions": data["impressions"],
        "recommendations": ["Cross-sectional 3D CBCT recommended prior to surgical extraction."],
        "raw_dictation": dictation_text
    }
    confirm_resp = client.put("/dictation/confirm", json=confirm_payload)
    assert confirm_resp.status_code == 200
    conf_data = confirm_resp.json()

    assert conf_data["report_metadata"]["status"] == "CONFIRMED_BY_CLINICIAN"
    assert conf_data["report_metadata"]["clinician_review_required"] is False
    assert conf_data["patient_context"]["patient_name"] == "Priya Sharma"
    assert conf_data["patient_context"]["patient_id"] == "PAT-1001"
    assert "[X] Confirmed & Electronically Signed" in conf_data["narrative_draft"]
    assert "CLINICALLY VERIFIED & FINAL" in conf_data["narrative_draft"]

    # Verify confirmed PDF exists and has updated verification
    conf_pdf_path = conf_data["dictation_pdf_path"]
    assert os.path.exists(conf_pdf_path)
    assert os.path.getsize(conf_pdf_path) > 1500
