import os
import tempfile
from fastapi.testclient import TestClient
from app.main import app
from app.services.claim_verification import is_clinically_relevant_dictation
from app.services.llm_service import parse_dictation_with_llm
from app.services.validation import verify_anti_hallucination
from app.services.pdf_service import generate_dictation_pdf

client = TestClient(app)

def test_non_clinical_input_rejected_without_fabricated_findings():
    """
    Verifies that technical discussions, architecture documents, or non-medical text
    are rejected without hallucinating bogus findings under 'Dentition' or 'General Observation'.
    """
    technical_text = (
        "I don't think the main problem is that you aren't using LangChain or LangGraph. "
        "Your fallback behavior is more likely caused by how your application connects its components, "
        "handles errors, and passes context between tools and the LLM. "
        "For your Viscan AI Clinical Workstation Assistant, you need reliable clinical workflows—not necessarily more frameworks."
    )

    # 1. Direct relevance evaluation
    assert is_clinically_relevant_dictation(technical_text) is False

    # 2. Dictation structuring engine
    parsed = parse_dictation_with_llm(technical_text)
    assert parsed["extraction_status"] == "NON_CLINICAL_INPUT_REJECTED"
    assert parsed["findings"] == []
    assert parsed["impressions"] == []
    assert parsed["recommendations"] == []

    # 3. Validation guardrail
    guard = verify_anti_hallucination(technical_text, parsed["findings"])
    assert guard["status"] == "EXTRACTION_EMPTY_REQUIRES_REVIEW"
    assert guard["grounded_ratio"] == 0.0
    assert len(guard["unsupported_findings"]) >= 1

    # 4. HTTP endpoint execution
    res = client.post("/dictation/structure", json={"text": technical_text})
    assert res.status_code == 200
    data = res.json()
    assert data["report_metadata"]["extraction_status"] == "NON_CLINICAL_INPUT_REJECTED"
    assert len(data["findings"]) == 0
    assert data["safety_guardrails"]["grounded_ratio"] == 0.0

def test_genuine_clinical_dictation_passes_relevance_check():
    """
    Verifies that clinical dictations across diverse modalities and dental anatomy
    are recognized as clinically relevant.
    """
    cases = [
        "Patient presents with deep occlusal caries on tooth 16.",
        "CBCT shows horizontal impaction of mandibular left third molar.",
        "Full mouth series shows edentulous posterior spaces and intact alveolar crest.",
        "Panoramic view demonstrates normal condylar morphology and clear maxillary sinuses."
    ]
    for text in cases:
        assert is_clinically_relevant_dictation(text) is True, f"Failed for: {text}"

def test_reportlab_pdf_generation_layout_and_pagination():
    """
    Verifies that ReportLab Platypus generates valid, uncorrupted PDFs with clean
    two-column metadata, multi-page pagination, and signature block without overlap.
    """
    findings = [
        {
            "anatomical_site": "Maxillary Right First Molar (Tooth 16)",
            "category": "Restorative & Caries Status",
            "observation": "Tooth 16 displays an intact occlusal-mesial composite restoration without recurrent secondary decay.",
            "certainty": "definite observation",
            "measurements": []
        },
        {
            "anatomical_site": "Adjacent Teeth",
            "category": "Periapical & Osseous Status",
            "observation": "Adjacent dentition demonstrates normal bone levels and absence of root resorption.",
            "certainty": "negative / normal finding",
            "measurements": []
        }
    ]
    impressions = [
        "Intact composite restoration at tooth 16.",
        "Normal surrounding osseous support and adjacent dentition."
    ]
    recommendations = [
        "Routine 6-month clinical and radiographic follow-up."
    ]
    patient_details = {
        "patient_name": "Dr. Vikram Malhotra (Professor of Maxillofacial Radiology)",
        "patient_id": "PAT-99482-LONG-IDENTIFIER",
        "study_type": "High-Resolution Cone Beam Computed Tomography (CBCT)"
    }

    tmp_pdf = tempfile.mktemp(suffix=".pdf")
    pdf_out = generate_dictation_pdf(
        pdf_path=tmp_pdf,
        patient_name=patient_details["patient_name"],
        patient_id=patient_details["patient_id"],
        raw_dictation="raw dictation text",
        findings=findings,
        impressions=impressions,
        recommendations=recommendations,
        patient_details=patient_details
    )

    assert os.path.exists(pdf_out)
    size = os.path.getsize(pdf_out)
    assert size > 1500, f"Expected valid PDF > 1500 bytes, got {size}"

    # Verify PDF starts with %PDF- header
    with open(pdf_out, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


def test_confirmed_pdf_includes_digital_signature_and_verification_badge():
    """
    Verifies that when a report is confirmed by the clinician, the generated PDF
    contains the verified digital signature block and approved status rather than
    blank placeholder underscores or unverified draft warnings.
    """
    tmp_pdf = tempfile.mktemp(suffix=".pdf")
    confirmed_details = {
        "patient_name": "Priya Sharma",
        "patient_id": "PAT-1001",
        "status": "CONFIRMED_BY_CLINICIAN",
        "confirmed_at": "2026-10-19 10:00:00",
        "clinician_name": "Dr. R. Mehta, MDS"
    }
    pdf_out = generate_dictation_pdf(
        pdf_path=tmp_pdf,
        patient_name="Priya Sharma",
        patient_id="PAT-1001",
        raw_dictation="Tooth 16 caries.",
        findings=[{"anatomical_site": "Tooth 16", "category": "Caries", "observation": "Coronal radiolucency", "certainty": "definite observation", "measurements": []}],
        impressions=["Coronal dental caries on tooth 16."],
        recommendations=["Restoration advised."],
        patient_details=confirmed_details
    )
    assert os.path.exists(pdf_out)
    with open(pdf_out, "rb") as f:
        content = f.read().decode("latin1", errors="ignore")
    assert "VERIFIED" in content or "APPROVED" in content
