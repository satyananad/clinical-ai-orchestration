import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.llm_service import parse_dictation_with_llm, extract_clinical_body
from backend.app.services.report_service import compose_narrative, is_symptom_or_history
from backend.app.services.pdf_service import generate_dictation_pdf

client = TestClient(app)

TEST_MIXED_PROMPT = (
    "Patient complains of loose teeth in the lower front region. "
    "Horizontal bone loss is noted around the lower central incisors, bone loss measurements were not provided. "
    "The upper right first molar is missing, reason for tooth loss is unknown. "
    "Metallic restoration is present in the lower left first molar, condition beneath the restoration was not described. "
    "Generate a clean, editable report with a clinic heading, report title, patient and examination details, clinical history, "
    "findings, impression, recommendations when supported by the dictation, and a final clinician-review statement. "
    "Keep patient details in a compact two-column layout and write findings as concise, coherent clinical paragraphs "
    "grouped by anatomical site or relevant clinical topic. The impression must summarize the important findings without "
    "unnecessarily repeating the entire findings section. Separate patient-reported symptoms from imaging findings, "
    "preserve laterality and anatomical terminology, and clearly state when information is indeterminate or was not provided. "
    "Do not invent negative findings, diagnoses, measurements, canal involvement, treatment recommendations, or imaging observations. "
    "If the information comes only from speech-to-text, explicitly indicate that the findings are based on supplied dictation and "
    "have not been independently verified against the images. Generate structured JSON as the source of truth and render the "
    "readable report consistently from that JSON. Keep the JSON and report editable and synchronize"
)

def test_extract_clinical_body_truncates_at_instruction_boundary():
    """Verify that extract_clinical_body stops at the imperative instruction boundary."""
    clinical_body = extract_clinical_body(TEST_MIXED_PROMPT)
    # The clinical body must contain the genuine dictation
    assert "Patient complains of loose teeth in the lower front region" in clinical_body
    assert "Horizontal bone loss is noted around the lower central incisors" in clinical_body
    assert "upper right first molar is missing" in clinical_body
    assert "Metallic restoration is present in the lower left first molar" in clinical_body
    # Must NOT contain instructions
    assert "Generate a clean, editable report" not in clinical_body
    assert "Keep patient details in a compact two-column layout" not in clinical_body
    assert "Separate patient-reported symptoms" not in clinical_body
    assert "Do not invent" not in clinical_body

def test_patient_demographics_does_not_extract_instructions_as_name():
    """Verify that 'patient-reported symptoms' is not extracted as patient_name."""
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    patient_details = res.get("patient_details", {})
    patient_name = patient_details.get("patient_name")
    assert patient_name is None, f"Expected patient_name to be None, got: {patient_name}"
    assert patient_name != "Reported Symptoms"

def test_clinical_history_preserves_loose_teeth_symptom():
    """Verify that the patient's complaint of loose teeth in lower front region is preserved."""
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    findings = res.get("findings", [])
    history_findings = [f for f in findings if is_symptom_or_history(f)]
    assert len(history_findings) >= 1
    symptom_text = history_findings[0]["observation"].lower()
    assert "loose teeth" in symptom_text
    assert "lower front region" in symptom_text or "mandibular" in symptom_text

def test_no_instruction_leakage_in_findings():
    """Verify that zero instruction sentences leak into the findings list."""
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    findings = res.get("findings", [])
    forbidden_instruction_keywords = [
        "generate a clean", "two-column layout", "clinic heading", "report title",
        "separate patient-reported", "do not invent", "speech-to-text", "source of truth",
        "keep the json", "clinician-review statement"
    ]
    for f in findings:
        obs = f.get("observation", "").lower()
        for kw in forbidden_instruction_keywords:
            assert kw not in obs, f"Instruction keyword '{kw}' leaked into observation: {obs}"

def test_imaging_findings_preserve_all_three_anatomical_sites_and_uncertainties():
    """
    Verify:
    1. Horizontal bone loss around lower central incisors (Tooth 31, 41) with bone loss measurements omitted/not provided.
    2. Upper right first molar missing (Tooth 16 / UR6) with unknown reason for tooth loss.
    3. Lower left first molar metallic restoration (Tooth 36 / LL6) with undescribed sub-restoration condition.
    """
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    findings = res.get("findings", [])
    imaging_findings = [f for f in findings if not is_symptom_or_history(f)]
    assert len(imaging_findings) == 3

    # Finding 1: Lower central incisors bone loss
    incisor_finding = next((f for f in imaging_findings if "lower central incisor" in f["observation"].lower() or "31" in str(f.get("anatomical_site"))), None)
    assert incisor_finding is not None
    assert "horizontal bone loss" in incisor_finding["observation"].lower()
    assert "bone loss measurements were not provided" in incisor_finding["observation"].lower()
    assert incisor_finding["measurements"] == []

    # Finding 2: Missing upper right first molar
    ur6_finding = next((f for f in imaging_findings if "upper right first molar" in f["observation"].lower() or "16" in str(f.get("tooth_number")) or "6 ┘" in str(f.get("anatomical_site"))), None)
    assert ur6_finding is not None
    assert "missing" in ur6_finding["observation"].lower()
    assert "reason for tooth loss is unknown" in ur6_finding["observation"].lower()

    # Finding 3: Lower left first molar metallic restoration
    ll6_finding = next((f for f in imaging_findings if "lower left first molar" in f["observation"].lower() or "36" in str(f.get("tooth_number")) or "┌ 6" in str(f.get("anatomical_site"))), None)
    assert ll6_finding is not None
    assert "metallic restoration" in ll6_finding["observation"].lower()
    assert "condition beneath the restoration was not described" in ll6_finding["observation"].lower()

def test_impressions_concise_and_free_of_instructions_or_symptoms():
    """Verify that impression concisely summarizes imaging findings without instruction leakage or symptoms."""
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    impressions = res.get("impressions", [])
    assert len(impressions) >= 2
    for imp in impressions:
        imp_l = imp.lower()
        # Must not contain instructions
        assert "generate" not in imp_l
        assert "compact" not in imp_l
        assert "source of truth" not in imp_l
        assert "two-column" not in imp_l
        # Must not contain patient complaint
        assert "patient complains of loose teeth" not in imp_l

def test_recommendations_no_hallucination():
    """Verify that recommendations are strictly empty since none were dictated."""
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    assert res.get("recommendations") == []

def test_narrative_and_pdf_export_synchronization():
    """Verify that the rendered consultation narrative and PDF export are synchronized."""
    res = parse_dictation_with_llm(TEST_MIXED_PROMPT)
    narrative = compose_narrative(
        "Pending Patient Linkage (Unassigned)",
        res.get("findings", []),
        res.get("impressions", []),
        res.get("recommendations", []),
        patient_details=res.get("patient_details")
    )
    # Check structure of narrative
    assert "VISCAN MAXILLOFACIAL & DENTAL RADIOLOGY CLINIC" in narrative
    assert "CLINICAL HISTORY & PATIENT-REPORTED SYMPTOMS:" in narrative
    assert "Patient complains of loose teeth in the lower front region." in narrative
    assert "DETAILED RADIOLOGICAL FINDINGS:" in narrative
    assert "Horizontal bone loss is noted around the lower central incisors" in narrative
    assert "The upper right first molar is missing" in narrative
    assert "Metallic restoration is present in the lower left first molar" in narrative
    assert "IMPRESSION:" in narrative
    assert "CLINICIAN REVIEW & ATTESTATION STATEMENT:" in narrative

    # Ensure no instructions appear in narrative
    assert "Generate a clean, editable report" not in narrative
    assert "Keep patient details in a compact two-column layout" not in narrative

    # Verify PDF generation succeeds
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_file = tmp.name
    res_path = generate_dictation_pdf(
        pdf_path=pdf_file,
        patient_name="Pending Patient Linkage (Unassigned)",
        patient_id="None",
        raw_dictation=TEST_MIXED_PROMPT,
        findings=res.get("findings", []),
        impressions=res.get("impressions", []),
        recommendations=res.get("recommendations", []),
        patient_details=res.get("patient_details")
    )
    with open(res_path, "rb") as f:
        pdf_bytes = f.read()
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

def test_api_endpoint_dictation_structure_with_mixed_prompt():
    """Verify the /dictation/structure FastAPI endpoint handles the mixed prompt cleanly."""
    response = client.post("/dictation/structure", json={"text": TEST_MIXED_PROMPT})
    assert response.status_code == 200
    data = response.json()
    assert data["patient_context"]["patient_name"] == "Unassigned / Pending Patient Selection"
    assert data["patient_context"]["patient_assigned"] is False
    assert len(data["findings"]) == 4
    assert data["recommendations"] == []
    assert "Generate a clean" not in data.get("narrative_report", "")
