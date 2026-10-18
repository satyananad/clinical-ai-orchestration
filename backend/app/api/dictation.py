import os
import json
import datetime
from typing import Optional
from fastapi import APIRouter
from ..schemas.report import DictationRequest, DictationReportResponse, ConfirmDictationRequest
from ..services.llm_service import parse_dictation_with_llm
from ..services.validation import verify_anti_hallucination
from ..services.report_service import compose_narrative
from ..services.pdf_service import generate_dictation_pdf, sanitize_pdf, wrap_lines

router = APIRouter(prefix="/dictation", tags=["Dictation"])

DICTATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dictation_reports")
os.makedirs(DICTATION_DIR, exist_ok=True)

@router.post("/structure", response_model=DictationReportResponse)
def structure_dictation_endpoint(payload: DictationRequest):
    """
    Feature B: Clinical dictation to validated editable structured JSON draft report.
    - Uses LLM instruction reasoning to extract sites, categories, and certainty.
    - Anti-hallucination verification.
    - Requires clinician review.
    """
    patient_id = payload.patient_id
    patient_name = payload.patient_name
    dictation_text = payload.text.strip()

    display_name = patient_name if patient_name else "Unassigned / Pending Patient Selection"
    display_id = patient_id if patient_id else "None"

    # 1. Invoke Clinical Scribe Agent via standard MCP tool execution
    from ..services.mcp_tools import execute_tool
    tool_res = execute_tool("structure_clinical_dictation", {
        "dictation_text": dictation_text,
        "patient_id": patient_id,
        "patient_name": patient_name
    })
    sites = tool_res.get("anatomical_sites", [])
    findings = tool_res.get("findings", [])
    impressions = tool_res.get("impressions", [])
    recommendations = tool_res.get("recommendations", [])
    guardrails = tool_res.get("safety_guardrails", {})
    patient_details = tool_res.get("patient_details") or {}
    age = patient_details.get("age")
    gender = patient_details.get("gender")
    demographics_summary = patient_details.get("demographics_summary")

    # [DIAGNOSTIC] Checkpoint 1: Immediately after parser returns
    print(f"\n[DIAGNOSTIC] Checkpoint 1 (Extraction): findings={len(findings)}, impressions={len(impressions)}, recommendations={len(recommendations)}, sites={len(sites)}")

    # [DIAGNOSTIC] Checkpoint 2: Immediately after verification runs
    print(f"[DIAGNOSTIC] Checkpoint 2 (Verification): status={guardrails.get('status')}, grounded_ratio={guardrails.get('grounded_ratio')}, claim_support={guardrails.get('claim_support_ratio')}, meas_preservation={guardrails.get('measurement_preservation')}")

    # Determine clinical completeness status
    report_status = "DRAFT_PENDING_REVIEW" if findings else "INCOMPLETE_REQUIRES_REVIEW"

    # 2. Compose narrative draft
    narrative = compose_narrative(display_name, findings, impressions, recommendations, patient_details=patient_details)

    # 4. Save JSON and PDF reports to dictation_reports/
    clean_name = (patient_name or "Unassigned_Patient").replace(" ", "_")
    pid = (patient_id or "UNASSIGNED").replace(" ", "_")
    json_filename = f"{pid}_{clean_name}_Dictation_Report.json"
    pdf_filename = f"{pid}_{clean_name}_Dictation_Report.pdf"
    json_path = os.path.join(DICTATION_DIR, json_filename)
    pdf_path = os.path.join(DICTATION_DIR, pdf_filename)

    # [DIAGNOSTIC] Checkpoint 3: Immediately before saving JSON and PDF
    print(f"[DIAGNOSTIC] Checkpoint 3 (Saving Artifacts): status={report_status}, json={json_filename}, pdf={pdf_filename}\n")

    structured_data = {
        "report_metadata": {
            "report_type": patient_details.get("study_type") or "Radiological Dictation Draft",
            "study_type": patient_details.get("study_type"),
            "status": report_status,
            "extraction_status": tool_res.get("extraction_status", "COMPLETE" if findings else "INCOMPLETE_REQUIRES_REVIEW"),
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_type": "Spoken Voice Dictation / Speech-to-Text Transcript",
            "clinician_review_required": True,
            "serial_number": patient_details.get("serial_number"),
            "study_date": patient_details.get("study_date"),
            "report_date": patient_details.get("report_date"),
            "provider_details": patient_details.get("provider_details"),
            "clinical_notes": patient_details.get("clinical_notes"),
            "medical_history": patient_details.get("medical_history"),
            "disclaimer": "ATTENTION: This is an unverified AI-generated draft created from clinical dictation. All anatomical findings and impressions must be reviewed and approved by the attending clinician."
        },
        "patient_context": {
            "patient_name": display_name,
            "patient_id": display_id,
            "patient_assigned": bool(patient_name or patient_id),
            "age": age,
            "gender": gender,
            "date_of_birth": patient_details.get("date_of_birth"),
            "demographics_summary": demographics_summary,
            "supplied_patient_name": patient_details.get("patient_name"),
            "supplied_patient_id": patient_details.get("patient_id"),
            "study_type": patient_details.get("study_type")
        },
        "raw_dictation": dictation_text,
        "anatomical_sites": sites,
        "findings": findings,
        "impressions": impressions,
        "recommendations": recommendations,
        "narrative_draft": narrative,
        "safety_guardrails": guardrails,
        "dictation_json_path": json_path,
        "dictation_pdf_path": pdf_path
    }

    with open(json_path, "w") as f:
        json.dump(structured_data, f, indent=2)

    # Generate authentic PDF report
    generate_dictation_pdf(pdf_path, display_name, display_id, dictation_text, findings, impressions, recommendations, patient_details=patient_details)

    return structured_data


@router.post("/confirm", response_model=DictationReportResponse)
@router.put("/confirm", response_model=DictationReportResponse)
def confirm_dictation_endpoint(payload: ConfirmDictationRequest):
    """
    Feature B: Confirmation & Edit Pipeline.
    - Accepts clinician review, manual edits, or patient assignment.
    - Re-validates safety guardrails and anti-hallucination guarantees.
    - Marks status as CONFIRMED_BY_CLINICIAN (clinician_review_required = False).
    - Persists confirmed JSON and re-renders PDF report.
    """
    base_data = {}
    json_path = payload.dictation_json_path
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                base_data = json.load(f)
        except Exception:
            base_data = {}

    # Extract or fallback values
    raw_dictation = payload.raw_dictation or base_data.get("raw_dictation", "")
    report_metadata = payload.report_metadata or base_data.get("report_metadata", {})
    patient_context = payload.patient_context or base_data.get("patient_context", {})
    anatomical_sites = base_data.get("anatomical_sites", [])

    # Update patient assignment if supplied
    p_name = payload.patient_name or patient_context.get("patient_name", "Unassigned / Pending Patient Selection")
    p_id = payload.patient_id or patient_context.get("patient_id", "None")
    is_assigned = bool(payload.patient_name or payload.patient_id or patient_context.get("patient_assigned", False))
    patient_context["patient_name"] = p_name
    patient_context["patient_id"] = p_id
    patient_context["patient_assigned"] = is_assigned

    # Update findings if supplied
    if payload.findings is not None:
        findings = [f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in payload.findings]
    else:
        findings = base_data.get("findings", [])

    # Update impressions if supplied
    if payload.impressions is not None:
        impressions = payload.impressions
    else:
        impressions = base_data.get("impressions", [])

    # Update recommendations if supplied
    if payload.recommendations is not None:
        recommendations = payload.recommendations
    else:
        recommendations = base_data.get("recommendations", [])

    # Re-run safety guardrails / claim verification
    guardrails = verify_anti_hallucination(raw_dictation, findings, impressions=impressions)

    # Update report metadata for confirmed status
    report_metadata["status"] = "CONFIRMED_BY_CLINICIAN"
    report_metadata["clinician_review_required"] = False
    report_metadata["confirmed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Re-compose narrative draft with confirmed details
    narrative = compose_narrative(p_name, findings, impressions, recommendations, patient_details=patient_context)

    # Determine file paths
    clean_name = (p_name or "Unassigned_Patient").replace(" ", "_")
    pid = (p_id or "UNASSIGNED").replace(" ", "_")
    if not json_path:
        json_path = os.path.join(DICTATION_DIR, f"{pid}_{clean_name}_Dictation_Report.json")
    pdf_path = base_data.get("dictation_pdf_path") or os.path.join(DICTATION_DIR, f"{pid}_{clean_name}_Dictation_Report.pdf")

    confirmed_data = {
        "report_metadata": report_metadata,
        "patient_context": patient_context,
        "raw_dictation": raw_dictation,
        "anatomical_sites": anatomical_sites,
        "findings": findings,
        "impressions": impressions,
        "recommendations": recommendations,
        "narrative_draft": narrative,
        "safety_guardrails": guardrails,
        "dictation_json_path": json_path,
        "dictation_pdf_path": pdf_path
    }

    with open(json_path, "w") as f:
        json.dump(confirmed_data, f, indent=2)

    # Re-render authentic PDF report from confirmed data
    generate_dictation_pdf(pdf_path, p_name, p_id, raw_dictation, findings, impressions, recommendations, patient_details=patient_context)

    return confirmed_data

