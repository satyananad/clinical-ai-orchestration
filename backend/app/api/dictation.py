import os
import json
import datetime
from fastapi import APIRouter
from ..schemas.report import DictationRequest, DictationReportResponse
from ..services.llm_service import parse_dictation_with_llm
from ..services.validation import verify_anti_hallucination

router = APIRouter(prefix="/dictation", tags=["Dictation"])

DICTATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dictation_reports")
os.makedirs(DICTATION_DIR, exist_ok=True)

def compose_narrative(patient_name: str, findings: list, impressions: list, recommendations: list) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("*** RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF ***")
    lines.append("=" * 68)
    lines.append(f"Patient Name:    {patient_name}")
    lines.append(f"Status:          DRAFT (AI Dictation Synthesis)")
    lines.append(f"Date of Draft:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("DETAILED FINDINGS:")
    for i, f in enumerate(findings, 1):
        lines.append(f"  {i}. [{f['anatomical_site']}] ({f['category']})")
        lines.append(f"     Observation: {f['observation']}")
        lines.append(f"     Certainty:   {f['certainty']}")
    lines.append("")
    lines.append("IMPRESSION:")
    for i, imp in enumerate(impressions, 1):
        lines.append(f"  {i}. {imp}")
    lines.append("")
    lines.append("RECOMMENDATIONS & CLINICAL CORRELATION:")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"  {i}. {rec}")
    lines.append("=" * 68)
    return "\n".join(lines)

def sanitize_pdf(text: str) -> str:
    if not text:
        return ""
    s = text.replace("•", "-").replace("—", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.encode("ascii", "replace").decode("ascii")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def wrap_lines(text: str, max_chars: int = 72) -> list:
    if not text:
        return []
    words = text.split(" ")
    lines = []
    curr = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + (1 if curr else 0) <= max_chars:
            curr.append(w)
            curr_len += len(w) + (1 if len(curr) > 1 else 0)
        else:
            if curr:
                lines.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
    if curr:
        lines.append(" ".join(curr))
    return lines

def generate_dictation_pdf(pdf_path: str, patient_name: str, patient_id: str, raw_dictation: str, findings: list, impressions: list, recommendations: list):
    cmds = [
        # Amber Banner for Draft
        "0.85 0.45 0.1 rg 45 760 505 48 re f",
        f"BT /F1 13 Tf 1 1 1 rg 55 785 Td ({sanitize_pdf('CLINICAL DICTATION DRAFT - MANDATORY CLINICIAN REVIEW')}) Tj ET",
        f"BT /F3 8.5 Tf 1 1 1 rg 55 771 Td ({sanitize_pdf('ViScan AI Scribe Network • Generated from Radiologist Speech / Dictation')}) Tj ET",

        # Patient Info Box
        "0.82 0.82 0.82 RG 1 w 45 675 505 75 re S",
        f"BT /F1 9.5 Tf 0 0 0 rg 55 732 Td ({sanitize_pdf('PATIENT: ' + patient_name.upper())}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 55 715 Td ({sanitize_pdf('PATIENT ID: ' + patient_id)}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 55 698 Td ({sanitize_pdf('STATUS: Unverified AI Synthesis (Pending Attending Sign-Off)')}) Tj ET",

        f"BT /F1 9.5 Tf 0 0 0 rg 310 732 Td ({sanitize_pdf('REPORT TYPE: Dictation Draft')}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 310 715 Td ({sanitize_pdf('DATE: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 310 698 Td ({sanitize_pdf('AI ENGINE: Llama-3.1 / Medical LLM')}) Tj ET",

        # Raw Dictation Transcript Box
        f"BT /F1 10 Tf 0.12 0.23 0.45 rg 50 655 Td ({sanitize_pdf('VERBATIM DICTATION AUDIO TRANSCRIPT')}) Tj ET"
    ]

    y = 642
    for rline in wrap_lines(f'"{raw_dictation}"', max_chars=80):
        cmds.append(f"BT /F3 8.5 Tf 0.3 0.3 0.3 rg 60 {y} Td ({sanitize_pdf(rline)}) Tj ET")
        y -= 11

    # Structured Observations
    y -= 8
    cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf('STRUCTURED FINDINGS & CERTAINTY')}) Tj ET")
    y -= 14

    for f in findings:
        site = f.get("anatomical_site", "Site")
        obs = f.get("observation", "")
        cert = f.get("certainty", "Documented")
        cat = f.get("category", "General")
        cmds.append(f"BT /F1 8.5 Tf 0.1 0.1 0.1 rg 60 {y} Td ({sanitize_pdf('- [' + site + '] (' + cat + '):')}) Tj ET")
        y -= 11

        for o_line in wrap_lines(obs, max_chars=78):
            cmds.append(f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 72 {y} Td ({sanitize_pdf(o_line)}) Tj ET")
            y -= 11

        cmds.append(f"BT /F3 8 Tf 0.4 0.4 0.4 rg 72 {y} Td ({sanitize_pdf('Diagnostic Certainty: ' + cert)}) Tj ET")
        y -= 13

    # Impressions
    y -= 4
    cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf('SYNTHESIZED IMPRESSION')}) Tj ET")
    y -= 13
    for imp in impressions:
        for iline in wrap_lines(imp, max_chars=78):
            cmds.append(f"BT /F1 8.5 Tf 0.65 0.1 0.1 rg 60 {y} Td ({sanitize_pdf('- ' + iline)}) Tj ET")
            y -= 11

    # Recommendations
    y -= 4
    cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf('RECOMMENDATIONS & REQUIRED ACTION')}) Tj ET")
    y -= 13
    for rec in recommendations:
        for r_line in wrap_lines(rec, max_chars=80):
            cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf(r_line)}) Tj ET")
            y -= 11

    # Signature Block firmly placed at bottom
    cmds.extend([
        "0.85 0.85 0.85 RG 1 w 45 105 505 0 m 550 105 l S",
        f"BT /F1 9 Tf 0 0 0 rg 50 85 Td ({sanitize_pdf('Attending Clinician Approval: ___________________________')}) Tj 230 0 Td ({sanitize_pdf('Sign-Off Date: _________________')}) Tj ET",
        f"BT /F3 8 Tf 0.4 0.4 0.4 rg 50 68 Td ({sanitize_pdf('ViScan Dental Assistant • Draft Document - Not Approved for Medical Records Without Clinician Signature')}) Tj ET"
    ])

    stream = "\n".join(cmds)
    stream_bytes = stream.encode("latin-1")
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream\nendobj\n",
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n",
        "6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        "7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>\nendobj\n"
    ]
    header = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = []
    curr = len(header.encode("latin-1"))
    body = ""
    for o in objects:
        offsets.append(curr)
        curr += len(o.encode("latin-1"))
        body += o
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = f"trailer\n<< /Size {len(objects) + 1}\n/Root 1 0 R >>\nstartxref\n{curr}\n%%EOF\n"

    with open(pdf_path, "wb") as f:
        f.write(header.encode("latin-1"))
        f.write(body.encode("latin-1"))
        f.write(xref.encode("latin-1"))
        f.write(trailer.encode("latin-1"))

@router.post("/structure", response_model=DictationReportResponse)
def structure_dictation_endpoint(payload: DictationRequest):
    """
    Feature B: Clinical dictation to validated editable structured JSON draft report.
    - Uses LLM instruction reasoning to extract sites, categories, and certainty.
    - Anti-hallucination verification.
    - Requires clinician review.
    """
    patient_id = payload.patient_id or "PAT-1001"
    patient_name = payload.patient_name or "Priya Sharma"
    dictation_text = payload.text.strip()

    # 1. Invoke Clinical Scribe Agent via standard MCP tool execution
    from ..services.mcp_tools import execute_tool
    tool_res = execute_tool("structure_clinical_dictation", {
        "dictation_text": dictation_text,
        "patient_id": patient_id
    })
    sites = tool_res.get("anatomical_sites", [])
    findings = tool_res.get("findings", [])
    impressions = tool_res.get("impressions", [])
    recommendations = tool_res.get("recommendations", [])
    guardrails = tool_res.get("safety_guardrails", {})

    # 2. Compose narrative draft
    narrative = compose_narrative(patient_name, findings, impressions, recommendations)

    # 4. Save JSON and PDF reports to dictation_reports/
    clean_name = patient_name.replace(" ", "_")
    pid = patient_id.replace(" ", "_")
    json_filename = f"{pid}_{clean_name}_Dictation_Report.json"
    pdf_filename = f"{pid}_{clean_name}_Dictation_Report.pdf"
    json_path = os.path.join(DICTATION_DIR, json_filename)
    pdf_path = os.path.join(DICTATION_DIR, pdf_filename)

    structured_data = {
        "report_metadata": {
            "report_type": "Radiological Dictation Draft",
            "status": "DRAFT_PENDING_REVIEW",
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_type": "Spoken Voice Dictation / Speech-to-Text Transcript",
            "clinician_review_required": True,
            "disclaimer": "ATTENTION: This is an unverified AI-generated draft created from clinical dictation. All anatomical findings and impressions must be reviewed and approved by the attending clinician."
        },
        "patient_context": {
            "patient_name": patient_name,
            "patient_id": patient_id
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
    generate_dictation_pdf(pdf_path, patient_name, patient_id, dictation_text, findings, impressions, recommendations)

    return structured_data
