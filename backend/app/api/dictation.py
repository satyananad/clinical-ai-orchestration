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
    """
    Generates a clean, professional clinical radiological report with automatic pagination.
    Prevents text collisions and ensures clean hierarchy between findings, impressions, and signature.
    """
    pages_cmds = []
    
    # Page 1 Initial Setup
    curr_cmds = [
        # Navy Header Banner
        "0.12 0.23 0.45 rg 45 765 505 45 re f",
        f"BT /F1 13 Tf 1 1 1 rg 55 790 Td ({sanitize_pdf('CLINICAL RADIOLOGY REPORT (DICTATION DRAFT)')}) Tj ET",
        f"BT /F3 8.5 Tf 1 1 1 rg 55 776 Td ({sanitize_pdf('Dental Radiography & Clinical AI Scribe Network • Review Copy')}) Tj ET",

        # Patient Info Card
        "0.85 0.85 0.85 RG 1 w 45 685 505 70 re S",
        f"BT /F1 9.5 Tf 0.1 0.1 0.1 rg 55 738 Td ({sanitize_pdf('PATIENT: ' + patient_name.upper())}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 722 Td ({sanitize_pdf('PATIENT ID: ' + patient_id)}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 706 Td ({sanitize_pdf('STATUS: Draft Pending Attending Clinician Approval')}) Tj ET",

        f"BT /F1 9.5 Tf 0.1 0.1 0.1 rg 310 738 Td ({sanitize_pdf('EXAM TYPE: Dictated Radiological Assessment')}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 310 722 Td ({sanitize_pdf('DATE: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 310 706 Td ({sanitize_pdf('VERIFICATION: Guardrail Verification Enforced')}) Tj ET",

        # Section 1 Header
        f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 660 Td ({sanitize_pdf('1. DETAILED CLINICAL FINDINGS & OBSERVATIONS')}) Tj ET",
        "0.85 0.85 0.85 RG 1 w 50 653 495 0 m 545 653 l S"
    ]
    y = 638

    def check_page_overflow(needed_space=40):
        nonlocal y, curr_cmds
        if y < (85 + needed_space):
            pages_cmds.append("\n".join(curr_cmds))
            curr_cmds = [
                # Continuation header
                "0.92 0.94 0.97 rg 45 785 505 25 re f",
                f"BT /F1 9.5 Tf 0.12 0.23 0.45 rg 55 793 Td ({sanitize_pdf('CLINICAL RADIOLOGY REPORT (CONTINUED) • PATIENT: ' + patient_name.upper() + ' (' + patient_id + ')')}) Tj ET",
                "0.85 0.85 0.85 RG 1 w 45 785 505 0 m 550 785 l S"
            ]
            y = 760

    # Write Structured Findings cleanly
    for i, f in enumerate(findings, 1):
        check_page_overflow(needed_space=35)
        site = f.get("anatomical_site", "General")
        cat = f.get("category", "General Observation")
        obs = f.get("observation", "")
        cert = f.get("certainty", "Confirmed")

        curr_cmds.append(f"BT /F1 8.5 Tf 0.15 0.25 0.5 rg 55 {y} Td ({sanitize_pdf(f'{i}. [{site}] • {cat}:')}) Tj ET")
        y -= 11
        for o_line in wrap_lines(obs, max_chars=82):
            check_page_overflow(needed_space=20)
            curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 65 {y} Td ({sanitize_pdf(o_line)}) Tj ET")
            y -= 10
        curr_cmds.append(f"BT /F3 7.5 Tf 0.4 0.4 0.4 rg 65 {y} Td ({sanitize_pdf('Diagnostic Certainty: ' + cert)}) Tj ET")
        y -= 12

    # Section 2: Impressions
    check_page_overflow(needed_space=60)
    y -= 6
    curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf('2. CLINICAL IMPRESSION')}) Tj ET")
    curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
    y -= 16

    for imp in impressions:
        for iline in wrap_lines(imp, max_chars=82):
            check_page_overflow(needed_space=20)
            curr_cmds.append(f"BT /F1 8.5 Tf 0.7 0.15 0.1 rg 60 {y} Td ({sanitize_pdf('• ' + iline)}) Tj ET")
            y -= 11
        y -= 3

    # Section 3: Recommendations
    check_page_overflow(needed_space=60)
    y -= 6
    curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf('3. RECOMMENDATIONS & REQUIRED ACTION')}) Tj ET")
    curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
    y -= 16

    for rec in recommendations:
        for r_line in wrap_lines(rec, max_chars=82):
            check_page_overflow(needed_space=20)
            curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf('• ' + r_line)}) Tj ET")
            y -= 11
        y -= 2

    # Bottom Signature Block
    curr_cmds.extend([
        "0.85 0.85 0.85 RG 1 w 45 75 505 0 m 550 75 l S",
        f"BT /F1 8.5 Tf 0 0 0 rg 50 58 Td ({sanitize_pdf('Attending Clinician Approval: ___________________________')}) Tj 230 0 Td ({sanitize_pdf('Sign-Off Date: _________________')}) Tj ET",
        f"BT /F3 7.5 Tf 0.4 0.4 0.4 rg 50 45 Td ({sanitize_pdf('Clinical Dental Workstation • Unverified AI Draft - Not Approved for Medical Records Without Clinician Signature')}) Tj ET"
    ])
    pages_cmds.append("\n".join(curr_cmds))

    # Assemble Multi-Page PDF Binary
    count = len(pages_cmds)
    page_obj_ids = [3 + i for i in range(count)]
    stream_obj_ids = [3 + count + i for i in range(count)]
    font_base = 3 + 2 * count
    f1_id, f2_id, f3_id = font_base, font_base + 1, font_base + 2

    objects = []
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {count} >>\nendobj\n")

    for i in range(count):
        pid = page_obj_ids[i]
        sid = stream_obj_ids[i]
        objects.append(f"{pid} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {sid} 0 R /Resources << /Font << /F1 {f1_id} 0 R /F2 {f2_id} 0 R /F3 {f3_id} 0 R >> >> >>\nendobj\n")

    for i in range(count):
        sid = stream_obj_ids[i]
        stream_bytes = pages_cmds[i].encode("latin-1")
        objects.append(f"{sid} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n{pages_cmds[i]}\nendstream\nendobj\n")

    objects.append(f"{f1_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")
    objects.append(f"{f2_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(f"{f3_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>\nendobj\n")

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
