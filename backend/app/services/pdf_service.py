import os
import datetime
from typing import Optional, List, Dict, Any

def sanitize_pdf(text: str) -> str:
    """Sanitize strings for PDF Type 1 encoding."""
    if not text:
        return ""
    s = text.replace("•", "-").replace("—", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.encode("ascii", "replace").decode("ascii")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def wrap_lines(text: str, max_chars: int = 70) -> List[str]:
    """Wrap lines at word boundaries."""
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

def _generate_reportlab_pdf(
    pdf_path: str,
    patient_name: Optional[str],
    patient_id: Optional[str],
    raw_dictation: str,
    findings: List[Dict[str, Any]],
    impressions: List[str],
    recommendations: List[str],
    patient_details: Optional[Dict[str, Any]] = None
) -> str:
    """Renders professional multi-page radiological draft report via ReportLab Platypus."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfgen import canvas

    class NumberedCanvas(canvas.Canvas):
        """Two-pass canvas for dynamic total page count."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(num_pages)
                super().showPage()
            super().save()

        def draw_page_decorations(self, page_count):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(letter[0] - 36, 20, page_text)
            self.drawString(36, 20, "Viscan AI Clinical Workstation • Unverified AI Draft - Clinician Review Required")
            self.restoreState()

    details = patient_details or {}
    supplied_name = details.get("patient_name")
    supplied_id = details.get("patient_id")
    study_type = details.get("study_type") or "Dictated Radiological Assessment"

    p_name = (patient_name or "Unassigned / Pending Patient Selection").upper()
    if "unassigned" in p_name.lower() and supplied_name:
        p_name = f"{p_name} ({supplied_name.upper()})"
    p_id = patient_id or "Pending Patient Assignment"
    if (not patient_id or patient_id == "None") and supplied_id:
        p_id = f"Supplied: {supplied_id} (Unverified)"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.white
    )
    header_sub_style = ParagraphStyle(
        "HeaderSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#cbd5e1")
    )
    card_label_bold = ParagraphStyle(
        "CardLabelBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    card_text_dim = ParagraphStyle(
        "CardTextDim",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569")
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=4
    )
    finding_title = ParagraphStyle(
        "FindingTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e40af")
    )
    finding_body = ParagraphStyle(
        "FindingBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    finding_meta = ParagraphStyle(
        "FindingMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#64748b")
    )
    bullet_item = ParagraphStyle(
        "BulletItem",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    sig_label = ParagraphStyle(
        "SigLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Header Banner Table
    banner_cell = [
        Paragraph("CLINICAL RADIOLOGY REPORT (DICTATION DRAFT)", header_title_style),
        Spacer(1, 2),
        Paragraph("Dental Radiography & Clinical AI Scribe Network • Review Copy", header_sub_style)
    ]
    banner_table = Table([[banner_cell]], colWidths=[540])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e3a8a")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 6))

    # 2. Patient Information Card (2-column structured table)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    col1 = [
        Paragraph(f"<b>PATIENT:</b> {p_name}", card_label_bold),
        Paragraph(f"<b>PATIENT ID:</b> {p_id}", card_text_dim),
        Paragraph("<b>STATUS:</b> Draft Pending Attending Clinician Approval", card_text_dim)
    ]
    col2 = [
        Paragraph(f"<b>EXAM TYPE:</b> {study_type}", card_label_bold),
        Paragraph(f"<b>DATE:</b> {now_str}", card_text_dim),
        Paragraph("<b>VERIFICATION:</b> Guardrail Verification Enforced", card_text_dim)
    ]
    card_table = Table([[col1, col2]], colWidths=[270, 270])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 10))

    # 3. Section 1: Detailed Clinical Findings & Observations
    story.append(Paragraph("1. DETAILED CLINICAL FINDINGS & OBSERVATIONS", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    if findings:
        for i, f in enumerate(findings, 1):
            site = f.get("anatomical_site", "General")
            cat = f.get("category", "General Observation")
            obs = f.get("observation", "")
            cert = f.get("certainty", "Confirmed")

            finding_block = [
                Paragraph(f"{i}. [{site}] • {cat}:", finding_title),
                Spacer(1, 1),
                Paragraph(obs, finding_body),
                Spacer(1, 1),
                Paragraph(f"Diagnostic Certainty: {cert}", finding_meta),
                Spacer(1, 4)
            ]
            story.append(KeepTogether(finding_block))
    else:
        story.append(Paragraph("No discrete clinical findings documented. Complete clinical dictation review required.", card_text_dim))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 6))

    # 4. Section 2: Clinical Impressions
    story.append(Paragraph("2. CLINICAL IMPRESSION", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    if impressions:
        for imp in impressions:
            story.append(Paragraph(f"• {imp}", bullet_item))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("Pending attending clinician synthesis.", card_text_dim))

    story.append(Spacer(1, 10))

    # 5. Section 3: Recommendations
    story.append(Paragraph("3. RECOMMENDATIONS & REQUIRED ACTION", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    if recommendations:
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", bullet_item))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("No specific intervention recommended at this time.", card_text_dim))

    story.append(Spacer(1, 16))

    # 6. Attending Clinician Approval Block
    sig_data = [
        [
            Paragraph("Attending Clinician Approval: ___________________________", sig_label),
            Paragraph("Sign-Off Date: _________________", sig_label)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[340, 200])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    legal_disclaimer = Paragraph(
        "Clinical Dental Workstation • Unverified AI Draft - Not Approved for Medical Records Without Clinician Signature",
        finding_meta
    )
    sig_block = [
        HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6),
        sig_table,
        Spacer(1, 4),
        legal_disclaimer
    ]
    story.append(KeepTogether(sig_block))

    doc.build(story, canvasmaker=NumberedCanvas)
    return pdf_path

def _generate_manual_canvas_pdf(
    pdf_path: str,
    patient_name: Optional[str],
    patient_id: Optional[str],
    raw_dictation: str,
    findings: List[Dict[str, Any]],
    impressions: List[str],
    recommendations: List[str],
    patient_details: Optional[Dict[str, Any]] = None
) -> str:
    """Fallback manual PDF canvas generator if ReportLab Platypus is unavailable."""
    details = patient_details or {}
    supplied_name = details.get("patient_name")
    supplied_id = details.get("patient_id")
    study_type = details.get("study_type") or "Dictated Radiological Assessment"

    p_name = (patient_name or "Unassigned / Pending Patient Selection").upper()
    if "unassigned" in p_name.lower() and supplied_name:
        p_name = f"{p_name} ({supplied_name.upper()})"
    p_id = patient_id or "Pending Patient Assignment"
    if (not patient_id or patient_id == "None") and supplied_id:
        p_id = f"Supplied: {supplied_id} (Unverified)"

    pages_cmds = []

    # Page 1 Initial Setup
    curr_cmds = [
        # Navy Header Banner
        "0.12 0.23 0.45 rg 45 765 505 45 re f",
        f"BT /F1 13 Tf 1 1 1 rg 55 790 Td ({sanitize_pdf('CLINICAL RADIOLOGY REPORT (DICTATION DRAFT)')}) Tj ET",
        f"BT /F3 8.5 Tf 1 1 1 rg 55 776 Td ({sanitize_pdf('Dental Radiography & Clinical AI Scribe Network • Review Copy')}) Tj ET",

        # Patient Info Card
        "0.85 0.85 0.85 RG 1 w 45 685 505 70 re S",
        f"BT /F1 9.5 Tf 0.1 0.1 0.1 rg 55 738 Td ({sanitize_pdf('PATIENT: ' + p_name[:35])}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 722 Td ({sanitize_pdf('PATIENT ID: ' + p_id[:35])}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 706 Td ({sanitize_pdf('STATUS: Draft Pending Attending Clinician Approval')}) Tj ET",

        f"BT /F1 9.5 Tf 0.1 0.1 0.1 rg 310 738 Td ({sanitize_pdf('EXAM TYPE: ' + study_type[:30])}) Tj ET",
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
                f"BT /F1 9.5 Tf 0.12 0.23 0.45 rg 55 793 Td ({sanitize_pdf('CLINICAL RADIOLOGY REPORT (CONTINUED) • PATIENT: ' + (patient_name or 'UNASSIGNED').upper() + ' (' + str(patient_id or '') + ')')}) Tj ET",
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
        for o_line in wrap_lines(obs, max_chars=72):
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
        for iline in wrap_lines(imp, max_chars=72):
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
        for r_line in wrap_lines(rec, max_chars=72):
            check_page_overflow(needed_space=20)
            curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf('• ' + r_line)}) Tj ET")
            y -= 11
        y -= 2

    # Bottom Signature Block
    curr_cmds.extend([
        "0.85 0.85 0.85 RG 1 w 45 75 505 0 m 550 75 l S",
        f"BT /F1 8.5 Tf 0 0 0 rg 50 58 Td ({sanitize_pdf('Attending Clinician Approval: ___________________________')}) Tj 200 0 Td ({sanitize_pdf('Sign-Off Date: _________________')}) Tj ET",
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

    return pdf_path

def generate_dictation_pdf(
    pdf_path: str,
    patient_name: Optional[str],
    patient_id: Optional[str],
    raw_dictation: str,
    findings: List[Dict[str, Any]],
    impressions: List[str],
    recommendations: List[str],
    patient_details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates a clean, professional clinical radiological report with automatic pagination.
    Uses ReportLab Platypus for guaranteed clean margins, automatic word wrapping,
    strict column isolation, and zero text collisions. Falls back to manual canvas renderer
    if ReportLab is unavailable.
    """
    try:
        return _generate_reportlab_pdf(
            pdf_path=pdf_path,
            patient_name=patient_name,
            patient_id=patient_id,
            raw_dictation=raw_dictation,
            findings=findings,
            impressions=impressions,
            recommendations=recommendations,
            patient_details=patient_details
        )
    except Exception as e:
        print(f"[PDF GENERATION WARNING] ReportLab generation encountered error ({e}); falling back to manual canvas.")
        return _generate_manual_canvas_pdf(
            pdf_path=pdf_path,
            patient_name=patient_name,
            patient_id=patient_id,
            raw_dictation=raw_dictation,
            findings=findings,
            impressions=impressions,
            recommendations=recommendations,
            patient_details=patient_details
        )
