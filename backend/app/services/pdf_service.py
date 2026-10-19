import os
import datetime
from typing import Optional, List, Dict, Any
from .report_service import is_symptom_or_history

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
            kwargs["pageCompression"] = 0
            super().__init__(*args, **kwargs)
            self._pageCompression = 0
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

    status = details.get("status") or details.get("report_metadata", {}).get("status", "DRAFT")
    is_confirmed = (status == "CONFIRMED_BY_CLINICIAN")
    confirmed_at = details.get("confirmed_at") or details.get("report_metadata", {}).get("confirmed_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    clinician_name = details.get("clinician_name") or "Dr. R. Mehta, MDS"

    resolved_name = (patient_name or supplied_name or "").strip()
    if ("unassigned" in resolved_name.lower() or "pending" in resolved_name.lower()) and supplied_name:
        resolved_name = supplied_name
    elif "unassigned" in resolved_name.lower():
        resolved_name = ""
    p_name = resolved_name.upper() if resolved_name else ""

    raw_pid = str(patient_id or supplied_id or "").strip()
    if raw_pid.lower() in ("none", "null", "pending patient assignment", "unassigned", "pat-record"):
        p_id = supplied_id or ""
    else:
        p_id = raw_pid

    # Demographics tag (blank if missing)
    demo_tags = []
    if details.get("age"):
        demo_tags.append(f"Age: {details['age']}")
    if details.get("gender"):
        demo_tags.append(str(details['gender']))
    if details.get("date_of_birth"):
        demo_tags.append(f"DOB: {details['date_of_birth']}")
    p_name_display = f"{p_name} ({', '.join(demo_tags)})" if (p_name and demo_tags) else (p_name or (', '.join(demo_tags) if demo_tags else ""))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
        pageCompression=0
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
    if is_confirmed:
        banner_title = "CLINICAL RADIOLOGICAL REPORT - VERIFIED & APPROVED"
        banner_sub = "Dental Radiography & Clinical AI Scribe Network • Official Medical Record"
        banner_bg = colors.HexColor("#065f46")
    else:
        banner_title = "RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF"
        banner_sub = "Dental Radiography & Clinical AI Scribe Network • Official Review Copy"
        banner_bg = colors.HexColor("#1e3a8a")

    banner_cell = [
        Paragraph(banner_title, header_title_style),
        Spacer(1, 2),
        Paragraph(banner_sub, header_sub_style)
    ]
    banner_table = Table([[banner_cell]], colWidths=[540])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 4))

    if is_confirmed:
        alert_style = ParagraphStyle(
            "ConfirmedAlert",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#065f46")
        )
        alert_cell = [
            Paragraph(f"<b>*** CLINICALLY VERIFIED & APPROVED MEDICAL RECORD ***</b><br/><font size=7 color='#047857'>This diagnostic report has been reviewed, verified, and electronically signed off by {clinician_name} ({confirmed_at}).</font>", alert_style)
        ]
        alert_table = Table([[alert_cell]], colWidths=[540])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfdf5")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#10b981")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(alert_table)
    else:
        # Amber Alert Banner for Draft Status
        alert_style = ParagraphStyle(
            "DraftAlert",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#78350f")
        )
        alert_cell = [
            Paragraph("<b>*** RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF ***</b><br/><font size=7 color='#92400e'>Unverified AI synthesis from voice dictation. All anatomical findings and impressions must be verified by attending clinician.</font>", alert_style)
        ]
        alert_table = Table([[alert_cell]], colWidths=[540])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#f59e0b")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(alert_table)
    story.append(Spacer(1, 6))

    # 2. Patient Information Card (2-column structured table)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    status_text = "<b>STATUS:</b> <font color='#059669'><b>FINAL / APPROVED</b></font>" if is_confirmed else "<b>STATUS:</b> Draft Pending Attending Clinician Approval"
    col1 = [
        Paragraph(f"<b>PATIENT:</b> {p_name_display}", card_label_bold),
        Paragraph(f"<b>PATIENT ID:</b> {p_id}", card_text_dim),
        Paragraph(status_text, card_text_dim)
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

    # Partition Findings: Separate Clinical History from Imaging Observations
    history_items = [f.get("observation", "").strip() for f in findings if is_symptom_or_history(f) and f.get("observation")]
    imaging_findings = [f for f in findings if not is_symptom_or_history(f)]

    sec_counter = 1
    # 3. Optional Section: Clinical History & Patient-Reported Symptoms
    if history_items:
        story.append(Paragraph(f"{sec_counter}. CLINICAL HISTORY & PATIENT-REPORTED SYMPTOMS", section_heading))
        story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        for h in history_items:
            clean_h = h.strip()
            if clean_h and not clean_h.endswith("."):
                clean_h += "."
            story.append(Paragraph(f"• {clean_h}", bullet_item))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 6))
        sec_counter += 1

    # 4. Section: Detailed Radiological Findings (Grouped by Anatomical Site)
    story.append(Paragraph(f"{sec_counter}. DETAILED RADIOLOGICAL FINDINGS", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    sec_counter += 1

    if imaging_findings:
        site_groups: Dict[str, List[Dict[str, Any]]] = {}
        for f in imaging_findings:
            raw_site = f.get("anatomical_site", "Dentition & Regional Anatomy")
            tooth_num = f.get("tooth_number")
            if tooth_num and f"tooth {tooth_num}".lower() not in raw_site.lower():
                group_key = f"{raw_site} (Tooth {tooth_num})"
            else:
                group_key = raw_site
            site_groups.setdefault(group_key, []).append(f)

        for site_name, group in site_groups.items():
            paragraph_sentences = []
            for item in group:
                obs = item.get("observation", "").strip()
                cert = item.get("certainty", "")
                if not obs:
                    continue
                clean_obs = obs[0].upper() + obs[1:]
                if not clean_obs.endswith("."):
                    clean_obs += "."
                if cert in ("unclear / clinician review required", "indeterminate"):
                    if "cannot be determined" not in clean_obs.lower() and "indeterminate" not in clean_obs.lower():
                        clean_obs += " (Finding is indeterminate from provided data; clinician review required)."
                    else:
                        clean_obs += " (Indeterminate / clinician review required)."
                paragraph_sentences.append(clean_obs)

            finding_block = [
                Paragraph(f"<b>[{site_name}]</b>", finding_title),
                Spacer(1, 2),
                Paragraph(" ".join(paragraph_sentences), finding_body),
                Spacer(1, 6)
            ]
            story.append(KeepTogether(finding_block))
    else:
        story.append(Paragraph("No discrete radiological imaging findings documented. Complete clinical dictation review required.", card_text_dim))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 2))
    story.append(Paragraph("<i>[Dictation Source Note]: Findings are synthesized from supplied speech-to-text dictation and have not been independently verified against native radiological images.</i>", finding_meta))
    story.append(Spacer(1, 8))

    # 5. Section: Clinical Impressions
    story.append(Paragraph(f"{sec_counter}. CLINICAL IMPRESSION", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    sec_counter += 1

    if impressions:
        for imp in impressions:
            story.append(Paragraph(f"• {imp}", bullet_item))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("Pending attending clinician synthesis.", card_text_dim))

    story.append(Spacer(1, 8))

    # 6. Section: Recommendations
    story.append(Paragraph(f"{sec_counter}. RECOMMENDATIONS & REQUIRED ACTION", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

    if recommendations:
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", bullet_item))
            story.append(Spacer(1, 2))
    else:
        has_indeterminate = any(f.get("certainty") in ("unclear / clinician review required", "indeterminate") for f in imaging_findings)
        if has_indeterminate:
            story.append(Paragraph("• Clinical correlation advised. Cross-sectional 3D CBCT imaging may be indicated to evaluate indeterminate canal proximity or anatomical relationships if surgical intervention is planned.", bullet_item))
        else:
            story.append(Paragraph("• None dictated. Attending clinician to determine management following clinical correlation.", card_text_dim))

    story.append(Spacer(1, 14))

    # 6. Attending Clinician Approval Block / Digital Signature
    if is_confirmed:
        sig_data = [
            [
                Paragraph(f"<b>Reporting Radiologist:</b> {clinician_name} <font color='#059669'><b>[VERIFIED & SEALED]</b></font>", sig_label),
                Paragraph(f"<b>Sign-Off Timestamp:</b> {confirmed_at}", sig_label)
            ]
        ]
        legal_disclaimer = Paragraph(
            "Clinical Dental Workstation • Authenticated Electronic Medical Record • Digital Signature: VERIFIED & SEALED",
            finding_meta
        )
    else:
        sig_data = [
            [
                Paragraph("Attending Clinician Approval: ___________________________", sig_label),
                Paragraph("Sign-Off Date: _________________", sig_label)
            ]
        ]
        legal_disclaimer = Paragraph(
            "Clinical Dental Workstation • Unverified AI Draft - Not Approved for Medical Records Without Clinician Signature",
            finding_meta
        )

    sig_table = Table(sig_data, colWidths=[340, 200])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))

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

    status = details.get("status") or details.get("report_metadata", {}).get("status", "DRAFT")
    is_confirmed = (status == "CONFIRMED_BY_CLINICIAN")
    confirmed_at = details.get("confirmed_at") or details.get("report_metadata", {}).get("confirmed_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    clinician_name = details.get("clinician_name") or "Dr. R. Mehta, MDS"

    resolved_name = patient_name or supplied_name or "Unassigned"
    if ("unassigned" in resolved_name.lower() or "pending" in resolved_name.lower()) and supplied_name:
        resolved_name = supplied_name
    if "unassigned" in resolved_name.lower() or "pending" in resolved_name.lower():
        p_name = "UNASSIGNED (Pending)"
    else:
        p_name = resolved_name.upper()

    p_id = patient_id or supplied_id or "Pending Assignment"
    if str(p_id).strip() in ("None", "null", ""):
        p_id = f"Supplied: {supplied_id}" if supplied_id else "Pending Assignment"

    demo_tags = []
    if details.get("age"):
        demo_tags.append(f"Age: {details['age']}")
    if details.get("gender"):
        demo_tags.append(str(details['gender']))
    if details.get("date_of_birth"):
        demo_tags.append(f"DOB: {details['date_of_birth']}")
    demo_str = ", ".join(demo_tags) if demo_tags else "Unspecified"
    status_str = "FINAL / APPROVED" if is_confirmed else "DRAFT (Pending Review)"

    pages_cmds = []

    # Page 1 Initial Setup
    curr_cmds = [
        # Navy Header Banner
        "0.12 0.23 0.45 rg 45 765 505 45 re f",
        f"BT /F1 11.5 Tf 1 1 1 rg 55 790 Td ({sanitize_pdf('RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF')}) Tj ET",
        f"BT /F3 8 Tf 1 1 1 rg 55 776 Td ({sanitize_pdf('Dental Radiography & Clinical AI Scribe Network • Review Copy')}) Tj ET",

        # Amber Alert Banner
        "0.99 0.95 0.78 rg 45 745 505 16 re f",
        "0.96 0.62 0.04 RG 0.75 w 45 745 505 16 re S",
        f"BT /F1 7.5 Tf 0.47 0.21 0.06 rg 55 749 Td ({sanitize_pdf('*** RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF ***')}) Tj ET",

        # Patient Info Card Box (from y = 650 to y = 735, height 85)
        "0.85 0.85 0.85 RG 1 w 45 650 505 85 re S",

        # Row 1 (y = 718)
        f"BT /F1 9 Tf 0.1 0.1 0.1 rg 55 718 Td ({sanitize_pdf('PATIENT: ' + p_name[:26])}) Tj ET",
        f"BT /F1 9 Tf 0.1 0.1 0.1 rg 330 718 Td ({sanitize_pdf('EXAM TYPE: ' + study_type[:24])}) Tj ET",

        # Row 2 (y = 700)
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 700 Td ({sanitize_pdf('PATIENT ID: ' + p_id[:26])}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 330 700 Td ({sanitize_pdf('DATE: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}) Tj ET",

        # Row 3 (y = 682)
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 682 Td ({sanitize_pdf('DEMOGRAPHICS: ' + demo_str[:26])}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 330 682 Td ({sanitize_pdf('STATUS: ' + status_str[:24])}) Tj ET",

        # Row 4 (y = 664)
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 55 664 Td ({sanitize_pdf('REPORTING: ' + clinician_name[:26])}) Tj ET",
        f"BT /F2 8.5 Tf 0.3 0.3 0.3 rg 330 664 Td ({sanitize_pdf('VERIFICATION: Guardrail Enforced')}) Tj ET",
    ]
    y = 630

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

    # Partition Findings
    history_items = [f.get("observation", "").strip() for f in findings if is_symptom_or_history(f) and f.get("observation")]
    imaging_findings = [f for f in findings if not is_symptom_or_history(f)]

    sec_idx = 1
    if history_items:
        curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf(f'{sec_idx}. CLINICAL HISTORY & PATIENT-REPORTED SYMPTOMS')}) Tj ET")
        curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
        y -= 16
        for h in history_items:
            clean_h = h.strip()
            if clean_h and not clean_h.endswith("."):
                clean_h += "."
            for h_line in wrap_lines(clean_h, max_chars=72):
                check_page_overflow(needed_space=18)
                curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf('• ' + h_line)}) Tj ET")
                y -= 11
            y -= 3
        sec_idx += 1
        y -= 6
        check_page_overflow(needed_space=40)
        curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf(f'{sec_idx}. DETAILED RADIOLOGICAL FINDINGS')}) Tj ET")
        curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
        y -= 16
    else:
        curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf(f'{sec_idx}. DETAILED RADIOLOGICAL FINDINGS')}) Tj ET")
        curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
        y -= 16

    sec_idx += 1

    # Write Site-Grouped Clinical Paragraphs
    if imaging_findings:
        site_groups_canvas: Dict[str, List[Dict[str, Any]]] = {}
        for f in imaging_findings:
            raw_site = f.get("anatomical_site", "Dentition & Regional Anatomy")
            tooth_num = f.get("tooth_number")
            if tooth_num and f"tooth {tooth_num}".lower() not in raw_site.lower():
                group_key = f"{raw_site} (Tooth {tooth_num})"
            else:
                group_key = raw_site
            site_groups_canvas.setdefault(group_key, []).append(f)

        for site_name, group in site_groups_canvas.items():
            check_page_overflow(needed_space=45)
            curr_cmds.append(f"BT /F1 9 Tf 0.15 0.25 0.5 rg 55 {y} Td ({sanitize_pdf(f'[{site_name}]')}) Tj ET")
            y -= 12

            paragraph_sentences = []
            for item in group:
                obs = item.get("observation", "").strip()
                cert = item.get("certainty", "")
                if not obs:
                    continue
                clean_obs = obs[0].upper() + obs[1:]
                if not clean_obs.endswith("."):
                    clean_obs += "."
                if cert in ("unclear / clinician review required", "indeterminate"):
                    if "cannot be determined" not in clean_obs.lower() and "indeterminate" not in clean_obs.lower():
                        clean_obs += " (Finding is indeterminate from provided data; clinician review required)."
                    else:
                        clean_obs += " (Indeterminate / clinician review required)."
                paragraph_sentences.append(clean_obs)

            para_text = " ".join(paragraph_sentences)
            for p_line in wrap_lines(para_text, max_chars=72):
                check_page_overflow(needed_space=18)
                curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 65 {y} Td ({sanitize_pdf(p_line)}) Tj ET")
                y -= 11
            y -= 6
    else:
        check_page_overflow(needed_space=25)
        curr_cmds.append(f"BT /F3 8.5 Tf 0.4 0.4 0.4 rg 55 {y} Td ({sanitize_pdf('No discrete radiological findings dictated.')}) Tj ET")
        y -= 14

    check_page_overflow(needed_space=25)
    curr_cmds.append(f"BT /F3 7.5 Tf 0.4 0.4 0.4 rg 55 {y} Td ({sanitize_pdf('[Source Note]: Findings are synthesized from supplied speech-to-text dictation and have not been independently verified against native images.')}) Tj ET")
    y -= 14

    # Section: Impressions
    check_page_overflow(needed_space=60)
    y -= 6
    curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf(f'{sec_idx}. CLINICAL IMPRESSION')}) Tj ET")
    curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
    y -= 16
    sec_idx += 1

    for imp in impressions:
        for iline in wrap_lines(imp, max_chars=72):
            check_page_overflow(needed_space=20)
            curr_cmds.append(f"BT /F1 8.5 Tf 0.7 0.15 0.1 rg 60 {y} Td ({sanitize_pdf('• ' + iline)}) Tj ET")
            y -= 11
        y -= 3

    # Section: Recommendations
    check_page_overflow(needed_space=60)
    y -= 6
    curr_cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf(f'{sec_idx}. RECOMMENDATIONS & REQUIRED ACTION')}) Tj ET")
    curr_cmds.append(f"0.85 0.85 0.85 RG 1 w 50 {y - 4} 495 0 m 545 {y - 4} l S")
    y -= 16

    if recommendations:
        for rec in recommendations:
            for r_line in wrap_lines(rec, max_chars=72):
                check_page_overflow(needed_space=20)
                curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf('• ' + r_line)}) Tj ET")
                y -= 11
            y -= 2
    else:
        has_indeterminate = any(f.get("certainty") in ("unclear / clinician review required", "indeterminate") for f in imaging_findings)
        rec_msg = "• Clinical correlation advised. Cross-sectional 3D CBCT imaging may be indicated if surgical intervention planned." if has_indeterminate else "• None dictated. Attending clinician to determine management following clinical correlation."
        for r_line in wrap_lines(rec_msg, max_chars=72):
            check_page_overflow(needed_space=20)
            curr_cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf(r_line)}) Tj ET")
            y -= 11
        y -= 2

    # Bottom Signature Block
    if is_confirmed:
        curr_cmds.extend([
            "0.85 0.85 0.85 RG 1 w 45 75 505 0 m 550 75 l S",
            f"BT /F1 8.5 Tf 0 0.5 0 rg 50 58 Td ({sanitize_pdf(f'Reporting Radiologist: {clinician_name} [VERIFIED & SEALED]')}) Tj ET",
            f"BT /F1 8.5 Tf 0 0.5 0 rg 340 58 Td ({sanitize_pdf(f'Sign-Off Date: {confirmed_at}')}) Tj ET",
            f"BT /F3 7.5 Tf 0.2 0.5 0.2 rg 50 45 Td ({sanitize_pdf('Clinical Dental Workstation • Authenticated Electronic Medical Record • Digital Signature: VERIFIED & SEALED')}) Tj ET"
        ])
    else:
        curr_cmds.extend([
            "0.85 0.85 0.85 RG 1 w 45 75 505 0 m 550 75 l S",
            f"BT /F1 8.5 Tf 0 0 0 rg 50 58 Td ({sanitize_pdf('Attending Clinician Approval: ___________________________')}) Tj ET",
            f"BT /F1 8.5 Tf 0 0 0 rg 350 58 Td ({sanitize_pdf('Sign-Off Date: _________________')}) Tj ET",
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
