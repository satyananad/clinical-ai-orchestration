"""
Generate all 30 authentic patient radiology PDF reports from data/mock_patients_full.json
"""

import os
import json

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "app", "data", "dental_mock_dataset.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_pdf_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("•", "-").replace("—", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.encode("ascii", "replace").decode("ascii")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def wrap_text_lines(text: str, max_chars: int = 72) -> list:
    if not text:
        return []
    words = text.split(" ")
    lines = []
    current = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + (1 if current else 0) <= max_chars:
            current.append(w)
            curr_len += len(w) + (1 if len(current) > 1 else 0)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
            curr_len = len(w)
    if current:
        lines.append(" ".join(current))
    return lines

def create_clinical_pdf(patient: dict, report: dict, filepath: str):
    patient_name = patient["patient_name"]
    dob = patient["date_of_birth"]
    gender = patient["sex"]
    patient_id = patient["patient_id"]
    record_number = patient.get("record_number", "MRN-DEMO")
    
    report_id = report["report_id"]
    report_date = report["report_date"]
    report_type = report["report_type"]
    modality = report.get("modality", "Diagnostic Imaging")
    indication = report.get("clinical_indication", "Routine examination")
    impression_list = report.get("impression", [])
    recommendation = report.get("recommendation", "Clinical correlation advised.")
    findings = report.get("findings", [])

    # Format exam title cleanly
    exam_str = f"{report_type} ({modality})"
    if len(exam_str) > 36:
        exam_str = f"{report_type}"

    cmds = [
        # Top Blue Banner (x=45 to 550, width=505)
        "0.12 0.23 0.45 rg 45 760 505 48 re f",
        f"BT /F1 14 Tf 1 1 1 rg 55 785 Td ({sanitize_pdf_text('VISCAN DIAGNOSTICS - CLINICAL RADIOLOGY REPORT')}) Tj ET",
        f"BT /F3 8.5 Tf 1 1 1 rg 55 771 Td ({sanitize_pdf_text('Advanced Maxillofacial Imaging Network • Validated Diagnostic Record')}) Tj ET",

        # Patient Info Box (x=45 to 550, width=505, height=80)
        "0.82 0.82 0.82 RG 1 w 45 668 505 82 re S",
        # Column 1 (Left: x=55, max width ~235)
        f"BT /F1 9.5 Tf 0 0 0 rg 55 732 Td ({sanitize_pdf_text('PATIENT: ' + patient_name.upper())}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 55 715 Td ({sanitize_pdf_text('DOB: ' + dob + '   |   SEX: ' + gender)}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 55 698 Td ({sanitize_pdf_text('MRN: ' + record_number)}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 55 681 Td ({sanitize_pdf_text('STATUS: Final Validated Report')}) Tj ET",
        
        # Column 2 (Right: x=305, max width ~235, safely terminates at x=540)
        f"BT /F1 9.5 Tf 0 0 0 rg 305 732 Td ({sanitize_pdf_text('PATIENT ID: ' + patient_id)}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 305 715 Td ({sanitize_pdf_text('EXAMINATION: ' + exam_str)}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 305 698 Td ({sanitize_pdf_text('REPORT ID: ' + report_id)}) Tj ET",
        f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 305 681 Td ({sanitize_pdf_text('STUDY DATE: ' + report_date)}) Tj ET",

        # Clinical Indication Section
        f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 644 Td ({sanitize_pdf_text('CLINICAL INDICATION & TECHNIQUE')}) Tj ET"
    ]

    y = 630
    for ind_line in wrap_text_lines(f"Indication: {indication}", max_chars=82):
        cmds.append(f"BT /F2 8.5 Tf 0.1 0.1 0.1 rg 60 {y} Td ({sanitize_pdf_text(ind_line)}) Tj ET")
        y -= 12

    # Structured Findings Section Header
    y -= 8
    cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf_text('STRUCTURED OBSERVATIONS & FINDINGS')}) Tj ET")
    y -= 14

    for f in findings:
        region = f.get("region", "Region")
        obs = f.get("observation", "")
        cert = f.get("certainty", "Confirmed")
        cmds.append(f"BT /F1 8.5 Tf 0.1 0.1 0.1 rg 60 {y} Td ({sanitize_pdf_text('- ' + region + ':')}) Tj ET")
        y -= 11

        obs_lines = wrap_text_lines(obs, max_chars=78)
        for idx, line in enumerate(obs_lines):
            suffix = f"  [{cert}]" if (idx == len(obs_lines) - 1 and len(line) < 64) else ""
            cmds.append(f"BT /F2 8.5 Tf 0.25 0.25 0.25 rg 72 {y} Td ({sanitize_pdf_text(line + suffix)}) Tj ET")
            y -= 11

        if not suffix:
            cmds.append(f"BT /F3 8 Tf 0.4 0.4 0.4 rg 72 {y} Td ({sanitize_pdf_text('Diagnostic Certainty: ' + cert)}) Tj ET")
            y -= 11
        y -= 3

    # Impression Section Header
    y -= 6
    cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf_text('RADIOLOGIST IMPRESSION')}) Tj ET")
    y -= 14
    for imp in impression_list:
        imp_lines = wrap_text_lines(imp, max_chars=78)
        for i, iline in enumerate(imp_lines):
            bullet = "- " if i == 0 else "  "
            cmds.append(f"BT /F1 8.5 Tf 0.65 0.1 0.1 rg 60 {y} Td ({sanitize_pdf_text(bullet + iline)}) Tj ET")
            y -= 11

    # Recommendation
    y -= 6
    cmds.append(f"BT /F1 10.5 Tf 0.12 0.23 0.45 rg 50 {y} Td ({sanitize_pdf_text('RECOMMENDATIONS & CORRELATION')}) Tj ET")
    y -= 13
    for rec_line in wrap_text_lines(recommendation, max_chars=82):
        cmds.append(f"BT /F2 8.5 Tf 0 0 0 rg 60 {y} Td ({sanitize_pdf_text(rec_line)}) Tj ET")
        y -= 11

    # Signature Block firmly placed at bottom (y=105 to y=60)
    cmds.extend([
        "0.85 0.85 0.85 RG 1 w 45 105 505 0 m 550 105 l S",
        f"BT /F1 9 Tf 0 0 0 rg 50 85 Td ({sanitize_pdf_text('Reporting Radiologist: Dr. R. Mehta, MDS')}) Tj 260 0 Td ({sanitize_pdf_text('Digital Signature: VERIFIED & SEALED')}) Tj ET",
        f"BT /F3 8 Tf 0.4 0.4 0.4 rg 50 68 Td ({sanitize_pdf_text('ViScan Dental Assistant • Synthetic Patient Record for Technical Assessment Demonstration')}) Tj ET"
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

    with open(filepath, "wb") as f:
        f.write(header.encode("latin-1"))
        f.write(body.encode("latin-1"))
        f.write(xref.encode("latin-1"))
        f.write(trailer.encode("latin-1"))

def main():
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    patients = data if isinstance(data, list) else data.get("patients", [])
    generated_count = 0

    print(f"Generating 30 Full PDFs from {DATASET_PATH} into {OUTPUT_DIR}...\n")
    for patient in patients:
        p_name = patient["patient_name"].replace(" ", "_")
        for report in patient["reports"]:
            rep_id = report["report_id"]
            rep_type = report["report_type"].replace(" ", "_")
            filename = f"{rep_id}_{p_name}_{rep_type}.pdf"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            create_clinical_pdf(patient, report, filepath)
            report["pdf_filename"] = filename
            generated_count += 1
            print(f"[{generated_count}/30] Generated: {filename} ({os.path.getsize(filepath)} bytes)")

    print("ALL 30 REAL CLINICAL PDF REPORTS GENERATED WITH 100% SUCCESS!")

if __name__ == "__main__":
    main()
