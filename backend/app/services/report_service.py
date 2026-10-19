import re
import datetime
from typing import List, Dict, Any, Optional

def is_symptom_or_history(finding: Dict[str, Any]) -> bool:
    """Identifies if a finding represents patient-reported symptoms or clinical history rather than an objective imaging observation."""
    from .claim_verification import is_meta_or_instruction_sentence
    obs = str(finding.get("observation", "")).strip()
    if not obs or is_meta_or_instruction_sentence(obs):
        return False
    cat = str(finding.get("category", "")).lower()
    site = str(finding.get("anatomical_site", "")).lower()
    obs_lower = obs.lower()
    if "symptom" in cat or "history" in cat:
        return True
    if "symptom" in site or "history" in site:
        return True
    if re.search(r"\b(?:patient\s+(?:reports?|complains?|presents?|presented)|complains?\s+of|chief\s+complaint|history\s+of|presented\s+with|presents\s+with|experiencing\s+(?:pain|discomfort|swelling|loose|mobility)|reports\s+(?:discomfort|pain|loose)|loose\s+teeth)\b", obs_lower):
        return True
    return False


def compose_narrative(
    patient_name: str,
    findings: List[Dict[str, Any]],
    impressions: List[str],
    recommendations: List[str],
    patient_details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Composes a readable, standardized radiological consultation narrative report for clinician review.
    Enforces separation between administrative metadata, patient symptoms, imaging observations, impressions, and recommendations.
    Renders findings grouped into coherent clinical paragraphs by anatomical region rather than raw numbered lists.
    """
    from .claim_verification import is_meta_or_instruction_sentence

    details = patient_details or {}
    supplied_name = details.get("patient_name")
    supplied_id = details.get("patient_id")
    study_type = details.get("study_type") or "Dictated Radiological Assessment"
    status = details.get("status") or details.get("report_metadata", {}).get("status", "DRAFT")
    is_confirmed = (status == "CONFIRMED_BY_CLINICIAN")
    confirmed_at = details.get("confirmed_at") or details.get("report_metadata", {}).get("confirmed_at")
    clinician_name = details.get("clinician_name") or "Dr. R. Mehta, MDS"

    # Demographics formatting (blank if missing)
    demo_parts = []
    if details.get("age"):
        demo_parts.append(f"Age: {details['age']}")
    if details.get("gender"):
        demo_parts.append(f"Gender: {details['gender']}")
    if details.get("date_of_birth"):
        demo_parts.append(f"DOB: {details['date_of_birth']}")
    demo_str = " | ".join(demo_parts) if demo_parts else ""

    lines = []
    lines.append("=" * 72)
    lines.append("VISCAN MAXILLOFACIAL & DENTAL RADIOLOGY CLINIC")
    lines.append("Department of Diagnostic Radiology & Oral Imaging")
    if is_confirmed:
        lines.append("*** RADIOLOGICAL CONSULTATION REPORT - CLINICALLY VERIFIED & FINAL ***")
    else:
        lines.append("*** RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF ***")
    lines.append("=" * 72)

    # 1. Compact Two-Column Patient & Examination Details (blank missing fields)
    lines.append("PATIENT & EXAMINATION DETAILS:")
    lines.append("-" * 72)
    
    # Resolve clean display values: blank if missing or generic placeholder
    raw_pname = str(patient_name or "").strip()
    raw_pid = str(details.get("patient_id") or "").strip()
    if raw_pid.lower() in ("none", "null", "pat-record", "unassigned", "pending patient assignment"):
        raw_pid = ""

    if supplied_name:
        p_name_display = supplied_name
    elif raw_pname and "unassigned" not in raw_pname.lower():
        p_name_display = raw_pname
    elif "pending" in raw_pname.lower():
        p_name_display = raw_pname
    else:
        p_name_display = ""

    if supplied_id:
        p_id_display = supplied_id
    elif raw_pid:
        p_id_display = raw_pid
    else:
        p_id_display = ""

    date_str = confirmed_at if (is_confirmed and confirmed_at) else datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    lines.append(f"Patient Name:    {p_name_display}")
    lines.append(f"Patient ID:      {p_id_display}")
    lines.append(f"Demographics:    {demo_str}")
    lines.append(f"Examination:     {study_type}")
    lines.append(f"Date of Draft:   {date_str}")
    if is_confirmed:
        lines.append(f"Report Status:   FINAL / CLINICALLY VERIFIED ({clinician_name})")
    else:
        lines.append("Report Status:   DRAFT (AI Dictation Synthesis - Pending Clinician Review)")
    lines.append("-" * 72)
    lines.append("")

    # 2. Partition Findings: Separate Clinical History & Symptoms from Imaging Observations
    history_items = []
    imaging_findings = []
    for f in findings:
        obs = str(f.get("observation", "")).strip()
        if not obs or is_meta_or_instruction_sentence(obs):
            continue
        if is_symptom_or_history(f):
            history_items.append(obs)
        else:
            imaging_findings.append(f)

    # Section: Clinical History
    lines.append("CLINICAL HISTORY & PATIENT-REPORTED SYMPTOMS:")
    if history_items:
        for item in history_items:
            clean_item = item.strip()
            if clean_item and not clean_item.endswith("."):
                clean_item += "."
            lines.append(f"  • {clean_item}")
    else:
        lines.append("  • No prior clinical history or patient-reported symptoms were supplied in the dictation.")
    lines.append("")

    # 3. Section: Detailed Radiological Findings (Grouped by Anatomical Site/Region as Clinical Paragraphs)
    lines.append("DETAILED RADIOLOGICAL FINDINGS:")
    if imaging_findings:
        # Group findings by anatomical site / region
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
            lines.append(f"  [{site_name}]")
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
            lines.append(f"    {' '.join(paragraph_sentences)}")
            lines.append("")
    else:
        lines.append("  • No discrete radiological imaging findings dictated.")
        lines.append("")

    lines.append("  [Source Note]: Findings are synthesized from supplied speech-to-text dictation and have not been independently verified against native radiological images.")
    lines.append("")

    # 4. Section: Impression
    lines.append("IMPRESSION:")
    if impressions:
        for i, imp in enumerate(impressions, 1):
            clean_imp = imp.strip()
            lines.append(f"  {i}. {clean_imp}")
    else:
        lines.append("  1. Incomplete clinical dictation; attending clinician to synthesize diagnostic impression.")
    lines.append("")

    # 5. Section: Recommendations
    lines.append("RECOMMENDATIONS & CLINICAL CORRELATION:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
    else:
        # Check if any indeterminate findings exist to provide conservative, indication-based guidance
        has_indeterminate = any(f.get("certainty") in ("unclear / clinician review required", "indeterminate") for f in imaging_findings)
        if has_indeterminate:
            lines.append("  • Clinical correlation advised. Cross-sectional 3D CBCT imaging may be indicated to evaluate indeterminate canal proximity or anatomical relationships if surgical intervention is planned.")
        else:
            lines.append("  • None dictated. Attending clinician to determine management following clinical correlation.")
    lines.append("")

    # 6. Section: Clinician Review & Attestation
    lines.append("CLINICIAN REVIEW & ATTESTATION STATEMENT:")
    if is_confirmed:
        lines.append(f"  [X] Confirmed & Electronically Signed by: {clinician_name} ({confirmed_at or datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})")
        lines.append("      Official Medical Record - Authenticated Diagnostic Consultation")
    else:
        lines.append("  [ ] I confirm that I have reviewed this AI-synthesized radiological draft report, verified all anatomical observations and indeterminate findings, and approve this report for clinical documentation.")
        lines.append("      Attending Clinician / Radiologist: ___________________________   Date: _______________")
    lines.append("=" * 72)
    return "\n".join(lines)


def synthesize_impression_from_findings(
    findings: List[Dict[str, Any]],
    artifacts: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    """
    Synthesizes impressions dynamically from structured clinical findings without hardcoded disease lists.
    Groups significant observations, structural dimensions, negative assurances, and speech clarifications.
    Excludes administrative metadata, headers, recommendations, and repetitive site prefixes.
    """
    imp_items = []

    from .claim_verification import is_meta_or_instruction_sentence

    # 1. Significant observations (positive, indeterminate, or differential findings)
    positive_findings = [
        f for f in findings
        if f.get("certainty") in ("definite observation", "definite observation (etiology uncertain)", "suspected / differential diagnosis")
        and not f.get("requires_clarification")
        and not is_symptom_or_history(f)
        and not is_meta_or_instruction_sentence(str(f.get("observation", "")))
        and not re.search(r"^(?:patient\s*id|patient\s*name|study|study\s*type|scan\s*type|mrn|name:|id:|serial\s*#?|radiology\s+report\s+details|conducted\s+by)\b", str(f.get("observation", "")).lower().strip())
        and not re.search(r"\b(further\s+evaluation\s+is\s+recommended|routine\s+follow-up\s+is\s+advised)\b", str(f.get("observation", "")).lower().strip())
    ]

    for f in positive_findings:
        obs = f.get("observation", "").strip()
        site = f.get("anatomical_site", "").strip()

        # Clean section header prefix or introductory boilerplate
        cleaned_obs = re.sub(r"^(?:IMPRESSIONS?\s*(?:&|and)\s*SPECIFIC\s+FINDINGS|SPECIFIC\s+FINDINGS|DENTAL\s+FINDINGS)\s*[:\-]?\s*", "", obs, flags=re.IGNORECASE).strip()
        cleaned_obs = re.sub(r"^(?:at\s+(?:the\s+)?[^,]+,\s*|the\s+|there\s+is\s+|there\s+are\s+)", "", cleaned_obs, flags=re.IGNORECASE).strip()
        if cleaned_obs:
            cleaned_obs = cleaned_obs[0].upper() + cleaned_obs[1:]

        # Avoid redundant site name prefix if already in observation or tooth/quadrant is named
        norm_site = re.sub(r"[&,]", " and ", site.lower()).strip() if site else ""
        norm_obs = re.sub(r"[&,]", " and ", cleaned_obs.lower()).strip()
        site_in_obs = (norm_site in norm_obs) or any(w in norm_obs for w in norm_site.split() if len(w) > 4) if site else False
        has_tooth_in_obs = bool(re.search(r"\b(?:tooth|teeth|#\d+|UR|UL|LL|LR|molar|premolar|incisor|canine)\b", cleaned_obs, re.I))
        if site and not site_in_obs and not has_tooth_in_obs and site.lower() not in ("dentition", "regional anatomy", "general", "soft tissue", "multiple teeth", "general observation", "clinical symptoms & history"):
            item_text = f"{site}: {cleaned_obs}"
        else:
            item_text = cleaned_obs

        # Deduplicate
        if not any(item_text.lower() == p.lower() for p in imp_items):
            imp_items.append(item_text)

    # 2. Key negative findings (complications / abnormalities ruled out)
    neg_findings = [
        f for f in findings
        if f.get("certainty") == "negative / normal finding"
        and not f.get("requires_clarification")
        and not is_meta_or_instruction_sentence(str(f.get("observation", "")))
        and "no other abnormal finding" not in f.get("observation", "").lower()
        and not re.search(r"^(?:patient\s*id|patient\s*name|study|mrn|serial)\b", str(f.get("observation", "")).lower().strip())
    ]

    if neg_findings:
        neg_descs = [nf.get("observation", "").strip().rstrip(".") for nf in neg_findings]
        if all(re.match(r"^no\s+", d, re.IGNORECASE) for d in neg_descs):
            cleaned_phrases = [re.sub(r"^no\s+", "", d, flags=re.IGNORECASE) for d in neg_descs]
            if any(re.search(r"\b(?:is|are|was|were)\s+(?:observed|identified|seen|noted|dictated)\b", cp, re.I) for cp in cleaned_phrases) or any(cp.strip().endswith(("observed", "identified", "seen", "noted", "dictated")) for cp in cleaned_phrases):
                imp_items.append("; ".join(f"No {cp.strip().rstrip('.')}" for cp in cleaned_phrases) + ".")
            else:
                cleaned_phrases = [re.sub(r"\s+(?:is|are)\s+(?:observed|identified|seen|noted)$", "", cp, flags=re.IGNORECASE) for cp in cleaned_phrases]
                imp_items.append(f"No {' or '.join(cleaned_phrases)} observed.")
        else:
            imp_items.append("; ".join(neg_descs) + ".")

    # 3. Indeterminate observations requiring clinical correlation
    indeterminate_findings = [
        f for f in findings
        if f.get("certainty") in ("unclear / clinician review required", "indeterminate")
        and not is_meta_or_instruction_sentence(str(f.get("observation", "")))
        and not f.get("requires_clarification")
    ]
    for f in indeterminate_findings:
        obs = f.get("observation", "").strip()
        if obs:
            clean_obs = obs[0].upper() + obs[1:]
            if not clean_obs.endswith("."):
                clean_obs += " (indeterminate, clinician review advised)."
            if not any(clean_obs.lower() in p.lower() for p in imp_items):
                imp_items.append(clean_obs)

    # 4. Flagged speech transcription artifacts
    if artifacts:
        for art in artifacts:
            imp_items.append(f"Unclear phrase '{art['matched_phrase']}' flagged for clinician clarification.")

    if not imp_items:
        return ["Radiological assessment completed."]

    return imp_items
