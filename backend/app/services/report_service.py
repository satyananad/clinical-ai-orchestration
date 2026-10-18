import re
import datetime
from typing import List, Dict, Any, Optional

def compose_narrative(
    patient_name: str,
    findings: List[Dict[str, Any]],
    impressions: List[str],
    recommendations: List[str],
    patient_details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Composes a readable, standardized radiological narrative report for clinician review.
    Enforces separation between administrative metadata, observations, impressions, and recommendations.
    """
    details = patient_details or {}
    supplied_name = details.get("patient_name")
    supplied_id = details.get("patient_id")
    study_type = details.get("study_type")

    lines = []
    lines.append("=" * 68)
    lines.append("*** RADIOLOGICAL DRAFT REPORT - REQUIRES CLINICIAN SIGN-OFF ***")
    lines.append("=" * 68)
    if "unassigned" in patient_name.lower():
        if supplied_name or supplied_id:
            lines.append(f"Patient Name:    Unassigned (Supplied in transcript: {supplied_name or 'N/A'}; ID: {supplied_id or 'N/A'} - Pending Clinician Verification)")
            lines.append("Patient ID:      None (Requires patient selection before permanent record entry)")
        else:
            lines.append("Patient Name:    Unassigned (Transcript not linked to a registered patient)")
            lines.append("Patient ID:      None (Requires patient selection before permanent record entry)")
    else:
        lines.append(f"Patient Name:    {patient_name}")
        if supplied_id:
            lines.append(f"Patient ID:      {supplied_id}")
    if study_type:
        lines.append(f"Study Type:      {study_type}")
    lines.append("Status:          DRAFT (AI Dictation Synthesis - Pending Clinician Review)")
    lines.append(f"Date of Draft:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("DETAILED FINDINGS:")
    for i, f in enumerate(findings, 1):
        site_str = f"[{f.get('anatomical_site', 'Dentition')}]"
        if f.get("tooth_number"):
            site_str += f" (Tooth {f['tooth_number']})"
        lines.append(f"  {i}. {site_str} • {f.get('category', 'General Observation')}")
        lines.append(f"     Observation: {f['observation']}")
        lines.append(f"     Certainty:   {f['certainty']}")
    lines.append("")
    lines.append("IMPRESSION:")
    for i, imp in enumerate(impressions, 1):
        lines.append(f"  {i}. {imp}")
    lines.append("")
    lines.append("RECOMMENDATIONS & CLINICAL CORRELATION:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
    else:
        lines.append("  • None dictated. Attending clinician to determine management following clinical correlation.")
    lines.append("=" * 68)
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

    # 1. Significant observations (positive, indeterminate, or differential findings)
    positive_findings = [
        f for f in findings
        if f.get("certainty") in ("definite observation", "definite observation (etiology uncertain)", "suspected / differential diagnosis")
        and not f.get("requires_clarification")
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
        site_in_obs = site.lower() in cleaned_obs.lower() if site else False
        has_tooth_in_obs = bool(re.search(r"\b(?:tooth|teeth|#\d+|UR|UL|LL|LR|molar|premolar|incisor|canine)\b", cleaned_obs, re.I))
        if site and not site_in_obs and not has_tooth_in_obs and site.lower() not in ("dentition", "regional anatomy", "general", "soft tissue", "multiple teeth", "general observation"):
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
        and "no other abnormal finding" not in f.get("observation", "").lower()
        and not re.search(r"^(?:patient\s*id|patient\s*name|study|mrn|serial)\b", str(f.get("observation", "")).lower().strip())
    ]

    if neg_findings:
        neg_descs = [nf.get("observation", "").strip().rstrip(".") for nf in neg_findings]
        if all(re.match(r"^no\s+", d, re.IGNORECASE) for d in neg_descs):
            cleaned_phrases = [re.sub(r"^no\s+", "", d, flags=re.IGNORECASE) for d in neg_descs]
            cleaned_phrases = [re.sub(r"\s+(?:is|are)\s+(?:observed|identified|seen|noted)$", "", cp, flags=re.IGNORECASE) for cp in cleaned_phrases]
            imp_items.append(f"No {' or '.join(cleaned_phrases)} observed.")
        else:
            imp_items.append("; ".join(neg_descs) + ".")

    # 3. Flagged speech transcription artifacts
    if artifacts:
        for art in artifacts:
            imp_items.append(f"Unclear phrase '{art['matched_phrase']}' flagged for clinician clarification.")

    if not imp_items:
        return ["Radiological assessment completed."]

    return imp_items
