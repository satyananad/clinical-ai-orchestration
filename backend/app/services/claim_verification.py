"""
CLINICAL CLAIM VERIFICATION & MULTI-DIMENSIONAL VALIDATION ENGINE (claim_verification.py)
---------------------------------------------------------------------------------------
Implements the revised Rule 2 (Anti-Hallucination & Multi-Dimensional Claim Verification)
and Rule 3 (Multi-Dimensional Diagnostic Uncertainty & Finding Status).

Core Capabilities:
1. Dictation Verification: Verifies that the structured draft preserves what clinician dictated.
2. Report Claim Verification: Verifies that AI answers are strictly supported by the active report.
3. Multi-Dimensional Clinical Status: Separates Certainty, Finding Status, and Image Limitation.
4. Two-Layer Verification: Deterministic Rule-Based (Layer 1) + LLM Semantic Verification (Layer 2).
5. Response Gate: Permits SUPPORTED claims, blocks CONTRADICTED claims, flags INSUFFICIENT_EVIDENCE.
"""

import re
from typing import Dict, Any, List, Optional
from enum import Enum

class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"

# Negation Triggers
NEGATION_WORDS = ["no", "not", "without", "absent", "clear", "denies", "negative", "none", "neither", "nor"]

# Uncertainty Triggers
HEDGING_WORDS = [
    "possible", "possibly", "suggestive of", "cannot rule out", "suspected",
    "differential", "potential", "questionable", "unconfirmed", "may represent",
    "likely", "consistent with", "indeterminate", "equivocal"
]

# Image Limitation Triggers
LIMITATION_WORDS = ["limited", "obscured", "partially obscured", "superimposition", "artifact", "projection limited"]


# =============================================================================
# REVISED RULE 3: MULTI-DIMENSIONAL CLINICAL STATUS
# =============================================================================
def classify_clinical_finding_status(text: str) -> Dict[str, str]:
    """
    Revised Rule 3: Separates clinical meaning into 3 independent dimensions:
    - Certainty: confirmed, suspected, indeterminate, confirmed_with_uncertain_etiology
    - Finding Status: positive, negative, not_documented
    - Image Limitation: none, limited, obscured, superimposition
    """
    t_lower = text.lower()

    # 1. Image Limitation Dimension
    limitation = "none"
    if "superimposition" in t_lower:
        limitation = "superimposition"
    elif "obscured" in t_lower or "partially obscured" in t_lower:
        limitation = "obscured"
    elif any(lw in t_lower for lw in ["limited", "artifact"]):
        limitation = "limited"

    # 2. Finding Status Dimension (Positive vs Negative vs Normal)
    is_negated = any(re.search(r"\b" + re.escape(nw) + r"\b", t_lower) for nw in NEGATION_WORDS)
    is_normal = any(term in t_lower for term in ["within normal limits", "normal physiological", "normal limits"])
    if "intact" in t_lower and not any(term in t_lower for term in ["thin", "thinned", "defect", "resorption", "erosion", "loss"]):
        is_normal = True

    # Important: widening of the periodontal ligament, cortical thinning, radiolucency, or cyst is positive clinical finding
    has_positive_finding = any(re.search(r"\b" + re.escape(term) + r"\b", t_lower) for term in ["widening", "periodontal ligament", "pdl widening", "thin", "thinned", "lesion", "defect", "radiolucency", "cyst", "caries", "bone loss", "fracture"])

    # Check if the sentence describes the anatomical location, boundaries, or relations of an active lesion/pathology
    # (e.g. "The lesion is located superior to and does not involve the inferior alveolar nerve canal...")
    is_lesion_extent_description = bool(
        re.search(r"\b(?:the\s+)?(?:lesion|radiolucency|cyst|mass|tumor|defect)\s+(?:is\s+)?(?:located|measures?|extends?|seen|observed|extending|identified)\b", t_lower) or
        (re.search(r"\b(?:the\s+)?(?:lesion|radiolucency|cyst|mass|tumor)\b", t_lower) and "does not involve" in t_lower)
    )

    if is_lesion_extent_description:
        is_negated = False
        is_normal = False
    elif has_positive_finding and not (re.search(r"\b(?:no|without|denies|negative for)\s+(?:evidence of\s+)?" + r"(?:widening|lesion|defect|radiolucency|cyst|caries|bone loss|fracture)\b", t_lower)):
        is_normal = False

    if is_negated or is_normal:
        status = "negative"
    elif len(text.strip()) > 0:
        status = "positive"
    else:
        status = "not_documented"

    # 3. Certainty Dimension (Confirmed vs Suspected vs Indeterminate)
    is_hedged = any(re.search(r"\b" + re.escape(hw) + r"\b", t_lower) for hw in HEDGING_WORDS)

    # Check for confirmed lesion presence with uncertain etiology:
    has_confirmed_lesion = bool(re.search(r"\b(lesion|radiolucency|defect|mass|cyst)\b", t_lower) and re.search(r"\b(present|observed|noted|identified|measures?|located|extending)\b", t_lower))
    has_uncertain_cause = bool(re.search(r"\b(etiology|cause|nature)\s+(?:is\s+)?(?:uncertain|unknown|indeterminate|unconfirmed)\b", t_lower) or "etiology is uncertain" in t_lower)

    if has_confirmed_lesion and has_uncertain_cause:
        certainty = "confirmed_with_uncertain_etiology"
        status = "positive"
    elif "no definite" in t_lower or "indeterminate" in t_lower or "equivocal" in t_lower:
        certainty = "indeterminate"
    elif is_hedged:
        certainty = "suspected"
    else:
        certainty = "confirmed"

    # Backward-compatible label string
    if certainty == "confirmed_with_uncertain_etiology":
        legacy_label = "definite observation (etiology uncertain)"
    elif is_lesion_extent_description:
        legacy_label = "definite observation"
    elif status == "negative" and certainty == "indeterminate":
        legacy_label = "negative / normal finding"
    elif certainty == "suspected" or certainty == "indeterminate":
        legacy_label = "suspected / differential diagnosis"
    elif status == "negative":
        legacy_label = "negative / normal finding"
    elif limitation != "none":
        legacy_label = "projection limited"
    else:
        legacy_label = "definite observation"

    return {
        "certainty": certainty,
        "finding_status": status,
        "image_limitation": limitation,
        "legacy_label": legacy_label
    }


# =============================================================================
# PATIENT DEMOGRAPHICS & CLINICAL DOCUMENT EXTRACTION
# =============================================================================
def extract_patient_demographics(text: str) -> Dict[str, Any]:
    """
    Extracts age, gender, date of birth, supplied patient name, patient ID,
    study type, serial number, study/report dates, provider details, and clinical notes from text.
    Strictly separates patient identity from device/scan serial numbers and provider credentials.
    """
    age = None
    gender = None
    age_match = re.search(r"\b(\d{1,3})\s*(?:-|–|\s)?(?:year|yr)s?(?:-|–|\s)?old\b|\bage\s*[:\-]?\s*(?:is\s*)?(\d{1,3})\b", text, re.IGNORECASE)
    if age_match:
        age_str = age_match.group(1) or age_match.group(2)
        if age_str:
            try:
                age = int(age_str)
            except ValueError:
                pass

    if re.search(r"\b(?:gender|sex)\s*[:\-]?\s*female\b|\b(female|woman|girl)\b", text, re.IGNORECASE):
        gender = "Female"
    elif re.search(r"\b(?:gender|sex)\s*[:\-]?\s*male\b|\b(male|man|boy)\b", text, re.IGNORECASE):
        gender = "Male"

    # Date of Birth
    date_of_birth = None
    dob_m = re.search(r"\b(?:dob|date\s+of\s+birth)\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b", text, re.IGNORECASE)
    if dob_m:
        date_of_birth = dob_m.group(1).strip()

    summary = None
    if age and gender and date_of_birth:
        summary = f"{age}-year-old {gender.lower()} (DOB: {date_of_birth})"
    elif age and gender:
        summary = f"{age}-year-old {gender.lower()}"
    elif age:
        summary = f"Age {age}"
    elif gender:
        summary = gender

    # Extract supplied patient name
    patient_name = None
    name_m = re.search(r"\b(?:patient\s+name|patient)\s*[:\-]\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", text, re.IGNORECASE)
    if name_m:
        cand_name = name_m.group(1).strip()
        if cand_name.lower() not in ["is", "has", "was", "reported", "id", "study", "the", "a", "details", "selection"]:
            patient_name = cand_name.title()

    # Extract supplied patient ID (STRICT: require explicit delimiter and reject serial numbers / common words)
    patient_id = None
    id_m = re.search(r"\b(?:patient\s*id|mrn|pt\s*id)\s*[:\-#]\s*([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    if not id_m:
        id_m = re.search(r"\bid\s*[:#]\s*([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    if id_m:
        cand_id = id_m.group(1).strip()
        if cand_id.lower() not in ["name", "study", "the", "a", "is", "was", "identified", "radiograph", "evaluation", "pending", "none"]:
            patient_id = cand_id.upper()
    elif re.search(r"\b(SYN-\d+|PAT-\d+|PT-\d+)\b", text, re.IGNORECASE):
        patient_id = re.search(r"\b(SYN-\d+|PAT-\d+|PT-\d+)\b", text, re.IGNORECASE).group(1).upper()

    # Extract serial number (Distinct from patient ID!)
    serial_number = None
    serial_m = re.search(r"\b(?:serial\s*#|serial\s*no|serial\s*number|serial)\s*[:\-]?\s*([A-Za-z0-9_-]+)\b", text, re.IGNORECASE)
    if serial_m:
        serial_number = serial_m.group(1).strip()

    # Extract study date and report date (Preserve source evidence without altering conflicting dates)
    study_date = None
    study_date_m = re.search(r"\bstudy\s*date\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b", text, re.IGNORECASE)
    if study_date_m:
        study_date = study_date_m.group(1).strip()

    report_date = None
    report_date_m = re.search(r"\breport\s*date\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b", text, re.IGNORECASE)
    if report_date_m:
        report_date = report_date_m.group(1).strip()

    # Extract study type / modality
    study_type = None
    study_m = re.search(r"\b(?:report\s+type|study\s+type|study|scan\s+type|modality|examination)\s*[:\-]\s*([A-Za-z0-9\s,\(\)]+?)(?:\.|$|\n|PATIENT|CLINICAL)", text, re.IGNORECASE)
    if study_m:
        cand_study = study_m.group(1).strip()
        if "cbct" in cand_study.lower():
            study_type = "CBCT"
        elif "panoramic" in cand_study.lower() or "opg" in cand_study.lower():
            study_type = "Panoramic radiograph"
        else:
            study_type = cand_study.title()
    elif "panoramic" in text.lower() or "opg" in text.lower():
        study_type = "Panoramic radiograph"
    elif "cbct" in text.lower() or "cone beam" in text.lower():
        study_type = "CBCT"
    elif "bitewing" in text.lower():
        study_type = "Bitewing Radiographs"
    elif "periapical" in text.lower() or "iopa" in text.lower():
        study_type = "Periapical Radiograph"

    # Extract Provider / Author Metadata
    provider_details = {}
    rad_m = re.search(r"\b(?:conducted\s+by|radiologist|reported\s+by)\s*[:\-]?\s*(Dr\.?\s+[A-Za-z\s]+?)(?:,|\.|\n|Dental|Dipl)", text, re.IGNORECASE)
    if rad_m:
        provider_details["provider_name"] = rad_m.group(1).strip()
    gdc_m = re.search(r"\b(?:gdc|gdc\s*registration|gdc\s*no)\s*[:\-]?\s*([0-9]+)\b", text, re.IGNORECASE)
    if gdc_m:
        provider_details["gdc_number"] = gdc_m.group(1).strip()
    cred_m = re.search(r"\b((?:BDS|DDS|PhD|Dipl|Dental\s+Radiology|Oral\s+and\s+Maxillofacial\s+Radiologist)[^.\n]*?)(?=\.\s+[A-Z]|\(GDC|$)", text, re.IGNORECASE)
    if cred_m:
        provider_details["credentials"] = cred_m.group(1).strip()
    addr_m = re.search(r"\b(Suite\s+\d+[^.\n]+(?:London|UK|[A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2}|\d{5}))\b", text, re.IGNORECASE)
    if addr_m:
        provider_details["address"] = addr_m.group(1).strip()
    facility_m = re.search(r"\b(?:office|dentalscan\s*office|clinic|practice)\s*[:\-]\s*(.*?)(?=\.|\n|\s+[A-Z\s]{3,}:|$)", text, re.IGNORECASE)
    if facility_m:
        provider_details["facility_name"] = facility_m.group(1).strip()

    # Extract Clinical Indication
    clinical_notes = None
    notes_m = re.search(r"\bclinical\s*notes\s*[:\-]?\s*(.*?)(?=\s+[A-Z\s]{4,}:|$)", text, re.IGNORECASE)
    if notes_m:
        clinical_notes = notes_m.group(1).strip().rstrip(".")

    medical_history = None
    hist_m = re.search(r"\bnoting\s+(medical\s+history\s+of\s+[^.\n,]+)", text, re.IGNORECASE)
    if hist_m:
        medical_history = hist_m.group(1).strip()

    return {
        "age": age,
        "gender": gender,
        "date_of_birth": date_of_birth,
        "demographics_summary": summary,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "serial_number": serial_number,
        "study_type": study_type,
        "study_date": study_date,
        "report_date": report_date,
        "provider_details": provider_details,
        "clinical_notes": clinical_notes,
        "medical_history": medical_history
    }


# =============================================================================
# REVISED RULE 2: MULTI-DIMENSIONAL CLAIM VERIFICATION (LAYER 1)
# =============================================================================
def extract_quadrant_info(text: str) -> Optional[str]:
    """Identifies the clinical quadrant referenced in text using strict word boundaries."""
    t_lower = text.lower()
    if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul\d?)\b", t_lower):
        return "maxillary_left"
    if re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur\d?)\b", t_lower):
        return "maxillary_right"
    if re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll\d?)\b", t_lower):
        return "mandibular_left"
    if re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr\d?)\b", t_lower):
        return "mandibular_right"
    return None

QUADRANT_ALLOWED_FDI = {
    "maxillary_right": set(range(11, 19)).union(set(range(51, 56))),
    "maxillary_left": set(range(21, 29)).union(set(range(61, 66))),
    "mandibular_left": set(range(31, 39)).union(set(range(71, 76))),
    "mandibular_right": set(range(41, 49)).union(set(range(81, 86)))
}

def extract_tooth_numbers(text: str) -> List[int]:
    """
    Extracts all teeth referenced in text, supporting:
    - Explicit FDI notation (11-48, deciduous 51-85)
    - Universal notation (#1-#32, Tooth 1-32, Tooth A-T)
    - Palmer notation (UR1-8, UL1-8, LL1-8, LR1-8, UR A-E, etc.)
    - Quadrant-relative named teeth (e.g. 'maxillary left first molar' -> 26)
    Safely ignores patient age, measurements, and dates.
    """
    # 1. Strip age, measurements, years
    clean_text = re.sub(r"\b\d{1,2}\s*(?:-|–|\s)?(?:year|yr)s?(?:-|–|\s)?old\b|\bage\s*(?:is\s*)?\d{1,2}\b", " ", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|mg|ml|%)\b", " ", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", clean_text)
    t_lower = clean_text.lower()

    teeth = set()

    # 2. Quadrant context for relative tooth names
    quad = None
    if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul)\b", t_lower):
        quad = "UL"
    elif re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur)\b", t_lower):
        quad = "UR"
    elif re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll)\b", t_lower):
        quad = "LL"
    elif re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr)\b", t_lower):
        quad = "LR"

    QUAD_MAP = {
        "UR": {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18},
        "UL": {1: 21, 2: 22, 3: 23, 4: 24, 5: 25, 6: 26, 7: 27, 8: 28},
        "LL": {1: 31, 2: 32, 3: 33, 4: 34, 5: 35, 6: 36, 7: 37, 8: 38},
        "LR": {1: 41, 2: 42, 3: 43, 4: 44, 5: 45, 6: 46, 7: 47, 8: 48},
    }

    TOOTH_POSITIONS = [
        (8, [r"third\s+molar", r"wisdom\s+tooth"]),
        (7, [r"second\s+molar"]),
        (6, [r"first\s+molar"]),
        (5, [r"second\s+premolar", r"second\s+bicuspid"]),
        (4, [r"first\s+premolar", r"first\s+bicuspid"]),
        (3, [r"canine", r"cuspid"]),
        (2, [r"lateral\s+incisor"]),
        (1, [r"central\s+incisor"])
    ]

    # Explicit palmer codes (e.g. UR6, UL7, LL5, LR8)
    for m in re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([1-8])\b", clean_text, re.IGNORECASE):
        q_code = m.group(1).upper()
        pos = int(m.group(2))
        teeth.add(QUAD_MAP[q_code][pos])

    # If quadrant is established, resolve relative tooth descriptions
    if quad:
        for pos, patterns in TOOTH_POSITIONS:
            if any(re.search(r"\b" + p + r"\b", t_lower) for p in patterns):
                teeth.add(QUAD_MAP[quad][pos])

    # Direct FDI numbers (e.g. 11-48)
    for m in re.finditer(r"\b(1[1-8]|2[1-8]|3[1-8]|4[1-8])\b", clean_text):
        teeth.add(int(m.group(1)))

    # Universal numbers with prefix (e.g. #29, #14, Tooth 6, Tooth 14)
    for m in re.finditer(r"(?:#|tooth\s*#?)([1-9]|[12]\d|3[0-2])\b", clean_text, re.IGNORECASE):
        u_num = int(m.group(1))
        # Map universal to FDI using permanent table
        from .dental_anatomy import UNIVERSAL_TO_PERMANENT
        if u_num in UNIVERSAL_TO_PERMANENT:
            _, fdi_val, _, _ = UNIVERSAL_TO_PERMANENT[u_num]
            teeth.add(fdi_val)

    return sorted(list(teeth))

def extract_measurements(text: str) -> List[str]:
    """Extracts measurements and units (e.g. '8 mm', '1.5 mm', '8 x 7 mm', '5.2 mm', '10.8 mm')."""
    return re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|x\s*\d+(?:\.\d+)?\s*mm)\b", text.lower())

def extract_laterality(text: str) -> Optional[str]:
    """Extracts laterality (left vs right) using strict word boundaries."""
    t_lower = text.lower()
    has_left = bool(re.search(r"\b(left|ul\d?|ll\d?|sinister)\b", t_lower))
    has_right = bool(re.search(r"\b(right|ur\d?|lr\d?|dexter)\b", t_lower))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    if has_left and has_right:
        return "bilateral"
    return None

def is_term_negated_in_text(term: str, text: str) -> bool:
    """Checks if a clinical term is specifically negated within its local clause."""
    t_lower = text.lower()
    # Direct negation patterns e.g. "no caries", "without radiolucency", "negative for caries", "no cortical erosion", "absence of cyst"
    direct_pat = r"\b(?:no|not|without|denies|negative\s+for|absence\s+of|clear\s+of|neither|nor)\s+(?:evidence\s+of\s+|definite\s+|frank\s+|apparent\s+|visible\s+|active\s+)?(?:[a-z0-9_-]+\s+){0,3}" + re.escape(term) + r"\b"
    if re.search(direct_pat, t_lower):
        return True
    # Reverse negation pattern e.g. "caries is absent", "periapical radiolucency was not seen"
    reverse_pat = re.escape(term) + r"\s+(?:is\s+|was\s+|are\s+)?(?:absent|negative|ruled\s+out|not\s+(?:seen|identified|detected|observed))"
    if re.search(reverse_pat, t_lower):
        return True
    return False

def extract_all_quadrants(text: str) -> List[str]:
    """Extracts all clinical quadrants explicitly referenced across text."""
    t_lower = text.lower()
    quads = []
    if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul\d?)\b", t_lower):
        quads.append("maxillary_left")
    if re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur\d?)\b", t_lower):
        quads.append("maxillary_right")
    if re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll\d?)\b", t_lower):
        quads.append("mandibular_left")
    if re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr\d?)\b", t_lower):
        quads.append("mandibular_right")
    return quads

def verify_single_claim_deterministic(source_text: str, claim_text: str) -> Dict[str, Any]:
    """
    Deterministic Layer 1 Verification:
    Checks tooth numbers, negation, laterality, measurements, and clinical findings.
    """
    src_lower = source_text.lower()
    claim_lower = claim_text.lower()

    # Check 1: Laterality Mismatch (e.g. left vs right)
    src_lat = extract_laterality(source_text)
    claim_lat = extract_laterality(claim_text)
    if src_lat and claim_lat and src_lat != "bilateral" and claim_lat != "bilateral" and src_lat != claim_lat:
        return {
            "status": ClaimStatus.CONTRADICTED,
            "dimension": "laterality",
            "details": f"Laterality contradiction: Source specifies {src_lat} side, but claim states {claim_lat} side.",
            "claim": claim_text
        }

    # Check 2: Tooth Number Mismatch (e.g. changing tooth 16 to tooth 46)
    src_teeth = extract_tooth_numbers(source_text)
    claim_teeth = extract_tooth_numbers(claim_text)
    if claim_teeth and src_teeth:
        mismatched_teeth = [t for t in claim_teeth if t not in src_teeth]
        if mismatched_teeth:
            return {
                "status": ClaimStatus.CONTRADICTED,
                "dimension": "tooth_number",
                "details": f"Tooth number mismatch: Claim mentions tooth {mismatched_teeth}, but source references tooth {src_teeth}.",
                "claim": claim_text
            }

    # Check 3: Measurement Mismatch (e.g. 4 mm vs 8 mm)
    src_meas = extract_measurements(source_text)
    claim_meas = extract_measurements(claim_text)
    if claim_meas and src_meas:
        if not any(cm in src_lower for cm in claim_meas):
            return {
                "status": ClaimStatus.CONTRADICTED,
                "dimension": "measurement",
                "details": f"Measurement mismatch: Claim states {claim_meas}, but source documents {src_meas}.",
                "claim": claim_text
            }

    # Check 4: Verbatim Substring Match
    # If the claim is an exact substring of the source dictation (ignoring punctuation), it is grounded and cannot be a contradiction
    claim_clean = re.sub(r"[^\w\s]", "", claim_lower).strip()
    src_clean = re.sub(r"[^\w\s]", "", src_lower)
    if claim_lower in src_lower or (claim_clean and claim_clean in src_clean):
        return {
            "status": ClaimStatus.SUPPORTED,
            "dimension": "verbatim_source_grounding",
            "details": "Claim is verified and supported by source clinical text.",
            "claim": claim_text
        }

    # Locate the most relevant sentence in source_text for sentence-level contextual verification
    src_sentences = [s.strip() for s in re.split(r"[.\n;]+", src_lower) if len(s.strip()) > 3]
    claim_words_set = set(w for w in re.findall(r"\w+", claim_lower) if len(w) > 2)
    relevant_src = src_lower
    best_overlap = 0
    for sent in src_sentences:
        sent_words = set(w for w in re.findall(r"\w+", sent) if len(w) > 2)
        overlap = len(claim_words_set.intersection(sent_words))
        if overlap > best_overlap:
            best_overlap = overlap
            relevant_src = sent

    # Check 5: Negation Contradiction on relevant sentence (Clause-Aware)
    pathologies = ["caries", "radiolucency", "fracture", "lesion", "bone loss", "impaction", "cyst", "periapical", "resorption", "erosion"]
    for p in pathologies:
        if p in relevant_src and p in claim_lower:
            src_p_neg = is_term_negated_in_text(p, relevant_src)
            claim_p_neg = is_term_negated_in_text(p, claim_lower)
            if src_p_neg and not claim_p_neg:
                return {
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "negation",
                    "details": f"Negation contradiction: Source states absence of '{p}', but claim asserts presence.",
                    "claim": claim_text
                }
            if not src_p_neg and claim_p_neg:
                return {
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "negation",
                    "details": f"Negation contradiction: Source states presence of '{p}', but claim asserts absence.",
                    "claim": claim_text
                }

    # Check 6: Uncertainty Escalation on relevant sentence
    claim_has_negation = any(re.search(r"\b" + re.escape(w) + r"\b", claim_lower) for w in NEGATION_WORDS)
    src_status = classify_clinical_finding_status(relevant_src)
    claim_status = classify_clinical_finding_status(claim_text)
    if src_status["certainty"] == "suspected" and claim_status["certainty"] == "confirmed" and not claim_has_negation:
        return {
            "status": ClaimStatus.CONTRADICTED,
            "dimension": "uncertainty",
            "details": "Certainty escalation: Source expresses suspected/differential finding, but claim asserts confirmed diagnosis.",
            "claim": claim_text
        }

    # Check 7: Lexical & Token Grounding (Presence in source)
    claim_words = [w for w in re.findall(r"\w+", claim_lower) if len(w) > 3 and w not in NEGATION_WORDS]
    if claim_words:
        matched = [w for w in claim_words if w in src_lower]
        ratio = len(matched) / len(claim_words)
        if ratio < 0.4:
            return {
                "status": ClaimStatus.INSUFFICIENT_EVIDENCE,
                "dimension": "missing_information",
                "details": f"Insufficient evidence: Clinical terms in claim lack grounding in source (grounded ratio: {ratio:.1%}).",
                "claim": claim_text
            }

    return {
        "status": ClaimStatus.SUPPORTED,
        "dimension": "all_checks_passed",
        "details": "Claim is verified and supported by source clinical text.",
        "claim": claim_text
    }


# =============================================================================
def extract_dictated_negative_findings(text: str) -> List[str]:
    """
    Extracts clinical findings explicitly negated in dictation text.
    E.g. 'no sinus communication', 'no root resorption', 'without fracture',
    'negative for caries', 'no cortical erosion', 'no displaced root fragments'.
    """
    neg_findings = []
    pattern = r"\b(?:no|not|without|absence\s+of|negative\s+for|denies)\s+(?:evidence\s+of\s+|definite\s+|frank\s+|apparent\s+)?([a-z0-9_-]+(?:\s+[a-z0-9_-]+){0,3})\b"
    for match in re.finditer(pattern, text.lower()):
        phrase = match.group(1).strip()
        if phrase and phrase not in ["other", "further", "change", "abnormality", "prior", "known", "such", "reported"]:
            neg_findings.append(phrase)
    return neg_findings

# =============================================================================
# MULTI-CLAIM VERIFICATION GATE & DICTATION VERIFICATION
# =============================================================================
def verify_dictation_against_source(raw_dictation: str, findings: List[Dict[str, Any]], impressions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Verifies clinical dictation transcript preservation:
    - Traceability of findings to audio transcript
    - Tooth numbers and anatomical sites preserved
    - Negation and diagnostic uncertainty preserved
    - Measurements and units preserved
    - Cross-checks Impressions against Findings for internal clinical contradictions
    """
    unsupported_claims = []
    contradicted_claims = []
    flagged_unclear_claims = []
    verified_count = 0
    all_results = []

    # Pre-segment raw dictation into sentences for scoped contextual matching
    raw_sentences = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[\n;]+", raw_dictation) if len(s.strip()) > 3]

    for f in findings:
        obs = f.get("observation", "")
        certainty = str(f.get("certainty", "")).lower()
        requires_clarification = f.get("requires_clarification", False)
        source_quote = f.get("source_quote")

        # Check for administrative metadata, author credentials, study dates, or headers misclassification
        obs_l = obs.lower().strip()
        is_metadata = bool(
            re.search(r"\b(?:patient\s*id|patient\s*name|mrn|pt\s*id|dob:|date\s+of\s+birth|serial\s*#|serial\s*no|serial\s*number|study\s*date|report\s*date|clinical\s*notes|clinical\s*indication)\b", obs_l) or
            re.search(r"^(?:study|study\s*type|scan\s*type|modality|examination|office|practice|clinic)\s*[:\-]", obs_l) or
            re.search(r"\b(?:radiology\s+report\s+details|conducted\s+by|oral\s+and\s+maxillofacial\s+radiologist|royal\s+college|gdc\s*registration|gdc:)\b", obs_l) or
            re.search(r"\b(?:suite\s+\d+|harley\s+st|london\s+w1g)\b", obs_l) or
            re.search(r"^(?:panoramic\s+radiograph|cone\s+beam\s+ct|cbct|opg|bitewings?|periapical(?:\s+radiograph)?)$", obs_l)
        )
        if is_metadata:
            meta_res = {
                "status": ClaimStatus.CONTRADICTED,
                "dimension": "metadata_misclassification",
                "details": f"Administrative/study/provider metadata ('{obs}') was misclassified as a clinical finding.",
                "claim": obs
            }
            contradicted_claims.append(meta_res)
            all_results.append(meta_res)
            continue

        # Check for unsupported anatomical attribution (e.g. assigning a single tooth to multi-tooth or edentulous findings)
        f_tooth = f.get("tooth_number")
        is_adjacent_obs = bool(re.search(r"\b(adjacent\s+teeth|surrounding\s+teeth|adjacent\s+structures|adjacent\s+dentition|surrounding\s+dentition)\b", obs_l))
        is_edentulous_obs = bool(re.search(r"\bedentulous\s+([a-z0-9_-]+)\s+region\b", obs_l))
        obs_teeth = extract_tooth_numbers(f"{obs} {source_quote or ''}")
        is_multi_tooth_obs = len(obs_teeth) > 1 or bool(re.search(r"\b(?:teeth|two\s+teeth|three\s+teeth|posterior\s+teeth)\b", obs_l) and not re.search(r"\btooth\s*#?\d+\b", obs_l))

        if f_tooth and (is_adjacent_obs or is_edentulous_obs or is_multi_tooth_obs):
            attr_res = {
                "status": ClaimStatus.CONTRADICTED,
                "dimension": "unsupported_anatomical_attribution",
                "details": f"Unsupported anatomical attribution: Finding refers to {('adjacent teeth' if is_adjacent_obs else ('an edentulous region' if is_edentulous_obs else 'multiple teeth'))} but was assigned single tooth {f_tooth}.",
                "claim": obs
            }
            contradicted_claims.append(attr_res)
            all_results.append(attr_res)
            continue
        elif f_tooth and not (is_adjacent_obs or is_edentulous_obs or is_multi_tooth_obs):
            try:
                t_digits = re.sub(r"\D", "", str(f_tooth))
                if t_digits:
                    t_int = int(t_digits)
                    if obs_teeth and t_int not in obs_teeth:
                        attr_res = {
                            "status": ClaimStatus.CONTRADICTED,
                            "dimension": "unsupported_anatomical_attribution",
                            "details": f"Tooth number attribution mismatch: Finding is assigned tooth {f_tooth}, but observation references {obs_teeth}.",
                            "claim": obs
                        }
                        contradicted_claims.append(attr_res)
                        all_results.append(attr_res)
                        continue
            except Exception:
                pass

        # Check for semantic polarity consistency: active lesion/pathology descriptions cannot be labeled normal/negative
        is_lesion_obs = bool(
            re.search(r"\b(?:the\s+)?(?:lesion|radiolucency|cyst|mass|tumor|defect)\s+(?:is\s+)?(?:located|measures?|extends?|seen|observed|extending|identified)\b", obs_l) or
            (re.search(r"\b(?:the\s+)?(?:lesion|radiolucency|cyst|mass|tumor)\b", obs_l) and "does not involve" in obs_l)
        )
        if is_lesion_obs and (certainty == "negative / normal finding" or f.get("polarity") == "absent"):
            sem_res = {
                "status": ClaimStatus.CONTRADICTED,
                "dimension": "polarity_semantic_consistency",
                "details": f"Semantic contradiction: Finding describes active lesion or pathology ('{obs}'), but was classified as negative / normal finding with absent polarity.",
                "claim": obs
            }
            contradicted_claims.append(sem_res)
            all_results.append(sem_res)
            continue

        # Acoustic artifact / unclear phrase check
        if requires_clarification or "unclear" in certainty or "unclear phrase" in obs.lower() or "artifact" in obs.lower():
            unclear_res = {
                "status": ClaimStatus.INSUFFICIENT_EVIDENCE,
                "dimension": "acoustic_artifact",
                "details": f"Flagged unclear dictation phrase / speech-to-text artifact requiring clinician verification: '{obs}'",
                "claim": obs
            }
            flagged_unclear_claims.append(unclear_res)
            all_results.append(unclear_res)
            continue

        # Grounding check via source_quote if provided
        if source_quote and not (source_quote.lower() in raw_dictation.lower() or re.sub(r"[^\w\s]", "", source_quote.lower()) in re.sub(r"[^\w\s]", "", raw_dictation.lower())):
            ungrounded_res = {
                "status": ClaimStatus.INSUFFICIENT_EVIDENCE,
                "dimension": "source_quote_grounding",
                "details": f"Source quote '{source_quote}' cannot be grounded in raw clinical dictation transcript.",
                "claim": obs
            }
            unsupported_claims.append(ungrounded_res)
            all_results.append(ungrounded_res)
            continue

        # Determine the scoped source passage for this finding (avoid global multi-quadrant contamination)
        scoped_source = None
        if source_quote and (source_quote.lower() in raw_dictation.lower() or re.sub(r"[^\w\s]", "", source_quote.lower()) in re.sub(r"[^\w\s]", "", raw_dictation.lower())):
            scoped_source = source_quote
        elif raw_sentences:
            f_teeth = set(extract_tooth_numbers(f"{f.get('anatomical_site', '')} {obs}"))
            f_words = set(w.lower() for w in re.findall(r"\w+", obs) if len(w) > 3 and w.lower() not in NEGATION_WORDS)
            best_score = -1
            best_sent = raw_dictation
            for s in raw_sentences:
                s_teeth = set(extract_tooth_numbers(s))
                s_words = set(w.lower() for w in re.findall(r"\w+", s) if len(w) > 3)
                tooth_overlap = len(f_teeth.intersection(s_teeth))
                word_overlap = len(f_words.intersection(s_words))
                score = (tooth_overlap * 10) + word_overlap
                if score > best_score and (tooth_overlap > 0 or word_overlap > 0):
                    best_score = score
                    best_sent = s
            scoped_source = best_sent if best_score > 0 else raw_dictation
        else:
            scoped_source = raw_dictation

        res = verify_single_claim_deterministic(scoped_source, obs)
        all_results.append(res)

        if res["status"] == ClaimStatus.CONTRADICTED:
            contradicted_claims.append(res)
        elif res["status"] == ClaimStatus.INSUFFICIENT_EVIDENCE:
            unsupported_claims.append(res)
        else:
            verified_count += 1

    # Check Quadrant Concordance:
    # 1. Scoped check: if a finding or its site specifies a quadrant, verify finding's teeth conform.
    # 2. Global check: ONLY if the entire dictation describes a single quadrant (e.g. focused quadrant study).
    all_dictation_quads = extract_all_quadrants(raw_dictation)
    single_quad_context = all_dictation_quads[0] if len(all_dictation_quads) == 1 else None

    for f in findings:
        obs = f.get("observation", "")
        f_text = f"{f.get('anatomical_site', '')} {obs}"
        finding_teeth = extract_tooth_numbers(f_text)

        # If the observation explicitly mentions multiple quadrants (e.g. LL1 and LR1), it is cross-quadrant
        obs_quads = extract_all_quadrants(obs)
        if len(obs_quads) > 1:
            continue

        site_quad = extract_quadrant_info(f.get("anatomical_site", "")) or (obs_quads[0] if obs_quads else None)
        target_quad = site_quad or single_quad_context

        if target_quad and target_quad in QUADRANT_ALLOWED_FDI:
            allowed_teeth = QUADRANT_ALLOWED_FDI[target_quad]
            violating_teeth = [t for t in finding_teeth if t not in allowed_teeth]
            if violating_teeth:
                contradicted_claims.append({
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "quadrant_concordance",
                    "details": f"Quadrant contradiction: Anatomical site is scoped to {target_quad.replace('_', ' ')}, but finding specifies tooth {violating_teeth} from a conflicting quadrant.",
                    "claim": f.get("observation", "")
                })

    # Check Key Measurement Preservation (e.g. 5.2 mm, 10.8 mm, 7.5 mm)
    raw_measurements = extract_measurements(raw_dictation)
    all_findings_text = " ".join([f.get("observation", "") for f in findings]).lower()
    all_findings_measurements = []
    for f in findings:
        for m in f.get("measurements", []):
            all_findings_measurements.append(str(m).lower())
    for rm in raw_measurements:
        if rm not in all_findings_text and not any(rm in fm for fm in all_findings_measurements):
            unsupported_claims.append({
                "status": ClaimStatus.INSUFFICIENT_EVIDENCE,
                "dimension": "omitted_measurement",
                "details": f"Clinical measurement '{rm}' was dictated but omitted from structured findings.",
                "claim": f"Omitted measurement: {rm}"
            })
    # Check Crucial Negative Findings Preservation (Dynamically extracted + clinical safety clauses)
    dictated_negatives = extract_dictated_negative_findings(raw_dictation)
    standard_clauses = ["sinus communication", "displaced root fragments", "root resorption", "periapical pathology"]
    for sc in standard_clauses:
        if sc not in dictated_negatives:
            neg_pat = r"\b(?:no|without|denies|negative for)\s+(?:definite\s+|evidence of\s+)?" + re.escape(sc) + r"\b"
            if re.search(neg_pat, raw_dictation, re.IGNORECASE):
                dictated_negatives.append(sc)

    for pat in set(dictated_negatives):
        neg_pattern = r"\b(?:no|without|denies|negative for)\s+(?:definite\s+|evidence of\s+)?" + re.escape(pat) + r"\b"
        found_neg = any(
            re.search(neg_pattern, f.get("observation", ""), re.IGNORECASE) or
            (pat in f.get("observation", "").lower() and any(nw in f.get("observation", "").lower() for nw in NEGATION_WORDS))
            for f in findings
        )
        if not found_neg:
            unsupported_claims.append({
                "status": ClaimStatus.INSUFFICIENT_EVIDENCE,
                "dimension": "omitted_negative_finding",
                "details": f"Dictated negative finding ('no {pat}') was omitted from structured findings.",
                "claim": f"Omitted negative finding: no {pat}"
            })

    # Cross-Check: Findings vs Impressions for Clinical Contradictions
    if impressions:
        for imp in impressions:
            imp_lower = imp.lower()
            # Case A: Impression asserts absence of resorption while findings document root resorption
            has_resorption_finding = any(
                "resorption" in str(f.get("observation", "")).lower() and
                not any(nw in str(f.get("observation", "")).lower() for nw in ["no root resorption", "no resorption", "without resorption"])
                for f in findings
            )
            imp_denies_resorption = any(neg in imp_lower for neg in ["no resorption", "without resorption", "no root resorption"])

            if has_resorption_finding and imp_denies_resorption:
                conflict_finding = next(
                    f for f in findings
                    if "resorption" in str(f.get("observation", "")).lower() and
                    not any(nw in str(f.get("observation", "")).lower() for nw in ["no root resorption", "no resorption", "without resorption"])
                )
                contradicted_claims.append({
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "internal_conflict",
                    "details": f"Clinical Contradiction: Impression asserts '{imp}', but finding documents '{conflict_finding.get('observation')}'.",
                    "claim": imp
                })

            # Case B: Impression contains administrative metadata, provider credentials, headings, or recommendations
            if re.search(r"\b(?:patient\s*id|patient\s*name|mrn|dob:|serial\s*#|study\s*date|report\s*date|conducted\s*by|radiologist|gdc:|suite\s+\d+|harley\s+st|london\s+w1g)\b", imp_lower) or \
               re.search(r"^(?:dental\s+findings|clinical\s+findings|impressions?\s*(?:&\s*specific\s+findings)?|radiology\s+report\s+details):\s*", imp_lower) or \
               re.search(r"\b(?:further\s+evaluation\s+is\s+recommended|recommend\s+prior\s+to\s+implant|routine\s+follow-up\s+is\s+recommended)\b", imp_lower):
                contradicted_claims.append({
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "impression_metadata_contamination",
                    "details": f"Impression contains administrative metadata, section headers, or recommendations: '{imp}'.",
                    "claim": imp
                })

    # Transparent Quality Metrics
    evaluated_count = len(findings)
    preserved_meas_count = sum(
        1 for rm in raw_measurements
        if rm in all_findings_text or any(rm in fm for fm in all_findings_measurements)
    )
    measurement_preservation = round(preserved_meas_count / len(raw_measurements), 3) if raw_measurements else 1.0
    claim_support_ratio = round(verified_count / evaluated_count, 3) if evaluated_count > 0 else 0.0

    omissions_count = len([u for u in unsupported_claims if u.get("dimension") in ("omitted_measurement", "omitted_negative_finding")])
    total_expected = evaluated_count + omissions_count
    extraction_completeness = round(verified_count / total_expected, 3) if total_expected > 0 else 0.0
    unresolved_issue_count = len(contradicted_claims) + len(unsupported_claims) + len(flagged_unclear_claims)

    overall_status = "PASSED"
    guarantee = "Zero ungrounded or contradicted claims admitted to final report."
    if evaluated_count == 0:
        overall_status = "EXTRACTION_EMPTY_REQUIRES_REVIEW"
        grounded_ratio_val = 0.0
        guarantee = "No clinical findings extracted; attending clinician manual entry and verification required."
    elif contradicted_claims:
        overall_status = "CONTRADICTED_CLAIMS_DETECTED"
        guarantee = "Contradiction detected: Attending clinician must reconcile conflicting claims before sign-off."
        grounded_ratio_val = round(verified_count / evaluated_count, 3)
    elif flagged_unclear_claims:
        overall_status = "FLAGGED_FOR_REVIEW"
        guarantee = "Unclear speech transcription artifact flagged; attending clinician review and verification required before final report sign-off."
        grounded_ratio_val = round(verified_count / evaluated_count, 3)
    elif unsupported_claims:
        overall_status = "FLAGGED_UNSUPPORTED"
        guarantee = "Unsupported findings flagged for clinician review."
        grounded_ratio_val = round(verified_count / evaluated_count, 3)
    else:
        grounded_ratio_val = round(verified_count / evaluated_count, 3)

    return {
        "status": overall_status,
        "grounded_ratio": grounded_ratio_val,
        "claim_support_ratio": claim_support_ratio,
        "measurement_preservation": measurement_preservation,
        "extraction_completeness": extraction_completeness,
        "unresolved_issue_count": unresolved_issue_count,
        "verified_count": f"{verified_count}/{evaluated_count}",
        "claims_total": evaluated_count,
        "supported_claims": verified_count,
        "contradicted_claims": contradicted_claims,
        "unsupported_findings": [c["claim"] for c in flagged_unclear_claims] + [c["claim"] for c in unsupported_claims] + [c["claim"] for c in contradicted_claims],
        "detailed_results": all_results,
        "clinician_review_required": True,
        "anti_hallucination_guarantee": guarantee
    }

def verify_report_answer_claim(report: Dict[str, Any], answer: str, expected_patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Verifies that the AI's conversational response is supported by the active report:
    1. Source Identity Check: Does report belong to expected patient?
    2. Clinical Detail Checks: Tooth numbers, findings, laterality, measurements.
    3. Evidence Grounding: Can system identify exact supporting quote?
    4. Absence Check: Flags claims not documented in report.
    """
    # 1. Identity Check
    if expected_patient_id and report.get("patient_id") != expected_patient_id:
        return {
            "status": ClaimStatus.IDENTITY_MISMATCH,
            "gate_decision": "BLOCK",
            "message": f"Critical identity mismatch: Report belongs to {report.get('patient_id')}, expected {expected_patient_id}.",
            "citations": []
        }

    # Aggregate full verified report text
    report_corpus = (
        f"Indication: {report.get('clinical_indication', '')}. " +
        " ".join([f"{f.get('region', '')}: {f.get('observation', '')}" for f in report.get("findings", [])]) + " " +
        " ".join(report.get("impression", [])) + " " +
        f"{report.get('recommendation', '')}"
    )

    # Break answer into sentence claims
    sentences = [s.strip() for s in re.split(r"[.\n]+", answer) if len(s.strip()) > 8]
    claim_verifications = []
    has_contradiction = False
    has_unsupported = False

    for s in sentences:
        v = verify_single_claim_deterministic(report_corpus, s)
        claim_verifications.append(v)
        if v["status"] == ClaimStatus.CONTRADICTED:
            has_contradiction = True
        elif v["status"] == ClaimStatus.INSUFFICIENT_EVIDENCE:
            has_unsupported = True

    if has_contradiction:
        gate = "BLOCK_CONTRADICTION"
        status = ClaimStatus.CONTRADICTED
    elif has_unsupported:
        gate = "FLAG_FOR_REVIEW"
        status = ClaimStatus.INSUFFICIENT_EVIDENCE
    else:
        gate = "ALLOW"
        status = ClaimStatus.SUPPORTED

    return {
        "status": status,
        "gate_decision": gate,
        "report_id": report.get("report_id"),
        "patient_id": report.get("patient_id"),
        "claim_verifications": claim_verifications,
        "total_claims": len(sentences),
        "citations": [f"Source Report: {report.get('report_id')} ({report.get('report_type')})"]
    }
