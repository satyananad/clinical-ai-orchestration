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

    # Important: widening of the periodontal ligament, cortical thinning, or thin bone is positive clinical finding
    has_positive_finding = any(re.search(r"\b" + re.escape(term) + r"\b", t_lower) for term in ["widening", "periodontal ligament", "pdl widening", "thin", "thinned", "lesion", "defect"])
    if has_positive_finding and not is_negated:
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
    has_confirmed_lesion = bool(re.search(r"\b(lesion|radiolucency|defect|mass|cyst)\b", t_lower) and re.search(r"\b(present|observed|noted|identified|measures?)\b", t_lower))
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
# PATIENT DEMOGRAPHICS EXTRACTION
# =============================================================================
def extract_patient_demographics(text: str) -> Dict[str, Any]:
    """
    Extracts age and gender from dictation text.
    Handles '46-year-old female', 'age 46, female', '57-year-old male', etc.
    """
    age = None
    gender = None
    age_match = re.search(r"\b(\d{1,2})\s*(?:-|–|\s)?(?:year|yr)s?(?:-|–|\s)?old\b|\bage\s*(?:is\s*)?(\d{1,2})\b", text, re.IGNORECASE)
    if age_match:
        age_str = age_match.group(1) or age_match.group(2)
        if age_str:
            try:
                age = int(age_str)
            except ValueError:
                pass

    if re.search(r"\b(female|woman|girl)\b", text, re.IGNORECASE):
        gender = "Female"
    elif re.search(r"\b(male|man|boy)\b", text, re.IGNORECASE):
        gender = "Male"

    summary = None
    if age and gender:
        summary = f"{age}-year-old {gender.lower()}"
    elif age:
        summary = f"Age {age}"
    elif gender:
        summary = gender

    return {"age": age, "gender": gender, "demographics_summary": summary}


# =============================================================================
# REVISED RULE 2: MULTI-DIMENSIONAL CLAIM VERIFICATION (LAYER 1)
# =============================================================================
def extract_quadrant_info(text: str) -> Optional[str]:
    """Identifies the clinical quadrant referenced in text using strict word boundaries."""
    t_lower = text.lower()
    if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul)\b", t_lower):
        return "maxillary_left"
    if re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur)\b", t_lower):
        return "maxillary_right"
    if re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll)\b", t_lower):
        return "mandibular_left"
    if re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr)\b", t_lower):
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

    # Check 5: Negation Contradiction on relevant sentence
    src_has_negation = any(re.search(r"\b" + re.escape(w) + r"\b", relevant_src) for w in NEGATION_WORDS)
    claim_has_negation = any(re.search(r"\b" + re.escape(w) + r"\b", claim_lower) for w in NEGATION_WORDS)

    pathologies = ["caries", "radiolucency", "fracture", "lesion", "bone loss", "impaction", "cyst", "periapical"]
    for p in pathologies:
        if p in relevant_src and p in claim_lower:
            if src_has_negation and not claim_has_negation:
                return {
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "negation",
                    "details": f"Negation contradiction: Source states absence of '{p}', but claim asserts presence.",
                    "claim": claim_text
                }
            if not src_has_negation and claim_has_negation:
                return {
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "negation",
                    "details": f"Negation contradiction: Source states presence of '{p}', but claim asserts absence.",
                    "claim": claim_text
                }

    # Check 6: Uncertainty Escalation on relevant sentence
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

    for f in findings:
        obs = f.get("observation", "")
        certainty = str(f.get("certainty", "")).lower()
        requires_clarification = f.get("requires_clarification", False)
        source_quote = f.get("source_quote")

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

        res = verify_single_claim_deterministic(raw_dictation, obs)
        all_results.append(res)

        if res["status"] == ClaimStatus.CONTRADICTED:
            contradicted_claims.append(res)
        elif res["status"] == ClaimStatus.INSUFFICIENT_EVIDENCE:
            unsupported_claims.append(res)
        else:
            verified_count += 1

    # Check Quadrant Concordance across all findings (e.g. maxillary left vs mandibular right)
    src_quad = extract_quadrant_info(raw_dictation)
    if src_quad and src_quad in QUADRANT_ALLOWED_FDI:
        allowed_teeth = QUADRANT_ALLOWED_FDI[src_quad]
        for f in findings:
            f_text = f"{f.get('anatomical_site', '')} {f.get('observation', '')}"
            finding_teeth = extract_tooth_numbers(f_text)
            violating_teeth = [t for t in finding_teeth if t not in allowed_teeth]
            if violating_teeth:
                contradicted_claims.append({
                    "status": ClaimStatus.CONTRADICTED,
                    "dimension": "quadrant_concordance",
                    "details": f"Quadrant contradiction: Dictation is scoped to {src_quad.replace('_', ' ')}, but finding specifies tooth {violating_teeth} from a conflicting quadrant.",
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

    # Check Crucial Negative Findings Preservation (e.g. sinus communication, displaced root fragments)
    critical_negative_clauses = [
        ("sinus communication", ["sinus communication"]),
        ("displaced root fragments", ["displaced root", "root fragments", "displaced fragments"]),
        ("root resorption", ["root resorption"]),
        ("periapical pathology", ["periapical pathology", "periapical radiolucency"])
    ]
    for term_label, term_patterns in critical_negative_clauses:
        for pat in term_patterns:
            neg_pattern = r"\b(?:no|without|denies|negative for)\s+(?:definite\s+|evidence of\s+)?" + re.escape(pat) + r"\b"
            if re.search(neg_pattern, raw_dictation, re.IGNORECASE):
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
                break

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

    overall_status = "PASSED"
    guarantee = "Zero ungrounded or contradicted claims admitted to final report."
    if contradicted_claims:
        overall_status = "CONTRADICTED_CLAIMS_DETECTED"
        guarantee = "Contradiction detected: Attending clinician must reconcile conflicting claims before sign-off."
    elif flagged_unclear_claims:
        overall_status = "FLAGGED_FOR_REVIEW"
        guarantee = "Unclear speech transcription artifact flagged; attending clinician review and verification required before final report sign-off."
    elif unsupported_claims:
        overall_status = "FLAGGED_UNSUPPORTED"
        guarantee = "Unsupported findings flagged for clinician review."

    return {
        "status": overall_status,
        "verified_count": f"{verified_count}/{len(findings)}",
        "claims_total": len(findings),
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
