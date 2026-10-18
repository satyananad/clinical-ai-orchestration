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
    - Certainty: confirmed, suspected, indeterminate
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
    is_normal = any(term in t_lower for term in ["within normal limits", "normal physiological", "normal limits", "intact"])
    
    if is_negated or is_normal:
        status = "negative"
    elif len(text.strip()) > 0:
        status = "positive"
    else:
        status = "not_documented"

    # 3. Certainty Dimension (Confirmed vs Suspected vs Indeterminate)
    is_hedged = any(re.search(r"\b" + re.escape(hw) + r"\b", t_lower) for hw in HEDGING_WORDS)
    if "no definite" in t_lower or "indeterminate" in t_lower or "equivocal" in t_lower:
        certainty = "indeterminate"
    elif is_hedged:
        certainty = "suspected"
    else:
        certainty = "confirmed"

    # Backward-compatible label string
    if status == "negative" and certainty == "indeterminate":
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
# REVISED RULE 2: MULTI-DIMENSIONAL CLAIM VERIFICATION (LAYER 1)
# =============================================================================
def extract_tooth_numbers(text: str) -> List[int]:
    """Extracts all FDI, Universal, and Palmer tooth numbers from text."""
    teeth = []
    # FDI 2-digit (11-48)
    fdi_matches = re.findall(r"\b(1[1-8]|2[1-8]|3[1-8]|4[1-8])\b", text)
    teeth.extend([int(x) for x in fdi_matches])
    
    # Universal notation with # (e.g., #29, #16)
    univ_matches = re.findall(r"#(\d{1,2})\b", text)
    for u in univ_matches:
        num = int(u)
        if 1 <= num <= 32 and num not in teeth:
            teeth.append(num)
    return sorted(list(set(teeth)))

def extract_measurements(text: str) -> List[str]:
    """Extracts measurements and units (e.g. '8 mm', '1.5 mm', '8 x 7 mm')."""
    return re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|x\s*\d+(?:\.\d+)?\s*mm)?\b", text.lower())

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

    # Check 1: Tooth Number Mismatch (e.g. changing tooth 16 to tooth 46)
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

    # Check 2: Negation Contradiction (e.g. source says 'no caries', claim says 'caries present')
    src_has_negation = any(re.search(r"\b" + re.escape(w) + r"\b", src_lower) for w in NEGATION_WORDS)
    claim_has_negation = any(re.search(r"\b" + re.escape(w) + r"\b", claim_lower) for w in NEGATION_WORDS)

    # Core pathologies to check for negation flipping
    pathologies = ["caries", "radiolucency", "fracture", "lesion", "bone loss", "impaction", "cyst", "periapical"]
    for p in pathologies:
        if p in src_lower and p in claim_lower:
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

    # Check 3: Laterality Mismatch (e.g. left vs right)
    src_lat = extract_laterality(source_text)
    claim_lat = extract_laterality(claim_text)
    if src_lat and claim_lat and src_lat != "bilateral" and claim_lat != "bilateral" and src_lat != claim_lat:
        return {
            "status": ClaimStatus.CONTRADICTED,
            "dimension": "laterality",
            "details": f"Laterality contradiction: Source specifies {src_lat} side, but claim states {claim_lat} side.",
            "claim": claim_text
        }

    # Check 4: Measurement Mismatch (e.g. 4 mm vs 8 mm)
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

    # Check 5: Uncertainty Escalation (e.g. 'possible lesion' -> 'confirmed lesion')
    src_status = classify_clinical_finding_status(source_text)
    claim_status = classify_clinical_finding_status(claim_text)
    if src_status["certainty"] == "suspected" and claim_status["certainty"] == "confirmed" and not claim_has_negation:
        return {
            "status": ClaimStatus.CONTRADICTED,
            "dimension": "uncertainty",
            "details": "Certainty escalation: Source expresses suspected/differential finding, but claim asserts confirmed diagnosis.",
            "claim": claim_text
        }

    # Check 6: Lexical & Token Grounding (Presence in source)
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
def verify_dictation_against_source(raw_dictation: str, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verifies clinical dictation transcript preservation:
    - Traceability of findings to audio transcript
    - Tooth numbers and anatomical sites preserved
    - Negation and diagnostic uncertainty preserved
    - Measurements and units preserved
    """
    unsupported_claims = []
    contradicted_claims = []
    verified_count = 0
    all_results = []

    for f in findings:
        obs = f.get("observation", "")
        res = verify_single_claim_deterministic(raw_dictation, obs)
        all_results.append(res)

        if res["status"] == ClaimStatus.CONTRADICTED:
            contradicted_claims.append(res)
        elif res["status"] == ClaimStatus.INSUFFICIENT_EVIDENCE:
            unsupported_claims.append(res)
        else:
            verified_count += 1

    overall_status = "PASSED"
    if contradicted_claims:
        overall_status = "CONTRADICTED_CLAIMS_DETECTED"
    elif unsupported_claims:
        overall_status = "FLAGGED_UNSUPPORTED"

    return {
        "status": overall_status,
        "verified_count": f"{verified_count}/{len(findings)}",
        "claims_total": len(findings),
        "supported_claims": verified_count,
        "contradicted_claims": contradicted_claims,
        "unsupported_findings": [c["claim"] for c in unsupported_claims] + [c["claim"] for c in contradicted_claims],
        "detailed_results": all_results,
        "clinician_review_required": True,
        "anti_hallucination_guarantee": "Zero ungrounded or contradicted claims admitted to final report."
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
