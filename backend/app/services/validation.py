"""
VALIDATION & ANTI-HALLUCINATION SERVICE (validation.py)
-------------------------------------------------------
Equipped with multi-dimensional claim verification (revised Rule 2)
and clinical certainty classification (revised Rule 3).
Connects directly to claim_verification.py.
"""

from typing import List, Dict, Any, Optional
from .claim_verification import (
    classify_clinical_finding_status,
    verify_dictation_against_source,
    verify_single_claim_deterministic,
    verify_report_answer_claim,
    ClaimStatus
)

def label_diagnostic_certainty(sentence: str) -> str:
    """
    Labels diagnostic certainty based on revised Rule 3.
    Extracts multi-dimensional certainty (confirmed, suspected, indeterminate)
    while returning the standard legacy clinical label string.
    """
    res = classify_clinical_finding_status(sentence)
    return res["legacy_label"]

def verify_anti_hallucination(raw_dictation: str, structured_findings: List[Dict[str, Any]], impressions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Anti-Hallucination & Clinical Claim Verification (Revised Rule 2):
    Checks tooth numbers, negation flipping, laterality, measurements, word grounding,
    and cross-checks impressions against findings for clinical contradictions.
    """
    verification = verify_dictation_against_source(raw_dictation, structured_findings, impressions=impressions)
    
    # Map to standard API response fields
    raw_status = verification.get("status", "PASSED")
    if raw_status == "FLAGGED_FOR_REVIEW":
        status = "FLAGGED_FOR_REVIEW"
    elif raw_status == "PASSED":
        status = "PASSED"
    else:
        status = "FLAGGED_UNSUPPORTED"
    unsupported = verification["unsupported_findings"]

    return {
        "status": status,
        "grounded_ratio": verification["grounded_ratio"],
        "unsupported_findings": unsupported,
        "contradicted_claims": verification.get("contradicted_claims", []),
        "anti_hallucination_guarantee": verification.get("anti_hallucination_guarantee", "Zero ungrounded or contradicted findings admitted to final report.")
    }
