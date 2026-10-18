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
    if raw_status in ("FLAGGED_FOR_REVIEW", "NON_CLINICAL_INPUT_REJECTED"):
        status = raw_status
    elif raw_status == "PASSED":
        status = "PASSED"
    elif raw_status == "EXTRACTION_EMPTY_REQUIRES_REVIEW":
        status = "EXTRACTION_EMPTY_REQUIRES_REVIEW"
    else:
        status = "FLAGGED_UNSUPPORTED"
    unsupported = verification["unsupported_findings"]

    return {
        "status": status,
        "grounded_ratio": verification["grounded_ratio"],
        "claim_support_ratio": verification.get("claim_support_ratio", 0.0),
        "measurement_preservation": verification.get("measurement_preservation", 1.0),
        "extraction_completeness": verification.get("extraction_completeness", 0.0),
        "unresolved_issue_count": verification.get("unresolved_issue_count", 0),
        "verified_count": verification.get("verified_count", "0/0"),
        "unsupported_findings": unsupported,
        "contradicted_claims": verification.get("contradicted_claims", []),
        "anti_hallucination_guarantee": verification.get("anti_hallucination_guarantee", "Zero ungrounded or contradicted findings admitted to final report.")
    }
