"""
AUTOMATED TEST SUITE FOR VISCAN DENTAL ASSISTANT
------------------------------------------------
Covers the 3 mandatory evaluation criteria from the technical assessment specification:
1. Correct Patient Report Retrieval (Feature A)
2. Ambiguous Patient Identity Handling (Feature A Safety)
3. Unsupported Clinical Claims / Anti-Hallucination Guardrail (Feature B Safety)
Plus additional coverage for Feature C (Conversational Citations) and Tooth Mapping.
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.retrieval import resolve_report_retrieval, get_all_patients
from backend.app.services.llm_service import parse_dictation_with_llm
from backend.app.services.validation import verify_anti_hallucination
from backend.app.services.conversation import handle_assistant_interaction, get_or_create_session

# =====================================================================
# TEST 1: CORRECT PATIENT REPORT RETRIEVAL (MANDATORY)
# =====================================================================
def test_correct_retrieval_unique_patient():
    """
    Verifies that a specific query for an unambiguous patient name
    correctly resolves the patient ID and retrieves their latest report.
    """
    # 1. Search for Vikram Malhotra's latest CBCT
    res = resolve_report_retrieval("Open Vikram Malhotra's latest CBCT report")
    assert res["status"] == "success", "Retrieval failed for Vikram Malhotra"
    assert res["patient"]["patient_id"] == "PAT-1009"
    assert res["patient"]["patient_name"] == "Vikram Malhotra"
    assert res["report"]["report_type"] == "CBCT"
    assert res["report"]["report_id"] == "RPT-2026"

    # 2. Search for Priya Sharma's latest OPG report (exact match among Priyas)
    res_priya = resolve_report_retrieval("Open Priya Sharma's latest OPG report")
    assert res_priya["status"] == "success", "Retrieval failed for Priya Sharma"
    assert res_priya["patient"]["patient_id"] == "PAT-1001"
    assert res_priya["report"]["report_type"] == "OPG"
    assert res_priya["report"]["report_id"] == "RPT-2001"


# =====================================================================
# TEST 2: AMBIGUOUS PATIENT IDENTITY HANDLING (MANDATORY)
# =====================================================================
def test_ambiguous_patient_identity_handling():
    """
    Verifies clinical safety: When a user query specifies an ambiguous name
    (e.g., 'Priya' matching Priya Sharma, Priya Sharmila, and Priya Nair),
    the system MUST NOT open a random report. It must safely pause and
    identify all matching candidates for disambiguation.
    """
    # Query with ambiguous name 'Priya'
    ambiguous_res = resolve_report_retrieval("Open Priya's latest OPG report")
    
    # Must return clarification_required rather than guessing
    assert ambiguous_res["status"] == "clarification_required", "Safety violation: Ambiguous query returned a single report instead of stopping for clarification!"
    assert len(ambiguous_res["candidates"]) == 3, f"Expected 3 Priya records in dataset, found {len(ambiguous_res['candidates'])}"

    names = {c["patient_name"] for c in ambiguous_res["candidates"]}
    assert "Priya Sharma" in names
    assert "Priya Sharmila" in names
    assert "Priya Nair" in names


# =====================================================================
# TEST 3: UNSUPPORTED CLINICAL CLAIMS GUARDRAIL (MANDATORY)
# =====================================================================
def test_unsupported_clinical_claims_guardrail():
    """
    Verifies that the anti-hallucination guardrail strictly flags or rejects
    any clinical claims not grounded in the dictated audio/transcript.
    """
    dictation = "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal."
    
    # 1. Authentic grounded findings should pass
    grounded_findings = [
        {
            "anatomical_site": "Tooth 48",
            "observation": "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal"
        }
    ]
    check_pass = verify_anti_hallucination(dictation, grounded_findings)
    assert check_pass["status"] == "PASSED"
    assert len(check_pass["unsupported_findings"]) == 0

    # 2. Fabricated / ungrounded findings MUST be caught and flagged
    hallucinated_findings = [
        {
            "anatomical_site": "Tooth 48",
            "observation": "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal"
        },
        {
            "anatomical_site": "Tooth 11",
            "observation": "Extensive periapical radiolucency and external inflammatory root resorption"
        },
        {
            "anatomical_site": "Mandible",
            "observation": "Displaced fracture through the left mandibular angle"
        }
    ]
    check_fail = verify_anti_hallucination(dictation, hallucinated_findings)
    assert check_fail["status"] == "FLAGGED_UNSUPPORTED", "Safety violation: Hallucinated claims were not flagged!"
    assert len(check_fail["unsupported_findings"]) == 2
    assert "Extensive periapical radiolucency" in check_fail["unsupported_findings"][0]
    assert "Displaced fracture" in check_fail["unsupported_findings"][1]


# =====================================================================
# TEST 4: ANATOMICAL TEETH EXTRACTION (FEATURE B)
# =====================================================================
def test_anatomical_teeth_mapping():
    """
    Verifies clinical extraction across natural language, Palmer, and Universal numbering.
    """
    text = "Lower right third molar is impacted. Also evaluate UR5, UR4, and tooth #29."
    parsed = parse_dictation_with_llm(text)
    
    sites = parsed.get("anatomical_sites", [])
    fdi_numbers = {s["fdi_number"] for s in sites if s.get("fdi_number")}
    assert 48 in fdi_numbers  # Lower right third molar = FDI 48
    assert 15 in fdi_numbers  # UR5 = FDI 15
    assert 14 in fdi_numbers  # UR4 = FDI 14
    assert 45 in fdi_numbers  # Tooth #29 (Universal) = FDI 45


# =====================================================================
# TEST 5: CONVERSATIONAL CITATION & CONTEXT CONTINUITY (FEATURE C)
# =====================================================================
def test_conversational_follow_up_with_citations():
    """
    Verifies Feature C: Asking 'Summarize her report' resolves the active patient context
    and returns exact citations without re-prompting for the patient name.
    """
    session_id = "test_pipeline_session"
    get_or_create_session(session_id)

    # Step 1: Open Priya Sharma's report to set active session
    r1 = handle_assistant_interaction("Open Priya Sharma's latest OPG report", session_id=session_id)
    assert r1["status"] == "success"

    # Step 2: Ask context-dependent query
    res = handle_assistant_interaction("Summarize her report", session_id=session_id)
    assert res["status"] == "success"
    assert res["patient"]["patient_name"] == "Priya Sharma"
    assert res["patient"]["patient_id"] == "PAT-1001"
    assert len(res["citations"]) > 0
    assert "Tooth 48" in res["answer"]
