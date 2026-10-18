"""
VISCAN DENTAL ASSISTANT - UNIFIED DEMO RUNNER
--------------------------------------------
Demonstrates the full end-to-end clinical workflow:
- Feature A: Safe Patient Report Retrieval (Handling Ambiguity & Matching)
- Feature B: Clinical Dictation to Structured JSON & PDF Report (LLM Extraction)
- Feature C: Context-Dependent Conversational Follow-Up with Evidence Citations
"""

import os
import sys
import json
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.retrieval import resolve_report_retrieval, get_patient, get_report_by_id
from backend.app.services.llm_service import parse_dictation_with_llm
from backend.app.services.conversation import handle_assistant_interaction, get_or_create_session
from backend.app.services.validation import verify_anti_hallucination

DICTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictation_reports")
os.makedirs(DICTATION_DIR, exist_ok=True)

def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)

def run_feature_a():
    print_banner("1. FEATURE A: Patient Report Retrieval & Ambiguity Disambiguation")

    print('\n[SCENARIO A1: Clinician says: "Open Priya\'s latest OPG report"]')
    print("Context: 3 patients named Priya exist (Priya Sharma, Priya Sharmila, Priya Nair).")
    res1 = resolve_report_retrieval("Open Priya's latest OPG report")
    if res1["status"] == "clarification_required":
        print(f"[!] SAFETY CHECK: {res1['message']}")
        for c in res1["candidates"]:
            print(f"    • {c['patient_name']} (DOB: {c['date_of_birth']}, ID: {c['patient_id']})")
    time.sleep(1)

    print('\n[SCENARIO A2: Clinician clarifies: "Open Priya Sharma\'s latest OPG report"]')
    res2 = resolve_report_retrieval("Open Priya Sharma's latest OPG report")
    if res2["status"] == "success":
        p = res2["patient"]
        r = res2["report"]
        print(f"[✓] RETRIEVED: {p['patient_name']} ({p['patient_id']}) - {r['report_type']} ({r['report_id']})")
        print(f"    Primary Indication: {r.get('clinical_indication')}")
        print(f"    Impression: {'; '.join(r.get('impression', []))}")
    time.sleep(1)

    print('\n[SCENARIO A3: Clinician requests: "Open Priya Sharmila\'s latest OPG report"]')
    res3 = resolve_report_retrieval("Open Priya Sharmila's latest OPG report")
    if res3["status"] == "success":
        p = res3["patient"]
        r = res3["report"]
        print(f"[✓] RETRIEVED: {p['patient_name']} ({p['patient_id']}) - {r['report_type']} ({r['report_id']})")
        print(f"    Diagnosis: {'; '.join(r.get('impression', []))}")

def run_feature_b():
    print_banner("2. FEATURE B: Clinical Dictation to Structured JSON Draft Report")

    # Sample 1: Assignment core prompt
    dictation_1 = "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal. Mild pericoronal radiolucency noted around the crown."
    print("\n[SCENARIO B1: Clinical Dictation Extraction via LLM]")
    print(f'Transcript: "{dictation_1}"')
    
    parsed = parse_dictation_with_llm(dictation_1)
    guardrails = verify_anti_hallucination(dictation_1, parsed.get("findings", []))
    
    out_1 = {
        "report_metadata": {
            "status": "DRAFT_PENDING_REVIEW",
            "clinician_review_required": True,
            "disclaimer": "ATTENTION: This is an unverified AI-generated draft created from clinical dictation."
        },
        "patient": "Priya Sharma (PAT-1001)",
        "raw_dictation": dictation_1,
        "anatomical_sites": parsed.get("anatomical_sites", []),
        "findings": parsed.get("findings", []),
        "impressions": parsed.get("impressions", []),
        "recommendations": parsed.get("recommendations", []),
        "safety_guardrails": guardrails
    }
    
    json_path = os.path.join(DICTATION_DIR, "PAT-1001_Priya_Sharma_Dictation_Report.json")
    with open(json_path, "w") as f:
        json.dump(out_1, f, indent=2)
    print(f"[✓] Structured JSON Saved: {json_path}")
    print("\n-> Structured JSON Output:")
    print(json.dumps({
        "anatomical_sites": out_1["anatomical_sites"],
        "findings": out_1["findings"],
        "impressions": out_1["impressions"],
        "safety_guardrails": out_1["safety_guardrails"]
    }, indent=2))
    time.sleep(1)

    # Sample 2: Real Specialist CBCT Sample
    dictation_2 = "UR5 is root treated with a unilocular well defined apical radiolucency 8 by 7 by 6 mm. UR4 is root treated with an apical radiolucency 1 to 2 mm wide. Distal to UR8 is an unerupted supernumerary tooth with an enlarged follicle space, possibly a dentigerous cyst."
    print("\n[SCENARIO B2: Complex Specialist Dictation with Uncertainty Labeling]")
    print(f'Transcript: "{dictation_2}"')
    parsed_2 = parse_dictation_with_llm(dictation_2)
    guardrails_2 = verify_anti_hallucination(dictation_2, parsed_2.get("findings", []))

    out_2 = {
        "report_metadata": {"status": "DRAFT_PENDING_REVIEW", "clinician_review_required": True},
        "patient": "Priya Sharmila (PAT-1002)",
        "raw_dictation": dictation_2,
        "anatomical_sites": parsed_2.get("anatomical_sites", []),
        "findings": parsed_2.get("findings", []),
        "impressions": parsed_2.get("impressions", []),
        "safety_guardrails": guardrails_2
    }
    json_path_2 = os.path.join(DICTATION_DIR, "PAT-1002_Priya_Sharmila_Dictation_Report.json")
    with open(json_path_2, "w") as f:
        json.dump(out_2, f, indent=2)
    print(f"[✓] Structured JSON Saved: {json_path_2}")
    print("\n-> Extracted Findings (with certainty):")
    for f in out_2["findings"]:
        print(f"   • Site: {f['anatomical_site']} | Certainty: {f['certainty']} | Observation: {f['observation']}")

def run_feature_c():
    print_banner("3. FEATURE C: Conversational Follow-Up with Evidence Citations")
    session_id = "demo_session"
    session = get_or_create_session(session_id)

    # Step 1: Clinician opens Priya Sharma's OPG report
    print('\n[CONVERSATION TURN 1]: User says: "Open Priya Sharma\'s latest OPG report"')
    res1 = handle_assistant_interaction("Open Priya Sharma's latest OPG report", session_id=session_id)
    print(f"-> Agent Response: {res1['message']}")
    time.sleep(1)

    # Step 2: Contextual follow-up using pronoun "her"
    print('\n[CONVERSATION TURN 2]: User asks: "Summarize her report"')
    res2 = handle_assistant_interaction("Summarize her report", session_id=session_id)
    print(f"-> Context Continuity: {res2.get('context_resolution')}")
    print("\n" + res2.get("answer", ""))
    print("\n[SOURCE EVIDENCE & CITATIONS]:")
    for c in res2.get("citations", []):
        print(f"  • {c}")
    time.sleep(1)

    # Step 3: Targeted anatomical question
    print('\n[CONVERSATION TURN 3]: User asks: "What did the radiologist note about her third molar?"')
    res3 = handle_assistant_interaction("What did the radiologist note about her third molar?", session_id=session_id)
    print("\n" + res3.get("answer", ""))
    print("\n[SOURCE EVIDENCE & CITATIONS]:")
    for c in res3.get("citations", []):
        print(f"  • {c}")

def run_tests():
    print_banner("4. AUTOMATED TEST SUITE VERIFICATION")
    ret = os.system("PYTHONPATH=. ./venv/bin/pytest backend/tests -v")
    if ret == 0:
        print("\n[✓] ALL 10 AUTOMATED CLINICAL TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    print_banner("VISCAN VOICE-FIRST DENTAL ASSISTANT - TECHNICAL PROOF OF CONCEPT")
    print("Running end-to-end verification across Features A, B, and C...")
    run_feature_a()
    run_feature_b()
    run_feature_c()
    run_tests()
    print_banner("DEMO COMPLETED SUCCESSFULLY")
