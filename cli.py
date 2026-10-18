"""
VISCAN DENTAL ASSISTANT - INTERACTIVE COMMAND LINE INTERFACE (CLI)
-----------------------------------------------------------------
Provides a live, interactive terminal interface for:
- Feature A: Safe patient report retrieval & identity disambiguation
- Feature B: Spoken/typed clinical dictation structuring & PDF export
- Feature C: Context-dependent follow-up ("Summarize her report")
- Bonus Feature: Report-grounded Q&A ("What about tooth 48?")
- Direct launch of verified clinical PDFs on macOS
"""

import sys
import os
import json
import subprocess
from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
from backend.app.services.conversation_service import get_or_create_session

SESSION_ID = "cli_interactive_session"
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_reports")

def print_banner():
    print("\n" + "=" * 76)
    print("  VISCAN AI VOICE-FIRST DENTAL ASSISTANT — INTERACTIVE CLINICAL CLI")
    print("=" * 76)
    print("  Multi-Agent Orchestrator • MCP Protocol Active • Zero PHI Egress")
    print("  Type any command or clinical dictation. Type 'help' for examples, 'exit' to quit.")
    print("=" * 76 + "\n")

def print_help():
    print("\n--- SAMPLE COMMANDS YOU CAN TYPE IN THIS CLI ---")
    print("1. AMBIGUITY CHECK (Feature A):")
    print("   > Open Priya's latest report")
    print("   (Triggers safety stop and presents 3 matching Priyas with DOBs)")
    print("")
    print("2. EXACT REPORT RETRIEVAL (Feature A):")
    print("   > Open Priya Sharma's latest OPG report")
    print("   > Open Vikram Malhotra's CBCT report")
    print("")
    print("3. CONTEXTUAL FOLLOW-UP (Feature C Mandatory):")
    print("   > Summarize her report")
    print("   (Resolves 'her' to Priya Sharma without asking for patient name again)")
    print("")
    print("4. REPORT-AWARE CHATBOT (Bonus Feature):")
    print("   > What did the report say about tooth 48?")
    print("   > Does the report mention an earlier scan?")
    print("")
    print("5. CLINICAL DICTATION (Feature B Mandatory):")
    print("   > Lower right third molar is horizontally impacted with roots approximating the canal.")
    print("   > UR5 is root treated with apical radiolucency 8 by 7 mm. UR4 apical lesion.")
    print("   (Generates structured JSON & PDF in dictation_reports/)")
    print("")
    print("6. SPECIAL CLI COMMANDS:")
    print("   > open pdf    - Opens the currently active PDF report in Mac Preview")
    print("   > status      - Shows the active patient and report in session memory")
    print("   > reset       - Clears active conversation memory")
    print("   > exit / quit - Exits the CLI\n")

def format_agent_trace(trace: dict):
    if not trace:
        return ""
    agent = trace.get("active_agent", "Agent")
    tool = trace.get("tool_called", "tool")
    return f"\033[94m[Agent: {agent} | MCP Tool: {tool}]\033[0m"

def handle_cli_interaction(query: str):
    session = get_or_create_session(SESSION_ID)
    q_strip = query.strip()
    q_lower = q_strip.lower()

    if q_lower in ["exit", "quit", "q"]:
        print("\nExiting Viscan Dental Assistant CLI. Goodbye, Doctor.\n")
        sys.exit(0)

    if q_lower in ["help", "?"]:
        print_help()
        return

    if q_lower == "reset":
        session.clear()
        print("\n\033[93m[✓] Session context cleared. No patient report currently active.\033[0m\n")
        return

    if q_lower == "status":
        if session.active_patient_id and session.active_report_id:
            print(f"\n\033[92m[STATUS] Active Patient: {session.active_patient_id} | Active Report: {session.active_report_id}\033[0m\n")
        else:
            print("\n\033[90m[STATUS] No patient report currently selected.\033[0m\n")
        return

    if q_lower in ["open pdf", "pdf"]:
        if not session.active_report_id:
            print("\n\033[91m[!] No report is currently active. Open a report first.\033[0m\n")
            return
        from backend.app.services.retrieval import get_report_by_id
        rep = get_report_by_id(session.active_report_id)
        if rep and rep.get("pdf_filename"):
            pdf_path = os.path.join(REPORTS_DIR, rep["pdf_filename"])
            if os.path.exists(pdf_path):
                print(f"\nOpening {rep['pdf_filename']} in system PDF viewer...")
                subprocess.run(["open", pdf_path])
                return
        print("\n\033[91m[!] PDF file not found.\033[0m\n")
        return

    # Run Multi-Agent Orchestrator Pipeline
    res = run_multi_agent_pipeline(q_strip, session_id=SESSION_ID)
    trace_badge = format_agent_trace(res.get("agent_trace", {}))

    print(f"\n{trace_badge}")

    # Case 1: Clarification Required (Ambiguity Stop)
    if res.get("status") == "clarification_required":
        if res.get("candidates"):
            print(f"\033[93m[!] AMBIGUITY SAFETY STOP: {res['message']}\033[0m")
            print("Matching Patients in Registry:")
            for idx, c in enumerate(res["candidates"], 1):
                print(f"    {idx}. \033[1m{c['patient_name']}\033[0m (ID: {c['patient_id']}, DOB: {c['date_of_birth']}, Sex: {c['sex']})")
            print("\n-> Type the full name to open the intended report (e.g. 'Open Priya Sharma's report').\n")
        else:
            print(f"\033[93m[!] {res['message']}\033[0m\n")
        return

    # Case 2: Dictation Structured Result
    if "dictation_result" in res:
        d = res["dictation_result"]
        print(f"\033[92m[✓] CLINICAL DICTATION STRUCTURED SUCCESSFULLY\033[0m")
        print(f"Raw Input: \"{d.get('raw_dictation')}\"")
        
        print("\nExtracted Anatomical Sites:")
        for s in d.get("anatomical_sites", []):
            fdi = f"(FDI {s['fdi_number']})" if s.get("fdi_number") else ""
            print(f"  • {s.get('anatomical_name')} {fdi}")

        print("\nStructured Findings:")
        for f in d.get("findings", []):
            print(f"  • [{f.get('anatomical_site')}] ({f.get('category')}): {f.get('observation')}")
            print(f"    Certainty: \033[96m{f.get('certainty')}\033[0m")

        print("\nImpression / Synthesis:")
        for imp in d.get("impressions", []):
            print(f"  • {imp}")

        # Guardrail check
        gr = d.get("safety_guardrails", {})
        print(f"\nAnti-Hallucination Guardrail: \033[92m{gr.get('status')} ({gr.get('grounded_ratio')} claims grounded)\033[0m")
        print("\033[93m[!] DRAFT REPORT - MANDATORY CLINICIAN REVIEW REQUIRED PRIOR TO SIGN-OFF\033[0m\n")
        return

    # Case 3: Report Successfully Retrieved
    if res.get("status") == "success" and res.get("report"):
        rep = res["report"]
        pat = res.get("patient", {})
        pname = pat.get("patient_name", "Verified Patient")
        print(f"\033[92m[✓] REPORT RETRIEVED: {pname} • {rep['report_type']} ({rep['report_id']})\033[0m")
        print(f"Exam Date: {rep['report_date']} | Indication: {rep.get('clinical_indication')}")
        print("\nKey Findings:")
        for f in rep.get("findings", []):
            print(f"  • {f.get('region')}: {f.get('observation')} [{f.get('certainty', 'Confirmed')}]")
        print("\nImpression:")
        for imp in rep.get("impression", []):
            print(f"  • {imp}")
        print(f"\nRecommendation: {rep.get('recommendation')}")
        if rep.get("pdf_filename"):
            print(f"\033[90mPDF File on Disk: patient_reports/{rep['pdf_filename']} (Type 'open pdf' to view)\033[0m\n")
        return

    # Case 4: Contextual Follow-Up Answer & Citations
    if res.get("answer"):
        if res.get("context_resolution"):
            print(f"\033[90m[{res['context_resolution']}]\033[0m\n")
        print(f"\033[1m{res['answer']}\033[0m")
        if res.get("citations"):
            print("\n\033[94mSource Evidence Citations:\033[0m")
            for cit in res["citations"]:
                print(f"  [Evidence] {cit}")
        print("")
        return

    # Default
    print(f"{res.get('message', 'Done')}\n")

def main():
    print_banner()
    session = get_or_create_session(SESSION_ID)

    while True:
        try:
            # Dynamic prompt showing active context
            if session.active_patient_id and session.active_report_id:
                prompt_label = f"\033[92m[{session.active_patient_id}:{session.active_report_id}]\033[0m Clinician> "
            else:
                prompt_label = "\033[94m[No Report Active]\033[0m Clinician> "

            user_input = input(prompt_label).strip()
            if not user_input:
                continue

            handle_cli_interaction(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting Viscan Dental Assistant CLI. Goodbye.\n")
            break

if __name__ == "__main__":
    main()
