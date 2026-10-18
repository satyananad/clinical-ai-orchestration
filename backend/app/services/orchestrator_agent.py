"""
MULTI-AGENT ORCHESTRATOR & MCP TOOL DISPATCHER (orchestrator_agent.py)
-----------------------------------------------------------------------
Orchestrates specialized clinical sub-agents using Model Context Protocol (MCP)
and LLM Function/Tool Calling:
1. Triage Orchestrator Agent: Analyzes clinician intent and selects the optimal agent/tool.
2. Retrieval Agent: Executes safe patient report retrieval with identity disambiguation.
3. Clinical Scribe Agent: Ingests audio transcripts and produces structured findings with certainty.
4. Consultation Agent: Conducts grounded Q&A over the active report with source citations.
"""

import json
import re
from typing import Dict, Any, List, Optional
from .mcp_tools import DENTAL_ASSISTANT_TOOLS, execute_tool, handle_mcp_jsonrpc_request
from .conversation_service import get_or_create_session, is_contextual_query
from .retrieval import resolve_report_retrieval
from .report_qa_service import answer_report_question
from .llm_service import query_ollama

TRIAGE_SYSTEM_PROMPT = """You are the Lead Clinical Triage AI Agent for a dental voice assistant.
Your job is to analyze the clinician's speech or text input and decide which specialized sub-agent and tool to invoke.

AVAILABLE TOOLS & SPECIALIZED AGENTS:
1. "search_patient_reports": Use when the clinician wants to locate, open, or view a patient's scans or reports (e.g. "Open Priya's report", "Show Vikram Malhotra's CBCT").
   Arguments: {"patient_name": "string", "modality": "OPG"|"CBCT"|"Bitewing"|"Periapical"|"Cephalogram"|"Any"}

2. "structure_clinical_dictation": Use when the clinician is dictating physical examination findings, dental observations, pathology, or tooth conditions (e.g. "Lower right third molar is impacted", "UR5 root treated with apical radiolucency").
   Arguments: {"dictation_text": "verbatim text"}

3. "summarize_active_report": Use when the clinician asks a follow-up question or summary about the currently open report (e.g. "Summarize her report", "What did the radiologist say about tooth 48?", "Are there any caries?").
   Arguments: {"report_id": "active", "query": "string"}

4. "general_query": Use when the clinician asks general questions about the system (e.g. "how many reports have", "what scans exist", "how many patients", "who are you", "help", greetings).
   Arguments: {"query": "string"}

Output strictly a JSON object with this format:
{
  "selected_agent": "RetrievalAgent" | "ClinicalScribeAgent" | "ConsultationAgent" | "GeneralAssistantAgent",
  "tool_name": "search_patient_reports" | "structure_clinical_dictation" | "summarize_active_report" | "general_query",
  "arguments": { ... },
  "rationale": "Brief reason for agent selection"
}
"""

def triage_clinician_request(query: str, session_id: str = "default_session") -> Dict[str, Any]:
    """
    Decides which specialized agent and MCP tool to invoke.
    Combines LLM intent classification with deterministic routing rules.
    """
    session = get_or_create_session(session_id)
    q_lower = query.lower().strip()

    # Step 1: LLM Dynamic Triage (Primary AI Agent Decision)
    llm_triage_raw = query_ollama(
        f"Clinician Input: \"{query}\"\nActive Session Report: {session.active_report_id or 'None'}",
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        json_format=True
    )
    if llm_triage_raw:
        try:
            parsed = json.loads(llm_triage_raw)
            if "tool_name" in parsed and parsed["tool_name"] in [
                "search_patient_reports", 
                "structure_clinical_dictation", 
                "summarize_active_report",
                "general_query"
            ]:
                # Ensure arguments dictionary exists
                if "arguments" not in parsed or not isinstance(parsed["arguments"], dict):
                    parsed["arguments"] = {}
                return parsed
        except Exception:
            pass

    # Step 2: Fallback deterministic routing if LLM service is unavailable
    # Clinical Dictation Phrases -> Scribe Agent
    clinical_keywords = [
        "impacted", "radiolucency", "root treated", "carious", "caries",
        "periapical", "canal", "osteitis", "follicle", "supernumerary",
        "alveolar ridge", "condyle", "tonsillolith", "bone loss"
    ]
    if any(kw in q_lower for kw in clinical_keywords) and not any(verb in q_lower for verb in ["open", "show", "find", "get", "retrieve", "search"]):
        return {
            "selected_agent": "ClinicalScribeAgent",
            "tool_name": "structure_clinical_dictation",
            "arguments": {
                "dictation_text": query,
                "patient_id": session.active_patient_id or "PAT-1001"
            },
            "rationale": "Fallback: Input describes clinical pathology findings."
        }

    # Contextual Follow-up on Active Report -> Consultation Agent
    if is_contextual_query(q_lower):
        return {
            "selected_agent": "ConsultationAgent",
            "tool_name": "summarize_active_report",
            "arguments": {
                "report_id": session.active_report_id or "unselected",
                "query": query
            },
            "rationale": "Detected contextual pronoun targeting active patient report."
        }

    # General queries (e.g. how many reports, what reports, help, stats)
    general_pattern = r"\b(how many|how much|what report|which report|list reports?|database|who are you|what can you|help|hello|\bhi\b|hey)\b"
    if re.search(general_pattern, q_lower) and not any(verb in q_lower for verb in ["open", "retrieve"]):
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "general_query",
            "arguments": {"query": query},
            "rationale": "General clinic inquiry regarding system records or capabilities."
        }

    # Default Deterministic Fallback: Retrieval Agent
    # Extract patient name candidate
    clean_query = re.sub(r"\b(open|show|find|get|latest|report|scan|opg|cbct|bitewing|periapical|please)\b", "", q_lower).strip()
    name_guess = clean_query.title() if clean_query else query.title()
    return {
        "selected_agent": "RetrievalAgent",
        "tool_name": "search_patient_reports",
        "arguments": {
            "patient_name": name_guess,
            "modality": "Any"
        },
        "rationale": "Defaulting to patient and report retrieval search."
    }

def run_multi_agent_pipeline(query: str, session_id: str = "default_session") -> Dict[str, Any]:
    """
    Executes the complete Multi-Agent Workflow:
    1. Triage Orchestrator Agent assigns task and builds MCP JSON-RPC payload.
    2. Invokes specialized sub-agent via standard MCP Tool Execution.
    3. Enforces deterministic clinical safety gates.
    4. Updates session state and returns full agentic execution trace.
    """
    session = get_or_create_session(session_id)
    triage = triage_clinician_request(query, session_id=session_id)

    selected_agent = triage.get("selected_agent", "RetrievalAgent")
    tool_name = triage.get("tool_name", "search_patient_reports")
    arguments = triage.get("arguments", {})
    rationale = triage.get("rationale", "Automated clinical dispatch.")

    # Build Standard MCP JSON-RPC 2.0 Request Payload
    mcp_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    # =========================================================================
    # AGENT 1: RETRIEVAL AGENT
    # =========================================================================
    if selected_agent == "RetrievalAgent":
        retrieval_res = resolve_report_retrieval(query)
        
        # Only update active context when retrieval is unambiguous and successful
        if retrieval_res.get("status") == "success" and retrieval_res.get("report"):
            session.set_active(
                retrieval_res["patient"]["patient_id"],
                retrieval_res["report"]["report_id"]
            )

        return {
            **retrieval_res,
            "agent_trace": {
                "orchestrator": "TriageAgent",
                "active_agent": "RetrievalAgent",
                "tool_called": tool_name,
                "arguments": arguments,
                "rationale": rationale,
                "mcp_jsonrpc": mcp_request
            }
        }

    # =========================================================================
    # AGENT 2: CLINICAL SCRIBE AGENT
    # =========================================================================
    elif selected_agent == "ClinicalScribeAgent":
        dictation_res = execute_tool("structure_clinical_dictation", arguments)
        return {
            "status": "success",
            "message": "Clinical observations structured into validated draft report.",
            "dictation_result": dictation_res,
            "agent_trace": {
                "orchestrator": "TriageAgent",
                "active_agent": "ClinicalScribeAgent",
                "tool_called": tool_name,
                "arguments": arguments,
                "rationale": rationale,
                "mcp_jsonrpc": mcp_request
            }
        }

    # =========================================================================
    # AGENT 3: GENERAL ASSISTANT AGENT (System Stats & Natural Conversation)
    # =========================================================================
    elif selected_agent == "GeneralAssistantAgent" or tool_name == "general_query":
        from .retrieval import get_all_patients
        patients = get_all_patients()
        total_patients = len(patients)
        total_reports = sum(len(p.get("reports", [])) for p in patients)
        patient_names = ", ".join(p.get("patient_name") for p in patients)

        system_context = f"""You are the ViScan Dental AI Assistant.
The clinic diagnostic database currently contains:
- Total Registered Patients: {total_patients} patients ({patient_names})
- Total Radiological Reports: {total_reports} diagnostic imaging scans (covering OPG panoramic, CBCT 3D volume, Bitewing, Periapical, and Cephalograms)

Answer the clinician's question helpfully, professionally, and concisely using these facts. If they ask how many reports exist, state that there are {total_reports} reports across {total_patients} patients."""

        llm_reply = query_ollama(f"Clinician Question: {query}", system_prompt=system_context, json_format=False)
        answer_text = llm_reply.strip() if llm_reply else f"There are currently {total_reports} radiological reports across {total_patients} registered patients in the clinic database (including OPG, CBCT, and Bitewing imaging). You can ask me to open any patient's report (e.g. 'Open Vikram Malhotra's CBCT')."

        return {
            "status": "success",
            "message": "Answered by General Assistant Agent.",
            "answer": answer_text,
            "citations": [f"ViScan Clinic Database: {total_reports} total scans across {total_patients} registered patients"],
            "agent_trace": {
                "orchestrator": "TriageAgent",
                "active_agent": "GeneralAssistantAgent",
                "tool_called": tool_name,
                "arguments": arguments,
                "rationale": rationale,
                "mcp_jsonrpc": mcp_request
            }
        }

    # =========================================================================
    # AGENT 4: CONSULTATION & REPORT-AWARE AGENT
    # =========================================================================
    else:
        # Check active report guardrail
        if not session.active_patient_id or not session.active_report_id:
            return {
                "status": "clarification_required",
                "message": "No active patient report is currently selected. Please select a patient from the directory above or ask me to open a specific report (e.g. 'Open Vikram Malhotra's CBCT').",
                "candidates": None,
                "report": None,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ConsultationAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        # Answer question grounded in active report
        from .retrieval import get_patient, get_report_by_id
        patient = get_patient(session.active_patient_id)
        report = get_report_by_id(session.active_report_id)
        qa_result = answer_report_question(session.active_report_id, query)
        
        return {
            "status": "success",
            "message": "Answered from active report context.",
            "answer": qa_result.get("answer"),
            "citations": qa_result.get("citations", []),
            "context_resolution": f"Active Context -> Patient ID: {session.active_patient_id}, Report ID: {session.active_report_id}",
            "patient": {
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "date_of_birth": patient["date_of_birth"],
                "sex": patient["sex"]
            } if patient else None,
            "report": report,
            "agent_trace": {
                "orchestrator": "TriageAgent",
                "active_agent": "ConsultationAgent",
                "tool_called": tool_name,
                "arguments": arguments,
                "rationale": rationale,
                "mcp_jsonrpc": mcp_request
            }
        }
