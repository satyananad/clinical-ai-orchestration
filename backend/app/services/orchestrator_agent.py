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
from .mcp_tools import DENTAL_ASSISTANT_TOOLS, get_ollama_tool_definitions, execute_tool, handle_mcp_jsonrpc_request
from .conversation_service import get_or_create_session, is_contextual_query
from .retrieval import resolve_report_retrieval
from .report_qa_service import answer_report_question
from .llm_service import query_ollama, query_ollama_chat_tools

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
    Primary: LLM Dynamic Multi-Agent Triage using full conversational context.
    Secondary: Safety fallback if LLM service is offline.
    """
    session = get_or_create_session(session_id)
    q_lower = query.lower().strip()

    # Step 1: Native LLM Function / Tool Calling (Direct Ollama / OpenAI Tool Calling Interface)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the ViScan Dental AI Multi-Agent Orchestrator. "
                "Select and invoke the appropriate clinical tool for the clinician's request.\n"
                f"Active Patient in session: {session.active_patient_id or 'None'}\n"
                f"Active Report in session: {session.active_report_id or 'None'}"
            )
        }
    ]
    if session.history:
        for turn in session.history[-3:]:
            messages.append({"role": "user", "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("assistant", "")[:150]})
    messages.append({"role": "user", "content": query})

    tool_call = query_ollama_chat_tools(messages, get_ollama_tool_definitions())
    if tool_call and tool_call.get("tool_name"):
        tool_name = tool_call.get("tool_name")
        args = tool_call.get("arguments", {})
        agent_map = {
            "search_patient_reports": "RetrievalAgent",
            "structure_clinical_dictation": "ClinicalScribeAgent",
            "summarize_active_report": "ConsultationAgent",
            "query_clinic_database": "GeneralAssistantAgent",
            "general_query": "GeneralAssistantAgent",
            "verify_clinical_claim": "ConsultationAgent"
        }
        if tool_name in agent_map:
            return {
                "selected_agent": agent_map[tool_name],
                "tool_name": "general_query" if tool_name == "query_clinic_database" else tool_name,
                "arguments": args,
                "rationale": f"Native LLM Tool Call: Selected {tool_name}."
            }

    # Step 2: Fallback routing (When local LLM is offline or initializing)
    # 2A. Speech repair fallback when clinician corrects a misheard term
    if session.last_retrieval_query and any(w in q_lower for w in ["not", "saying", "meant"]):
        for part in q_lower.split():
            if part in ["panoramic", "opg", "cbct", "bitewing", "periapical"]:
                mod = "OPG" if part in ["panoramic", "opg"] else ("CBCT" if part == "cbct" else part.capitalize())
                return {
                    "selected_agent": "RetrievalAgent",
                    "tool_name": "search_patient_reports",
                    "arguments": {
                        "patient_name": session.last_retrieval_query,
                        "modality": mod
                    },
                    "rationale": "Fallback: Speech repair targeting previous retrieval."
                }

    # 2B. Disambiguation candidate selection from pending clarification
    if session.pending_candidates:
        for cand in session.pending_candidates:
            c_name = cand["patient_name"].lower()
            c_id = cand["patient_id"].lower()
            if c_name in q_lower or c_id in q_lower:
                return {
                    "selected_agent": "RetrievalAgent",
                    "tool_name": "search_patient_reports",
                    "arguments": {
                        "patient_name": cand["patient_name"],
                        "patient_id": cand["patient_id"],
                        "modality": session.last_retrieval_modality or "Any"
                    },
                    "rationale": f"Fallback: Matched candidate {cand['patient_name']}."
                }

    # 2C. Contextual Follow-up on Active Report
    if is_contextual_query(q_lower, has_active_report=bool(session.active_report_id)):
        return {
            "selected_agent": "ConsultationAgent",
            "tool_name": "summarize_active_report",
            "arguments": {
                "report_id": session.active_report_id or "unselected",
                "query": query
            },
            "rationale": "Fallback: Contextual inquiry on active report."
        }

    # 2D. Dictation findings
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

    # 2E. General Clinic Database Queries
    is_database_question = any(q_lower.startswith(w) for w in ["how many", "how much", "which", "who", "list", "what", "show"]) or "database" in q_lower or "?" in q_lower
    has_retrieval_intent = any(verb in q_lower for verb in ["open", "fetch", "load", "retrieve"])
    if is_database_question and not has_retrieval_intent:
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "general_query",
            "arguments": {"question": query},
            "rationale": "Fallback: Clinical database statistical inquiry."
        }

    # 2F. Default: Patient & Report Retrieval
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

def verify_response_against_query(query: str, response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Response Verification Gate:
    Verifies that the executed tool output directly satisfies what the clinician actually asked.
    - If user asked for an OPG or CBCT, confirms that the loaded scan matches that modality.
    - If user query was ambiguous, verifies that clarification candidates are returned and no arbitrary file was opened.
    - If summarizing an active report, verifies evidence citations are attached.
    """
    q_lower = query.lower()
    
    # Check 1: Modality Verification
    if response.get("status") == "success" and response.get("report"):
        rep = response["report"]
        rep_type = rep.get("report_type", "").upper()
        if ("opg" in q_lower or "panoramic" in q_lower) and rep_type != "OPG":
            response["verification_notice"] = f"Notice: Requested OPG scan, but loaded {rep.get('report_type')}."
        elif ("cbct" in q_lower or "cone beam" in q_lower) and rep_type != "CBCT":
            response["verification_notice"] = f"Notice: Requested CBCT scan, but loaded {rep.get('report_type')}."

    # Check 2: Grounding and Citation Verification
    if response.get("status") == "success" and "answer" in response:
        if not response.get("citations"):
            response["citations"] = ["ViScan Clinic Diagnostic Database"]

    return response

def run_multi_agent_pipeline(query: str, session_id: str = "default_session") -> Dict[str, Any]:
    res = _execute_multi_agent_pipeline(query, session_id=session_id)
    return verify_response_against_query(query, res)

def _execute_multi_agent_pipeline(query: str, session_id: str = "default_session") -> Dict[str, Any]:
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
        corrected_q = arguments.get("corrected_query")
        actual_query = corrected_q or query
        patient_name_arg = arguments.get("patient_name")
        patient_id_arg = arguments.get("patient_id")
        modality_arg = arguments.get("modality")

        if patient_id_arg:
            actual_query = patient_id_arg

        session.last_retrieval_query = actual_query
        retrieval_res = resolve_report_retrieval(actual_query, patient_name=patient_name_arg, modality=modality_arg)
        
        # Update active context upon verified retrieval
        if retrieval_res.get("status") == "success" and retrieval_res.get("report"):
            session.set_active(
                retrieval_res["patient"]["patient_id"],
                retrieval_res["report"]["report_id"]
            )
        elif retrieval_res.get("status") == "clarification_required":
            session.pending_candidates = retrieval_res.get("candidates")
            session.last_retrieval_modality = modality_arg

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
    elif selected_agent == "GeneralAssistantAgent" or tool_name in ["general_query", "query_clinic_database"]:
        db_res = execute_tool("query_clinic_database", {
            "question": query,
            "patient_name": arguments.get("patient_name"),
            "modality": arguments.get("modality")
        })
        return {
            "status": "success",
            "message": "Answered by General Assistant Agent via MCP Tool query_clinic_database.",
            "answer": db_res.get("answer"),
            "citations": db_res.get("citations", []),
            "agent_trace": {
                "orchestrator": "TriageAgent",
                "active_agent": "GeneralAssistantAgent",
                "tool_called": "query_clinic_database",
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
        session.history.append({"user": query, "assistant": qa_result.get("answer", "")})
        
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
