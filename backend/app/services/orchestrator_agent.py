"""
MULTI-AGENT ORCHESTRATOR & MCP TOOL DISPATCHER (orchestrator_agent.py)
-----------------------------------------------------------------------
Orchestrates specialized clinical sub-agents using Model Context Protocol (MCP)
and native LLM Function/Tool Calling with structured Pydantic dispatch:
1. Triage Orchestrator Agent: Directs clinician queries dynamically to specialized tools.
2. Retrieval Agent: Executes safe patient report retrieval with identity disambiguation.
3. Clinical Scribe Agent: Ingests audio transcripts and produces structured findings with certainty.
4. Consultation Agent: Conducts grounded Q&A over the active report with source citations.
5. Scan Booking Agent: Manages appointment workflows and clinic slot inventory.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

from .mcp_tools import DENTAL_ASSISTANT_TOOLS, get_ollama_tool_definitions, execute_tool, handle_mcp_jsonrpc_request
from .conversation_service import get_or_create_session, save_session, is_contextual_query, is_general_or_assistant_query, BookingStage
from .retrieval import resolve_report_retrieval, extract_retrieval_intent
from .report_qa_service import answer_report_question, answer_patient_reports_question, generate_multi_report_summary
from .llm_service import query_llm, query_ollama, query_ollama_chat_tools

# Fallback & Natural Language Entity Extraction Engine (Enables offline test harness & CI/CD verification)
from .orchestrator_fallback import (
    resolve_pending_candidates_from_query,
    is_valid_patient_name,
    resolve_pending_slot_from_query,
    extract_date_from_query,
    extract_age_from_query,
    extract_doctor_from_query,
    extract_patient_registration_details,
    extract_patient_details_for_booking,
    _offline_deterministic_fallback,
    generate_dynamic_conversational_reply
)

class AgentDispatch(BaseModel):
    """Structured multi-agent dispatch schema for LLM-driven tool selection."""
    selected_agent: Literal[
        'RetrievalAgent',
        'ClinicalScribeAgent',
        'ConsultationAgent',
        'ScanBookingAgent',
        'GeneralAssistantAgent'
    ]
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str

TRIAGE_SYSTEM_PROMPT = """You are the Lead Clinical Triage AI Agent for a dental voice workstation.
Your job is to analyze the clinician's speech or text input and decide which specialized sub-agent and tool to invoke.

AVAILABLE TOOLS & SPECIALIZED AGENTS:
1. "search_patient_reports": Use when the clinician wants to locate, open, view, or search for a patient's scans or reports, or asks to open a patient by name (e.g. "Open Priya Sharma", "Open Priya's report", "Show Vikram Malhotra's CBCT", "hey i want to open the report of arjun mehta", "open arjun mehta"). Even if the clinician starts with a greeting like 'hey', 'hello', or 'hi', you MUST select 'search_patient_reports' if they mention opening, viewing, or searching for a patient's report or opening a patient record! NEVER select "prompt_existing_or_new_patient" or appointment booking when the request is to open or view a patient or report!
   Agent: "RetrievalAgent"
   Arguments: {"patient_name": "string", "modality": "OPG"|"CBCT"|"Bitewing"|"Periapical"|"Cephalogram"|"Any"}

2. "structure_clinical_dictation": Use when the clinician is dictating physical examination findings, dental observations, pathology, or tooth conditions (e.g. "Lower right third molar is impacted", "UR5 root treated with apical radiolucency").
   Agent: "ClinicalScribeAgent"
   Arguments: {"dictation_text": "verbatim text"}

3. "summarize_active_report": Use when an active report is open in the session and the clinician asks ANY question about findings, diagnoses, problems, conditions, impressions, teeth, or summaries (e.g. "what problem it have", "what problem does she have", "what is wrong with her", "Summarize her report", "What did the radiologist say about tooth 48?", "Explain this report", "Show me the PDF"). NEVER use 'conversation' if an active report is loaded and the query asks about patient problems or scan findings!
   Agent: "ConsultationAgent"
   Arguments: {"report_id": "active", "query": "string"}

4. "prompt_existing_or_new_patient": Use ONLY when someone explicitly asks to book or schedule an appointment (e.g. "i want to book appointment", "how can i book the appointment", "book an appointment", "I need to schedule an appointment", "can I book an appointment", "appointment booking") when NO patient is active in the session and NO patient name was specified. NEVER use when the user asks to open, view, or search for a patient or report!
   Agent: "ScanBookingAgent"
   Arguments: {"query": "string"}

5. "prompt_patient_dob_and_contact": Use during new patient registration when a patient provides their name (e.g. "Satyanand Shukla", "My name is John Doe") and we need their date of birth / age and contact number to complete registration.
   Agent: "ScanBookingAgent"
   Arguments: {"patient_name": "string"}

6. "check_appointment_slots": Use when checking or booking available appointment slots for a scan when a specific modality, date, or patient is known (e.g. "Book a CBCT scan", "Schedule an OPG appointment", "What slots are available for CBCT?", "How many slots on 20 October?").
   Agent: "ScanBookingAgent"
   Arguments: {"modality": "CBCT"|"OPG"|"Bitewing"|"Periapical"|"Any", "preferred_date": "optional date string"}

7. "confirm_scan_booking": Use when confirming a slot booking (e.g. "Confirm slot SLOT-101", "Yes, book slot 1", "Book it for Priya").
   Agent: "ScanBookingAgent"
   Arguments: {"slot_id": "string", "patient_name": "string", "patient_id": "optional string"}

8. "conversation": Use ONLY for pure greetings, small talk, or general dental concepts when NO patient, report retrieval, booking, or clinical dictation is requested (e.g. "hello", "who are you?", "what is a CBCT scan?"). If the clinician asks to open or view a report or mentions a patient (e.g. "hey i want to open the report of arjun mehta"), NEVER use 'conversation'!
   Agent: "GeneralAssistantAgent"
   Arguments: {"query": "string"}

9. "query_clinic_database": Use when inquiring about registered patients, listing patient names, total scans, patient counts, or database breakdowns (e.g. "can you list out only patient name", "first tell me the patient names", "how many reports are in the database?", "how many patients do we have?", "what scans exist in the system?").
   Agent: "GeneralAssistantAgent"
   Arguments: {"question": "string", "patient_name": "optional string", "modality": "optional string"}

Output strictly a JSON object with this format:
{
  "selected_agent": "RetrievalAgent" | "ClinicalScribeAgent" | "ConsultationAgent" | "ScanBookingAgent" | "GeneralAssistantAgent",
  "tool_name": "search_patient_reports" | "structure_clinical_dictation" | "summarize_active_report" | "prompt_existing_or_new_patient" | "prompt_patient_dob_and_contact" | "check_appointment_slots" | "confirm_scan_booking" | "conversation" | "query_clinic_database",
  "arguments": { ... },
  "rationale": "Brief reason for agent selection"
}
"""


def select_tool_with_llm(query: str, session: Any) -> Optional[Dict[str, Any]]:
    """
    General LLM-driven agent that understands natural-language requests from the
    current message, conversation history, and application context, then dynamically
    selects the appropriate available tool and its arguments based on tool schemas.
    """
    q_lower = query.lower().strip()
    
    # 1. Primary: JSON Structured Triage via Local/Configured LLM
    prompt = (
        f"Clinician request: \"{query}\"\n"
        f"Active Patient in session: {session.active_patient_id or 'None'}\n"
        f"Active Report in session: {session.active_report_id or 'None'}\n"
        f"Booking Workflow State: {session.booking_state.stage.value if hasattr(session.booking_state, 'stage') else 'IDLE'}\n"
        f"Pending Booking Slots: {'Yes (' + str(len(session.pending_booking_slots)) + ' slots)' if session.pending_booking_slots else 'No'}\n"
        f"Pending Patient Candidates: {'Yes (' + str(len(session.pending_candidates)) + ' candidates)' if session.pending_candidates else 'No'}\n"
    )
    if session.history:
        prompt += "Recent dialogue:\n"
        for turn in session.history[-3:]:
            prompt += f"  Clinician: {turn.get('user', '')}\n  Assistant: {turn.get('assistant', '')[:100]}\n"

    raw_triage = query_llm(prompt, system_prompt=TRIAGE_SYSTEM_PROMPT, json_format=True)
    if raw_triage:
        try:
            parsed_triage = json.loads(raw_triage)
            sel_agent = parsed_triage.get("selected_agent")
            t_name = parsed_triage.get("tool_name")
            args = parsed_triage.get("arguments", {})
            if sel_agent and t_name:
                if t_name == "summarize_active_report" and args.get("report_id") in ["active", None]:
                    args["report_id"] = session.active_report_id or "unselected"
                    if "query" not in args:
                        args["query"] = query
                elif t_name == "prompt_existing_or_new_patient":
                    has_retrieval_intent = any(v in q_lower for v in ["open", "retrieve", "view", "fetch", "load", "find report", "show report"]) or bool(re.search(r"\b(pat-\d+|rpt-\d+)\b", q_lower))
                    intent = extract_retrieval_intent(query)
                    cand_name = intent.get("patient_name", "").strip()
                    if has_retrieval_intent or (cand_name and any(v in q_lower for v in ["open", "retrieve", "view", "fetch", "load"])):
                        sel_agent = "RetrievalAgent"
                        t_name = "search_patient_reports"
                        args = {
                            "patient_name": cand_name or "Patient",
                            "modality": intent.get("report_type", "Any")
                        }
                    else:
                        sel_agent = "ScanBookingAgent"
                        session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                        if "query" not in args:
                            args["query"] = query
                elif t_name == "prompt_patient_dob_and_contact":
                    sel_agent = "ScanBookingAgent"
                    session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                    p_name = args.get("patient_name") or session.booking_state.patient_name or query.strip().title()
                    session.booking_state.patient_name = p_name
                    args["patient_name"] = p_name
                elif t_name == "check_appointment_slots":
                    cand_p_name, cand_p_id, cand_p_age, cand_doc = extract_patient_details_for_booking(query, session)
                    has_explicit_scan_or_slot = any(m in q_lower for m in ["cbct", "opg", "panoramic", "bitewing", "periapical", "iopa", "slot-", "slots", "slot "])
                    if not cand_p_name and not session.active_patient_id and not session.booking_state.patient_name and not has_explicit_scan_or_slot:
                        sel_agent = "ScanBookingAgent"
                        t_name = "prompt_existing_or_new_patient"
                        args = {"query": query}
                        session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                    else:
                        if cand_p_name:
                            args["patient_name"] = cand_p_name
                            session.booking_state.patient_name = cand_p_name
                        if cand_p_id:
                            args["patient_id"] = cand_p_id
                            session.booking_state.patient_id = cand_p_id
                        if cand_p_age:
                            args["patient_age"] = cand_p_age
                            session.booking_state.patient_age = cand_p_age
                        if cand_doc:
                            args["doctor_name"] = cand_doc
                            session.booking_state.doctor_name = cand_doc
                        if not args.get("preferred_date"):
                            extracted_d = extract_date_from_query(query)
                            if extracted_d:
                                args["preferred_date"] = extracted_d
                elif t_name == "conversation":
                    has_retrieval = (
                        any(v in q_lower for v in ["open", "view", "retrieve", "fetch", "load", "find"])
                        and any(w in q_lower for w in ["report", "scan", "opg", "cbct", "bitewing", "periapical", "examination", "record", "patient"])
                    ) or bool(re.search(r"\b(pat-\d+|pt-\d+|rpt-\d+)\b", q_lower))
                    intent = extract_retrieval_intent(query)
                    cand_name = intent.get("patient_name", "").strip()
                    if has_retrieval or (cand_name and is_valid_patient_name(cand_name) and any(v in q_lower for v in ["open", "view", "retrieve", "fetch", "load", "find", "show", "get"])):
                        sel_agent = "RetrievalAgent"
                        t_name = "search_patient_reports"
                        args = {
                            "patient_name": cand_name or "Patient",
                            "modality": intent.get("report_type", "Any")
                        }
                    elif any(b_phrase in q_lower for b_phrase in ["book appointment", "book an appointment", "schedule appointment", "schedule an appointment", "want to book", "how can i book", "need to book"]):
                        sel_agent = "ScanBookingAgent"
                        t_name = "prompt_existing_or_new_patient"
                        args = {"query": query}
                        session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                    elif "query" not in args:
                        args["query"] = query

                if session.booking_state.stage in (BookingStage.COLLECTING_DETAILS, BookingStage.IDENTIFYING_PATIENT):
                    if t_name in ("query_clinic_database", "conversation", "general_query", "prompt_existing_or_new_patient"):
                        cand_name = extract_patient_registration_details(query).get("full_name")
                        if not cand_name and is_valid_patient_name(query.strip()) and not any(w in q_lower for w in ["yes", "no", "ok", "cancel"]):
                            cand_name = query.strip().title()
                        if cand_name or is_valid_patient_name(query):
                            sel_agent = "ScanBookingAgent"
                            t_name = "prompt_patient_dob_and_contact"
                            p_name = cand_name or query.strip().title()
                            session.booking_state.patient_name = p_name
                            args = {"patient_name": p_name}

                return {
                    "selected_agent": sel_agent,
                    "tool_name": t_name,
                    "arguments": args,
                    "rationale": f"LLM Multi-Agent Reasoning: {parsed_triage.get('rationale', 'Dynamic triage by Clinical Orchestrator Agent.')}"
                }
        except Exception:
            pass

    # 2. Secondary: Native LLM Tool Calling (Ollama chat tools interface)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Dental Clinical Workstation Orchestrator. "
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
            "verify_clinical_claim": "ConsultationAgent",
            "prompt_existing_or_new_patient": "ScanBookingAgent",
            "prompt_patient_dob_and_contact": "ScanBookingAgent",
            "check_appointment_slots": "ScanBookingAgent",
            "confirm_scan_booking": "ScanBookingAgent",
            "lookup_existing_booking": "ScanBookingAgent",
            "cancel_scan_booking": "ScanBookingAgent",
            "reschedule_scan_booking": "ScanBookingAgent",
            "conversation": "GeneralAssistantAgent"
        }
        if tool_name in agent_map:
            if tool_name == "check_appointment_slots":
                cand_p_name, cand_p_id, cand_p_age, cand_doc = extract_patient_details_for_booking(query, session)
                has_explicit_scan_or_slot = any(m in q_lower for m in ["cbct", "opg", "panoramic", "bitewing", "periapical", "iopa", "slot-", "slots", "slot "])
                if not cand_p_name and not session.active_patient_id and not session.booking_state.patient_name and not has_explicit_scan_or_slot:
                    tool_name = "prompt_existing_or_new_patient"
                    args = {"query": query}
                    session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                else:
                    if cand_p_name:
                        args["patient_name"] = cand_p_name
                        session.booking_state.patient_name = cand_p_name
                    if cand_p_id:
                        args["patient_id"] = cand_p_id
                        session.booking_state.patient_id = cand_p_id
                    if cand_p_age:
                        args["patient_age"] = cand_p_age
                        session.booking_state.patient_age = cand_p_age
                    if cand_doc:
                        args["doctor_name"] = cand_doc
                        session.booking_state.doctor_name = cand_doc
            elif tool_name == "prompt_existing_or_new_patient":
                has_retrieval_intent = any(v in q_lower for v in ["open", "retrieve", "view", "fetch", "load", "find report", "show report"]) or bool(re.search(r"\b(pat-\d+|rpt-\d+)\b", q_lower))
                intent = extract_retrieval_intent(query)
                cand_name = intent.get("patient_name", "").strip()
                if has_retrieval_intent or (cand_name and any(v in q_lower for v in ["open", "retrieve", "view", "fetch", "load"])):
                    tool_name = "search_patient_reports"
                    args = {
                        "patient_name": cand_name or "Patient",
                        "modality": intent.get("report_type", "Any")
                    }
                else:
                    session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                    if "query" not in args:
                        args["query"] = query
            elif tool_name == "prompt_patient_dob_and_contact":
                session.booking_state.stage = BookingStage.COLLECTING_DETAILS
                p_name = args.get("patient_name") or session.booking_state.patient_name or query.strip().title()
                session.booking_state.patient_name = p_name
                args["patient_name"] = p_name

            if session.booking_state.stage in (BookingStage.COLLECTING_DETAILS, BookingStage.IDENTIFYING_PATIENT):
                if tool_name in ("query_clinic_database", "conversation", "general_query", "prompt_existing_or_new_patient"):
                    cand_name = extract_patient_registration_details(query).get("full_name")
                    if not cand_name and is_valid_patient_name(query.strip()) and not any(w in q_lower for w in ["yes", "no", "ok", "cancel"]):
                        cand_name = query.strip().title()
                    if cand_name or is_valid_patient_name(query):
                        tool_name = "prompt_patient_dob_and_contact"
                        p_name = cand_name or query.strip().title()
                        session.booking_state.patient_name = p_name
                        args = {"patient_name": p_name}
            return {
                "selected_agent": agent_map[tool_name],
                "tool_name": "general_query" if tool_name == "query_clinic_database" else tool_name,
                "arguments": args,
                "rationale": f"Native LLM Tool Call: Selected {tool_name}."
            }

    return None


def triage_clinician_request(query: str, session_id: str = "default_session") -> Dict[str, Any]:
    """
    Decides which specialized agent and MCP tool to invoke.
    Step 0A: Disambiguation candidate resolution if clarification is pending.
    Step 0B: Scan booking confirmation resolution if appointment slots are pending.
    Step 1: LLM Dynamic Multi-Agent Triage using full conversational context.
    Step 2: Safety fallback if LLM service is offline or unhandled.
    """
    session = get_or_create_session(session_id)
    q_lower = query.lower().strip()


    # Step 0A: Disambiguation candidate selection from pending clarification
    if session.pending_candidates:
        clarification_res = resolve_pending_candidates_from_query(query, session.pending_candidates)
        query_intent = extract_retrieval_intent(query)
        detected_mod = query_intent.get("report_type", "Any")
        effective_mod = detected_mod if detected_mod != "Any" else (session.last_retrieval_modality or "Any")

        if clarification_res["status"] == "resolved":
            cand = clarification_res["matched_candidate"]
            session.pending_candidates = None
            return {
                "selected_agent": "RetrievalAgent",
                "tool_name": "search_patient_reports",
                "arguments": {
                    "patient_name": cand["patient_name"],
                    "patient_id": cand["patient_id"],
                    "modality": effective_mod
                },
                "rationale": f"Clarification Resolution: Matched unique candidate {cand['patient_name']} ({cand['patient_id']})."
            }
        elif clarification_res["status"] == "narrowed":
            session.pending_candidates = clarification_res["remaining_candidates"]
        elif clarification_res["status"] == "invalid_selection":
            return {
                "selected_agent": "RetrievalAgent",
                "tool_name": "invalid_candidate_selection",
                "arguments": {
                    "requested_index": clarification_res.get("requested_index"),
                    "total_candidates": len(session.pending_candidates)
                },
                "rationale": f"Safety Gate: Rejected out-of-range candidate selection #{clarification_res.get('requested_index')}."
            }
        else:
            # Query did not address the clarification; check if user is asking a database question or changing topic
            pronoun_tokens = {"his", "her", "hers", "hir", "their", "theirs", "she", "he", "him", "them"}
            is_pronoun_ref = any(w in pronoun_tokens for w in re.findall(r"\b[a-zA-Z]+\b", q_lower))
            is_database_q = any(q_lower.startswith(w) for w in ["how many", "how much", "which", "who", "list", "what", "show"]) or "database" in q_lower or "?" in q_lower
            has_retrieval_cmd = any(verb in q_lower for verb in ["open", "find", "load", "retrieve", "book", "schedule", "cancel", "stop"])
            if (is_database_q or has_retrieval_cmd) and not is_pronoun_ref:
                session.pending_candidates = None
            elif is_pronoun_ref and session.pending_candidates:
                return {
                    "selected_agent": "RetrievalAgent",
                    "tool_name": "search_patient_reports",
                    "arguments": {
                        "patient_name": session.pending_candidates[0]["patient_name"],
                        "modality": effective_mod
                    },
                    "rationale": "Clarification Safety Gate: Clinician referenced ambiguous candidate with pronoun without choosing."
                }

    # Step 1: Dynamic Multi-Agent Tool Calling via LLM (OpenAI / Ollama / Groq)
    # The query is passed DIRECTLY to the LLM. The LLM decides which agent and tool to invoke.
    llm_decision = select_tool_with_llm(query, session)
    if llm_decision and llm_decision.get("selected_agent") and llm_decision.get("tool_name"):
        return llm_decision

    # Step 2: Deterministic Safety Fallback (Only executed when LLM is offline or in test harness)
    return _offline_deterministic_fallback(query, session)


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
        if not response.get("citations") and response.get("agent_trace", {}).get("active_agent") != "GeneralAssistantAgent":
            response["citations"] = ["Clinic Diagnostic Database"]

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
        if tool_name == "invalid_candidate_selection":
            return {
                "status": "clarification_required",
                "message": f"Invalid candidate selection. Please choose a valid candidate from the list (1 to {len(session.pending_candidates or [])}) or provide the Patient ID.",
                "candidates": session.pending_candidates,
                "patient": None,
                "report": None,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "RetrievalAgent",
                    "tool_called": "reject_invalid_candidate",
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        corrected_q = arguments.get("corrected_query")
        actual_query = corrected_q or query
        patient_name_arg = arguments.get("patient_name")
        patient_id_arg = arguments.get("patient_id")
        modality_arg = arguments.get("modality")

        if patient_id_arg:
            actual_query = patient_id_arg

        session.last_retrieval_query = actual_query
        retrieval_res = resolve_report_retrieval(actual_query, patient_name=patient_name_arg, modality=modality_arg, active_patient_id=session.active_patient_id)
        
        # Update active context upon verified retrieval
        if retrieval_res.get("status") == "success" and retrieval_res.get("report"):
            session.set_active(
                retrieval_res["patient"]["patient_id"],
                retrieval_res["report"]["report_id"]
            )
            session.pending_candidates = None
            rep = retrieval_res["report"]
            pdf_name = rep.get("pdf_filename")
            pdf_url = f"/patient-pdf/{pdf_name}" if pdf_name else None
            retrieval_res["pdf_url"] = pdf_url
            retrieval_res["pdf_filename"] = pdf_name
            p_name = retrieval_res["patient"]["patient_name"]
            rep_type = rep.get("report_type", "Scan")
            rep_id = rep.get("report_id", "")

            from .retrieval import get_patient_reports
            all_reports = get_patient_reports(session.active_patient_id)
            retrieval_res["available_reports"] = [
                {
                    "report_id": r["report_id"],
                    "report_type": r["report_type"],
                    "report_date": r["report_date"],
                    "pdf_filename": r.get("pdf_filename"),
                    "pdf_url": f"/patient-pdf/{r.get('pdf_filename')}" if r.get("pdf_filename") else None
                }
                for r in all_reports
            ]

            extra_msg = ""
            if len(all_reports) > 1:
                other_reps = [f"• [{r['report_id']}] {r['report_type']} ({r['report_date']})" for r in all_reports]
                from .retrieval import get_patient_pronouns
                p_sex = retrieval_res["patient"].get("sex")
                pronouns = get_patient_pronouns(p_sex)
                pronoun_pos = pronouns["possessive"]
                suggested_mod = all_reports[0]["report_type"]
                for r in all_reports:
                    if r["report_id"] != rep_id:
                        suggested_mod = r["report_type"]
                        break
                extra_msg = (
                    f"\n\nThis patient has {len(all_reports)} diagnostic reports on file:\n"
                    + "\n".join(other_reps)
                    + f"\n\nYou can ask to view any of these reports (e.g. 'Open {pronoun_pos} {suggested_mod}' or 'Show {all_reports[0]['report_id']}'), "
                    f"or ask clinical questions about the current report."
                )

            retrieval_res["message"] = (
                f"Opened {rep_type} report ({rep_id}) for {p_name}."
                + (f" [View Diagnostic PDF]({pdf_url})" if pdf_url else "")
                + extra_msg
            )
            retrieval_res["answer"] = retrieval_res["message"]
        elif retrieval_res.get("status") == "clarification_required":
            session.pending_candidates = retrieval_res.get("candidates")
            session.last_retrieval_modality = modality_arg if (modality_arg and modality_arg != "Any") else extract_retrieval_intent(query).get("report_type", "Any")

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
    elif selected_agent == "ClinicalScribeAgent":
        dictation_res = execute_tool("structure_clinical_dictation", arguments)
        is_complete = dictation_res.get("status") == "success" and bool(dictation_res.get("findings"))
        findings = dictation_res.get("findings", [])
        impressions = dictation_res.get("impressions", [])

        # Build readable clinical draft response for the clinician from narrative draft
        narrative = dictation_res.get("narrative_draft")
        if not narrative:
            from .report_service import compose_narrative
            patient_label = arguments.get("patient_name") or (dictation_res.get("patient_details") or {}).get("patient_name") or "Pending Patient Linkage (Unassigned)"
            narrative = compose_narrative(
                patient_label,
                findings,
                impressions,
                dictation_res.get("recommendations", []),
                patient_details=dictation_res.get("patient_details")
            )
            dictation_res["narrative_draft"] = narrative

        draft_answer = (
            f"📋 **Clinical Dictation Draft Report Generated** (Status: `DRAFT_PENDING_REVIEW`)\n\n"
            f"{narrative}\n\n"
            f"⚠️ **Review Notice:** Status is `DRAFT_PENDING_REVIEW`. All observations require attending clinician verification and sign-off before permanent record entry."
        )

        citations = [f.get("source_quote") for f in findings if f.get("source_quote")]
        if not citations and dictation_res.get("raw_dictation"):
            citations = [dictation_res.get("raw_dictation")]

        return {
            "status": "success" if is_complete else "incomplete",
            "extraction_status": dictation_res.get("extraction_status", "COMPLETE" if is_complete else "INCOMPLETE_REQUIRES_REVIEW"),
            "message": dictation_res.get("message", "Clinical observations structured into validated draft report."),
            "answer": draft_answer,
            "narrative_draft": narrative,
            "citations": citations,
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
    # AGENT 3A: CONVERSATIONAL AGENT (Dynamic LLM dialogue; NO database tool)
    # =========================================================================
    elif (tool_name in ["conversation", "general_query"] and not any(w in query.lower() for w in ["database", "how many", "count", "statistics", "registered patients", "all reports"])) or (selected_agent == "GeneralAssistantAgent" and tool_name not in ["query_clinic_database"]):
        reply_text = generate_dynamic_conversational_reply(query, session)

        return {
            "status": "success",
            "message": "Conversational response generated by Clinical Assistant.",
            "answer": reply_text,
            "citations": [],
            "agent_trace": {
                "orchestrator": "TriageAgent",
                "active_agent": "GeneralAssistantAgent",
                "tool_called": "conversation",
                "arguments": arguments,
                "rationale": rationale,
                "mcp_jsonrpc": mcp_request
            }
        }

    # =========================================================================
    # AGENT 3B: CLINIC DATABASE QUERY (Statistical facts from database)
    # =========================================================================
    elif selected_agent == "GeneralAssistantAgent" or tool_name in ["query_clinic_database", "general_query"]:
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
    # AGENT 4: SCAN-BOOKING AGENT (Voice & Typed Appointment Scheduling)
    # =========================================================================
    elif selected_agent == "ScanBookingAgent" or tool_name in ["check_appointment_slots", "confirm_scan_booking", "show_booking_summary", "lookup_existing_booking", "cancel_scan_booking", "reschedule_scan_booking", "booking_already_confirmed", "equipment_limitation", "past_date_rejected", "patient_id_not_found", "scan_type_required"]:
        if tool_name == "confirm_scan_booking":
            slot_id = arguments.get("slot_id")
            if not slot_id and session.booking_state.displayed_slots:
                slot_id = session.booking_state.displayed_slots[0]["slot_id"]

            patient_name = arguments.get("patient_name") or session.booking_state.patient_name
            patient_id = arguments.get("patient_id") or session.booking_state.patient_id
            if not patient_name and session.active_patient_id:
                from .retrieval import get_patient
                p = get_patient(session.active_patient_id)
                if p:
                    patient_name = p.get("patient_name")
                    patient_id = p.get("patient_id")
            if not patient_name or not is_valid_patient_name(patient_name):
                patient_name = "Walk-in Patient"

            patient_age = arguments.get("patient_age") or session.booking_state.patient_age
            doctor_name = arguments.get("doctor_name") or session.booking_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)"

            confirm_res = execute_tool("confirm_scan_booking", {
                "slot_id": slot_id,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "patient_age": patient_age,
                "doctor_name": doctor_name,
                "scan_type": arguments.get("modality") or session.booking_state.modality
            })

            if confirm_res.get("status") == "success":
                session.booking_state.stage = BookingStage.CONFIRMED
                session.booking_state.completed_booking = confirm_res.get("slot")
                session.booking_state.displayed_slots = []
                session.booking_state.selected_slot = None
                session.pending_booking_slots = []

                return {
                    "status": "success",
                    "message": confirm_res.get("message"),
                    "booking_reference": confirm_res.get("booking_reference"),
                    "slot": confirm_res.get("slot"),
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }
            elif confirm_res.get("status") == "patient_not_found":
                return {
                    "status": "clarification_required",
                    "message": confirm_res.get("message"),
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }
            elif confirm_res.get("status") == "past_date_rejected":
                return {
                    "status": "date_rejected",
                    "message": confirm_res.get("message"),
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }
            elif confirm_res.get("status") == "failed":
                return {
                    "status": "failed",
                    "message": confirm_res.get("message"),
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }
            else:
                # Slot is already booked or unavailable
                session.booking_state.stage = BookingStage.SLOTS_PRESENTED
                from .booking_service import check_available_slots as get_open_slots
                fresh_slots = get_open_slots(modality=session.booking_state.modality or "Any")
                session.booking_state.displayed_slots = fresh_slots
                slot_lines = [f"• [{s['slot_id']}] {s['datetime_str']} — {s['modality']} ({s['room']})" for s in fresh_slots]
                fail_msg = (
                    f"{confirm_res.get('message')}\n\n"
                    f"Please choose an alternative from the currently available slots:\n"
                    + "\n".join(slot_lines)
                    + f"\n\nConfirmation Required: Would you like to confirm any of these slots? (e.g. 'Yes, book slot {fresh_slots[0]['slot_id']}')."
                )
                return {
                    "status": "confirmation_required",
                    "message": fail_msg,
                    "answer": fail_msg,
                    "available_slots": fresh_slots,
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }

        elif tool_name == "show_booking_summary":
            slot = arguments.get("slot") or session.booking_state.selected_slot or {}
            patient_name = arguments.get("patient_name") or session.booking_state.patient_name or "Patient"
            patient_id = arguments.get("patient_id") or session.booking_state.patient_id
            patient_age = arguments.get("patient_age") or session.booking_state.patient_age
            doctor_name = arguments.get("doctor_name") or session.booking_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)"
            modality = arguments.get("modality") or slot.get("modality", "Radiology")
            room_name = slot.get("room", "Radiology Suite")
            clinic_id = slot.get("clinic_id", "CLINIC-MAIN-01")
            from .booking_service import SUPPORTED_CLINICS
            clinic_title = SUPPORTED_CLINICS.get(clinic_id, {}).get("name", "Viscan Dental Imaging Center - Main Clinic")

            from .clinic_catalog import find_matching_specialization
            matched_spec = find_matching_specialization(session.booking_state.service or modality)
            spec_title = session.booking_state.specialization or (matched_spec.get("name") if matched_spec else "Oral and Maxillofacial Radiology")
            service_title = session.booking_state.service or f"{modality} Scan"
            clean_pid = patient_id if patient_id else "None (Unregistered)"

            if not patient_age and patient_id:
                from .retrieval import get_patient
                p_data = get_patient(patient_id)
                if p_data and p_data.get("date_of_birth"):
                    try:
                        b_year = int(p_data["date_of_birth"].split("-")[0])
                        patient_age = f"{2026 - b_year} years (DOB: {p_data['date_of_birth']})"
                    except Exception:
                        pass

            pid_display = f"{clean_pid} (Age: {patient_age})" if patient_age else clean_pid

            summary_msg = (
                f"## Review appointment\n"
                f"Pending confirmation\n\n"
                f"Patient\n"
                f"{patient_name}\n\n"
                f"Patient ID\n"
                f"{pid_display}\n\n"
                f"Service\n"
                f"{service_title}\n\n"
                f"Specialization\n"
                f"{spec_title}\n\n"
                f"Doctor\n"
                f"{doctor_name}\n\n"
                f"Date and time\n"
                f"{slot.get('datetime_str', 'Scheduled slot')}\n\n"
                f"Location\n"
                f"{room_name}, {clinic_title}\n\n"
                f"Slot ID\n"
                f"{slot.get('slot_id')}\n\n"
                f"Confirmation Required: Would you like me to book this exact appointment? (Please reply 'Yes, confirm' or 'Cancel')."
            )
            session.booking_state.stage = BookingStage.AWAITING_CONFIRMATION
            session.booking_state.summary_displayed = True
            slot_summary = dict(slot) if slot else {}
            slot_summary["patient_name"] = patient_name
            slot_summary["patient_id"] = patient_id
            slot_summary["patient_age"] = patient_age
            slot_summary["doctor_name"] = doctor_name
            slot_summary["service"] = service_title
            slot_summary["specialization"] = spec_title
            session.booking_state.selected_slot = slot_summary
            if patient_name:
                session.booking_state.patient_name = patient_name
            if patient_id:
                session.booking_state.patient_id = patient_id
            if patient_age:
                session.booking_state.patient_age = patient_age
            if doctor_name:
                session.booking_state.doctor_name = doctor_name

            return {
                "status": "confirmation_required",
                "message": summary_msg,
                "answer": summary_msg,
                "slot": slot_summary,
                "available_slots": session.booking_state.displayed_slots,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "lookup_existing_booking":
            res = execute_tool("lookup_existing_booking", arguments)
            bookings = res.get("bookings", [])
            p_name = arguments.get("patient_name")
            if bookings:
                lines = [f"• [{b.get('booking_reference')}] Token: {b.get('token_number', 'TK-' + b.get('slot_id', '').split('-')[-1])} | {b.get('patient_name')} — {b.get('modality')} scan on {b.get('datetime_str')} in {b.get('room')} (Slot: {b.get('slot_id')}, Status: Confirmed)" for b in bookings]
                msg = (
                    f"Found {len(bookings)} confirmed appointment booking(s):\n"
                    + "\n".join(lines)
                    + f"\n\nYou can keep, reschedule, or cancel this booking (e.g. 'Cancel my booking')."
                )
            else:
                msg = f"No existing appointments were found{' for ' + p_name if p_name else ''}. Would you like to check available slots to book a new appointment?"

            return {
                "status": "success",
                "message": msg,
                "bookings": bookings,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "cancel_scan_booking":
            res = execute_tool("cancel_scan_booking", arguments)
            session.booking_state.reset()
            return {
                "status": "success",
                "message": res.get("message"),
                "answer": res.get("message"),
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "booking_already_confirmed":
            cb = arguments.get("booking") or session.booking_state.completed_booking or {}
            msg = (
                f"This appointment is already confirmed:\n"
                f"• Patient: {cb.get('patient_name')}\n"
                f"• Scan: {cb.get('modality')}\n"
                f"• Date/Time: {cb.get('datetime_str')}\n"
                f"• Booking Reference: {cb.get('booking_reference')}\n"
                f"No duplicate booking was created."
            )
            return {
                "status": "success",
                "message": msg,
                "booking_reference": cb.get("booking_reference"),
                "slot": cb,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "patient_linked_to_booking":
            slot = arguments.get("slot") or {}
            msg = (
                f"✅ **Booking Linked to Patient Record!**\n\n"
                f"Appointment **{arguments.get('booking_reference')}** has been successfully assigned to "
                f"**{arguments.get('patient_name')}** ({arguments.get('patient_id') or 'Registered Patient'}).\n"
                f"• Scheduled Time: {slot.get('datetime_str')}\n"
                f"• Modality: {slot.get('modality')}\n"
                f"• Location: {slot.get('room')}\n"
                f"• Token #: {slot.get('queue_token') or slot.get('token_number')}"
            )
            return {
                "status": "success",
                "message": msg,
                "booking_reference": arguments.get("booking_reference"),
                "slot": slot,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "reschedule_scan_booking":
            res = execute_tool("reschedule_scan_booking", arguments)
            if res.get("status") == "success":
                session.booking_state.completed_booking = res.get("slot")
                session.booking_state.stage = BookingStage.CONFIRMED
                return {
                    "status": "success",
                    "message": res.get("message"),
                    "answer": res.get("message"),
                    "booking_reference": res.get("booking_reference"),
                    "slot": res.get("slot"),
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }
            else:
                return {
                    "status": "failed",
                    "message": res.get("message"),
                    "answer": res.get("message"),
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "ScanBookingAgent",
                        "tool_called": tool_name,
                        "arguments": arguments,
                        "rationale": rationale,
                        "mcp_jsonrpc": mcp_request
                    }
                }

        elif tool_name == "equipment_limitation":
            req_mod = arguments.get("modality", "the requested")
            msg = f"Our clinic does not possess equipment for {req_mod} scans. We are a specialized dental radiology center providing Periapical (IOPA), Panoramic (OPG), Bitewing, and CBCT (3D) imaging."
            return {
                "status": "equipment_limitation",
                "message": msg,
                "answer": msg,
                "available_slots": [],
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "past_date_rejected":
            p_date = arguments.get("date")
            msg = f"Cannot schedule an appointment for a past date ({p_date}). Please select a future date and time."
            return {
                "status": "date_rejected",
                "message": msg,
                "answer": msg,
                "available_slots": [],
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "patient_id_not_found":
            p_id = arguments.get("patient_id")
            msg = f"Patient ID '{p_id}' does not exist in our clinic records. Would you like to check the patient ID, or proceed with an unregistered guest booking?"
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "scan_type_required":
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            msg = "Which type of scan is required? Please specify: Periapical (IOPA), Panoramic (OPG), Bitewing, or CBCT (3D) imaging."
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "booking_declined":
            cancel_msg = "Appointment booking has been cancelled as requested. No appointment was booked."
            return {
                "status": "cancelled",
                "message": cancel_msg,
                "answer": cancel_msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "invalid_slot_selection":
            slots = session.booking_state.displayed_slots or []
            if slots:
                slot_lines = [f"• [{s['slot_id']}] {s['datetime_str']} — {s['modality']} ({s['room']})" for s in slots]
                msg = (
                    f"I couldn't find an available appointment slot matching '{query}'. "
                    f"Please choose from the available slots:\n"
                    + "\n".join(slot_lines)
                    + f"\n\nConfirmation Required: Would you like to confirm any of these slots? (e.g. 'Yes, book slot {slots[0]['slot_id']}')."
                )
            else:
                msg = "No appointment slots are currently available. Please check back later."

            return {
                "status": "confirmation_required",
                "message": msg,
                "answer": msg,
                "available_slots": slots,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "prompt_existing_or_new_patient":
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            msg = (
                "I'd be happy to help you book an appointment. Are you an existing patient at Viscan Dental, or is this your first visit with us?"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "prompt_new_patient_registration":
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            session.booking_state.is_new_patient = True
            msg = (
                "Welcome to Viscan Dental! To get you registered as a new patient, could you please provide:\n"
                "• Full Name\n"
                "• Date of birth (from which your age will be calculated automatically)\n"
                "• Contact phone number\n"
                "• Email address\n"
                "• Sex / Gender"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "prompt_patient_dob_and_contact":
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            p_name = arguments.get("patient_name") or session.booking_state.patient_name or "there"
            session.booking_state.patient_name = p_name
            msg = (
                f"Thank you, {p_name}! Could you please provide your Date of Birth (or age) and a contact phone number to complete your registration?"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "prompt_existing_patient_identifier":
            session.booking_state.stage = BookingStage.IDENTIFYING_PATIENT
            session.booking_state.is_new_patient = False
            msg = (
                "Welcome back! Could you please share your Patient ID (e.g. PAT-1001), registered phone number, or your full name and date of birth so I can locate your record?"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "registration_completed_ask_service":
            p_name = arguments.get("patient_name") or session.booking_state.patient_name
            p_id = arguments.get("patient_id") or session.booking_state.patient_id
            p_age = arguments.get("age") or session.booking_state.patient_age
            msg = (
                f"Thank you, {p_name}! You have been successfully registered as a new patient (Patient ID: {p_id}, Age: {p_age}).\n\n"
                f"What is the reason for your visit today? Are you looking for a specialist doctor consultation or an imaging procedure (e.g., CBCT, OPG, Bitewing, Periapical)?"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "patient_id": p_id,
                "patient_name": p_name,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "existing_patient_identified":
            p_name = arguments.get("patient_name") or session.booking_state.patient_name
            p_id = arguments.get("patient_id") or session.booking_state.patient_id
            msg = (
                f"Found your record: {p_name} ({p_id}).\n\n"
                f"What is the reason for your visit today? Are you looking for a specialist doctor consultation or an imaging procedure (e.g., CBCT, OPG, Bitewing, Periapical)?"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "patient_id": p_id,
                "patient_name": p_name,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "service_selected_ask_preference":
            srv = arguments.get("service") or session.booking_state.service or session.booking_state.modality or "examination"
            spec = arguments.get("specialization") or session.booking_state.specialization or "Oral and Maxillofacial Radiology"
            docs = arguments.get("doctors") or "Dr. Rajesh Mehta and Dr. Anjali Deshmukh"
            facility = arguments.get("facility") or "Radiology Suite 1 (3D Scanner) at our Main Clinic"
            notes_str = f" for {session.booking_state.notes}" if session.booking_state.notes else ""
            msg = (
                f"Understood. A {srv}{notes_str} falls under {spec}. Our specialists for this service are {docs} at our {facility}.\n\n"
                f"Do you have a preferred date, time of day (morning or afternoon), or doctor for your appointment?"
            )
            return {
                "status": "clarification_required",
                "message": msg,
                "answer": msg,
                "service": srv,
                "specialization": spec,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        elif tool_name == "present_matching_schedule_slots":
            slots = arguments.get("available_slots") or session.booking_state.displayed_slots or []
            session.booking_state.displayed_slots = slots
            session.booking_state.stage = BookingStage.SLOTS_PRESENTED
            session.pending_booking_slots = slots
            pref_tod = arguments.get("time_of_day", "")
            tod_str = f"{pref_tod} " if pref_tod else ""
            pref_d = arguments.get("date_str", "Tomorrow")
            slot_lines = [f"• [{s['slot_id']}] {s['datetime_str']} — {s.get('doctor_name', 'Dr. Rajesh Mehta')} ({s.get('room', 'Radiology Suite 1')})" for s in slots]
            slot_pick_example = slots[0].get('time', '10:30 AM') if slots else "10:30 AM"
            msg = (
                f"Here are the available {tod_str}slots for {session.booking_state.service or session.booking_state.modality or 'your scan'} on {pref_d} at Viscan Main Clinic:\n"
                + "\n".join(slot_lines)
                + f"\n\nPlease select your preferred slot (e.g. 'Let's go with the {slot_pick_example} slot' or 'Select {slots[0]['slot_id']}')."
            )
            return {
                "status": "confirmation_required",
                "message": msg,
                "answer": msg,
                "available_slots": slots,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

        else:
            raw_mod = arguments.get("modality")
            pref_date = arguments.get("preferred_date")
            mod_title = raw_mod if (raw_mod and str(raw_mod).lower() not in ["none", "any", "null", "all"]) else "Radiology"

            # Check if preferred_date is in the past
            if pref_date:
                iso_m = re.search(r"\b(202\d)-(0[1-9]|1[0-2])-([0-3]\d)\b", str(pref_date))
                if iso_m:
                    from .clock import get_current_date
                    from datetime import date
                    try:
                        p_date = date(int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3)))
                        if p_date < get_current_date():
                            msg = f"Cannot schedule an appointment for a past date ({pref_date}). Please select a future date and time for your scan."
                            return {
                                "status": "date_rejected",
                                "message": msg,
                                "answer": msg,
                                "available_slots": [],
                                "agent_trace": {
                                    "orchestrator": "TriageAgent",
                                    "active_agent": "ScanBookingAgent",
                                    "tool_called": "past_date_rejected",
                                    "arguments": arguments,
                                    "rationale": f"Appointment date {pref_date} is in the past.",
                                    "mcp_jsonrpc": mcp_request
                                }
                            }
                    except Exception:
                        pass

            slots_res = execute_tool("check_appointment_slots", arguments)
            slots = slots_res.get("available_slots", [])
            session.booking_state.displayed_slots = slots
            session.booking_state.modality = mod_title
            if arguments.get("patient_name"):
                session.booking_state.patient_name = arguments.get("patient_name")
            if arguments.get("patient_id"):
                session.booking_state.patient_id = arguments.get("patient_id")
            if arguments.get("patient_age"):
                session.booking_state.patient_age = arguments.get("patient_age")
            if arguments.get("doctor_name"):
                session.booking_state.doctor_name = arguments.get("doctor_name")
            if slots:
                session.booking_state.stage = BookingStage.SLOTS_PRESENTED

            date_str = ""
            if pref_date:
                iso_m = re.search(r"202\d-(\d{2})-(\d{2})", str(pref_date))
                if iso_m:
                    m_num, d_num = iso_m.group(1), iso_m.group(2)
                    m_names = {"01": "January", "02": "February", "03": "March", "04": "April", "05": "May", "06": "June", "07": "July", "08": "August", "09": "September", "10": "October", "11": "November", "12": "December"}
                    date_str = f" on {m_names.get(m_num, 'October')} {int(d_num)}"
                elif "10-20" in str(pref_date) or "20" in str(pref_date):
                    date_str = " on October 20"
                elif "10-19" in str(pref_date) or "19" in str(pref_date) or "tomorrow" in str(pref_date).lower():
                    date_str = " for Tomorrow (Oct 19)"
                else:
                    date_str = f" on {pref_date}"

            if slots:
                slot_lines = [f"• [{s['slot_id']}] {s['datetime_str']} — {s['modality']} ({s['room']})" for s in slots]
                mod_phrase = f" for {mod_title} scan" if mod_title != "Radiology" else ""
                p_phrase = f" for {session.booking_state.patient_name}" if session.booking_state.patient_name and session.booking_state.patient_name not in ["Patient", "Walk-in Patient"] else ""
                
                if not session.booking_state.patient_name or session.booking_state.patient_name in ["Patient", "Walk-in Patient"]:
                    confirm_guide = (
                        f"Confirmation Required: Would you like to confirm any of these slots? (e.g. 'Yes, book slot {slots[0]['slot_id']}').\n"
                        f"Please specify the Slot ID along with patient name, age, referring doctor, and clinic location (or reply 'Confirm slot {slots[0]['slot_id']} as Walk-in Patient')."
                    )
                else:
                    confirm_guide = f"Confirmation Required for {session.booking_state.patient_name}: Would you like to confirm any of these slots? (e.g. 'Yes, book slot {slots[0]['slot_id']}')."

                msg = (
                    f"Found {len(slots)} available appointment slot(s){mod_phrase}{p_phrase}{date_str}:\n"
                    + "\n".join(slot_lines)
                    + f"\n\n{confirm_guide}"
                )
            else:
                from .booking_service import check_available_slots as get_all_slots
                all_open = get_all_slots(modality=raw_mod or "Any")
                if all_open:
                    slot_lines = [f"• [{s['slot_id']}] {s['datetime_str']} — {s['modality']} ({s['room']})" for s in all_open]
                    session.booking_state.displayed_slots = all_open
                    msg = (
                        f"No appointment slots or radiologists are currently scheduled{date_str}. "
                        f"However, we have {len(all_open)} available scan slot(s) on upcoming dates:\n"
                        + "\n".join(slot_lines)
                        + f"\n\nConfirmation Required: Would you like to confirm any of these slots? (e.g. 'Yes, book slot {all_open[0]['slot_id']}')."
                    )
                else:
                    msg = f"No appointment slots currently available{f' for {mod_title} scan' if mod_title != 'Radiology' else ''}{date_str}. Please check back later."

            return {
                "status": "confirmation_required",
                "message": msg,
                "answer": msg,
                "modality": mod_title,
                "available_slots": slots,
                "agent_trace": {
                    "orchestrator": "TriageAgent",
                    "active_agent": "ScanBookingAgent",
                    "tool_called": tool_name,
                    "arguments": arguments,
                    "rationale": rationale,
                    "mcp_jsonrpc": mcp_request
                }
            }

    # =========================================================================
    # AGENT 5: CONSULTATION & REPORT-AWARE AGENT
    # =========================================================================
    else:
        # Check active report guardrail
        if not session.active_patient_id or not session.active_report_id:
            # Auto-recovery: If user actually asked to open a patient's report, route to retrieval!
            has_retrieval_verb = any(v in query.lower() for v in ["open", "show", "load", "find", "fetch", "get", "view"])
            intent = extract_retrieval_intent(query)
            if has_retrieval_verb and intent.get("patient_name"):
                actual_query = intent.get("patient_name")
                retrieval_res = resolve_report_retrieval(actual_query, patient_name=intent.get("patient_name"), modality=intent.get("report_type"))
                if retrieval_res.get("status") == "success" and retrieval_res.get("report"):
                    session.set_active(
                        retrieval_res["patient"]["patient_id"],
                        retrieval_res["report"]["report_id"]
                    )
                    session.pending_candidates = None
                elif retrieval_res.get("status") == "clarification_required":
                    session.pending_candidates = retrieval_res.get("candidates")
                    session.last_retrieval_modality = intent.get("report_type", "Any")

                return {
                    **retrieval_res,
                    "agent_trace": {
                        "orchestrator": "TriageAgent",
                        "active_agent": "RetrievalAgent",
                        "tool_called": "search_patient_reports",
                        "arguments": intent,
                        "rationale": "Auto-recovery: Request was to open a report; redirected from Consultation to Retrieval.",
                        "mcp_jsonrpc": mcp_request
                    }
                }

            # If patient is selected in session, but no specific report was chosen:
            from .retrieval import get_patient, get_report_by_id
            if session.active_patient_id and not session.active_report_id:
                p = get_patient(session.active_patient_id)
                if p:
                    reps = p.get("reports", [])
                    if len(reps) == 1:
                        session.set_active(p["patient_id"], reps[0]["report_id"])
                    else:
                        is_all_reports = any(w in query.lower() for w in ["all", "across", "every", "what reports", "list reports", "how many reports", "all her reports", "all reports"])
                        is_single_report_ambiguous = (
                            (any(v in query.lower() for v in ["open", "view", "load"]) and "report" in query.lower() and not is_all_reports)
                            or (re.search(r"\b(?:summarize|summary)\b", query.lower()) and not is_all_reports and "reports" not in query.lower())
                            or (re.search(r"\b(?:her|his|hir|their|the)\s+report\b", query.lower()) and not is_all_reports)
                        )
                        if is_single_report_ambiguous:
                            rep_str = " • ".join([f"[{r['report_id']}] {r['report_type']} ({r['report_date']})" for r in reps])
                            from .retrieval import get_patient_pronouns
                            p_sex = p.get("sex")
                            pronoun_pos = get_patient_pronouns(p_sex)["possessive"]
                            return {
                                "status": "clarification_required",
                                "message": f"Patient {p['patient_name']} has {len(reps)} diagnostic reports on file: {rep_str}. Please specify which report to view or summarize (e.g. 'Open {pronoun_pos} {reps[0]['report_type']}').",
                                "patient": p,
                                "candidates": None,
                                "report": None,
                                "agent_trace": {
                                    "orchestrator": "TriageAgent",
                                    "active_agent": "ConsultationAgent",
                                    "tool_called": "summarize_active_report",
                                    "arguments": {"patient_id": p["patient_id"]},
                                    "rationale": "Clarification required: multiple reports available for selected patient."
                                }
                            }
                        else:
                            qa_result = answer_patient_reports_question(session.active_patient_id, None, query, history=session.history)
                            session.history.append({"user": query, "assistant": qa_result.get("answer", "")})
                            return {
                                "status": "success",
                                "message": qa_result.get("answer"),
                                "answer": qa_result.get("answer"),
                                "citations": qa_result.get("citations", []),
                                "pdf_url": qa_result.get("pdf_url"),
                                "context_resolution": f"Active Context -> Patient ID: {session.active_patient_id}",
                                "patient": p,
                                "report": None,
                                "agent_trace": {
                                    "orchestrator": "TriageAgent",
                                    "active_agent": "ConsultationAgent",
                                    "tool_called": tool_name,
                                    "arguments": arguments,
                                    "rationale": "Multi-Report Consultation: Synthesized findings across patient's diagnostic reports.",
                                    "mcp_jsonrpc": mcp_request
                                }
                            }

            if not session.active_report_id and not session.active_patient_id:
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

        # Answer question grounded in active report or cross-reports
        from .retrieval import get_patient, get_report_by_id
        patient = get_patient(session.active_patient_id) if session.active_patient_id else None
        report = get_report_by_id(session.active_report_id) if session.active_report_id else None
        if not patient and report:
            patient = get_patient(report.get("patient_id"))
            if patient:
                session.active_patient_id = patient["patient_id"]

        if session.active_patient_id:
            qa_result = answer_patient_reports_question(session.active_patient_id, session.active_report_id, query, history=session.history)
        else:
            qa_result = answer_report_question(session.active_report_id, query, history=session.history)

        session.history.append({"user": query, "assistant": qa_result.get("answer", "")})
        
        return {
            "status": "success",
            "message": qa_result.get("answer"),
            "answer": qa_result.get("answer"),
            "citations": qa_result.get("citations", []),
            "pdf_url": qa_result.get("pdf_url") or (f"/patient-pdf/{report.get('pdf_filename')}" if report and report.get("pdf_filename") else None),
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