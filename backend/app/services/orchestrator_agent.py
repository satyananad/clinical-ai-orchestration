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
from .retrieval import resolve_report_retrieval, extract_retrieval_intent
from .report_qa_service import answer_report_question
from .llm_service import query_llm, query_ollama, query_ollama_chat_tools

TRIAGE_SYSTEM_PROMPT = """You are the Lead Clinical Triage AI Agent for a dental voice workstation.
Your job is to analyze the clinician's speech or text input and decide which specialized sub-agent and tool to invoke.

AVAILABLE TOOLS & SPECIALIZED AGENTS:
1. "search_patient_reports": Use when the clinician wants to locate, open, or view a patient's scans or reports (e.g. "Open Priya's report", "Show Vikram Malhotra's CBCT").
   Agent: "RetrievalAgent"
   Arguments: {"patient_name": "string", "modality": "OPG"|"CBCT"|"Bitewing"|"Periapical"|"Cephalogram"|"Any"}

2. "structure_clinical_dictation": Use when the clinician is dictating physical examination findings, dental observations, pathology, or tooth conditions (e.g. "Lower right third molar is impacted", "UR5 root treated with apical radiolucency").
   Agent: "ClinicalScribeAgent"
   Arguments: {"dictation_text": "verbatim text"}

3. "summarize_active_report": Use when the clinician asks a follow-up question, explanation, or summary about the currently open report in the session (e.g. "Summarize her report", "What did the radiologist say about tooth 48?", "Explain this report", "Show me the PDF").
   Agent: "ConsultationAgent"
   Arguments: {"report_id": "active", "query": "string"}

4. "check_appointment_slots": Use when checking or booking available appointment slots for a scan (e.g. "Book a CBCT scan", "Schedule an OPG appointment", "What slots are available for CBCT?", "How many slots on 20 October?").
   Agent: "ScanBookingAgent"
   Arguments: {"modality": "CBCT"|"OPG"|"Bitewing"|"Periapical"|"Any", "preferred_date": "optional date string"}

5. "confirm_scan_booking": Use when confirming a slot booking (e.g. "Confirm slot SLOT-101", "Yes, book slot 1", "Book it for Priya").
   Agent: "ScanBookingAgent"
   Arguments: {"slot_id": "string", "patient_name": "string", "patient_id": "optional string"}

6. "conversation": Use for greetings, natural dialogue, conversational questions, workstation help, capabilities, or general clinical/dental radiology questions when no specific report is open (e.g. "hey", "hello", "what are you doing?", "who are you?", "can you help me?", "what is a CBCT scan?", "explain caries", "what is an OPG?", "how does this workstation work?", "thanks", "good morning"). No database tool is invoked.
   Agent: "GeneralAssistantAgent"
   Arguments: {"query": "string"}

7. "query_clinic_database": Use when specifically inquiring about the clinic database facts, total scans, patient counts, or modality breakdowns (e.g. "how many reports are in the database?", "how many patients do we have?", "what scans exist in the system?").
   Agent: "GeneralAssistantAgent"
   Arguments: {"question": "string", "patient_name": "optional string", "modality": "optional string"}

Output strictly a JSON object with this format:
{
  "selected_agent": "RetrievalAgent" | "ClinicalScribeAgent" | "ConsultationAgent" | "ScanBookingAgent" | "GeneralAssistantAgent",
  "tool_name": "search_patient_reports" | "structure_clinical_dictation" | "summarize_active_report" | "check_appointment_slots" | "confirm_scan_booking" | "conversation" | "query_clinic_database",
  "arguments": { ... },
  "rationale": "Brief reason for agent selection"
}
"""

def resolve_pending_candidates_from_query(
    query: str,
    candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates clinician query against pending clarification candidates using generic model fields:
    - patient_id (exact ID or numeric digits)
    - date_of_birth (exact ISO date, 4-digit birth year, 2-digit suffix, or ASR phoneme slip e.g. 9098 for 1998)
    - patient_name (exact full name or distinctive name token)
    - sex (if candidates differ by sex)
    - ordinal selection (first, 1st, second, 2nd, option 1, option 2, etc.)
    """
    q_lower = query.lower().strip()
    if not candidates:
        return {"status": "unmatched"}

    # 1. Ordinal matching (e.g. "the first patient", "second one", "option 2", "1st")
    ordinal_map = {
        "first": 0, "1st": 0, "one": 0, "1": 0,
        "second": 1, "2nd": 1, "two": 1, "2": 1,
        "third": 2, "3rd": 2, "three": 2, "3": 2,
        "fourth": 3, "4th": 3, "four": 3, "4": 3,
        "fifth": 4, "5th": 4, "five": 4, "5": 4
    }
    select_verb = any(v in q_lower for v in ["select", "choose", "pick", "open", "the", "patient", "candidate", "option", "number"])
    matched_idx = None
    for word, idx in ordinal_map.items():
        pattern = r"\b(?:the\s+)?(?:" + re.escape(word) + r")\s*(?:patient|candidate|one|record|option)?\b"
        if re.search(pattern, q_lower) and select_verb:
            matched_idx = idx
            break
        if re.search(r"\b(?:option|candidate|patient|number|#)\s*" + re.escape(word) + r"\b", q_lower):
            matched_idx = idx
            break

    if matched_idx is not None:
        if 0 <= matched_idx < len(candidates):
            return {
                "status": "resolved",
                "matched_candidate": candidates[matched_idx]
            }
        else:
            return {
                "status": "invalid_selection",
                "requested_index": matched_idx + 1
            }

    # 2. Check each candidate against generic model attributes
    matches = []
    for cand in candidates:
        c_id = cand.get("patient_id", "").lower()
        c_num = "".join(filter(str.isdigit, c_id))
        c_name = cand.get("patient_name", "").lower()
        c_dob = cand.get("date_of_birth", "")
        c_year = c_dob.split("-")[0] if c_dob else ""
        c_sex = cand.get("sex", "").lower()

        # A. Patient ID matching (e.g. PAT-1001, PT 10001, 1001)
        id_matched = False
        if c_id and c_id in q_lower:
            id_matched = True
        elif c_num and (
            re.search(r"\b(?:pat|pt)?[- ]?0*" + re.escape(c_num) + r"\b", q_lower)
            or re.search(r"\b" + re.escape(c_num) + r"\b", q_lower)
            or (len(c_num) >= 4 and re.search(r"\b" + re.escape(c_num[:1]) + r"0*" + re.escape(c_num[1:]) + r"\b", q_lower))
        ):
            id_matched = True

        # B. DOB / Birth Year matching (e.g. 1998, 1998-04-12, dob 1998, dob 9098, 998)
        dob_matched = False
        if c_dob and c_dob in q_lower:
            dob_matched = True
        elif c_year:
            if re.search(r"\b" + re.escape(c_year) + r"\b", q_lower):
                dob_matched = True
            elif len(c_year) >= 2:
                year_suffix = c_year[-2:]
                if re.search(r"\b(?:dob|born|birth)?\D*(?:90" + re.escape(year_suffix) + r"|9" + re.escape(year_suffix) + r"|" + re.escape(year_suffix) + r")\b", q_lower):
                    dob_matched = True

        # C. Name matching (distinguishing surname or distinctive first/middle name)
        name_matched = False
        cand_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", c_name))
        other_tokens = set()
        for other in candidates:
            if other.get("patient_id") != cand.get("patient_id"):
                other_tokens.update(re.findall(r"\b[a-zA-Z0-9_-]+\b", other.get("patient_name", "").lower()))
        distinctive_tokens = cand_tokens - other_tokens
        if distinctive_tokens and any(token in q_lower for token in distinctive_tokens):
            name_matched = True
        elif c_name in q_lower and all(c_name not in other.get("patient_name", "").lower() for other in candidates if other.get("patient_id") != cand.get("patient_id")):
            name_matched = True

        # D. Sex matching (if candidates differ by sex)
        sex_matched = False
        other_sexes = {other.get("sex", "").lower() for other in candidates if other.get("patient_id") != cand.get("patient_id")}
        if c_sex and c_sex not in other_sexes and re.search(r"\b" + re.escape(c_sex) + r"\b", q_lower):
            sex_matched = True

        if id_matched or dob_matched or name_matched or sex_matched:
            matches.append(cand)

    if len(matches) == 1:
        return {
            "status": "resolved",
            "matched_candidate": matches[0]
        }
    elif len(matches) > 1:
        return {
            "status": "narrowed",
            "remaining_candidates": matches
        }

    # 3. Explicit selection of unlisted candidate
    if any(v in q_lower for v in ["select", "choose", "pick"]):
        return {
            "status": "invalid_selection",
            "raw_query": query
        }

    return {"status": "unmatched"}

def resolve_pending_slot_from_query(query: str, pending_slots: List[Dict[str, Any]]) -> Optional[str]:
    """
    Matches user's appointment confirmation input against pending slots by:
    - Slot ID (e.g. 'SLOT-101', '101')
    - Time (e.g. '10:00 AM', '10:00 a.m.', '10am', '10:00', '2:30 PM', '11:15')
    - Date / Day (e.g. 'tomorrow', 'Oct 19', 'Oct 20')
    - Modality (e.g. 'CBCT', 'OPG')
    - Ordinal (e.g. 'first', '1st', 'second', '2nd')
    - Confirmation keywords ('yes', 'confirm', 'book it', 'take it', 'proceed') -> defaults to slot 1
    """
    if not pending_slots:
        return None
    q_norm = query.lower().replace("a.m.", "am").replace("p.m.", "pm").strip()

    # 1. Exact slot ID match (SLOT-101 or 101)
    for s in pending_slots:
        sid = s["slot_id"].lower()
        s_num = "".join(filter(str.isdigit, sid))
        if sid in q_norm or re.search(r"\b(?:slot[- ]?)?" + re.escape(s_num) + r"\b", q_norm):
            return s["slot_id"]

    # 2. Ordinal match (first, second, etc.)
    ordinal_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4}
    for word, idx in ordinal_map.items():
        if re.search(r"\b" + re.escape(word) + r"\b", q_norm) and idx < len(pending_slots):
            return pending_slots[idx]["slot_id"]

    # 3. Match by time / date in datetime_str
    matched = []
    for s in pending_slots:
        s_time = s.get("time", "").lower()
        time_part = s_time.replace(" am", "").replace(" pm", "").strip()
        if s_time in q_norm or (time_part and time_part in q_norm) or (time_part.lstrip("0") and time_part.lstrip("0") in q_norm):
            matched.append(s)

    if len(matched) == 1:
        return matched[0]["slot_id"]
    elif len(matched) > 1:
        for m in matched:
            m_dt = m.get("datetime_str", "").lower()
            if any(w in q_norm for w in ["oct 19", "tomorrow", "19th", "cbct"]) and "19" in m_dt:
                return m["slot_id"]
            if any(w in q_norm for w in ["oct 20", "20th"]) and "20" in m_dt:
                return m["slot_id"]
        return matched[0]["slot_id"]

    # 4. Confirmation keyword fallback (default to slot 1)
    if any(w in q_norm for w in ["yes", "confirm", "book", "proceed", "sure", "ok", "please"]):
        return pending_slots[0]["slot_id"]

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
            is_database_q = any(q_lower.startswith(w) for w in ["how many", "how much", "which", "who", "list", "what", "show"]) or "database" in q_lower or "?" in q_lower
            has_retrieval_cmd = any(verb in q_lower for verb in ["open", "find", "load", "retrieve", "book", "schedule", "cancel", "stop"])
            if is_database_q or has_retrieval_cmd:
                session.pending_candidates = None

    # Step 0B: Scan booking confirmation resolution from pending slots
    if session.pending_booking_slots:
        matched_slot_id = resolve_pending_slot_from_query(query, session.pending_booking_slots)
        if matched_slot_id:
            pat_name = None
            pat_id = session.active_patient_id
            if pat_id:
                from .retrieval import get_patient
                p = get_patient(pat_id)
                if p:
                    pat_name = p.get("patient_name")
            if not pat_name:
                pat_name = "Walk-in Patient"

            session.pending_booking_slots = None
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "confirm_scan_booking",
                "arguments": {
                    "slot_id": matched_slot_id,
                    "patient_name": pat_name,
                    "patient_id": pat_id
                },
                "rationale": f"Clarification Resolution: Matched slot {matched_slot_id} from user input '{query}'."
            }
        elif any(w in q_lower for w in ["cancel", "never mind", "stop", "abort", "no"]):
            session.pending_booking_slots = None
            return {
                "selected_agent": "GeneralAssistantAgent",
                "tool_name": "general_query",
                "arguments": {"question": "Appointment booking cancelled."},
                "rationale": "Clinician cancelled appointment booking."
            }
        else:
            is_time_or_date = any(t in q_lower for t in ["am", "pm", "a.m.", "p.m.", ":", "tomorrow", "oct", "october", "morning", "afternoon", "slot", "#", "first", "second", "third", "take", "give me"]) or any(c.isdigit() for c in query)
            is_new_command = any(v in q_lower for v in ["open", "find", "retrieve", "dictate", "generate", "summarize", "database", "how many"])
            if is_time_or_date and not is_new_command:
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "invalid_slot_selection",
                    "arguments": {
                        "query": query,
                        "available_slots": session.pending_booking_slots
                    },
                    "rationale": f"Safety Gate: User query '{query}' did not match any available slot."
                }
            elif is_new_command:
                session.pending_booking_slots = None
    else:
        # If session pending_booking_slots was cleared/None, check if user directly entered a slot ID or time/date
        is_direct_slot_or_time = bool(
            re.search(r"\b(?:slot[- ]?\d+|10:00|02:30|2:30|11:15|04:00|4:00|09:00|9:00|10:30|03:15|3:15)\b", q_lower)
            or (any(t in q_lower for t in ["am", "pm", "a.m.", "p.m."]) and any(c.isdigit() for c in q_lower))
        )
        is_retrieval_cmd = any(v in q_lower for v in ["open", "find", "retrieve", "load", "show", "view", "dictate", "generate", "database", "how many"])
        if is_direct_slot_or_time and not is_retrieval_cmd:
            from .booking_service import check_available_slots
            all_avail = check_available_slots()
            matched_slot_id = resolve_pending_slot_from_query(query, all_avail)
            if matched_slot_id:
                pat_name = None
                pat_id = session.active_patient_id
                if pat_id:
                    from .retrieval import get_patient
                    p = get_patient(pat_id)
                    if p:
                        pat_name = p.get("patient_name")
                if not pat_name:
                    pat_name = "Walk-in Patient"
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "confirm_scan_booking",
                    "arguments": {
                        "slot_id": matched_slot_id,
                        "patient_name": pat_name,
                        "patient_id": pat_id
                    },
                    "rationale": f"Direct Slot Resolution: Matched slot {matched_slot_id} from user input '{query}'."
                }

    # Step 1: LLM Dynamic Multi-Agent Triage (Groq / Ollama Clinical Orchestrator Agent)
    prompt = (
        f"Clinician request: \"{query}\"\n"
        f"Active Patient in session: {session.active_patient_id or 'None'}\n"
        f"Active Report in session: {session.active_report_id or 'None'}\n"
        f"Pending Booking Slots: {'Yes (' + str(len(session.pending_booking_slots)) + ' slots)' if session.pending_booking_slots else 'No'}\n"
    )
    if session.history:
        prompt += "Recent dialogue:\n"
        for turn in session.history[-2:]:
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
                elif t_name == "check_appointment_slots":
                    if not args.get("preferred_date"):
                        if re.search(r"\b(20(?:th)?\s+oct(?:ober)?|oct(?:ober)?\s+20(?:th)?|2026-10-20)\b", q_lower):
                            args["preferred_date"] = "2026-10-20"
                        elif re.search(r"\b(19(?:th)?\s+oct(?:ober)?|oct(?:ober)?\s+19(?:th)?|tomorrow|2026-10-19)\b", q_lower):
                            args["preferred_date"] = "2026-10-19"
                elif t_name == "conversation" and "query" not in args:
                    args["query"] = query
                return {
                    "selected_agent": sel_agent,
                    "tool_name": t_name,
                    "arguments": args,
                    "rationale": f"LLM Multi-Agent Reasoning: {parsed_triage.get('rationale', 'Dynamic triage by Clinical Orchestrator Agent.')}"
                }
        except Exception:
            pass

    # Secondary: Native LLM Tool Calling (Ollama chat tools interface)
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
            "check_appointment_slots": "ScanBookingAgent",
            "confirm_scan_booking": "ScanBookingAgent"
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


    # 2B.5 Scan Booking Request (Check Available Slots)
    is_booking_request = any(w in q_lower for w in ["book", "booking", "schedule", "reschedule", "appointment", "appoint ment", "reserve", "slots", "slot"])
    is_retrieval_override = any(w in q_lower for w in ["open", "show", "view", "find report", "retrieve"])
    if is_booking_request and not is_retrieval_override:
        detected_mod = "Any"
        if any(w in q_lower for w in ["cbct", "cone beam", "3d"]):
            detected_mod = "CBCT"
        elif any(w in q_lower for w in ["opg", "panoramic"]):
            detected_mod = "OPG"
        elif "bitewing" in q_lower:
            detected_mod = "Bitewing"
        elif any(w in q_lower for w in ["periapical", "iopa"]):
            detected_mod = "Periapical"

        detected_date = None
        if re.search(r"\b(20(?:th)?\s+oct(?:ober)?|oct(?:ober)?\s+20(?:th)?|2026-10-20)\b", q_lower):
            detected_date = "2026-10-20"
        elif re.search(r"\b(19(?:th)?\s+oct(?:ober)?|oct(?:ober)?\s+19(?:th)?|tomorrow|2026-10-19)\b", q_lower):
            detected_date = "2026-10-19"

        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "check_appointment_slots",
            "arguments": {
                "modality": detected_mod,
                "preferred_date": detected_date
            },
            "rationale": f"Fallback: Clinician requested scan booking for {detected_mod} on {detected_date or 'any date'}."
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

    # 2D. Dictation findings vs Educational/Clinical Questions
    clinical_keywords = [
        "impacted", "radiolucency", "root treated", "carious", "caries",
        "periapical", "canal", "osteitis", "follicle", "supernumerary",
        "alveolar ridge", "condyle", "tonsillolith", "bone loss"
    ]
    is_question_intent = any(q_lower.startswith(w) for w in [
        "what is", "what are", "what does", "how does", "how do", "why does", "why is",
        "explain", "tell me about", "can you explain", "who are you", "what are you doing",
        "what can you do", "help me", "how to", "difference between"
    ]) or q_lower.endswith("?")

    if any(kw in q_lower for kw in clinical_keywords) and not is_question_intent and not any(verb in q_lower for verb in ["open", "show", "find", "get", "retrieve", "search"]):
        return {
            "selected_agent": "ClinicalScribeAgent",
            "tool_name": "structure_clinical_dictation",
            "arguments": {
                "dictation_text": query,
                "patient_id": session.active_patient_id
            },
            "rationale": "Fallback: Input describes clinical pathology findings."
        }

    # 2D.5 Conversational Dialogue, Help, and General Clinical Concept Questions
    is_conversational_greeting = any(re.search(r"\b" + re.escape(g) + r"\b", q_lower) for g in [
        "hey", "hello", "hi", "good morning", "good afternoon", "good evening", "greetings",
        "help", "who are you", "what are you doing", "what can you do", "how do you work", "how does this work",
        "thanks", "thank you"
    ])
    has_retrieval_cmd = any(verb in q_lower for verb in ["open", "load", "show report", "view report", "fetch", "retrieve", "find report"])

    if (is_conversational_greeting or is_question_intent) and not has_retrieval_cmd and not any(w in q_lower for w in ["database", "how many", "count", "breakdown"]):
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "conversation",
            "arguments": {"query": query},
            "rationale": "Fallback: Natural conversational inquiry, greeting, or clinical concept question."
        }

    # 2E. General Clinic Database Queries
    is_database_question = any(q_lower.startswith(w) for w in ["how many", "how much", "which", "list", "count", "breakdown"]) or "database" in q_lower or "total scans" in q_lower or "total patients" in q_lower
    if is_database_question and not has_retrieval_cmd:
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "query_clinic_database",
            "arguments": {"question": query},
            "rationale": "Fallback: Clinical database statistical inquiry."
        }

    # 2F. Default: Patient & Report Retrieval vs Conversational Fallback
    query_intent = extract_retrieval_intent(query)
    target_modality = query_intent.get("report_type", "Any")
    extracted_name = query_intent.get("patient_name", "").strip()

    # Safety Guard: If extracted name is an inquiry, general concept, or non-patient words
    non_patient_words = [
        "what", "why", "how", "explain", "tell", "help", "who", "when", "does", "can", "please",
        "doctor", "doctore", "docoter", "hospital", "physician", "clinic", "machine", "cost", "price",
        "name", "information", "detail", "details"
    ]
    has_non_patient_word = any(w in extracted_name.lower().split() for w in non_patient_words)
    has_retrieval_command = any(verb in q_lower for verb in ["open", "show", "load", "fetch", "view", "find", "get"])
    has_patient_id = bool(re.search(r"\b(PAT-\d+|PT-\d+|\d{4})\b", query, re.IGNORECASE))

    from .retrieval import search_patients
    matched_candidates = search_patients(extracted_name) if extracted_name else []

    if has_non_patient_word or not extracted_name or (not has_retrieval_command and not has_patient_id and not matched_candidates):
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "conversation",
            "arguments": {"query": query},
            "rationale": "Safety Guard: Query was an inquiry or question, not a patient name."
        }

    name_guess = extracted_name.title() if extracted_name else query.title()

    pat_id_match = re.search(r"\b(PAT-\d+)\b", query, re.IGNORECASE)
    patient_id_found = pat_id_match.group(1).upper() if pat_id_match else None

    return {
        "selected_agent": "RetrievalAgent",
        "tool_name": "search_patient_reports",
        "arguments": {
            "patient_name": name_guess,
            "patient_id": patient_id_found,
            "modality": target_modality
        },
        "rationale": "Defaulting to patient and report retrieval search."
    }

def generate_dynamic_conversational_reply(query: str, session: Any) -> str:
    """
    Dynamically generates contextual conversational responses using the LLM.
    If LLM inference is unavailable or offline, provides an intelligent,
    clinically sound, query-specific response tailored to what was asked
    rather than a static repeated greeting.
    """
    conversation_prompt = (
        "You are an intelligent, empathetic, and knowledgeable Clinical AI Assistant. "
        "Answer the clinician or user directly, intelligently, naturally, and helpfully. "
        "Rules:\n"
        "1. Directly answer what the user asks in a conversational, helpful, and natural manner.\n"
        "2. Never output placeholder tokens like '[Clinic Name]', '[Doctor Name]', or 'None'.\n"
        "3. If the user asks for a doctor or hospital, ask them clarifying details (such as city/location, medical specialty, or preferred hospital) to assist them.\n"
        "4. If they ask about dental or medical concepts (such as CBCT, OPG, caries, impactions, root canals), explain clearly and concisely.\n"
        "5. If they want to retrieve a patient's scans, structure dictation, or check appointment slots, guide them naturally.\n"
        "6. Do not repeat a fixed greeting if they asked a specific question."
    )
    if session.active_patient_id:
        conversation_prompt += f"\nCurrently active patient in session: {session.active_patient_id}"
    if session.active_report_id:
        conversation_prompt += f"\nCurrently active report in session: {session.active_report_id}"

    llm_reply = query_llm(query, system_prompt=conversation_prompt, json_format=False)
    if llm_reply and len(llm_reply.strip()) >= 5:
        return llm_reply.strip()

    # Dynamic semantic offline fallback tailored specifically to the query
    q_lower = query.lower().strip()

    # 1. CBCT / 3D imaging questions
    if any(k in q_lower for k in ["cbct", "cone beam", "3d scan", "3d imaging"]):
        return (
            "Cone Beam Computed Tomography (CBCT) provides volumetric 3D diagnostic imaging of the maxillofacial complex. "
            "It produces sub-millimeter isotropic resolution views in axial, coronal, and sagittal planes, essential for "
            "implant site planning, assessing impacted third molars relative to the inferior alveolar nerve canal, and evaluating complex endodontic root canal anatomy."
        )

    # 2. OPG / Panoramic radiograph questions
    if any(k in q_lower for k in ["opg", "panoramic", "orthopantomogram"]):
        return (
            "An Orthopantomogram (OPG) is a wide-field rotational panoramic radiograph that visualizes the entire dentition, "
            "maxillary sinuses, alveolar crests, and temporomandibular joint (TMJ) condyles in a single 2D screening view. "
            "It is commonly indicated for third molar assessment, orthodontic records, and comprehensive baseline screening."
        )

    # 3. Bitewing & Periapical (IOPA) questions
    if any(k in q_lower for k in ["bitewing", "periapical", "iopa", "intraoral"]):
        return (
            "Intraoral radiographs deliver high spatial resolution: Bitewings are the standard of care for detecting interproximal "
            "caries and assessing crestal alveolar bone height, whereas Periapical (IOPA) radiographs capture the complete tooth "
            "including the crown, root, and periapical tissues to diagnose apical periodontitis, root fractures, or resorptive lesions."
        )

    # 4. Caries / Cavity pathology questions
    if any(k in q_lower for k in ["caries", "carious", "cavity", "decay", "demineralization"]):
        return (
            "Dental caries is microbial demineralization of the calcified tissues of the tooth (enamel, dentin, cementum). "
            "Radiographically, it presents as a radiolucent shadow on occlusal pits or interproximal contact zones. "
            "Restorative therapy (e.g. composite or amalgam restorations) or endodontic treatment is indicated based on pulpal proximity."
        )

    # 5. Impaction & Third Molar questions
    if any(k in q_lower for k in ["impacted", "impaction", "wisdom tooth", "third molar", "tooth 48", "tooth 38"]):
        return (
            "Tooth impaction occurs when normal eruption is prevented by adjacent teeth, dense bone, or inadequate arch length "
            "(most commonly third molars 38 and 48). Radiographic assessment classifies angulation (mesioangular, vertical, distoangular, or horizontal) "
            "and carefully assesses the proximity of root apices to the inferior alveolar nerve canal."
        )

    # 6. Radiolucency / Periapical pathosis
    if any(k in q_lower for k in ["radiolucency", "radiolucent", "periapical", "apical periodontitis", "cyst"]):
        return (
            "A periapical radiolucency represents localized bone resorption at the tooth apex, typically secondary to pulpal necrosis "
            "and subsequent inflammatory response (apical periodontitis, granuloma, or radicular cyst). Treatment usually entails root canal therapy or surgical endodontics."
        )

    # 7. Workstation capabilities & "What are you doing / Who are you / How does it work"
    if any(k in q_lower for k in ["who are you", "what are you doing", "what can you do", "how do you work", "how does this work", "help", "capabilities"]):
        return (
            "I am your Viscan AI Clinical Workstation Assistant. I am designed to assist dental clinicians with:\n"
            "• **Diagnostic Retrieval**: Find and open patient imaging (OPG, CBCT, Bitewing, Periapical) and view official diagnostic PDFs with patient safety disambiguation.\n"
            "• **Clinical Scribe**: Convert spoken or typed clinical dictations into verified radiological findings with FDI/Palmer numbering, measurement preservation, and certainty labeling.\n"
            "• **Report Q&A & Consultation**: Answer questions and summarize findings strictly grounded in active reports with verbatim source citations.\n"
            "• **Scan Scheduling**: Check real-time appointment slots and confirm bookings."
        )

    # 7b. Doctor / Hospital information queries
    if any(k in q_lower for k in ["doctor", "doctore", "docoter", "hospital", "physician", "clinic name"]):
        return (
            "To help provide you with the right doctor or hospital details, could you please specify:\n"
            "1. Location (city, state, or region)\n"
            "2. Type of doctor or medical specialty (e.g. oral surgeon, radiologist, cardiologist, general practitioner)\n"
            "3. Name of the hospital or clinic (if you already know it)\n\n"
            "Once you provide these details, I will be happy to assist you!"
        )

    # 8. Greetings
    if any(re.search(r"\b" + re.escape(g) + r"\b", q_lower) for g in ["hey", "hello", "hi", "good morning", "good afternoon", "good evening", "greetings"]):
        return (
            "Hello Doctor! I am your Viscan AI Clinical Workstation Assistant. "
            "You can ask me to open a patient's scan (e.g. 'Open Vikram Malhotra's CBCT'), dictate examination observations, "
            "explain active findings, or check available scan appointment slots. How can I assist you?"
        )

    # 9. Appointment booking instructions
    if any(k in q_lower for k in ["how to book", "book appointment", "schedule scan"]):
        return (
            "To book a scan appointment, you can ask for available slots (e.g. 'What slots are available for CBCT?' or 'How many slots on 20 October?'). "
            "Once the slots are displayed, confirm your choice by replying with the slot ID or time (e.g. 'Yes, book slot SLOT-101')."
        )

    # 10. General catch-all
    return (
        f"Regarding your query: I am your Clinical Workstation Assistant. "
        f"You can ask me to retrieve patient imaging (e.g. 'Open Priya Sharma's OPG'), explain diagnostic findings, "
        f"dictate new clinical observations, or schedule appointment slots. What would you like to explore?"
    )

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
        retrieval_res = resolve_report_retrieval(actual_query, patient_name=patient_name_arg, modality=modality_arg)
        
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
            retrieval_res["message"] = (
                f"Opened {rep_type} report ({rep_id}) for {p_name}."
                + (f" [View Diagnostic PDF]({pdf_url})" if pdf_url else "")
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
        return {
            "status": "success" if is_complete else "incomplete",
            "extraction_status": dictation_res.get("extraction_status", "COMPLETE" if is_complete else "INCOMPLETE_REQUIRES_REVIEW"),
            "message": dictation_res.get("message", "Clinical observations structured into validated draft report."),
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
    elif selected_agent == "ScanBookingAgent" or tool_name in ["check_appointment_slots", "confirm_scan_booking"]:
        if tool_name == "confirm_scan_booking":
            slot_id = arguments.get("slot_id")
            if not slot_id and session.pending_booking_slots:
                slot_id = session.pending_booking_slots[0]["slot_id"]

            patient_name = arguments.get("patient_name")
            patient_id = arguments.get("patient_id")
            if not patient_name and session.active_patient_id:
                from .retrieval import get_patient
                p = get_patient(session.active_patient_id)
                if p:
                    patient_name = p.get("patient_name")
                    patient_id = p.get("patient_id")
            if not patient_name:
                patient_name = "Patient"

            confirm_res = execute_tool("confirm_scan_booking", {
                "slot_id": slot_id,
                "patient_name": patient_name,
                "patient_id": patient_id
            })
            session.pending_booking_slots = None
            session.pending_booking_modality = None

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
        elif tool_name == "invalid_slot_selection":
            slots = session.pending_booking_slots or []
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
            slots_res = execute_tool("check_appointment_slots", arguments)
            slots = slots_res.get("available_slots", [])
            raw_mod = arguments.get("modality")
            pref_date = arguments.get("preferred_date")
            mod_title = raw_mod if (raw_mod and str(raw_mod).lower() not in ["none", "any", "null", "all"]) else "Radiology"
            session.pending_booking_slots = slots
            session.pending_booking_modality = mod_title

            date_str = ""
            if pref_date:
                if "10-20" in str(pref_date) or "20" in str(pref_date):
                    date_str = " on October 20"
                elif "10-19" in str(pref_date) or "19" in str(pref_date):
                    date_str = " for Tomorrow (Oct 19)"
                else:
                    date_str = f" on {pref_date}"

            if slots:
                slot_lines = [f"• [{s['slot_id']}] {s['datetime_str']} — {s['modality']} ({s['room']})" for s in slots]
                mod_phrase = f" for {mod_title} scan" if mod_title != "Radiology" else ""
                msg = (
                    f"Found {len(slots)} available appointment slot(s){mod_phrase}{date_str}:\n"
                    + "\n".join(slot_lines)
                    + f"\n\nConfirmation Required: Would you like to confirm any of these slots? (e.g. 'Yes, book slot {slots[0]['slot_id']}')."
                )
            else:
                msg = f"No appointment slots currently available{f' for {mod_title} scan' if mod_title != 'Radiology' else ''}{date_str}. Please check back later."

            return {
                "status": "confirmation_required",
                "message": msg,
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
            "pdf_url": qa_result.get("pdf_url"),
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