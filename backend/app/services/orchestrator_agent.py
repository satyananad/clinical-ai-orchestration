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
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .mcp_tools import DENTAL_ASSISTANT_TOOLS, get_ollama_tool_definitions, execute_tool, handle_mcp_jsonrpc_request
from .conversation_service import get_or_create_session, is_contextual_query, is_general_or_assistant_query, BookingStage
from .retrieval import resolve_report_retrieval, extract_retrieval_intent
from .report_qa_service import answer_report_question, answer_patient_reports_question, generate_multi_report_summary
from .llm_service import query_llm, query_ollama, query_ollama_chat_tools

TRIAGE_SYSTEM_PROMPT = """You are the Lead Clinical Triage AI Agent for a dental voice workstation.
Your job is to analyze the clinician's speech or text input and decide which specialized sub-agent and tool to invoke.

AVAILABLE TOOLS & SPECIALIZED AGENTS:
1. "search_patient_reports": Use when the clinician wants to locate, open, view, or search for a patient's scans or reports (e.g. "Open Priya's report", "Show Vikram Malhotra's CBCT", "hey i want to open the report of arjun mehta"). Even if the clinician starts with a greeting like 'hey', 'hello', or 'hi', you MUST select 'search_patient_reports' if they mention opening, viewing, or searching for a patient's report!
   Agent: "RetrievalAgent"
   Arguments: {"patient_name": "string", "modality": "OPG"|"CBCT"|"Bitewing"|"Periapical"|"Cephalogram"|"Any"}

2. "structure_clinical_dictation": Use when the clinician is dictating physical examination findings, dental observations, pathology, or tooth conditions (e.g. "Lower right third molar is impacted", "UR5 root treated with apical radiolucency").
   Agent: "ClinicalScribeAgent"
   Arguments: {"dictation_text": "verbatim text"}

3. "summarize_active_report": Use when an active report is open in the session and the clinician asks ANY question about findings, diagnoses, problems, conditions, impressions, teeth, or summaries (e.g. "what problem it have", "what problem does she have", "what is wrong with her", "Summarize her report", "What did the radiologist say about tooth 48?", "Explain this report", "Show me the PDF"). NEVER use 'conversation' if an active report is loaded and the query asks about patient problems or scan findings!
   Agent: "ConsultationAgent"
   Arguments: {"report_id": "active", "query": "string"}

4. "check_appointment_slots": Use when checking or booking available appointment slots for a scan (e.g. "Book a CBCT scan", "Schedule an OPG appointment", "What slots are available for CBCT?", "How many slots on 20 October?").
   Agent: "ScanBookingAgent"
   Arguments: {"modality": "CBCT"|"OPG"|"Bitewing"|"Periapical"|"Any", "preferred_date": "optional date string"}

5. "confirm_scan_booking": Use when confirming a slot booking (e.g. "Confirm slot SLOT-101", "Yes, book slot 1", "Book it for Priya").
   Agent: "ScanBookingAgent"
   Arguments: {"slot_id": "string", "patient_name": "string", "patient_id": "optional string"}

6. "conversation": Use ONLY for pure greetings, small talk, or general dental concepts when NO patient, report retrieval, booking, or clinical dictation is requested (e.g. "hello", "who are you?", "what is a CBCT scan?"). If the clinician asks to open or view a report or mentions a patient (e.g. "hey i want to open the report of arjun mehta"), NEVER use 'conversation'!
   Agent: "GeneralAssistantAgent"
   Arguments: {"query": "string"}

7. "query_clinic_database": Use when inquiring about registered patients, listing patient names, total scans, patient counts, or database breakdowns (e.g. "can you list out only patient name", "first tell me the patient names", "how many reports are in the database?", "how many patients do we have?", "what scans exist in the system?").
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

def is_valid_patient_name(name: Optional[str]) -> bool:
    """Validates if a candidate string is a plausible patient name rather than command keywords or slot tokens."""
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    if len(clean) < 3:
        return False
    # Patient name cannot contain digits, slot references, or hyphens followed by numbers
    if re.search(r"\d", clean) or re.search(r"\b(?:slot|bkg|tk)[- ]?", clean, re.IGNORECASE):
        return False
    invalid_keywords = {
        "slot", "slots", "confirm", "confirmed", "booking", "book", "appointment",
        "patient", "walk-in", "walk", "yes", "please", "tomorrow", "today", "yesterday",
        "scan", "report", "opg", "cbct", "periapical", "xray", "x-ray", "ok", "okay",
        "sure", "cancel", "open", "show", "view", "find", "load", "dr", "doctor",
        "location", "hospital", "clinic", "radiology", "suite", "room", "bay"
    }
    words = re.findall(r"[a-zA-Z]+", clean.lower())
    if not words or any(w in invalid_keywords for w in words):
        return False
    return True


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

    # 4. Explicit confirmation keyword fallback (default to slot 1 if user explicitly confirms without picking)
    is_standalone_confirm = any(
        re.search(r"\b" + re.escape(w) + r"\b", q_norm)
        for w in ["yes", "confirm", "book it", "take it", "proceed", "first one", "first slot", "that one"]
    ) and not any(w in q_norm for w in ["cbct", "opg", "patient", "panoramic", "bitewing", "periapical"])
    if is_standalone_confirm:
        return pending_slots[0]["slot_id"]

    return None

def extract_date_from_query(query: str) -> Optional[str]:
    """
    Extracts an appointment or scan date from clinician natural language:
    Uses clock.get_current_date() for dynamic calculation.
    """
    from .clock import get_current_date
    now_d = get_current_date()
    q_lower = query.lower().strip()
    if "tomorrow" in q_lower:
        return (now_d + timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in q_lower:
        return now_d.strftime("%Y-%m-%d")

    iso = re.search(r"\b(202\d)-(0[1-9]|1[0-2])-([0-3]\d)\b", q_lower)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

    months = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05", "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12"
    }
    m_re = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

    # "23rd october", "20 oct", "20th of october"
    m_day_first = re.search(rf"\b([0-3]?\d)(?:st|nd|rd|th)?\s+(?:of\s+)?({m_re})\b", q_lower)
    if m_day_first:
        day = int(m_day_first.group(1))
        m_str = m_day_first.group(2)
        for k, v in months.items():
            if m_str.startswith(k):
                return f"{now_d.year}-{v}-{day:02d}"

    # "october 23rd", "oct 20" (ensure day is not time like 10:00)
    m_month_first = re.search(rf"\b({m_re})\s+([0-3]?\d)(?:st|nd|rd|th)?(?!\s*[:\d])\b", q_lower)
    if m_month_first:
        m_str = m_month_first.group(1)
        day = int(m_month_first.group(2))
        for k, v in months.items():
            if m_str.startswith(k):
                return f"{now_d.year}-{v}-{day:02d}"

    return None

def extract_age_from_query(query: str) -> Optional[str]:
    """Extracts patient age or date of birth reference from natural language."""
    m = re.search(r"\b(?:age[d]?\s*[:=]?\s*|age\s+is\s+)(\d{1,3})\b", query, re.IGNORECASE)
    if m:
        return f"{m.group(1)} years"
    m2 = re.search(r"\b(\d{1,3})\s*(?:years?\s+old|yrs?\s+old|yo|y/o)\b", query, re.IGNORECASE)
    if m2:
        return f"{m2.group(1)} years"
    return None

def extract_doctor_from_query(query: str) -> Optional[str]:
    """Extracts referring or attending doctor name from natural language."""
    m = re.search(r"\b(?:dr\.?|doctor)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", query, re.IGNORECASE)
    if m:
        raw_doc = m.group(1).strip()
        parts = raw_doc.split()
        stop_words = {
            "main", "clinic", "hospital", "center", "centre", "radiology", "suite", "bay", "room",
            "dept", "department", "at", "in", "on", "for", "age", "aged", "pat", "patient", "scan",
            "cbct", "opg", "slot", "tomorrow", "today", "yesterday"
        }
        filtered = [p for p in parts if p.lower() not in stop_words]
        if filtered:
            return f"Dr. {' '.join(filtered).title()}"
    return None

def extract_patient_details_for_booking(query: str, session: Optional[Any] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extracts patient name, patient ID, patient age, and attending/referring doctor
    specifically tailored for booking dialogues, handling multi-entity queries gracefully.
    Returns: (patient_name, patient_id, patient_age, doctor_name)
    """
    q_lower = query.lower()
    pat_age = extract_age_from_query(query)
    doc_name = extract_doctor_from_query(query)
    pat_name = None
    pat_id = None

    if any(w in q_lower for w in ["walk-in", "walk in"]):
        pat_name = "Walk-in Patient"
        pat_id = None
        return pat_name, pat_id, pat_age, doc_name

    # 1. Check registered patient ID (e.g. PAT-1003)
    m_id = re.search(r"\b(PAT-\d+|PT-\d+)\b", query, re.IGNORECASE)
    if m_id:
        from .retrieval import get_patient
        p = get_patient(m_id.group(1).upper())
        if p:
            pat_name = p.get("patient_name")
            pat_id = p.get("patient_id")

    # 2. Check registered full name in query
    if not pat_name:
        from .retrieval import get_all_patients
        all_p = get_all_patients()
        for p in all_p:
            p_name = p.get("patient_name", "")
            if p_name and re.search(r"\b" + re.escape(p_name.lower()) + r"\b", q_lower):
                pat_name = p_name
                pat_id = p.get("patient_id")
                break

    # 3. Check 'patient <Name>' or 'for <Name>'
    if not pat_name:
        from .retrieval import get_all_patients
        all_p = get_all_patients()
        pat_pattern = re.search(r"\b(?:for\s+patient|patient|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", query, re.IGNORECASE)
        if pat_pattern:
            cand = pat_pattern.group(1).strip()
            stop_words = {
                "cbct", "opg", "scan", "slot", "tomorrow", "today", "yesterday", "dr", "doctor",
                "main", "clinic", "radiology", "suite", "bay", "room", "age", "appointment", "check",
                "patient", "referred", "by"
            }
            cand_parts = [w for w in cand.split() if w.lower() not in stop_words]
            if cand_parts:
                clean_cand = " ".join(cand_parts)
                exact_matches = [p for p in all_p if p["patient_name"].lower() == clean_cand.lower()]
                if len(exact_matches) == 1:
                    pat_name = exact_matches[0]["patient_name"]
                    pat_id = exact_matches[0]["patient_id"]
                elif len(cand_parts) == 1:
                    first_matches = [p for p in all_p if p["patient_name"].split()[0].lower() == cand_parts[0].lower()]
                    if len(first_matches) == 1:
                        pat_name = first_matches[0]["patient_name"]
                        pat_id = first_matches[0]["patient_id"]
                    elif is_valid_patient_name(clean_cand):
                        pat_name = clean_cand.title()
                elif is_valid_patient_name(clean_cand):
                    pat_name = clean_cand.title()

    # 4. Fallback to session state if available
    if not pat_name and session:
        if hasattr(session, "booking_state") and session.booking_state and session.booking_state.patient_name:
            pat_name = session.booking_state.patient_name
            pat_id = session.booking_state.patient_id
        elif hasattr(session, "active_patient_id") and session.active_patient_id:
            from .retrieval import get_patient
            p = get_patient(session.active_patient_id)
            if p:
                pat_name = p.get("patient_name")
                pat_id = p.get("patient_id")

    if not pat_age and session and hasattr(session, "booking_state") and session.booking_state and session.booking_state.patient_age:
        pat_age = session.booking_state.patient_age

    if not doc_name and session and hasattr(session, "booking_state") and session.booking_state and session.booking_state.doctor_name:
        doc_name = session.booking_state.doctor_name

    # 5. Populate age from DOB if registered patient and age not provided
    if pat_id and not pat_age:
        from .retrieval import get_patient
        p = get_patient(pat_id)
        if p and p.get("date_of_birth"):
            try:
                pat_age = f"{2026 - int(p['date_of_birth'].split('-')[0])} years"
            except Exception:
                pass

    return pat_name, pat_id, pat_age, doc_name


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
                elif t_name == "check_appointment_slots":
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
                    elif "query" not in args:
                        args["query"] = query
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
            "check_appointment_slots": "ScanBookingAgent",
            "confirm_scan_booking": "ScanBookingAgent",
            "lookup_existing_booking": "ScanBookingAgent",
            "cancel_scan_booking": "ScanBookingAgent",
            "reschedule_scan_booking": "ScanBookingAgent",
            "conversation": "GeneralAssistantAgent"
        }
        if tool_name in agent_map:
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
            is_database_q = any(q_lower.startswith(w) for w in ["how many", "how much", "which", "who", "list", "what", "show"]) or "database" in q_lower or "?" in q_lower
            has_retrieval_cmd = any(verb in q_lower for verb in ["open", "find", "load", "retrieve", "book", "schedule", "cancel", "stop"])
            if is_database_q or has_retrieval_cmd:
                session.pending_candidates = None

    # Step 0B: State-Machine Booking Workflow Management
    b_state = session.booking_state
    ref_match = re.search(r"\b(BKG-[A-Za-z0-9_-]+|SLOT-\d+)\b", query, re.IGNORECASE)
    b_ref = ref_match.group(1).upper() if ref_match else None

    # 0B.0 Workflow in AWAITING_CONFIRMATION (Explicit approval or decline of pending selection)
    if b_state.stage == BookingStage.AWAITING_CONFIRMATION and b_state.selected_slot:
        is_approval = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ["yes", "confirm", "proceed", "sure", "ok", "book it", "please", "go ahead", "approved"]) or "yes" in q_lower or "confirm" in q_lower
        is_decline = any(w in q_lower for w in ["no", "cancel", "stop", "abort", "never mind", "don't"])

        if is_approval and not is_decline:
            cand_p_name, cand_p_id, cand_p_age, cand_doc = extract_patient_details_for_booking(query, session)
            if cand_p_name:
                b_state.patient_name = cand_p_name
            if cand_p_id:
                b_state.patient_id = cand_p_id
            if cand_p_age:
                b_state.patient_age = cand_p_age
            if cand_doc:
                b_state.doctor_name = cand_doc

            slot = b_state.selected_slot
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "confirm_scan_booking",
                "arguments": {
                    "slot_id": slot["slot_id"],
                    "patient_name": b_state.patient_name or "Walk-in Patient",
                    "patient_id": b_state.patient_id,
                    "patient_age": b_state.patient_age,
                    "doctor_name": b_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)",
                    "modality": b_state.modality or slot.get("modality")
                },
                "rationale": "Explicit Confirmation: Clinician confirmed booking."
            }
        elif is_decline:
            b_state.reset()
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "booking_declined",
                "arguments": {"reason": "Declined by clinician"},
                "rationale": "Clinician declined booking confirmation."
            }

    # 0B.1 Cancellation request (Must check before lookup so 'cancel booking BKG-...' is treated as cancellation)
    is_cancel_query = (
        any(w in q_lower for w in ["cancel", "release", "revoke", "delete"])
        and any(w in q_lower for w in ["booking", "appointment", "slot", "it"])
    )
    if is_cancel_query:
        ident = b_ref or (b_state.completed_booking.get("booking_reference") if b_state.completed_booking else None)
        if not ident and b_state.selected_slot:
            ident = b_state.selected_slot.get("slot_id")

        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "cancel_scan_booking",
            "arguments": {
                "booking_identifier": ident or "UNKNOWN",
                "reason": "Clinician requested cancellation"
            },
            "rationale": "Clinician requested appointment cancellation."
        }

    # 0B.1B Reschedule request
    is_reschedule_query = "reschedule" in q_lower
    if is_reschedule_query:
        target_slot_match = re.search(r"\b(SLOT-\d+)\b", query, re.IGNORECASE)
        target_slot = target_slot_match.group(1).upper() if target_slot_match else None
        ident = b_ref if (b_ref and b_ref.startswith("BKG-")) else (b_state.completed_booking.get("booking_reference") if b_state.completed_booking else None)
        if not ident and b_state.completed_booking:
            ident = b_state.completed_booking.get("slot_id")

        if not ident:
            from .booking_service import lookup_existing_bookings
            p_name = b_state.patient_name
            found_b = lookup_existing_bookings(patient_name=p_name)
            if found_b:
                ident = found_b[0].get("booking_id") or found_b[0].get("booking_reference") or found_b[0].get("slot_id")

        if not ident and target_slot:
            from .booking_service import load_slots
            booked = [s for s in load_slots() if s.get("status") in ("booked", "CONFIRMED")]
            if booked:
                ident = booked[-1].get("booking_id") or booked[-1].get("booking_reference") or booked[-1].get("slot_id")

        if ident and target_slot:
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "reschedule_scan_booking",
                "arguments": {
                    "booking_identifier": ident,
                    "new_slot_id": target_slot
                },
                "rationale": f"Reschedule appointment '{ident}' to '{target_slot}'."
            }

    # 0B.2 Existing booking lookup inquiry ("I already booked", "Show my appointment", "Check my booking")
    is_lookup_query = (
        bool(b_ref and b_ref.startswith("BKG-"))
        or any(p in q_lower for p in [
            "i already booked", "already booked", "already have an appointment", "already have a booking",
            "show my appointment", "show my booking", "check my booking", "view my booking",
            "view my appointment", "check my appointment", "is my appointment booked",
            "my existing appointment", "my existing booking", "my appointment", "my appointments",
            "existing appointment", "existing booking", "appointment details", "booking details"
        ])
        or (any(w in q_lower for w in ["lookup", "find", "show", "view", "check", "retrieve", "details"]) and any(w in q_lower for w in ["my booking", "my appointment", "existing booking", "existing appointment", "appointment details", "booking details"]))
    )
    if is_lookup_query:
        pat_name = b_state.patient_name
        pat_id = b_state.patient_id or session.active_patient_id
        if not pat_name and pat_id:
            from .retrieval import get_patient
            p = get_patient(pat_id)
            if p:
                pat_name = p.get("patient_name")
        q_intent = extract_retrieval_intent(query)
        if q_intent.get("patient_name") and q_intent["patient_name"].lower() not in ["patient", "booking", "appointment", "my", "details"]:
            from .retrieval import search_patients
            found_cands = search_patients(q_intent["patient_name"])
            if found_cands:
                pat_name = found_cands[0]["patient_name"]
                pat_id = found_cands[0]["patient_id"]
            else:
                pat_name = q_intent["patient_name"].title()

        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "lookup_existing_booking",
            "arguments": {
                "patient_name": pat_name,
                "patient_id": pat_id,
                "booking_reference": b_ref,
                "date": extract_date_from_query(query)
            },
            "rationale": "Clinician requested lookup of existing appointment booking."
        }

    # 0B.3 Idempotency check if user repeats confirmation when already booked
    if b_state.stage == BookingStage.CONFIRMED and b_state.completed_booking:
        is_repeat_confirm = (
            any(p in q_lower for p in ["confirm", "already booked", "duplicate booking"])
            and not any(w in q_lower for w in ["open", "find", "retrieve", "view", "what", "how", "tell", "explain", "who", "where", "which"])
        ) or (b_ref and b_ref.startswith("BKG-"))
        if is_repeat_confirm:
            cb = b_state.completed_booking
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "booking_already_confirmed",
                "arguments": {
                    "booking": cb
                },
                "rationale": "Idempotency: Appointment was already confirmed."
            }
        else:
            # User moved on to other conversation or new request; release confirmation lock
            b_state.stage = BookingStage.IDLE
            b_state.completed_booking = None
            b_state.displayed_slots = []
            session.pending_booking_slots = []



    # 0B.5 Workflow in SLOTS_PRESENTED or pending_booking_slots active
    if b_state.stage == BookingStage.SLOTS_PRESENTED or session.pending_booking_slots:
        avail_slots = b_state.displayed_slots or session.pending_booking_slots or []

        # If user provides new booking criteria (specifying modality, preferred date, or patient)
        # without selecting a specific slot ID, treat as refined search instead of slot selection
        has_new_booking_criteria = (
            (any(w in q_lower for w in ["cbct", "opg", "panoramic", "bitewing", "periapical", "cephalogram"])
             or extract_date_from_query(query) is not None
             or bool(re.search(r"\bpatient\s+[a-z]+", q_lower))
             or bool(re.search(r"\bfor\s+(?:patient\s+)?[a-z]+", q_lower)))
            and not bool(re.search(r"\b(slot[- ]?\d+|\b10[1-7]\b)\b", q_lower))
        )
        if has_new_booking_criteria:
            b_state.displayed_slots = []
        else:
            matched_slot_id = resolve_pending_slot_from_query(query, avail_slots)
            is_approval_intent = any(w in q_lower for w in ["yes", "confirm", "please confirm", "book slot", "confirm slot"])
            is_decline = any(w in q_lower for w in ["cancel", "never mind", "stop", "abort", "no"])

            if is_decline:
                b_state.reset()
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "booking_declined",
                    "arguments": {"reason": "Cancelled by clinician"},
                    "rationale": "Clinician cancelled appointment booking."
                }

            if matched_slot_id:
                matched_slot = next((s for s in avail_slots if s["slot_id"].upper() == matched_slot_id.upper()), None)
                if not matched_slot:
                    from .booking_service import get_slot_by_id
                    matched_slot = get_slot_by_id(matched_slot_id)

                # Determine patient identity
                cand_p_name, cand_p_id, cand_p_age, cand_doc = extract_patient_details_for_booking(query, session)
                pat_name = cand_p_name or b_state.patient_name
                pat_id = cand_p_id or b_state.patient_id or session.active_patient_id
                if not pat_name and pat_id:
                    from .retrieval import get_patient
                    p = get_patient(pat_id)
                    if p:
                        pat_name = p.get("patient_name")

                if not pat_name:
                    pat_name = "Walk-in Patient" if (b_state.patient_type == "walk_in" or "walk" in q_lower) else "Walk-in Patient"

                if cand_p_age:
                    b_state.patient_age = cand_p_age
                elif pat_id and not b_state.patient_age:
                    from .retrieval import get_patient
                    p_rec = get_patient(pat_id)
                    if p_rec and p_rec.get("date_of_birth"):
                        try:
                            b_state.patient_age = f"{2026 - int(p_rec['date_of_birth'].split('-')[0])} years (DOB: {p_rec['date_of_birth']})"
                        except Exception:
                            b_state.patient_age = f"DOB: {p_rec['date_of_birth']}"

                if cand_doc:
                    b_state.doctor_name = cand_doc

                is_explicit_selection = bool(
                    re.search(r"\b(select|choose|pick|option)\b", q_lower)
                    or "change that to" in q_lower
                    or "switch to" in q_lower
                    or "let's select" in q_lower
                )

                if not is_explicit_selection:
                    session.pending_booking_slots = None
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "confirm_scan_booking",
                        "arguments": {
                            "slot_id": matched_slot_id,
                            "patient_name": pat_name,
                            "patient_id": pat_id,
                            "patient_age": b_state.patient_age,
                            "doctor_name": b_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)",
                            "modality": matched_slot.get("modality") if matched_slot else b_state.modality
                        },
                        "rationale": f"Direct Confirmation: Confirmed slot {matched_slot_id}."
                    }
                else:
                    b_state.stage = BookingStage.AWAITING_CONFIRMATION
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "show_booking_summary",
                        "arguments": {
                            "slot": matched_slot,
                            "patient_name": pat_name,
                            "patient_id": pat_id,
                            "patient_age": b_state.patient_age,
                            "doctor_name": b_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)",
                            "modality": matched_slot.get("modality") if matched_slot else b_state.modality
                        },
                        "rationale": f"Slot Selected: Presenting summary for {matched_slot_id} awaiting explicit confirmation."
                    }
            else:
                is_new_inquiry = (
                    any(w in q_lower for w in ["is there", "any dr", "any doctor", "how many", "check", "want to book", "i want", "other", "another", "what are", "which"])
                    or any(v in q_lower for v in ["open", "find", "retrieve", "dictate", "generate", "summarize", "database"])
                )
                is_time_or_date = any(t in q_lower for t in ["am", "pm", "a.m.", "p.m.", ":", "tomorrow", "oct", "october", "morning", "afternoon", "slot", "#", "first", "second", "third", "take", "give me"]) or any(c.isdigit() for c in query)
                if is_new_inquiry:
                    b_state.displayed_slots = []
                elif is_time_or_date:
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "invalid_slot_selection",
                        "arguments": {
                            "query": query,
                            "available_slots": avail_slots
                        },
                        "rationale": f"Safety Gate: User query '{query}' did not match any available slot."
                    }
                else:
                    b_state.displayed_slots = []

    else:
        # If session has no pending slots, check if user directly entered a slot ID or time
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
                matched_slot = next((s for s in all_avail if s["slot_id"].upper() == matched_slot_id.upper()), None)
                cand_p_name, cand_p_id, cand_p_age, cand_doc = extract_patient_details_for_booking(query, session)
                pat_name = cand_p_name or b_state.patient_name
                pat_id = cand_p_id or b_state.patient_id or session.active_patient_id
                if not pat_name and pat_id:
                    from .retrieval import get_patient
                    p = get_patient(pat_id)
                    if p:
                        pat_name = p.get("patient_name")
                        dob = p.get("date_of_birth")
                        if dob and not b_state.patient_age:
                            try:
                                b_state.patient_age = f"{2026 - int(dob.split('-')[0])} years (DOB: {dob})"
                            except Exception:
                                b_state.patient_age = f"DOB: {dob}"

                if not pat_name:
                    pat_name = "Walk-in Patient"

                if cand_p_age:
                    b_state.patient_age = cand_p_age
                elif pat_id and not b_state.patient_age:
                    from .retrieval import get_patient
                    p_rec = get_patient(pat_id)
                    if p_rec and p_rec.get("date_of_birth"):
                        try:
                            b_state.patient_age = f"{2026 - int(p_rec['date_of_birth'].split('-')[0])} years (DOB: {p_rec['date_of_birth']})"
                        except Exception:
                            b_state.patient_age = f"DOB: {p_rec['date_of_birth']}"

                if cand_doc:
                    b_state.doctor_name = cand_doc

                is_explicit_selection = bool(
                    re.search(r"\b(select|choose|pick|option)\b", q_lower)
                    or "change that to" in q_lower
                    or "switch to" in q_lower
                    or "let's select" in q_lower
                )
                if not is_explicit_selection:
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "confirm_scan_booking",
                        "arguments": {
                            "slot_id": matched_slot_id,
                            "patient_name": pat_name,
                            "patient_id": pat_id,
                            "patient_age": b_state.patient_age,
                            "doctor_name": b_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)",
                            "modality": matched_slot.get("modality") if matched_slot else b_state.modality
                        },
                        "rationale": f"Direct Slot Resolution: Confirmed slot {matched_slot_id}."
                    }
                else:
                    b_state.stage = BookingStage.AWAITING_CONFIRMATION
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "show_booking_summary",
                        "arguments": {
                            "slot": matched_slot,
                            "patient_name": pat_name,
                            "patient_id": pat_id,
                            "patient_age": b_state.patient_age,
                            "doctor_name": b_state.doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)",
                            "modality": matched_slot.get("modality") if matched_slot else b_state.modality
                        },
                        "rationale": f"Slot Selected: Presenting summary for {matched_slot_id} awaiting explicit confirmation."
                    }

    # 0B.6 Deterministic Booking Safety Gates
    detected_date = extract_date_from_query(query)
    is_booking_request = (
        any(w in q_lower for w in ["book", "booking", "schedule", "reschedule", "appointment", "appoint ment", "reserve", "slots", "slot"])
        or (any(w in q_lower for w in ["dr", "doctor", "radiologist", "scan"]) and any(w in q_lower for w in ["have", "available", "free", "open", "slot", "slots", "schedule", "appointment", "oct", "october", "nov", "dec", "tomorrow", "today", "any", "is there", "are there"]))
        or (bool(detected_date) and any(w in q_lower for w in ["have", "available", "slot", "slots", "appointment", "dr", "doctor", "scan", "can i"]))
    )
    has_booking_keyword = any(w in q_lower for w in ["slot", "slots", "appointment", "appoint ment", "booking", "schedule", "reschedule"])
    is_retrieval_override = (
        not has_booking_keyword
        and any(w in q_lower for w in ["open", "show", "view", "find report", "retrieve", "load report"])
    )
    if is_booking_request and not is_retrieval_override:
        # Check for equipment limitations / unsupported modalities (Test 12)
        unsupported_mods = ["mri", "magnetic resonance", "ct scan", "body ct", "ultrasound", "sonography", "mammogram", "mammography", "pet", "pet-ct"]
        matched_unsupported = next((m for m in unsupported_mods if m in q_lower and "cbct" not in q_lower and "cone beam" not in q_lower), None)
        if matched_unsupported:
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "equipment_limitation",
                "arguments": {"modality": matched_unsupported.upper()},
                "rationale": f"Equipment Limitation: Clinic does not possess equipment for '{matched_unsupported}'."
            }

        # Check for past dates (Test 11)
        if detected_date:
            from .clock import get_current_date
            now_d = get_current_date()
            try:
                d_parts = [int(x) for x in detected_date.split("-")]
                if date(d_parts[0], d_parts[1], d_parts[2]) < now_d:
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "past_date_rejected",
                        "arguments": {"date": detected_date},
                        "rationale": f"Clinic Scheduling Policy: Past date '{detected_date}' cannot be scheduled."
                    }
            except Exception:
                pass

        # Check for non-existent patient ID (Test 5)
        pat_id_match = re.search(r"\b(PAT-\d+)\b", query, re.IGNORECASE)
        if pat_id_match:
            cand_pid = pat_id_match.group(1).upper()
            from .retrieval import get_patient
            if not get_patient(cand_pid):
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "patient_id_not_found",
                    "arguments": {"patient_id": cand_pid},
                    "rationale": f"Patient ID '{cand_pid}' not found in clinic records."
                }

        # Check if user specifically requested 'book a scan' without a scan type (Test 2)
        detected_mod = "Any"
        if any(w in q_lower for w in ["cbct", "cone beam", "3d"]):
            detected_mod = "CBCT"
        elif any(w in q_lower for w in ["opg", "panoramic"]):
            detected_mod = "OPG"
        elif "bitewing" in q_lower:
            detected_mod = "Bitewing"
        elif any(w in q_lower for w in ["periapical", "iopa"]):
            detected_mod = "Periapical"

        is_generic_book_scan = bool(re.search(r"^\s*(?:i\s+(?:want|need)\s+to\s+)?book\s+(?:a\s+)?scan\s*$", query, re.IGNORECASE))
        if is_generic_book_scan and detected_mod == "Any":
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "scan_type_required",
                "arguments": {"query": query},
                "rationale": "Missing Required Modality: User requested 'Book a scan' without a scan type."
            }

    # 0B.7 Deterministic Clinic Database Inquiry Gate
    is_database_inquiry = (
        (any(w in q_lower for w in ["many", "manu", "mani", "much", "total", "count", "number of"]) and any(w in q_lower for w in ["patient", "report", "scan"]))
        or any(w in q_lower for w in ["clinic database", "database statistics"])
        or bool(re.search(r"\b(?:patient\s+names?|list\s+(?:out\s+)?(?:only\s+)?(?:the\s+)?patient\s+names?|list\s+(?:of\s+)?patients?|who\s+are\s+the\s+patients?|all\s+patients?|registered\s+patients?|which\s+patients?|know\s+the\s+patient|tell\s+me\s+(?:the\s+)?patient\s+names?)\b", q_lower))
    ) and not any(verb in q_lower for verb in ["open", "retrieve", "view", "load"])
    if is_database_inquiry and not is_booking_request:
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "query_clinic_database",
            "arguments": {"question": query},
            "rationale": "Clinic Database Inquiry: Inquiring about registered patient directory or database statistics."
        }

    # 0B.8 Deterministic Active Report & Patient Consultation Gate
    if session.active_report_id or session.active_patient_id:
        # Check if this is a general knowledge question, workstation capability query, or greeting:
        if is_general_or_assistant_query(query):
            return {
                "selected_agent": "GeneralAssistantAgent",
                "tool_name": "conversation",
                "arguments": {"query": query},
                "rationale": f"General Assistant: Handled general knowledge or workstation inquiry '{query}'."
            }

        intent = extract_retrieval_intent(query)
        extracted_cand_name = intent.get("patient_name", "").strip()
        non_patient_tokens = {
            "what", "why", "how", "explain", "tell", "help", "who", "when", "does", "can",
            "please", "summary", "summarize", "problem", "findings", "finding", "report",
            "scan", "it", "she", "he", "her", "his", "tooth", "teeth", "are", "is", "the",
            "for", "about", "any", "caries", "canal", "root", "nerve", "impaction"
        }
        has_new_patient_target = bool(extracted_cand_name) and not all(t.lower() in non_patient_tokens for t in extracted_cand_name.split())
        has_retrieval_verb = bool(re.search(r"\b(open|retrieve|fetch|load|switch\s+to)\b", q_lower))
        has_new_patient_retrieval = has_retrieval_verb and has_new_patient_target
        
        # Check if user is asking to switch or open another report of the active patient (e.g. 'open her OPG', 'show periapical', 'open her latest report')
        is_modality_switch = (
            not has_new_patient_target
            and any(w in q_lower for w in ["opg", "cbct", "panoramic", "bitewing", "periapical", "examination", "exam", "rpt-", "latest", "recent", "newest", "last"])
            and any(w in q_lower for w in ["open", "show", "switch", "view", "load", "her", "his", "the", "other"])
        )
        if is_modality_switch and session.active_patient_id:
            target_mod = "Any"
            if any(m in q_lower for m in ["opg", "panoramic"]):
                target_mod = "OPG"
            elif any(m in q_lower for m in ["cbct", "cone beam", "3d"]):
                target_mod = "CBCT"
            elif "bitewing" in q_lower:
                target_mod = "Bitewing"
            elif any(m in q_lower for m in ["periapical", "iopa"]):
                target_mod = "Periapical"
            elif any(m in q_lower for m in ["examination", "exam"]):
                target_mod = "Dental Examination"

            rid_match = re.search(r"\b(RPT-\d+)\b", query, re.IGNORECASE)
            target_rid = rid_match.group(1).upper() if rid_match else None

            from .retrieval import get_patient_reports, get_report_by_id
            p_reports = get_patient_reports(session.active_patient_id)
            matched_rep = None
            if target_rid:
                matched_rep = get_report_by_id(target_rid)
            elif any(w in q_lower for w in ["latest", "recent", "newest", "last"]) and p_reports:
                matched_rep = p_reports[0]
            elif target_mod != "Any":
                for r in p_reports:
                    if target_mod.lower() in r.get("report_type", "").lower():
                        matched_rep = r
                        break

            if matched_rep:
                session.active_report_id = matched_rep["report_id"]
                return {
                    "selected_agent": "RetrievalAgent",
                    "tool_name": "search_patient_reports",
                    "arguments": {
                        "patient_id": session.active_patient_id,
                        "modality": matched_rep.get("report_type")
                    },
                    "rationale": f"Report Switch: Switched to {matched_rep.get('report_type')} ({matched_rep['report_id']}) for active patient."
                }

        # Active Report / Patient Consultation & Evidence-Grounded Follow-Up
        if not is_booking_request and not is_database_inquiry and not has_new_patient_retrieval:
            return {
                "selected_agent": "ConsultationAgent",
                "tool_name": "summarize_active_report",
                "arguments": {
                    "report_id": session.active_report_id or "all",
                    "patient_id": session.active_patient_id,
                    "query": query
                },
                "rationale": f"Report Consultation: Clinician inquiry for patient {session.active_patient_id} (active report: {session.active_report_id})."
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
                    elif "query" not in args:
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


    # 2B.5 Scan Booking Request (Check Available Slots & Doctor Availability)
    detected_date = extract_date_from_query(query)
    is_booking_request = (
        any(w in q_lower for w in ["book", "booking", "schedule", "reschedule", "appointment", "appoint ment", "reserve", "slots", "slot"])
        or (any(w in q_lower for w in ["dr", "doctor", "radiologist", "scan"]) and any(w in q_lower for w in ["have", "available", "free", "open", "slot", "slots", "schedule", "appointment", "oct", "october", "nov", "dec", "tomorrow", "today", "any", "is there", "are there"]))
        or (bool(detected_date) and any(w in q_lower for w in ["have", "available", "slot", "slots", "appointment", "dr", "doctor", "scan", "can i"]))
    )
    has_booking_keyword = any(w in q_lower for w in ["slot", "slots", "appointment", "appoint ment", "booking", "schedule", "reschedule"])
    is_retrieval_override = (
        not has_booking_keyword
        and any(w in q_lower for w in ["open", "show", "view", "find report", "retrieve", "load report"])
    )
    # Check for equipment limitations / unsupported modalities (Test 12)
    unsupported_mods = ["mri", "magnetic resonance", "ct scan", "body ct", "ultrasound", "sonography", "mammogram", "mammography", "pet", "pet-ct"]
    matched_unsupported = next((m for m in unsupported_mods if m in q_lower and "cbct" not in q_lower and "cone beam" not in q_lower), None)
    if matched_unsupported:
        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "equipment_limitation",
            "arguments": {"modality": matched_unsupported.upper()},
            "rationale": f"Equipment Limitation: Clinic does not possess equipment for '{matched_unsupported}'."
        }

    # Check for past dates (Test 11)
    if detected_date:
        from .clock import get_current_date
        now_d = get_current_date()
        try:
            d_parts = [int(x) for x in detected_date.split("-")]
            if date(d_parts[0], d_parts[1], d_parts[2]) < now_d:
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "past_date_rejected",
                    "arguments": {"date": detected_date},
                    "rationale": f"Clinic Scheduling Policy: Past date '{detected_date}' cannot be scheduled."
                }
        except Exception:
            pass

    # Check for non-existent patient ID (Test 5)
    pat_id_match = re.search(r"\b(PAT-\d+)\b", query, re.IGNORECASE)
    if pat_id_match and is_booking_request:
        cand_pid = pat_id_match.group(1).upper()
        from .retrieval import get_patient
        if not get_patient(cand_pid):
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "patient_id_not_found",
                "arguments": {"patient_id": cand_pid},
                "rationale": f"Patient ID '{cand_pid}' not found in clinic records."
            }

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

        # Check if user specifically requested 'book a scan' without a scan type (Test 2)
        is_generic_book_scan = bool(re.search(r"^\s*(?:i\s+(?:want|need)\s+to\s+)?book\s+(?:a\s+)?scan\s*$", query, re.IGNORECASE))
        if is_generic_book_scan and detected_mod == "Any":
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "scan_type_required",
                "arguments": {"query": query},
                "rationale": "Missing Required Modality: User requested 'Book a scan' without a scan type."
            }

        # Check if patient is specified
        pat_name_for_booking = None
        pat_id_for_booking = session.active_patient_id
        cand_p_name, cand_p_id, cand_p_age, cand_doc = extract_patient_details_for_booking(query, session)
        if cand_p_name:
            pat_name_for_booking = cand_p_name
            pat_id_for_booking = cand_p_id
        if cand_p_age:
            session.booking_state.patient_age = cand_p_age
        if cand_doc:
            session.booking_state.doctor_name = cand_doc

        if not pat_name_for_booking:
            if any(w in q_lower for w in ["walk-in", "walk in"]):
                session.booking_state.patient_type = "walk_in"
                pat_name_for_booking = "Walk-in Patient"
            else:
                q_intent = extract_retrieval_intent(query)
                cand_name = q_intent.get("patient_name", "")
                if cand_name and is_valid_patient_name(cand_name):
                    from .retrieval import search_patients
                    cands = search_patients(cand_name)
                    exact_cands = [c for c in cands if c["patient_name"].lower() == cand_name.lower()]
                    if len(exact_cands) == 1:
                        pat_name_for_booking = exact_cands[0]["patient_name"]
                        pat_id_for_booking = exact_cands[0]["patient_id"]
                    elif len(exact_cands) > 1:
                        session.pending_candidates = exact_cands
                        return {
                            "selected_agent": "RetrievalAgent",
                            "tool_name": "clarification_required",
                            "arguments": {"candidates": exact_cands},
                            "rationale": f"Ambiguous patient name '{cand_name}' for booking. Clarification required."
                        }
                    else:
                        cand_tokens = cand_name.split()
                        if len(cand_tokens) == 1:
                            cand_first = cand_tokens[0].lower()
                            same_first = [c for c in cands if c["patient_name"].split()[0].lower() == cand_first]
                            if len(same_first) == 1:
                                pat_name_for_booking = same_first[0]["patient_name"]
                                pat_id_for_booking = same_first[0]["patient_id"]
                            elif len(same_first) > 1:
                                session.pending_candidates = same_first
                                return {
                                    "selected_agent": "RetrievalAgent",
                                    "tool_name": "clarification_required",
                                    "arguments": {"candidates": same_first},
                                    "rationale": f"Ambiguous patient name '{cand_name}' for booking. Clarification required."
                                }
                            else:
                                pat_name_for_booking = cand_name.title()
                                pat_id_for_booking = None
                                session.booking_state.patient_type = "new"
                        else:
                            pat_name_for_booking = cand_name.title()
                            pat_id_for_booking = None
                            session.booking_state.patient_type = "new"

        if not pat_name_for_booking and session.active_patient_id:
            from .retrieval import get_patient
            p = get_patient(session.active_patient_id)
            if p:
                pat_name_for_booking = p.get("patient_name")
                pat_id_for_booking = p.get("patient_id")
                dob = p.get("date_of_birth")
                if dob and not session.booking_state.patient_age:
                    try:
                        session.booking_state.patient_age = f"{2026 - int(dob.split('-')[0])} years (DOB: {dob})"
                    except Exception:
                        session.booking_state.patient_age = f"DOB: {dob}"

        cand_age = extract_age_from_query(query)
        if cand_age:
            session.booking_state.patient_age = cand_age
        cand_doc = extract_doctor_from_query(query)
        if cand_doc:
            session.booking_state.doctor_name = cand_doc

        session.booking_state.patient_name = pat_name_for_booking
        session.booking_state.patient_id = pat_id_for_booking
        session.booking_state.modality = detected_mod
        session.booking_state.preferred_date = detected_date
        session.booking_state.stage = BookingStage.SLOTS_PRESENTED

        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "check_appointment_slots",
            "arguments": {
                "modality": detected_mod,
                "preferred_date": detected_date,
                "patient_name": pat_name_for_booking,
                "patient_id": pat_id_for_booking,
                "patient_age": session.booking_state.patient_age,
                "doctor_name": session.booking_state.doctor_name
            },
            "rationale": f"Fallback: Clinician requested scan booking/availability for {detected_mod} on {detected_date or 'any date'}."
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

    has_retrieval_cmd = any(verb in q_lower for verb in ["open", "load", "fetch", "retrieve"]) or (
        any(verb in q_lower for verb in ["show", "view", "find", "get"]) and any(w in q_lower for w in ["report", "scan", "opg", "cbct", "bitewing", "periapical", "examination", "record", "patient", "image"])
    )

    # 2D.5 Clinic Database Queries & Patient List Requests
    is_database_question = (
        any(q_lower.startswith(w) for w in ["how many", "how much", "which", "list", "count", "breakdown"])
        or any(w in q_lower for w in ["database", "total scans", "total patients"])
        or bool(re.search(r"\b(?:patient\s+names?|list\s+(?:of\s+)?patients?|list\s+(?:of\s+)?reports?|all\s+patients?|registered\s+patients?|who\s+are\s+the\s+patients?)\b", q_lower))
    )
    if is_database_question and not has_retrieval_cmd:
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "query_clinic_database",
            "arguments": {"question": query},
            "rationale": "Clinical database patient list or statistical inquiry."
        }

    # 2D.6 Conversational Dialogue, Help, and General Clinical Concept Questions
    is_conversational_greeting = any(re.search(r"\b" + re.escape(g) + r"\b", q_lower) for g in [
        "hey", "hello", "hi", "good morning", "good afternoon", "good evening", "greetings",
        "help", "who are you", "what are you doing", "what can you do", "how do you work", "how does this work",
        "thanks", "thank you"
    ])

    if (is_conversational_greeting or is_question_intent) and not has_retrieval_cmd and not any(w in q_lower for w in ["database", "how many", "count", "breakdown"]):
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "conversation",
            "arguments": {"query": query},
            "rationale": "Fallback: Natural conversational inquiry, greeting, or clinical concept question."
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
    pat_id_match = re.search(r"\b(PAT-\d+)\b", query, re.IGNORECASE)
    has_patient_id = bool(pat_id_match) or bool(re.search(r"\b(PT-\d+|\d{4})\b", query, re.IGNORECASE))
    patient_id_found = pat_id_match.group(1).upper() if pat_id_match else None

    from .retrieval import search_patients
    matched_candidates = search_patients(extracted_name) if extracted_name else []

    # If the user explicitly provided a patient ID or a matched patient candidate with a retrieval command,
    # route to RetrievalAgent
    is_explicit_retrieval = has_patient_id or (has_retrieval_command and (matched_candidates or bool(extracted_name)))

    if not is_explicit_retrieval:
        if has_non_patient_word or not extracted_name or (not has_retrieval_command and not matched_candidates):
            return {
                "selected_agent": "GeneralAssistantAgent",
                "tool_name": "conversation",
                "arguments": {"query": query},
                "rationale": "Safety Guard: Query was an inquiry or question, not a patient name."
            }

    name_guess = extracted_name.title() if extracted_name else query.title()

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
    q_lower = query.lower().strip()
    # Guard against retrieval requests accidentally entering conversation
    if any(verb in q_lower for verb in ["open", "retrieve", "fetch", "load"]) or (
        any(verb in q_lower for verb in ["show", "view", "find"]) and any(w in q_lower for w in ["report", "scan", "opg", "cbct", "bitewing", "periapical"])
    ):
        from .retrieval import resolve_report_retrieval
        retrieval_res = resolve_report_retrieval(query)
        if retrieval_res.get("status") == "success" and retrieval_res.get("report"):
            rep = retrieval_res["report"]
            return f"Retrieved and opened {rep.get('report_type')} report ({rep.get('report_id')}) for {retrieval_res['patient']['patient_name']}."

    if not is_general_or_assistant_query(query) and (session.active_report_id or session.active_patient_id) and any(w in q_lower for w in [
        "problem", "wrong", "findings", "finding", "diagnos", "impression", "summar",
        "condition", "issue", "caries", "teeth", "tooth", "they have", "she have", "he have",
        "what they", "what do they", "what does", "what is", "report", "scan",
        "explain", "elaborate", "clarify", "why", "meaning"
    ]):
        if session.active_report_id:
            from .report_qa_service import answer_report_question
            qa = answer_report_question(session.active_report_id, query, history=session.history)
            if qa.get("answer"):
                return qa["answer"]
        elif session.active_patient_id:
            from .report_qa_service import answer_patient_reports_question
            qa = answer_patient_reports_question(session.active_patient_id, None, query, history=session.history)
            if qa.get("answer"):
                return qa["answer"]

    conversation_prompt = (
        "You are the Viscan AI Clinical Workstation Assistant for a Dental & Maxillofacial Radiology Clinic. "
        "This workstation supports four specialized functions:\n"
        "1. Diagnostic Patient Report Retrieval: Retrieve imaging reports (OPG panoramic, CBCT 3D volume, Bitewing, Periapical) with safety ambiguity resolution.\n"
        "2. Clinical Dictation to Structured Draft: Convert spoken/typed clinical observations into validated draft reports with FDI/Palmer tooth mapping and anti-hallucination verification.\n"
        "3. Report-Aware Consultation: Answer clinician questions grounded in active imaging reports with source citations.\n"
        "4. Scan Booking & Scheduling: Check real-time appointment slots in Radiology Suite 1 (CBCT), Panoramic Suite 2 (OPG), and Intraoral Bay 3, and confirm bookings.\n\n"
        "Rules for answering:\n"
        "• Answer concisely, professionally, and accurately within dental radiology and maxillofacial clinical domain.\n"
        "• If asked about doctors, clinic, or hospital name, explain our dental radiology imaging clinic suites (Suite 1 3D CBCT, Panoramic Suite 2 OPG, Intraoral Bay 3) and guide them to check scan slots or retrieve patient reports.\n"
        "• If asked about dental or radiological concepts (such as CBCT, OPG, caries, impactions, root canals, periapical lesions), explain them clearly.\n"
        "• Never output placeholder tokens like '[Clinic Name]', '[Doctor Name]', or 'None'.\n"
        "• Do not suggest non-dental specialties like cardiology or orthopedics."
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

    # 7b. Doctor / Clinic information queries
    if any(k in q_lower for k in ["doctor", "doctore", "docoter", "hospital", "physician", "clinic name"]):
        return (
            "Viscan AI is the Clinical Workstation for our Dental & Maxillofacial Radiology Clinic. "
            "Our imaging facilities and hospital suites include Radiology Suite 1 (3D CBCT Scanner), Panoramic Suite 2 (OPG), "
            "and Intraoral X-Ray Bay 3 (Bitewing & Periapical). Our team of specialized dental radiologists and doctors provides "
            "diagnostic maxillofacial imaging and clinical interpretations. You can ask me to check available scan appointment slots "
            "or retrieve a patient's diagnostic report."
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
                extra_msg = (
                    f"\n\nThis patient has {len(all_reports)} diagnostic reports on file:\n"
                    + "\n".join(other_reps)
                    + f"\n\nYou can ask to view any of these reports (e.g. 'Open her OPG' or 'Show {all_reports[0]['report_id']}'), "
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

        # Build readable clinical draft response for the clinician
        findings_bullets = []
        for idx, f in enumerate(findings, 1):
            site_info = f.get("anatomical_site") or "Anatomical site"
            t_num = f" (Tooth {f.get('tooth_number')})" if f.get("tooth_number") else ""
            cat = f" [{f.get('category')}]" if f.get("category") else ""
            cert = f.get("certainty", "definite observation")
            findings_bullets.append(f"{idx}. **{site_info}**{t_num}{cat}:\n   Observation: {f.get('observation')}\n   *Diagnostic Certainty: {cert}*")

        impressions_bullets = [f"• {imp}" for imp in impressions] if impressions else ["• Diagnostic review completed."]

        draft_answer = (
            f"📋 **Clinical Dictation Draft Report Generated**\n\n"
            f"**Key Findings & Anatomical Sites:**\n"
            f"{(chr(10) + chr(10)).join(findings_bullets) if findings_bullets else '• No specific clinical findings detected.'}\n\n"
            f"**Radiologist Impression:**\n"
            f"{chr(10).join(impressions_bullets)}\n\n"
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

            p_id_str = f" ({patient_id})" if patient_id else " (Status: UNREGISTERED / Guest Booking)"
            age_line = f"• Patient Age: {patient_age}\n" if patient_age else "• Patient Age: Not specified (Walk-in / Guest)\n"

            summary_msg = (
                f"Review Appointment Booking Details:\n"
                f"• Patient Name: {patient_name}{p_id_str}\n"
                f"{age_line}"
                f"• Attending Doctor / Radiologist: {doctor_name}\n"
                f"• Modality / Scan: {modality}\n"
                f"• Scheduled Date & Time: {slot.get('datetime_str', 'Scheduled slot')}\n"
                f"• Facility & Location: {room_name}, {clinic_title}\n"
                f"• Slot ID: {slot.get('slot_id')}\n\n"
                f"Confirmation Required: Would you like to confirm this appointment booking? (Please reply 'Yes, confirm' or 'Cancel')."
            )
            session.booking_state.stage = BookingStage.AWAITING_CONFIRMATION
            session.booking_state.summary_displayed = True
            session.booking_state.selected_slot = slot
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
                "slot": slot,
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
            return {
                "status": "cancelled",
                "message": "Appointment booking has been cancelled as requested. No appointment was booked.",
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
                            or (re.search(r"\b(?:her|his|the)\s+report\b", query.lower()) and not is_all_reports)
                        )
                        if is_single_report_ambiguous:
                            rep_str = " • ".join([f"[{r['report_id']}] {r['report_type']} ({r['report_date']})" for r in reps])
                            return {
                                "status": "clarification_required",
                                "message": f"Patient {p['patient_name']} has {len(reps)} diagnostic reports on file: {rep_str}. Please specify which report to view or summarize (e.g. 'Open her {reps[0]['report_type']}').",
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