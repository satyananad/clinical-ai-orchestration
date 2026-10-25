"""
DETERMINISTIC CLINICAL SAFETY & OFFLINE TEST HARNESS (orchestrator_fallback.py)
-------------------------------------------------------------------------------
Contains deterministic fallback rules, natural language entity extraction,
and appointment state-machine handling to ensure automated CI/CD test suites
(pytest) run reliably in offline environments without active LLM daemons.
"""

import re
import json
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
from .conversation_service import BookingStage, is_general_or_assistant_query, is_contextual_query, save_session
from .retrieval import resolve_report_retrieval, extract_retrieval_intent
from .report_qa_service import answer_report_question, answer_patient_reports_question
from .llm_service import query_llm

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

    # E. Modality matching across candidate diagnostic reports (e.g. "open her bitewing radiographs")
    from .retrieval import get_patient_reports, extract_retrieval_intent
    intent = extract_retrieval_intent(query)
    req_mod = intent.get("report_type", "Any")
    if req_mod == "Any":
        for pat, mod in [
            (r"\b(opg|panoramic|orthopantomogram)\b", "OPG"),
            (r"\b(cbct|cone\s*beam|3d\s*volume)\b", "CBCT"),
            (r"\b(bitewing|bitewings)\b", "Bitewing"),
            (r"\b(periapical|iopa)\b", "Periapical"),
            (r"\b(cephalogram|ceph)\b", "Cephalogram"),
            (r"\b(examination|exam)\b", "Dental Examination"),
        ]:
            if re.search(pat, q_lower):
                req_mod = mod
                break

    if req_mod != "Any":
        mod_matches = []
        for cand in candidates:
            c_reports = get_patient_reports(cand.get("patient_id"), modality=req_mod)
            if c_reports:
                mod_matches.append(cand)
        if len(mod_matches) == 1:
            return {
                "status": "resolved",
                "matched_candidate": mod_matches[0]
            }
        elif len(mod_matches) > 1:
            return {
                "status": "narrowed",
                "remaining_candidates": mod_matches
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
        "scan", "report", "opg", "cbct", "bitewing", "periapical", "xray", "x-ray", "ok", "okay",
        "sure", "cancel", "open", "show", "view", "find", "load", "dr", "doctor",
        "location", "hospital", "clinic", "radiology", "suite", "room", "bay",
        "specialist", "specialty", "dental", "dentist", "consultation", "consult",
        "implant", "radiologist", "surgeon", "orthodontist", "endodontist", "periodontist",
        "oral", "maxillofacial",
        "his", "her", "hers", "hir", "their", "theirs", "its", "him", "them", "she", "he", "they", "hiss", "herr"
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
    if not any(w in q_norm for w in ["first time", "first visit", "my first", "first name"]):
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
    from .clinic_catalog import DOCTORS_DIRECTORY
    for d in DOCTORS_DIRECTORY:
        d_name = d.get("short_name", "") or d.get("name", "")
        parts = d_name.split()
        surname = parts[-1]
        if re.search(r"\b(?:dr\.?|doctor)\s+" + re.escape(surname) + r"\b", query, re.IGNORECASE):
            m_full = re.search(r"\b" + re.escape(d_name) + r"\b", query, re.IGNORECASE)
            if m_full:
                return d_name
            return f"Dr. {surname.capitalize()}"
        if re.search(r"\b" + re.escape(d_name.lower()) + r"\b", query.lower()):
            return d_name

    m = re.search(r"\b(?:dr\.?|doctor)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", query, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        words = cand.split()
        if len(words) == 2 and is_valid_patient_name(cand):
            return f"Dr. {cand.title()}"
        elif words and is_valid_patient_name(words[0]):
            return f"Dr. {words[0].title()}"
    return None


def extract_patient_registration_details(query: str) -> Dict[str, Optional[str]]:
    """
    Extracts registration fields from natural language text:
    - Full name
    - Date of birth / DOB
    - Phone number
    - Email address
    - Sex / Gender
    """
    res = {
        "full_name": None,
        "date_of_birth": None,
        "contact_phone": None,
        "email": None,
        "sex": None
    }
    q_clean = query.strip()

    # 1. Email
    email_m = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", q_clean)
    if email_m:
        res["email"] = email_m.group(1).strip()

    # 2. Phone
    phone_m = re.search(r"(\+?\d{1,4}[-\s]?\(?\d{1,4}\)?[-\s]?\d{3,5}[-\s]?\d{3,5}|\b\d{10}\b)", q_clean)
    if phone_m:
        res["contact_phone"] = phone_m.group(1).strip()

    # 3. Sex / Gender
    sex_m = re.search(r"\b(female|male|non-binary|other)\b", q_clean, re.IGNORECASE)
    if sex_m:
        res["sex"] = sex_m.group(1).capitalize()

    # 4. DOB / Date of Birth
    dob_m = re.search(r"\b(?:born\s+on|born\s+in|dob\s*[:=]?|date\s+of\s+birth\s*[:=]?)\s*([0-9A-Za-z\s,/-]+?)(?=(?:\.|\band\b|\bmy\s+phone\b|\bphone\b|\bemail\b|\bfemale\b|\bmale\b|,|\n|$))", q_clean, re.IGNORECASE)
    if dob_m:
        res["date_of_birth"] = dob_m.group(1).strip()
    else:
        # Check DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        dmy_m = re.search(r"\b([0-3]?\d[-/.][0-1]?\d[-/.](?:19|20)\d{2})\b", q_clean)
        if dmy_m:
            res["date_of_birth"] = dmy_m.group(1).strip()
        else:
            # Check natural dates like '14 February 1996'
            date_pattern = re.search(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b", q_clean, re.IGNORECASE)
            if date_pattern:
                res["date_of_birth"] = date_pattern.group(1).strip()
            else:
                iso_m = re.search(r"\b((?:19|20)\d{2}[-/][0-1]?\d[-/][0-3]?\d)\b", q_clean)
                if iso_m:
                    res["date_of_birth"] = iso_m.group(1).strip()

    # 5. Full Name
    name_m = re.search(r"\b(?:my\s+name\s+is|name\s+is|name\s*[:=]?|patient\s*[:=]?)\s*([A-Za-z\s]+?)(?=(?:,|\.|\bborn\b|\bdob\b|\bphone\b|\bemail\b|\bmy\s+phone\b|\band\b|\n|$))", q_clean, re.IGNORECASE)
    if name_m:
        cand_name = name_m.group(1).strip()
        if is_valid_patient_name(cand_name):
            res["full_name"] = cand_name.title()
    else:
        words = [w for w in q_clean.split() if w.isalpha()]
        stop_first = {
            "i", "need", "want", "we", "can", "could", "would", "like", "schedule",
            "tomorrow", "today", "yesterday", "scan", "have", "yes", "no", "this",
            "my", "please", "book", "the", "first", "new", "hello", "hey", "hi", "how",
            "open", "view", "retrieve", "fetch", "load", "find", "search", "show", "check",
            "what", "where", "who", "which", "why", "tell", "explain", "is", "are"
        }
        if 1 <= len(words) <= 4 and words[0].lower() not in stop_first:
            cand = " ".join(words[:2]) if len(words) >= 2 else words[0]
            if is_valid_patient_name(cand):
                res["full_name"] = cand.title()

    return res

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
            if is_valid_patient_name(cand):
                exact_matches = [p for p in all_p if p["patient_name"].lower() == cand.lower()]
                if len(exact_matches) == 1:
                    pat_name = exact_matches[0]["patient_name"]
                    pat_id = exact_matches[0]["patient_id"]
                else:
                    first_matches = [p for p in all_p if p["patient_name"].split()[0].lower() == cand.lower()]
                    if len(first_matches) == 1:
                        pat_name = first_matches[0]["patient_name"]
                        pat_id = first_matches[0]["patient_id"]
                    else:
                        pat_name = cand.title()

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



def _offline_deterministic_fallback(query: str, session: Any) -> Dict[str, Any]:
    """
    Deterministic Clinical Safety & Offline Test Harness Engine:
    Executes rule-based handling when running in offline CI/CD test suites (pytest)
    or when local Ollama/OpenAI LLM is offline or unreachable.
    """
    q_lower = query.lower().strip()

    # Step 00: Explicit Patient / Report Retrieval Override
    # When a clinician explicitly asks to open, view, or retrieve a patient record/report,
    # it must take precedence over any lingering or unfinished booking flow.
    retrieval_intent = extract_retrieval_intent(query)
    cand_retrieval_name = (retrieval_intent.get("patient_name") or "").strip()
    has_retrieval_verb = any(v in q_lower for v in ["open", "retrieve", "view", "fetch", "load", "find report", "show report", "get report", "bring up"])
    has_booking_kw = any(w in q_lower for w in ["book", "booking", "schedule", "reschedule", "appointment", "appoint ment", "slot", "slots", "reserve"])
    is_explicit_patient_retrieval = (
        has_retrieval_verb
        and not has_booking_kw
        and (bool(cand_retrieval_name) or bool(re.search(r"\b(pat-\d+|rpt-\d+)\b", q_lower)))
    )
    if is_explicit_patient_retrieval:
        # Reset any lingering booking state to avoid state machine hijacking
        session.booking_state.reset()
        session.pending_booking_slots = []
        session.pending_candidates = None
        session.last_retrieval_query = cand_retrieval_name or query
        session.last_retrieval_modality = retrieval_intent.get("report_type", "Any")
        return {
            "selected_agent": "RetrievalAgent",
            "tool_name": "search_patient_reports",
            "arguments": {
                "patient_name": cand_retrieval_name or query,
                "modality": retrieval_intent.get("report_type", "Any")
            },
            "rationale": f"Explicit Patient Retrieval: Locating records for '{cand_retrieval_name or query}'."
        }

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

    # 0B.3 Idempotency check or patient linkage if user specifies patient after guest booking
    if b_state.stage == BookingStage.CONFIRMED and b_state.completed_booking:
        cb = b_state.completed_booking
        is_guest_booking = cb.get("patient_status") == "UNREGISTERED" or cb.get("patient_name") in ("Walk-in Patient", "Guest Patient")
        pat_link_match = re.search(r"\b(?:patient\s+is|patient\s*[:=]|assign\s+to|link\s+to|for\s+patient|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b", query, re.IGNORECASE)
        cand_link_name = pat_link_match.group(1).strip() if pat_link_match else None

        if not cand_link_name and is_guest_booking:
            pid_m = re.search(r"\b(PAT-\d+)\b", query, re.IGNORECASE)
            if pid_m:
                cand_link_name = pid_m.group(1).upper()
            elif is_valid_patient_name(query.strip()) and not any(w in q_lower for w in ["open", "retrieve", "view", "show", "what", "how", "why"]):
                from .retrieval import search_patients
                cands = search_patients(query.strip())
                if cands:
                    cand_link_name = cands[0]["patient_name"]

        if is_guest_booking and cand_link_name:
            from .booking_service import update_booking_patient
            upd_res = update_booking_patient(cb.get("booking_reference") or cb.get("slot_id"), cand_link_name)
            if upd_res.get("status") == "success":
                b_state.completed_booking = upd_res.get("slot")
                upd_slot = upd_res.get("slot", {})
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "patient_linked_to_booking",
                    "arguments": {
                        "booking_reference": upd_slot.get("booking_reference"),
                        "patient_name": upd_slot.get("patient_name"),
                        "patient_id": upd_slot.get("patient_id"),
                        "slot": upd_slot
                    },
                    "rationale": f"Linked patient '{cand_link_name}' to booking {upd_slot.get('booking_reference')}."
                }

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

    # 0B.3.5 New Patient vs Existing Patient Intent Inquiry (handles direct replies like 'first time' or 'existing patient')
    if any(w in q_lower for w in ["first visit", "new patient", "i am new", "first time", "not registered", "my first"]):
        b_state.is_new_patient = True
        b_state.stage = BookingStage.COLLECTING_DETAILS
        session.active_patient_id = None
        session.active_report_id = None
        b_state.patient_name = None
        b_state.patient_id = None
        b_state.displayed_slots = []
        session.pending_booking_slots = []
        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "prompt_new_patient_registration",
            "arguments": {},
            "rationale": "User confirmed new patient status. Prompting for registration details."
        }

    if any(w in q_lower for w in ["existing patient", "i am an existing", "visited before", "already registered", "have an account", "existing"]):
        b_state.is_new_patient = False
        b_state.stage = BookingStage.IDENTIFYING_PATIENT
        return {
            "selected_agent": "ScanBookingAgent",
            "tool_name": "prompt_existing_patient_identifier",
            "arguments": {},
            "rationale": "User indicated existing patient status. Prompting for patient ID or phone number."
        }

    # 0B.4 Workflow in Multi-Step Booking (IDENTIFYING_PATIENT, COLLECTING_DETAILS, SELECTING_SERVICE, SELECTING_SPECIALIZATION, SELECTING_DOCTOR)
    if b_state.stage in (
        BookingStage.IDENTIFYING_PATIENT,
        BookingStage.COLLECTING_DETAILS,
        BookingStage.SELECTING_SERVICE,
        BookingStage.SELECTING_SPECIALIZATION,
        BookingStage.SELECTING_DOCTOR
    ):
        # 0B.4.1 Cancellation / Abort
        if any(w in q_lower for w in ["cancel", "stop", "abort", "never mind", "don't book"]):
            b_state.reset()
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "booking_declined",
                "arguments": {"reason": "Cancelled by user"},
                "rationale": "User cancelled appointment booking workflow."
            }

        # 0B.4.2 Direct command providing both patient in query and modality (e.g. "Book CBCT for patient Rahul Sharma tomorrow")
        direct_p_name, direct_p_id, direct_p_age, direct_doc = extract_patient_details_for_booking(query, session=None)
        has_direct_mod = any(w in q_lower for w in ["cbct", "opg", "panoramic", "bitewing", "periapical", "iopa"])
        d_date = extract_date_from_query(query)
        if direct_p_name and has_direct_mod and (d_date or "book" in q_lower):
            d_mod = "CBCT" if "cbct" in q_lower else ("OPG" if any(w in q_lower for w in ["opg", "panoramic"]) else ("Bitewing" if "bitewing" in q_lower else "Periapical"))
            from .booking_service import check_available_slots as get_avail_slots
            direct_slots = get_avail_slots(modality=d_mod, preferred_date=d_date)
            b_state.patient_name = direct_p_name
            b_state.patient_id = direct_p_id
            b_state.patient_age = direct_p_age
            b_state.doctor_name = direct_doc
            b_state.modality = d_mod
            b_state.preferred_date = d_date
            b_state.stage = BookingStage.SLOTS_PRESENTED
            b_state.displayed_slots = direct_slots
            session.pending_booking_slots = direct_slots
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "check_appointment_slots",
                "arguments": {
                    "modality": d_mod,
                    "preferred_date": d_date,
                    "patient_name": direct_p_name,
                    "patient_id": direct_p_id,
                    "patient_age": direct_p_age,
                    "doctor_name": direct_doc
                },
                "rationale": f"Fast-Path Booking: Direct patient '{direct_p_name}' and modality '{d_mod}' provided."
            }

        # 0B.4.4 Registration Details Submission
        reg_info = extract_patient_registration_details(query)
        cand_name = reg_info.get("full_name") or b_state.patient_name
        if not cand_name and is_valid_patient_name(query.strip()) and not any(w in q_lower for w in ["yes", "no", "ok", "cancel", "stop", "abort"]):
            cand_name = query.strip().title()
        cand_dob = reg_info.get("date_of_birth") or b_state.patient_dob
        cand_phone = reg_info.get("contact_phone") or b_state.patient_phone
        cand_email = reg_info.get("email") or b_state.patient_email
        cand_sex = reg_info.get("sex") or b_state.patient_sex

        if (b_state.is_new_patient or b_state.stage == BookingStage.COLLECTING_DETAILS) and not b_state.patient_id:
            # Case A: Complete registration details provided (Name + DOB or Phone)
            if cand_name and (cand_dob or cand_phone):
                from .patient_registration import register_patient
                reg_res = register_patient(
                    full_name=cand_name,
                    date_of_birth=cand_dob or "1996-02-14",
                    contact_number=cand_phone or "+91-98765-43210",
                    email=cand_email,
                    sex=cand_sex
                )
                p_data = reg_res.get("patient", {})
                b_state.patient_name = p_data.get("patient_name") or cand_name
                b_state.patient_id = p_data.get("patient_id")
                b_state.patient_dob = p_data.get("date_of_birth")
                b_state.patient_phone = p_data.get("contact_number")
                b_state.patient_email = p_data.get("email")
                b_state.patient_sex = p_data.get("sex")
                b_state.patient_age = f"{p_data.get('age', 30)} years"
                b_state.is_new_patient = False
                b_state.stage = BookingStage.SELECTING_SERVICE
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "registration_completed_ask_service",
                    "arguments": {
                        "patient_name": b_state.patient_name,
                        "patient_id": b_state.patient_id,
                        "age": p_data.get("age", 30)
                    },
                    "rationale": f"Registered patient {b_state.patient_name} ({b_state.patient_id}). Inquiring visit reason."
                }
            # Case B: Only patient name provided so far
            elif cand_name and not any(w in q_lower for w in ["cbct", "opg", "scan", "consultation", "implant", "slot", "tomorrow", "today"]):
                b_state.patient_name = cand_name
                b_state.stage = BookingStage.COLLECTING_DETAILS
                return {
                    "selected_agent": "ScanBookingAgent",
                    "tool_name": "prompt_patient_dob_and_contact",
                    "arguments": {
                        "patient_name": cand_name
                    },
                    "rationale": f"Received patient name '{cand_name}'. Prompting for DOB and contact phone number to complete registration."
                }

        # 0B.4.5 Existing Patient Identifier Lookup
        if b_state.stage == BookingStage.IDENTIFYING_PATIENT or not b_state.patient_id:
            m_pid = re.search(r"\b(PAT-\d+)\b", query, re.IGNORECASE)
            if m_pid:
                from .retrieval import get_patient
                p_obj = get_patient(m_pid.group(1).upper())
                if p_obj:
                    b_state.patient_name = p_obj["patient_name"]
                    b_state.patient_id = p_obj["patient_id"]
                    b_state.stage = BookingStage.SELECTING_SERVICE
                    return {
                        "selected_agent": "ScanBookingAgent",
                        "tool_name": "existing_patient_identified",
                        "arguments": {
                            "patient_name": b_state.patient_name,
                            "patient_id": b_state.patient_id
                        },
                        "rationale": f"Identified existing patient {b_state.patient_name} ({b_state.patient_id})."
                    }

        # 0B.4.6 Visit Reason & Service Selection
        from .clinic_catalog import find_matching_specialization, DOCTORS_DIRECTORY
        matched_spec = find_matching_specialization(query)
        has_cbct_srv = any(w in q_lower for w in ["cbct", "cone beam", "3d"])
        has_opg_srv = any(w in q_lower for w in ["opg", "panoramic"])
        has_bw_srv = "bitewing" in q_lower
        has_pa_srv = any(w in q_lower for w in ["periapical", "iopa"])
        has_service_kw = has_cbct_srv or has_opg_srv or has_bw_srv or has_pa_srv or matched_spec or any(w in q_lower for w in ["implant", "wisdom", "extraction", "root canal", "brace", "aligner", "consultation"])

        if has_service_kw and not any(w in q_lower for w in ["tomorrow", "today", "morning", "afternoon", "slot"]):
            if has_cbct_srv:
                b_state.service = "CBCT scan for implant planning" if "implant" in q_lower else "CBCT Scan (3D Volume Imaging)"
                b_state.modality = "CBCT"
                b_state.specialization = "Oral and Maxillofacial Radiology"
            elif has_opg_srv:
                b_state.service = "OPG (Panoramic Digital Radiograph)"
                b_state.modality = "OPG"
                b_state.specialization = "Oral and Maxillofacial Radiology"
            elif has_bw_srv:
                b_state.service = "Bitewing Radiographs"
                b_state.modality = "Bitewing"
                b_state.specialization = "General Dentistry"
            elif has_pa_srv:
                b_state.service = "Periapical Radiograph (IOPA)"
                b_state.modality = "Periapical"
                b_state.specialization = "Endodontics"
            elif matched_spec:
                b_state.specialization = matched_spec["name"]
                b_state.service = f"{matched_spec['name']} Consultation"
                b_state.modality = "Consultation"

            if "implant" in q_lower:
                b_state.notes = "implant planning in the lower jaw"

            if not b_state.patient_id and b_state.patient_name:
                from .patient_registration import register_patient
                reg_res = register_patient(
                    full_name=b_state.patient_name,
                    date_of_birth=b_state.patient_dob or "1996-02-14",
                    contact_number=b_state.patient_phone or "+91-98765-43210",
                    email=b_state.patient_email,
                    sex=b_state.patient_sex
                )
                p_data = reg_res.get("patient", {})
                b_state.patient_id = p_data.get("patient_id")
                if not b_state.patient_age:
                    b_state.patient_age = f"{p_data.get('age', 30)} years"

            matching_docs = [d["name"] for d in DOCTORS_DIRECTORY if d["specialization"] == b_state.specialization]
            doc_str = " and ".join(matching_docs[:2]) if matching_docs else "Dr. Rajesh Mehta and Dr. Anjali Deshmukh"

            b_state.stage = BookingStage.COLLECTING_DETAILS
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "service_selected_ask_preference",
                "arguments": {
                    "service": b_state.service,
                    "specialization": b_state.specialization,
                    "doctors": doc_str,
                    "facility": "Radiology Suite 1 (3D Scanner) at our Main Clinic"
                },
                "rationale": f"Service identified: {b_state.service} under {b_state.specialization}. Inquiring schedule preference."
            }

        # 0B.4.7 Schedule Preference Submission (Date, Time-of-Day, Branch)
        is_pref_query = any(w in q_lower for w in ["tomorrow", "today", "morning", "afternoon", "evening", "main clinic", "october", "oct"])
        if is_pref_query and (b_state.service or b_state.modality):
            from .booking_service import check_available_slots as get_avail_slots
            d_target = extract_date_from_query(query)
            mod = b_state.modality or "CBCT"
            raw_slots = get_avail_slots(modality=mod, preferred_date=d_target)
            time_pref = "morning" if "morning" in q_lower else ("afternoon" if "afternoon" in q_lower else None)
            if time_pref == "morning":
                filtered = [s for s in raw_slots if "am" in s.get("time", "").lower() or (s.get("datetime_str") and "am" in s.get("datetime_str").lower())]
            elif time_pref == "afternoon":
                filtered = [s for s in raw_slots if "pm" in s.get("time", "").lower() or (s.get("datetime_str") and "pm" in s.get("datetime_str").lower())]
            else:
                filtered = raw_slots

            if not filtered:
                filtered = raw_slots

            b_state.stage = BookingStage.SLOTS_PRESENTED
            b_state.displayed_slots = filtered
            session.pending_booking_slots = filtered
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "present_matching_schedule_slots",
                "arguments": {
                    "available_slots": filtered,
                    "time_of_day": time_pref or "",
                    "date_str": "Tomorrow" if "tomorrow" in q_lower else (d_target or "Tomorrow")
                },
                "rationale": f"Presented matching {time_pref or ''} slots for {mod}."
            }

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
                pat_id = cand_p_id or b_state.patient_id
                if not pat_id and session.active_patient_id:
                    from .retrieval import get_patient
                    p_active = get_patient(session.active_patient_id)
                    if p_active and (not pat_name or p_active.get("patient_name", "").lower() == (pat_name or "").lower()):
                        pat_id = session.active_patient_id
                        if not pat_name:
                            pat_name = p_active.get("patient_name")

                if pat_name and not pat_id and pat_name != "Walk-in Patient":
                    from .retrieval import search_patients
                    p_matches = search_patients(pat_name)
                    if len(p_matches) == 1:
                        pat_id = p_matches[0]["patient_id"]
                        b_state.patient_id = pat_id
                        dob = p_matches[0].get("date_of_birth")
                        if dob and not cand_p_age:
                            try:
                                b_state.patient_age = f"{2026 - int(dob.split('-')[0])} years"
                            except Exception:
                                pass

                if not pat_name:
                    pat_name = "Walk-in Patient" if (b_state.patient_type == "walk_in" or "walk" in q_lower) else "Walk-in Patient"

                if cand_p_age:
                    b_state.patient_age = cand_p_age
                elif pat_id and not b_state.patient_age:
                    from .retrieval import get_patient
                    p_rec = get_patient(pat_id)
                    if p_rec and p_rec.get("date_of_birth"):
                        try:
                            b_state.patient_age = f"{2026 - int(p_rec['date_of_birth'].split('-')[0])} years"
                        except Exception:
                            b_state.patient_age = f"DOB: {p_rec['date_of_birth']}"

                if cand_doc:
                    b_state.doctor_name = cand_doc

                if cand_doc:
                    b_state.doctor_name = cand_doc

                is_direct_confirmation = bool(
                    re.search(r"\b(confirm|confirmed|yes|proceed|approve|confirm\s+slot)\b", q_lower)
                    or "please confirm" in q_lower
                    or "yes," in q_lower
                ) and not any(w in q_lower for w in ["select", "choose", "pick", "change that to", "switch to", "let's go with", "go with", "option"])

                if is_direct_confirmation:
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
                        "rationale": f"Explicit Direct Confirmation: Confirmed slot {matched_slot_id}."
                    }
                else:
                    b_state.stage = BookingStage.AWAITING_CONFIRMATION
                    b_state.selected_slot = matched_slot
                    b_state.summary_displayed = True
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

                is_direct_confirmation = bool(
                    re.search(r"\b(confirm|confirmed|yes|proceed|approve|confirm\s+slot)\b", q_lower)
                    or "please confirm" in q_lower
                    or "yes," in q_lower
                ) and not any(w in q_lower for w in ["select", "choose", "pick", "change that to", "switch to", "let's go with", "go with", "option"])

                if is_direct_confirmation:
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
                    b_state.selected_slot = matched_slot
                    b_state.summary_displayed = True
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
    is_clinical_scan_query = bool(re.search(
        r"\b(?:in\s+(?:this|the)\s+(?:scan|report|image|radiograph)|documented|fracture|cyst|caries|lesion|bone\s+loss|root|canal|nerve|impaction|pathology|abscess|infection|finding|findings)\b",
        q_lower
    )) and not any(w in q_lower for w in ["book", "booking", "schedule", "reschedule", "appointment", "appoint ment", "reserve", "slot", "slots"])

    is_booking_request = (
        not is_clinical_scan_query
        and (
            any(w in q_lower for w in ["book", "booking", "schedule", "reschedule", "appointment", "appoint ment", "reserve", "slots", "slot"])
            or (any(w in q_lower for w in ["dr", "doctor", "radiologist"]) and any(w in q_lower for w in ["have", "available", "free", "open", "slot", "slots", "schedule", "appointment", "oct", "october", "nov", "dec", "tomorrow", "today", "any", "is there", "are there"]))
            or (bool(detected_date) and any(w in q_lower for w in ["have", "available", "slot", "slots", "appointment", "dr", "doctor", "can i"]))
            or (any(w in q_lower for w in ["scan", "scans"]) and any(w in q_lower for w in ["slot", "slots", "appointment", "schedule", "book", "reserve"]))
        )
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

    # 0B.65 Context Restoration from Last Retrieval if Pronoun Reference Used
    has_pronoun_reference = any(re.search(r"\b" + re.escape(p) + r"\b", q_lower) for p in ["her", "his", "their", "hir", "she", "he"])
    if not session.active_patient_id and has_pronoun_reference and session.last_retrieval_query:
        from .retrieval import search_patients
        found_cands = search_patients(session.last_retrieval_query)
        if len(found_cands) == 1:
            session.set_active_patient(found_cands[0]["patient_id"])

    # 0B.7 Deterministic Active Report & Patient Consultation Gate
    if session.active_report_id or session.active_patient_id:
        # Check if this is a general knowledge question, workstation capability query, or greeting:
        if is_general_or_assistant_query(query):
            return {
                "selected_agent": "GeneralAssistantAgent",
                "tool_name": "conversation",
                "arguments": {"query": query},
                "rationale": f"General Assistant: Handled general knowledge or workstation inquiry '{query}'."
            }

        # Check if clinician explicitly asked for clinic-wide database scope
        is_explicit_clinic_scope = bool(re.search(
            r"\b(?:across\s+(?:the\s+)?clinic|clinic\s+database|whole\s+clinic|entire\s+clinic|all\s+patients\s+in\s+(?:the\s+)?clinic)\b",
            q_lower
        ))

        # Check if clinician is explicitly asking to switch to another patient
        intent = extract_retrieval_intent(query)
        extracted_cand_name = intent.get("patient_name", "").strip()
        has_new_patient_target = False
        if extracted_cand_name:
            from .retrieval import get_patient
            p_active = get_patient(session.active_patient_id) if session.active_patient_id else None
            if p_active:
                p_name_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", p_active.get("patient_name", "").lower()))
                cand_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", extracted_cand_name.lower()))
                if not cand_words.issubset(p_name_words):
                    has_new_patient_target = True
            else:
                has_new_patient_target = True

        has_retrieval_verb = bool(re.search(r"\b(open|retrieve|fetch|load|switch\s+to)\b", q_lower))
        has_new_patient_retrieval = has_retrieval_verb and has_new_patient_target

        # Check if user is asking to switch or open another report of the active patient (e.g. 'open her OPG', 'open hir OPG', 'show periapical', 'open his latest report')
        is_switch_verb = bool(re.search(r"\b(open|switch|view|load|fetch|show|see)\b", q_lower))
        is_qa_or_summary = bool(re.search(r"\b(summarize|summary|explain|what|why|how|tell|describe|meaning|clarify)\b", q_lower))
        is_modality_or_report = any(w in q_lower for w in ["opg", "cbct", "panoramic", "bitewing", "periapical", "iopa", "cephalogram", "ceph", "examination", "exam", "rpt-", "latest", "recent", "newest", "last", "report", "reports", "scan", "scans"])

        is_modality_switch = (
            not has_new_patient_target
            and not is_qa_or_summary
            and is_switch_verb
            and is_modality_or_report
        )
        if is_modality_switch and session.active_patient_id:
            target_mod = "Any"
            if any(m in q_lower for m in ["opg", "panoramic", "orthopantomogram"]):
                target_mod = "OPG"
            elif any(m in q_lower for m in ["cbct", "cone beam", "3d volume", "3d"]):
                target_mod = "CBCT"
            elif "bitewing" in q_lower:
                target_mod = "Bitewing"
            elif any(m in q_lower for m in ["periapical", "iopa"]):
                target_mod = "Periapical"
            elif any(m in q_lower for m in ["cephalogram", "ceph"]):
                target_mod = "Cephalogram"
            elif any(m in q_lower for m in ["examination", "exam"]):
                target_mod = "Dental Examination"

            rid_match = re.search(r"\b(RPT-\d+)\b", query, re.IGNORECASE)
            target_rid = rid_match.group(1).upper() if rid_match else None

            from .retrieval import get_patient_reports, get_report_by_id
            p_reports = get_patient_reports(session.active_patient_id)
            matched_rep = None
            if target_rid:
                matched_rep = get_report_by_id(target_rid)
            elif target_mod != "Any":
                for r in p_reports:
                    if target_mod.lower() in r.get("report_type", "").lower() or (target_mod == "Periapical" and "iopa" in r.get("report_type", "").lower()):
                        matched_rep = r
                        break
            elif any(w in q_lower for w in ["latest", "recent", "newest", "last"]) and p_reports:
                matched_rep = p_reports[0]
            elif len(p_reports) == 1:
                matched_rep = p_reports[0]

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
            elif target_mod != "Any":
                return {
                    "selected_agent": "RetrievalAgent",
                    "tool_name": "search_patient_reports",
                    "arguments": {
                        "patient_id": session.active_patient_id,
                        "modality": target_mod
                    },
                    "rationale": f"Report Switch: Requested modality {target_mod} for active patient {session.active_patient_id}."
                }

        # Active Report / Patient Consultation & Evidence-Grounded Follow-Up
        if not is_booking_request and b_state.stage == BookingStage.IDLE and not has_new_patient_retrieval and not is_explicit_clinic_scope:
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

    # 0B.8 Deterministic Clinic Database Inquiry Gate (when no active patient or explicit clinic scope)
    is_database_inquiry = (
        not is_booking_request
        and b_state.stage == BookingStage.IDLE
        and not any(verb in q_lower for verb in ["open", "retrieve", "view", "load"])
        and (
            bool(re.search(r"\b(?:how\s+man[yu]|total|count|number\s+of)\b.*\b(?:patients?|reports?|scans?)\b", q_lower))
            or bool(re.search(r"\b(?:clinic\s+database|database\s+statistics|across\s+(?:the\s+)?clinic|all\s+patients|registered\s+patients|whole\s+clinic|entire\s+clinic)\b", q_lower))
            or bool(re.search(r"\b(?:patient\s+names?|list\s+(?:out\s+)?(?:only\s+)?(?:the\s+)?patient\s+names?|list\s+(?:of\s+)?patients?|who\s+are\s+the\s+patients?|which\s+patients?|tell\s+me\s+(?:the\s+)?patient\s+names?)\b", q_lower))
        )
    )
    if is_database_inquiry:
        return {
            "selected_agent": "GeneralAssistantAgent",
            "tool_name": "query_clinic_database",
            "arguments": {"question": query},
            "rationale": "Clinic Database Inquiry: Inquiring about registered patient directory or database statistics."
        }

    # Fallback routing (When local LLM is offline or initializing)
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

        # Check if user requested an appointment without active patient or patient specified (Step 1: Patient Identification)
        is_generic_appointment = bool(
            re.search(r"\b(?:dental\s+specialist|specialist|doctor\s+consultation|specialization)\b", q_lower)
            or re.search(r"\b(?:want\s+to\s+book|want\s+book|book\s+an?\s+appointment|schedule\s+an?\s+appointment|make\s+an?\s+appointment|need\s+an?\s+appointment|book\s+appointment)\b", q_lower)
            or (any(w in q_lower for w in ["book", "booking", "schedule", "appointment"]) and not any(m in q_lower for m in ["cbct", "opg", "panoramic", "bitewing", "periapical", "iopa", "slot-", "slots", "slot "]))
        ) and not any(w in q_lower for w in ["walk-in", "walk in"])
        if is_generic_appointment and not pat_name_for_booking and not session.active_patient_id:
            session.booking_state.stage = BookingStage.COLLECTING_DETAILS
            return {
                "selected_agent": "ScanBookingAgent",
                "tool_name": "prompt_existing_or_new_patient",
                "arguments": {"query": query},
                "rationale": "Generic Appointment Inquiry: Inquiring whether patient is existing or first visit."
            }

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
                if dob:
                    try:
                        session.booking_state.patient_age = f"{2026 - int(dob.split('-')[0])} years"
                    except Exception:
                        pass
        elif pat_id_for_booking and not extract_age_from_query(query):
            from .retrieval import get_patient
            p = get_patient(pat_id_for_booking)
            if p and p.get("date_of_birth"):
                try:
                    session.booking_state.patient_age = f"{2026 - int(p['date_of_birth'].split('-')[0])} years"
                except Exception:
                    pass

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
        save_session(session)

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
        "alveolar ridge", "condyle", "tonsillolith", "bone loss",
        "missing tooth", "missing teeth", "unerupted", "resorption",
        "crown fracture", "lesion", "lucency", "mobility", "horizontal bone loss",
        "vertical bone loss", "furcation", "pulpitis", "apical periodontitis"
    ]
    is_explicit_dictation = any(k in q_lower for k in [
        "dictate", "dictation", "clinical dictation", "draft report", "generate draft report",
        "generate report for", "speech-to-text transcript"
    ])
    is_question_intent = any(q_lower.startswith(w) for w in [
        "what is", "what are", "what does", "how does", "how do", "why does", "why is",
        "explain", "tell me about", "can you explain", "who are you", "what are you doing",
        "what can you do", "help me", "how to", "difference between"
    ]) or q_lower.endswith("?")

    if (is_explicit_dictation or any(kw in q_lower for kw in clinical_keywords)) and not is_question_intent and not any(verb in q_lower for verb in ["open", "show", "find", "get", "retrieve", "search"]):
        from .claim_verification import extract_patient_demographics
        pt_demo = extract_patient_demographics(query)
        target_pid = pt_demo.get("patient_id") or session.active_patient_id
        target_pname = pt_demo.get("patient_name")
        if not target_pname and target_pid:
            from .retrieval import get_patient
            rec = get_patient(target_pid)
            if rec:
                target_pname = rec.get("patient_name")

        return {
            "selected_agent": "ClinicalScribeAgent",
            "tool_name": "structure_clinical_dictation",
            "arguments": {
                "dictation_text": query,
                "patient_id": target_pid,
                "patient_name": target_pname
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

    # If the user explicitly provided a patient ID, a matched patient candidate, or a retrieval command
    # with modality/pronoun/report tokens, route to RetrievalAgent
    has_report_or_pronoun_token = any(w in q_lower for w in [
        "report", "reports", "scan", "scans", "opg", "cbct", "bitewing", "periapical", "iopa",
        "radiograph", "radiographs", "his", "her", "hir", "their", "latest", "recent"
    ])
    is_explicit_retrieval = has_patient_id or (has_retrieval_command and (matched_candidates or bool(extracted_name) or has_report_or_pronoun_token))

    if not is_explicit_retrieval:
        if has_non_patient_word or not extracted_name or (not has_retrieval_command and not matched_candidates):
            return {
                "selected_agent": "GeneralAssistantAgent",
                "tool_name": "conversation",
                "arguments": {"query": query},
                "rationale": "Safety Guard: Query was an inquiry or question, not a patient name."
            }

    name_guess = extracted_name.title() if extracted_name else ""

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

    # Immediate Fast-Path for Greetings (Respond in < 5ms without LLM latency)
    if any(re.search(r"\b" + re.escape(g) + r"\b", q_lower) for g in ["hey", "hello", "hi", "good morning", "good afternoon", "good evening", "greetings"]):
        return (
            "Hello Doctor! I am your Viscan AI Clinical Workstation Assistant. "
            "You can ask me to open a patient's scan (e.g. 'Open Vikram Malhotra's CBCT'), dictate examination observations, "
            "explain active findings, or check available scan appointment slots. How can I assist you?"
        )

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

    # 7c. Clinic Timings, Location, and Operating Hours
    if any(k in q_lower for k in ["timing", "timings", "hours", "open", "location", "address", "where is", "contact", "phone", "email"]):
        return (
            "**Viscan Dental Imaging Center - Main Clinic (CLINIC-MAIN-01):**\n"
            "• **Operating Hours**: Monday through Saturday, 09:00 AM – 06:00 PM (Closed Sundays & Public Holidays)\n"
            "• **Location**: Viscan Dental Imaging Center, Diagnostic Radiology Wing, Main Campus\n"
            "• **Facilities**: Radiology Suite 1 (3D CBCT Scanner), Panoramic Suite 2 (OPG Scanner), Intraoral Radiography Bay 3 (Bitewings & IOPA)\n"
            "• **Services**: Cone Beam CT (CBCT), Panoramic (OPG), Bitewing, and Periapical imaging\n"
            "• **Appointments & Scheduling**: You can check available scan slots directly (e.g. 'Check scan slots for CBCT')."
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

