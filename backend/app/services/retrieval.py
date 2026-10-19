import os
import json
import re
from typing import List, Optional, Dict, Any

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dental_mock_dataset.json")

def load_patients_data() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    for p in data:
        for r in p.get("reports", []):
            r["patient_id"] = p["patient_id"]
            r["patient_name"] = p["patient_name"]
            r["date_of_birth"] = p.get("date_of_birth")
            r["sex"] = p.get("sex")
    return data

PATIENTS_DATA = load_patients_data()

def get_all_patients() -> List[Dict[str, Any]]:
    return PATIENTS_DATA

def search_patients(query: str) -> List[Dict[str, Any]]:
    """
    Searches patients by ID, full name, or partial name.
    Strictly differentiates between unambiguous full-name matches and ambiguous single-name queries.
    """
    clean_q = query.strip().lower()
    if not clean_q:
        return PATIENTS_DATA

    # 1. Exact or prefixed Patient ID match (e.g. PAT-1001, PT 10001, PAT 1001, PT-1001)
    for p in PATIENTS_DATA:
        p_id = p["patient_id"].lower()
        p_num = "".join(filter(str.isdigit, p_id))
        id_pattern = r"\b(?:pat|pt)[- ]?0*" + re.escape(p_num[:1]) + r"0*" + re.escape(p_num[1:]) + r"\b"
        if (
            clean_q == p_id
            or re.search(r"\b" + re.escape(p_id) + r"\b", clean_q)
            or re.search(id_pattern, clean_q)
            or clean_q == p_num
        ):
            p_copy = dict(p)
            p_copy["reports"] = [dict(r, patient_id=p["patient_id"]) for r in p.get("reports", [])]
            return [p_copy]

    # 2. Unambiguous Full Name Match (e.g. 'Priya Sharma')
    query_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", clean_q))
    full_matches = []
    for p in PATIENTS_DATA:
        p_name = p["patient_name"].lower()
        p_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", p_name))
        if p_name in clean_q or p_words.issubset(query_words):
            p_copy = dict(p)
            p_copy["reports"] = [dict(r, patient_id=p["patient_id"]) for r in p.get("reports", [])]
            full_matches.append(p_copy)

    # If full name matches are found:
    if full_matches:
        if len(full_matches) == 1:
            return full_matches

        # If multiple candidates share the exact same name (e.g. two Priya Sharmas),
        # check if query contains differentiating details (DOB, birth year, ID digits, sex).
        differentiated = []
        for cand in full_matches:
            c_id = cand["patient_id"].lower()
            c_num = "".join(filter(str.isdigit, c_id))
            c_dob = cand.get("date_of_birth", "")
            c_year = c_dob.split("-")[0] if c_dob else ""
            c_sex = cand.get("sex", "").lower()

            # ID / digit match
            id_match = (
                c_id in clean_q
                or bool(c_num and (
                    re.search(r"\b(?:pat|pt)?[- ]?0*" + re.escape(c_num) + r"\b", clean_q)
                    or re.search(r"\b" + re.escape(c_num[:1]) + r"0*" + re.escape(c_num[1:]) + r"\b", clean_q)
                ))
            )

            # DOB / year match (including STT slips like 9098 for 1998)
            dob_match = False
            if c_dob and c_dob in clean_q:
                dob_match = True
            elif c_year and re.search(r"\b" + re.escape(c_year) + r"\b", clean_q):
                dob_match = True
            elif c_year and len(c_year) >= 2 and re.search(r"\b(?:dob|born|birth)?\D*(?:90" + re.escape(c_year[-2:]) + r"|" + re.escape(c_year[-2:]) + r")\b", clean_q):
                dob_match = True

            # Sex match (if candidates differ)
            sex_match = bool(c_sex and re.search(r"\b" + re.escape(c_sex) + r"\b", clean_q))

            if id_match or dob_match or sex_match:
                differentiated.append(cand)

        if len(differentiated) == 1:
            return differentiated
        return full_matches

    # 3. Partial / Ambiguous First Name Match (e.g. 'Priya' matching Sharma, Sharmila, Nair)
    stop_words = {
        "can", "you", "open", "show", "get", "fetch", "view", "the", "a", "an",
        "report", "reports", "scan", "scans", "opg", "cbct", "bitewing", "periapical",
        "latest", "recent", "please", "for", "of", "to", "in", "and", "or",
        "radiograph", "radiographs", "examination", "panoramic", "orthopantomogram"
    }
    candidate_tokens = query_words - stop_words

    partial_matches = []
    for p in PATIENTS_DATA:
        p_name = p["patient_name"].lower()
        p_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", p_name))
        if any(token in p_words for token in candidate_tokens):
            p_copy = dict(p)
            p_copy["reports"] = [dict(r, patient_id=p["patient_id"]) for r in p.get("reports", [])]
            partial_matches.append(p_copy)

    if partial_matches:
        return partial_matches

    # 4. Search by Date of Birth / Birth Year alone (e.g. '1998', 'dob 1998', 'dob 9098')
    dob_matches = []
    for p in PATIENTS_DATA:
        p_dob = p.get("date_of_birth", "")
        if not p_dob:
            continue
        p_year = p_dob.split("-")[0]
        # Ensure year token is not part of a report identifier (e.g. RPT-2001)
        if re.search(r"\brpt[- ]?" + re.escape(p_year) + r"\b", clean_q):
            continue
        if p_dob in clean_q or re.search(r"\b" + re.escape(p_year) + r"\b", clean_q):
            p_copy = dict(p)
            p_copy["reports"] = [dict(r, patient_id=p["patient_id"]) for r in p.get("reports", [])]
            dob_matches.append(p_copy)
        elif re.search(r"\b(?:dob|born|birth)\D*(?:90" + re.escape(p_year[-2:]) + r"|" + re.escape(p_year[-2:]) + r")\b", clean_q):
            p_copy = dict(p)
            p_copy["reports"] = [dict(r, patient_id=p["patient_id"]) for r in p.get("reports", [])]
            dob_matches.append(p_copy)

    return dob_matches

def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    clean_id = patient_id.strip().upper()
    for p in PATIENTS_DATA:
        if p["patient_id"].upper() == clean_id:
            p_copy = dict(p)
            p_copy["reports"] = [dict(r, patient_id=p["patient_id"]) for r in p.get("reports", [])]
            return p_copy
    return None

def get_patient_reports(patient_id: str, modality: Optional[str] = None) -> List[Dict[str, Any]]:
    patient = get_patient(patient_id)
    if not patient:
        return []
    reports = patient.get("reports", [])
    if modality and modality.upper() != "ANY":
        mod_upper = modality.upper()
        if mod_upper in ["PANORAMIC", "ORTHOPANTOMOGRAM"]:
            mod_upper = "OPG"
        elif mod_upper in ["CONE BEAM", "3D", "3D VOLUME"]:
            mod_upper = "CBCT"
        reports = [r for r in reports if mod_upper in r["report_type"].upper()]
    enriched = [dict(r, patient_id=patient["patient_id"]) for r in reports]
    return sorted(enriched, key=lambda x: x["report_date"], reverse=True)

def get_report_by_id(report_id: str) -> Optional[Dict[str, Any]]:
    clean_rid = report_id.strip().upper()
    for p in PATIENTS_DATA:
        for r in p.get("reports", []):
            if r["report_id"].upper() == clean_rid:
                r_copy = dict(r)
                r_copy["patient_name"] = p["patient_name"]
                return r_copy
    return None

def extract_retrieval_intent(user_input: str) -> Dict[str, str]:
    """
    Extracts candidate patient name and scan modality with synonym support.
    Uses LLM instruction interpretation when available, falling back to deterministic extraction.
    """
    try:
        from .llm_service import extract_retrieval_intent_with_llm
        llm_intent = extract_retrieval_intent_with_llm(user_input)
        if llm_intent and llm_intent.get("patient_name"):
            return llm_intent
    except Exception:
        pass

    lower_input = user_input.lower()
    
    # Identify requested modality with clinical synonyms
    modality = "Any"
    if any(m in lower_input for m in ["opg", "panoramic", "orthopantomogram"]):
        modality = "OPG"
    elif any(m in lower_input for m in ["cbct", "cone beam", "3d volume"]):
        modality = "CBCT"
    elif "bitewing" in lower_input:
        modality = "Bitewing"
    elif any(m in lower_input for m in ["periapical", "iopa"]):
        modality = "Periapical"
    elif any(m in lower_input for m in ["cephalogram", "ceph"]):
        modality = "Cephalogram"
    elif "examination" in lower_input:
        modality = "Dental Examination"

    clean = lower_input
    # Remove slot references like SLOT-106, BKG-101-1234, TK-106
    clean = re.sub(r"\b(?:slot|bkg|tk)[- ]?[0-9a-z_-]+\b", " ", clean, flags=re.IGNORECASE)
    # Remove standalone numbers/digits (e.g. 106, 101)
    clean = re.sub(r"\b\d+\b", " ", clean)

    stop_tokens = [
        "can", "you", "could", "would", "please", "i", "want", "to", "need", "see",
        "open", "show", "get", "fetch", "view", "load", "find", "latest", "recent",
        "retrieve", "retrieving", "search", "searching",
        "report", "reports", "scan", "scans", "opg", "cbct", "bitewing", "periapical",
        "cephalogram", "iopa", "radiograph", "radiographs", "examination", "the",
        "a", "an", "for", "of", "with", "patient", "mrn", "id",
        "panoramic", "orthopantomogram", "not", "saying",
        "book", "booking", "schedule", "reschedule", "reserve", "reservation",
        "appointment", "appoint", "ment", "slot", "slots", "tomorrow", "today",
        "yesterday", "dr", "doctor", "check", "available", "availability", "time", "date",
        "confirm", "confirmed", "confirmation", "yes", "ok", "sure", "proceed", "cancel"
    ]
    for token in stop_tokens:
        clean = re.sub(r"\b" + re.escape(token) + r"\b", " ", clean)
    clean = clean.replace("'s", " ").replace("’s", " ")
    clean = " ".join(clean.split()).strip()

    return {"patient_name": clean, "report_type": modality}

def resolve_report_retrieval(query: str, patient_name: Optional[str] = None, modality: Optional[str] = None) -> Dict[str, Any]:
    """
    Feature A: Patient & Report Retrieval with strict safety ambiguity handling.
    """
    # Check if query contains an explicit Report ID (e.g. 'RPT-2001', 'Open OPG report RPT-2001')
    rid_match = re.search(r"\b(RPT-\d+)\b", query, re.IGNORECASE)
    if rid_match:
        target_rid = rid_match.group(1).upper()
        report = get_report_by_id(target_rid)
        if report:
            patient = get_patient(report["patient_id"])
            return {
                "status": "success",
                "message": f"Retrieved {report.get('report_type')} report ({report['report_id']}) for {patient['patient_name']}.",
                "candidates": None,
                "patient": {
                    "patient_id": patient["patient_id"],
                    "patient_name": patient["patient_name"],
                    "date_of_birth": patient["date_of_birth"],
                    "sex": patient["sex"]
                },
                "report": report
            }

    intent = extract_retrieval_intent(query)
    target_name = patient_name.strip() if patient_name else intent["patient_name"]
    requested_modality = modality if (modality and modality != "Any") else intent["report_type"]

    # Check if user invoked retrieval without specifying a patient name or ID
    if not target_name or target_name.lower() in ["retrieving", "retrieve", "patient report", "open report", "view report", "find report", "search", "loading"]:
        return {
            "status": "missing_patient_name",
            "message": "Please specify the patient name or ID whose report you'd like to retrieve (e.g. 'Open Vikram Malhotra's CBCT' or 'Show Priya Sharma's report'). You can also ask me to list available patients.",
            "candidates": None,
            "report": None,
            "patient": None
        }

    # First check query directly against patient database
    candidates = search_patients(query)
    if not candidates and target_name:
        candidates = search_patients(target_name)

    # 1. Ambiguity Detection: Multiple patients match
    if len(candidates) > 1:
        candidate_list = [
            {
                "patient_id": c["patient_id"],
                "patient_name": c["patient_name"],
                "date_of_birth": c["date_of_birth"],
                "sex": c["sex"]
            }
            for c in candidates
        ]
        display_name = target_name or query.strip()
        return {
            "status": "clarification_required",
            "message": f"I found {len(candidates)} matching patients for '{display_name}'. Please clarify which patient you mean.",
            "candidates": candidate_list,
            "report": None,
            "patient": None
        }

    # 2. Not Found
    if not candidates:
        return {
            "status": "not_found",
            "message": f"No patient records found matching '{target_name or query}'.",
            "candidates": None,
            "report": None,
            "patient": None
        }

    # 3. Exactly one patient found
    patient = candidates[0]
    reports = get_patient_reports(patient["patient_id"], modality=requested_modality)

    if not reports:
        return {
            "status": "missing_report",
            "message": f"Patient '{patient['patient_name']}' was found, but no {requested_modality} report exists on file.",
            "candidates": None,
            "report": None,
            "patient": {
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "date_of_birth": patient["date_of_birth"],
                "sex": patient["sex"]
            }
        }

    latest_report = reports[0]
    return {
        "status": "success",
        "message": f"Retrieved latest {latest_report['report_type']} report ({latest_report['report_id']}) for {patient['patient_name']}.",
        "candidates": None,
        "patient": {
            "patient_id": patient["patient_id"],
            "patient_name": patient["patient_name"],
            "date_of_birth": patient["date_of_birth"],
            "sex": patient["sex"]
        },
        "report": latest_report
    }
