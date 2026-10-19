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

def reset_patients() -> List[Dict[str, Any]]:
    global PATIENTS_DATA
    PATIENTS_DATA = load_patients_data()
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
    known_name_tokens = {w.lower() for p in PATIENTS_DATA for w in p["patient_name"].split()}
    candidate_tokens = {token for token in query_words if token in known_name_tokens}
    if not candidate_tokens:
        return []

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

    lower_input = user_input.lower().strip()

    # 1. Identify requested modality with clinical synonyms
    modality = "Any"
    modality_patterns = [
        (r"\b(opg|panoramic|orthopantomogram)\b", "OPG"),
        (r"\b(cbct|cone\s*beam|3d\s*volume)\b", "CBCT"),
        (r"\b(bitewing|bitewings)\b", "Bitewing"),
        (r"\b(periapical|iopa)\b", "Periapical"),
        (r"\b(cephalogram|ceph)\b", "Cephalogram"),
        (r"\b(examination|exam)\b", "Dental Examination"),
    ]
    for pattern, mod_name in modality_patterns:
        if re.search(pattern, lower_input):
            modality = mod_name
            break

    # 2. Explicit Patient ID (e.g. PAT-1001, PT-1001)
    id_match = re.search(r"\b(?:pat|pt)[- ]?0*(\d+)\b", user_input, re.IGNORECASE)
    if id_match:
        return {"patient_name": f"PAT-{id_match.group(1)}", "report_type": modality}

    # 3. Exact full-name matches against registered patient directory
    for p in PATIENTS_DATA:
        p_name = p["patient_name"]
        if re.search(r"\b" + re.escape(p_name.lower()) + r"\b", lower_input):
            return {"patient_name": p_name, "report_type": modality}

    pronoun_refs = {
        "he", "she", "his", "her", "hers", "hir", "their", "theirs", "its", "him", "them",
        "this", "that", "these", "those", "it", "hiss", "herr"
    }

    # 4. Known patient first names / surnames from registered directory
    known_name_tokens = {w.lower() for p in PATIENTS_DATA for w in p["patient_name"].split()}
    query_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", lower_input)
    matched_db_tokens = [t for t in query_tokens if t in known_name_tokens and t not in pronoun_refs]
    if matched_db_tokens:
        return {"patient_name": " ".join(matched_db_tokens).title(), "report_type": modality}

    # 5. Grammatical Request Framing for Seen & Unseen Patient Names:
    # A) Possessive: "Marcus Aurelius's CBCT", "Vikram's scan"
    pos_match = re.search(r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)(?:'s|’s)\b", user_input)
    if pos_match:
        cand = pos_match.group(1).strip()
        if cand.lower() not in pronoun_refs:
            return {"patient_name": cand, "report_type": modality}

    # B) Structural request framing: Strip polite/imperative request prefix and trailing modality/document nouns
    core_text = user_input.strip()
    core_text = re.sub(
        r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?|(?:please\s+)?(?:i\s+want\s+to\s+|i\s+need\s+to\s+)?)*(?:open|show|view|find|load|retrieve|fetch|get|display|search\s+for|look\s+for)\s+(?:(?:the|a|an)\s+)?(?:(?:patient|report|scan|radiograph|image)\s+(?:for\s+|of\s+)?)?",
        "",
        core_text,
        flags=re.IGNORECASE
    ).strip()

    core_text = re.sub(r"^(?:for\s+patient|patient|pt|for|of)\s+", "", core_text, flags=re.IGNORECASE).strip()

    core_text = re.sub(
        r"\s+(?:(?:cbct|opg|panoramic|orthopantomogram|bitewings?|periapical|iopa|cephalogram|ceph|examination|exam|latest|recent|newest|last|prior|previous)\s+)*(?:reports?|scans?|radiographs?|images?|x-?rays?)\s*$",
        "",
        core_text,
        flags=re.IGNORECASE
    ).strip()
    core_text = re.sub(
        r"\s+(?:cbct|opg|panoramic|orthopantomogram|bitewings?|periapical|iopa|cephalogram|ceph|examination|exam|latest|recent|newest|last|prior|previous)\s*$",
        "",
        core_text,
        flags=re.IGNORECASE
    ).strip()

    core_text = re.sub(r"['’]s\b", "", core_text).strip()

    if core_text and not any(w in pronoun_refs for w in core_text.lower().split()):
        words = set(core_text.lower().split())
        if not words.issubset({"opg", "cbct", "panoramic", "bitewing", "periapical", "iopa", "report", "scan", "exam", "examination", "latest", "recent", "newest", "last"}):
            if not re.search(r"\b(?:slot|bkg|tk)[- ]?\d+", core_text, re.IGNORECASE) and not re.search(r"\d", core_text):
                if re.search(r"[a-zA-Z]", core_text):
                    return {"patient_name": core_text, "report_type": modality}

    return {"patient_name": "", "report_type": modality}

def get_patient_pronouns(sex: Optional[str]) -> Dict[str, str]:
    """
    Returns pronouns based on patient sex.
    - Male: 'his', 'he', 'him'
    - Female: 'her', 'she', 'her'
    - Unspecified / unclear / other: 'his/her', 'he/she', 'him/her' (mentions both)
    """
    s = (sex or "").strip().lower()
    if s in ["male", "m"]:
        return {"possessive": "his", "subject": "he", "object": "him"}
    elif s in ["female", "f"]:
        return {"possessive": "her", "subject": "she", "object": "her"}
    else:
        return {"possessive": "his/her", "subject": "he/she", "object": "him/her"}

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

    # Check if user invoked retrieval without specifying a patient name or ID, or supplied only pronouns
    pronoun_tokens = {"his", "her", "hers", "hir", "their", "theirs", "its", "them", "him", "she", "he", "they", "hiss", "herr"}
    is_pronoun_only = bool(target_name) and all(w.lower() in pronoun_tokens for w in target_name.split())
    if not target_name or is_pronoun_only or target_name.lower() in ["retrieving", "retrieve", "patient report", "open report", "view report", "find report", "search", "loading"]:
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
