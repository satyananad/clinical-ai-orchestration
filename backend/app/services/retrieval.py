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
            if "patient_id" not in r or not r["patient_id"]:
                r["patient_id"] = p["patient_id"]
    return data

PATIENTS_DATA = load_patients_data()

def get_all_patients() -> List[Dict[str, Any]]:
    return PATIENTS_DATA

def search_patients(query: str) -> List[Dict[str, Any]]:
    """Searches patients by exact or partial name or patient_id."""
    clean_q = query.strip().lower()
    if not clean_q:
        return PATIENTS_DATA

    exact_matches = []
    partial_matches = []
    query_tokens = set(clean_q.split())

    for p in PATIENTS_DATA:
        p_name = p["patient_name"].lower()
        p_id = p["patient_id"].lower()
        p_name_tokens = set(p_name.split())
        
        if clean_q == p_name or clean_q == p_id:
            exact_matches.append(p)
        elif query_tokens.issubset(p_name_tokens):
            exact_matches.append(p)
        elif clean_q in p_name or clean_q in p_id or any(token in p_name for token in query_tokens):
            partial_matches.append(p)

    return exact_matches if exact_matches else partial_matches

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
    """Deterministically extracts candidate patient name and scan modality."""
    lower_input = user_input.lower()
    
    # Identify requested modality
    modality = "Any"
    for m in ["opg", "cbct", "bitewing", "periapical", "cephalogram", "iopa"]:
        if m in lower_input:
            modality = m.upper() if m in ["opg", "cbct", "iopa"] else m.capitalize()
            break

    clean = lower_input
    for token in ["open", "show", "get", "fetch", "view", "latest", "recent", "report", "reports", "scan", "scans", "opg", "cbct", "bitewing", "periapical", "cephalogram", "iopa", "radiograph", "radiographs", "examination", "the", "please"]:
        clean = re.sub(r"\b" + re.escape(token) + r"\b", " ", clean)
    clean = clean.replace("'s", " ").replace("’s", " ")
    clean = " ".join(clean.split()).strip()

    return {"patient_name": clean, "report_type": modality}

def resolve_report_retrieval(query: str) -> Dict[str, Any]:
    """
    Feature A: Patient & Report Retrieval with strict safety ambiguity handling.
    """
    intent = extract_retrieval_intent(query)
    target_name = intent["patient_name"]
    modality = intent["report_type"]

    if not target_name:
        return {
            "status": "not_found",
            "message": "No patient name could be extracted from your request. Please specify a patient name.",
            "candidates": None,
            "report": None
        }

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
        return {
            "status": "clarification_required",
            "message": f"I found {len(candidates)} matching patients for '{target_name}'. Please clarify which patient you mean.",
            "candidates": candidate_list,
            "report": None,
            "patient": None
        }

    # 2. Not Found
    if not candidates:
        return {
            "status": "not_found",
            "message": f"No patient records found matching '{target_name}'.",
            "candidates": None,
            "report": None,
            "patient": None
        }

    # 3. Exactly one patient found
    patient = candidates[0]
    reports = get_patient_reports(patient["patient_id"], modality=modality)

    if not reports:
        return {
            "status": "missing_report",
            "message": f"Patient '{patient['patient_name']}' was found, but no {modality} report exists on file.",
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
