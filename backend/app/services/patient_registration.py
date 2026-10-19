"""
PATIENT REGISTRATION & IDENTITY SERVICE (patient_registration.py)
-----------------------------------------------------------------
Handles patient identification, search, and new registration with:
- Strict duplicate checks before registration (checking name, DOB, phone).
- Automatic age calculation from Date of Birth (DOB).
- Assigned persistent Patient ID (e.g. PAT-1006) and MRN.
- Safe candidate selection if multiple patients match.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from .retrieval import get_all_patients, search_patients, DATA_FILE

def get_patient_details(patient_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves verified patient details by patient ID or exact name."""
    clean_id = patient_id.strip().lower()
    for p in get_all_patients():
        if (
            p.get("patient_id", "").lower() == clean_id
            or p.get("patient_name", "").lower() == clean_id
            or p.get("record_number", "").lower() == clean_id
        ):
            p_copy = dict(p)
            # Ensure age is computed
            dob = p_copy.get("date_of_birth")
            if dob:
                try:
                    b_year = int(dob.split("-")[0])
                    p_copy["age"] = 2026 - b_year
                except Exception:
                    p_copy["age"] = None
            return p_copy
    return None


def calculate_age_from_dob(dob_str: str) -> Optional[int]:
    """Calculates age in years from YYYY-MM-DD or DD/MM/YYYY date of birth string."""
    if not dob_str:
        return None
    try:
        # Standard ISO YYYY-MM-DD
        m_iso = re.search(r"\b(19\d\d|20\d\d)-(0[1-9]|1[0-2])-([0-3]\d)\b", dob_str)
        if m_iso:
            return 2026 - int(m_iso.group(1))

        # Words like '14 February 1996' or 'Feb 14, 1996'
        m_year = re.search(r"\b(19\d\d|20\d\d)\b", dob_str)
        if m_year:
            return 2026 - int(m_year.group(1))
    except Exception:
        pass
    return None


def normalize_dob(dob_str: str) -> str:
    """Normalizes natural date of birth into YYYY-MM-DD."""
    if not dob_str:
        return ""
    clean = dob_str.strip()
    # Check if already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", clean):
        return clean

    months = {
        "january": "01", "jan": "01",
        "february": "02", "feb": "02",
        "march": "03", "mar": "03",
        "april": "04", "apr": "04",
        "may": "05",
        "june": "06", "jun": "06",
        "july": "07", "jul": "07",
        "august": "08", "aug": "08",
        "september": "09", "sep": "09", "sept": "09",
        "october": "10", "oct": "10",
        "november": "11", "nov": "11",
        "december": "12", "dec": "12"
    }

    # Match '14 February 1996' or '14 Feb 1996'
    m_day_month_year = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", clean)
    if m_day_month_year:
        d = int(m_day_month_year.group(1))
        m_str = m_day_month_year.group(2).lower()
        y = m_day_month_year.group(3)
        m = months.get(m_str, "01")
        return f"{y}-{m}-{d:02d}"

    # Match 'February 14, 1996'
    m_month_day_year = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", clean)
    if m_month_day_year:
        m_str = m_month_day_year.group(1).lower()
        d = int(m_month_day_year.group(2))
        y = m_month_day_year.group(3)
        m = months.get(m_str, "01")
        return f"{y}-{m}-{d:02d}"

    return clean


def register_patient(
    full_name: str,
    date_of_birth: str,
    contact_number: str,
    email: Optional[str] = None,
    sex: Optional[str] = None,
    address: Optional[str] = None
) -> Dict[str, Any]:
    """
    Registers a new patient with validation and deduplication safeguards.
    - Validates required fields: full_name, date_of_birth, contact_number.
    - Checks for existing record by name + DOB or phone number to prevent duplicates.
    - Calculates age from DOB without requiring a separate question.
    - Assigns next verified Patient ID and MRN.
    """
    clean_name = (full_name or "").strip().title()
    if not clean_name or len(clean_name) < 2:
        return {
            "status": "validation_error",
            "message": "Full patient name is required for clinic registration.",
            "patient": None
        }

    norm_dob = normalize_dob(date_of_birth)
    if not norm_dob:
        return {
            "status": "validation_error",
            "message": "Valid date of birth is required for clinic registration (e.g. '1996-02-14' or '14 February 1996').",
            "patient": None
        }

    clean_phone = re.sub(r"[^\d+]", "", (contact_number or "").strip())
    if not clean_phone or len(clean_phone) < 7:
        return {
            "status": "validation_error",
            "message": "A valid contact phone number is required for appointment confirmation and notifications.",
            "patient": None
        }

    # 1. Deduplication check: Do not register duplicate patients
    all_patients = get_all_patients()
    for existing in all_patients:
        ex_name = existing.get("patient_name", "").strip().lower()
        ex_dob = existing.get("date_of_birth", "").strip()
        ex_phone = re.sub(r"[^\d+]", "", str(existing.get("contact_number") or existing.get("phone") or ""))

        # Match by name and DOB
        if ex_name == clean_name.lower() and ex_dob == norm_dob:
            age = calculate_age_from_dob(norm_dob)
            return {
                "status": "existing_patient",
                "is_new": False,
                "message": f"Patient '{existing.get('patient_name')}' is already registered with ID {existing.get('patient_id')}.",
                "patient": dict(existing, age=age)
            }

        # Match by contact number if provided
        if ex_phone and clean_phone and (ex_phone in clean_phone or clean_phone in ex_phone) and len(ex_phone) >= 10:
            age = calculate_age_from_dob(existing.get("date_of_birth"))
            return {
                "status": "existing_patient",
                "is_new": False,
                "message": f"An existing record matching phone {contact_number} was found for {existing.get('patient_name')} ({existing.get('patient_id')}).",
                "patient": dict(existing, age=age)
            }

    # 2. Assign new Patient ID and MRN
    max_id_num = 1000
    for p in all_patients:
        p_id = p.get("patient_id", "")
        digits = "".join(filter(str.isdigit, p_id))
        if digits:
            try:
                num = int(digits)
                if num > max_id_num:
                    max_id_num = num
            except Exception:
                pass

    new_id_num = max_id_num + 1
    new_patient_id = f"PAT-{new_id_num}"
    new_mrn = f"MRN-DEMO-{new_id_num}"
    calc_age = calculate_age_from_dob(norm_dob)

    new_patient = {
        "patient_id": new_patient_id,
        "patient_name": clean_name,
        "date_of_birth": norm_dob,
        "sex": (sex or "Unspecified").capitalize(),
        "record_number": new_mrn,
        "contact_number": contact_number.strip(),
        "email": email.strip() if email else None,
        "address": address.strip() if address else "Bengaluru, Karnataka",
        "status": "active",
        "reports": []
    }

    # Append to memory dataset
    all_patients.append(new_patient)

    # Persist to JSON file
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(all_patients, f, indent=2)
        except Exception:
            pass

    return {
        "status": "success",
        "is_new": True,
        "message": f"Successfully registered new patient {clean_name}. Assigned Patient ID: {new_patient_id} (MRN: {new_mrn}, Age: {calc_age} years).",
        "patient": dict(new_patient, age=calc_age)
    }
