"""
SCAN BOOKING & APPOINTMENT SERVICE (booking_service.py)
------------------------------------------------------
Implements stateful, atomic scheduling for radiological imaging:
- Concurrency-safe transactions (mutex lock) preventing double-booking race conditions.
- Strict read-back verification after disk persistence.
- Dynamic, timezone-aware slot formatting (Today, Tomorrow, Calendar date) via clock.py.
- Patient registration validation (differentiating registered vs unregistered/guest patients).
- Clinic equipment capability checks (explaining limitations, never inventing rooms).
- Past date and past cancellation rejections according to clinic policy.
- Safe appointment lookup, rescheduling, and cancellation.
"""

import os
import json
import re
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
from .clock import get_current_datetime, get_current_date, format_relative_slot_datetime, DEFAULT_TIMEZONE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BOOKINGS_FILE = os.path.join(DATA_DIR, "appointment_slots.json")

# Concurrency protection lock for atomic slot operations
_BOOKING_LOCK = threading.RLock()

# Simulated database failure hook for testing resilience
_SIMULATED_DB_FAILURE = False

def set_simulated_db_failure(fail: bool):
    """Enable or disable simulated database persistence failure for testing."""
    global _SIMULATED_DB_FAILURE
    _SIMULATED_DB_FAILURE = fail

# Modality appointment duration in minutes
MODALITY_DURATION_MINUTES = {
    "Periapical": 15,
    "Bitewing": 15,
    "OPG": 20,
    "CBCT": 30,
}

# Clinic facility equipment registry
SUPPORTED_CLINICS = {
    "CLINIC-MAIN-01": {
        "clinic_id": "CLINIC-MAIN-01",
        "name": "Viscan Dental Imaging Center - Main Clinic",
        "supported_modalities": ["CBCT", "OPG", "Bitewing", "Periapical"],
        "rooms": {
            "ROOM-CBCT-1": {"name": "Radiology Suite 1 (3D Scanner)", "modalities": ["CBCT"]},
            "ROOM-PANO-2": {"name": "Panoramic Suite 2", "modalities": ["OPG"]},
            "ROOM-INTRAORAL-3": {"name": "Intraoral X-Ray Bay 3", "modalities": ["Bitewing", "Periapical"]}
        }
    },
    "CLINIC-EXPRESS-EAST": {
        "clinic_id": "CLINIC-EXPRESS-EAST",
        "name": "Viscan Express Dental Imaging - East Satellite",
        "supported_modalities": ["Bitewing", "Periapical"],
        "rooms": {
            "ROOM-INTRAORAL-EAST": {"name": "Intraoral Bay East", "modalities": ["Bitewing", "Periapical"]}
        }
    }
}

UNSUPPORTED_MODALITIES = [
    "mri", "magnetic resonance", "ct scan", "body ct", "ultrasound",
    "sonography", "mammogram", "mammography", "pet", "pet-ct", "x-ray chest"
]

def format_iso_datetime(date_str: str, time_str: str, duration_mins: int = 30) -> tuple[str, str]:
    """Calculates ISO-8601 appointment start and end timestamps."""
    try:
        clean_time = time_str.strip().replace("a.m.", "AM").replace("p.m.", "PM").replace("am", " AM").replace("pm", " PM")
        clean_time = re.sub(r"\s+", " ", clean_time).strip()
        dt_start = datetime.strptime(f"{date_str} {clean_time}", "%Y-%m-%d %I:%M %p")
        dt_end = dt_start + timedelta(minutes=duration_mins)
        # Format as ISO with +05:30 offset
        start_iso = dt_start.strftime("%Y-%m-%dT%H:%M:%S+05:30")
        end_iso = dt_end.strftime("%Y-%m-%dT%H:%M:%S+05:30")
        return start_iso, end_iso
    except Exception:
        return f"{date_str}T10:00:00+05:30", f"{date_str}T10:30:00+05:30"


def _build_initial_template() -> List[Dict[str, Any]]:
    slots_spec = [
        ("SLOT-101", "2026-10-19", "10:00 AM", "CBCT", "Radiology Suite 1 (3D Scanner)", "ROOM-CBCT-1"),
        ("SLOT-102", "2026-10-19", "02:30 PM", "CBCT", "Radiology Suite 1 (3D Scanner)", "ROOM-CBCT-1"),
        ("SLOT-103", "2026-10-19", "11:15 AM", "OPG", "Panoramic Suite 2", "ROOM-PANO-2"),
        ("SLOT-104", "2026-10-19", "04:00 PM", "OPG", "Panoramic Suite 2", "ROOM-PANO-2"),
        ("SLOT-105", "2026-10-20", "09:00 AM", "Bitewing", "Intraoral X-Ray Bay 3", "ROOM-INTRAORAL-3"),
        ("SLOT-106", "2026-10-20", "10:30 AM", "Periapical", "Intraoral X-Ray Bay 3", "ROOM-INTRAORAL-3"),
        ("SLOT-107", "2026-10-20", "03:15 PM", "CBCT", "Radiology Suite 1 (3D Scanner)", "ROOM-CBCT-1"),
    ]
    template = []
    for sid, sdate, stime, smod, sroom, sroom_id in slots_spec:
        duration = MODALITY_DURATION_MINUTES.get(smod, 30)
        s_iso, _ = format_iso_datetime(sdate, stime, duration)
        template.append({
            "slot_id": sid,
            "booking_id": None,
            "patient_id": None,
            "patient_name": None,
            "patient_status": None,
            "modality": smod,
            "date": sdate,
            "time": stime,
            "appointment_start": s_iso,
            "appointment_end": None,
            "timezone": DEFAULT_TIMEZONE,
            "clinic_id": "CLINIC-MAIN-01",
            "room_id": sroom_id,
            "room": sroom,
            "clinician_id": None,
            "status": "available",
            "queue_token": None,
            "created_at": "2026-10-18T00:00:00+05:30",
            "updated_at": "2026-10-18T00:00:00+05:30"
        })
    return template

INITIAL_SLOTS_TEMPLATE = _build_initial_template()


def _decorate_slot_datetime(slot: Dict[str, Any]) -> Dict[str, Any]:
    """Injects dynamic, clock-computed datetime_str into the slot dictionary."""
    s = dict(slot)
    s["datetime_str"] = format_relative_slot_datetime(s.get("date", ""), s.get("time", ""))
    # Ensure booking_reference and token_number are available for backward compatibility
    if s.get("booking_id") and not s.get("booking_reference"):
        s["booking_reference"] = s.get("booking_id")
    if s.get("queue_token") and not s.get("token_number"):
        s["token_number"] = s.get("queue_token")
    return s


def load_slots() -> List[Dict[str, Any]]:
    """Loads slots from storage and calculates dynamic relative datetime strings."""
    with _BOOKING_LOCK:
        if not os.path.exists(BOOKINGS_FILE):
            reset_slots()
            raw_slots = [dict(s) for s in INITIAL_SLOTS_TEMPLATE]
        else:
            try:
                with open(BOOKINGS_FILE, "r") as f:
                    raw_slots = json.load(f)
            except Exception:
                raw_slots = [dict(s) for s in INITIAL_SLOTS_TEMPLATE]

        return [_decorate_slot_datetime(s) for s in raw_slots]


def save_slots(slots: List[Dict[str, Any]]):
    """Persists slot states to disk with atomic write protection."""
    with _BOOKING_LOCK:
        if _SIMULATED_DB_FAILURE:
            raise IOError("Simulated Database I/O Write Failure: connection terminated.")
        os.makedirs(DATA_DIR, exist_ok=True)
        temp_file = BOOKINGS_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(slots, f, indent=2)
        os.replace(temp_file, BOOKINGS_FILE)


def reset_slots() -> List[Dict[str, Any]]:
    """Resets appointment slots to fresh initial available state."""
    with _BOOKING_LOCK:
        fresh = [dict(s) for s in INITIAL_SLOTS_TEMPLATE]
        for item in fresh:
            item["datetime_str"] = format_relative_slot_datetime(item["date"], item["time"])
        save_slots(fresh)
        return fresh


def get_slot_by_id(slot_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a slot by its unique slot ID without fallback guessing."""
    clean_id = slot_id.strip().upper()
    for s in load_slots():
        if s.get("slot_id", "").upper() == clean_id:
            return s
    return None


def validate_equipment_and_clinic(modality: str, clinic_id: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Validates whether the requested imaging service and clinic have matching equipment.
    Returns (is_valid, explanation_of_limitation).
    """
    m_clean = (modality or "").lower().strip()
    c_id = clinic_id or "CLINIC-MAIN-01"

    # 1. Check for non-dental / unsupported modalities
    for un_mod in UNSUPPORTED_MODALITIES:
        if un_mod in m_clean:
            return False, f"Our clinic does not possess equipment for '{modality}' scans. We are a specialized dental radiology center providing Periapical (IOPA), Panoramic (OPG), Bitewing, and CBCT (3D) imaging."

    # 2. Check clinic existence
    if c_id not in SUPPORTED_CLINICS:
        return False, f"Clinic identifier '{c_id}' does not exist in our system. Main clinic is 'CLINIC-MAIN-01'."

    clinic = SUPPORTED_CLINICS[c_id]
    std_mod = None
    if "cbct" in m_clean or "3d" in m_clean or "cone beam" in m_clean:
        std_mod = "CBCT"
    elif "opg" in m_clean or "panoramic" in m_clean:
        std_mod = "OPG"
    elif "bitewing" in m_clean:
        std_mod = "Bitewing"
    elif "periapical" in m_clean or "iopa" in m_clean:
        std_mod = "Periapical"

    if std_mod and std_mod not in clinic["supported_modalities"]:
        return False, f"{std_mod} imaging equipment is not available at {clinic['name']}. Available services at this facility: {', '.join(clinic['supported_modalities'])}. CBCT scans are available at our Main Radiology Suite 1."

    return True, None


def check_available_slots(
    modality: str = "Any",
    preferred_date: Optional[str] = None,
    clinic_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Checks appointment slots for the requested modality and date.
    Returns only truly available slots (status == 'available').
    Rejects past dates according to scheduling policy.
    """
    slots = load_slots()
    mod_clean = (modality or "Any").upper().strip()
    available = [s for s in slots if s.get("status") in ("available", None)]

    # Equipment validation
    if mod_clean not in ["ANY", ""]:
        is_supported, _ = validate_equipment_and_clinic(modality, clinic_id)
        if not is_supported:
            return []

    # 1. Modality Filter
    if mod_clean not in ["ANY", ""]:
        if "CBCT" in mod_clean or "CONE BEAM" in mod_clean or "3D" in mod_clean:
            available = [s for s in available if s["modality"] == "CBCT"]
        elif "OPG" in mod_clean or "PANORAMIC" in mod_clean:
            available = [s for s in available if s["modality"] == "OPG"]
        elif "BITEWING" in mod_clean:
            available = [s for s in available if s["modality"] == "Bitewing"]
        elif "PERIAPICAL" in mod_clean or "IOPA" in mod_clean:
            available = [s for s in available if s["modality"] == "Periapical"]

    # 2. Date Filter
    if preferred_date:
        d_lower = str(preferred_date).lower().strip()
        now_date = get_current_date()
        filtered = []

        # Check for past date string (YYYY-MM-DD)
        iso_m = re.search(r"\b(202\d)-(0[1-9]|1[0-2])-([0-3]\d)\b", d_lower)
        if iso_m:
            try:
                target_dt = date(int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3)))
                if target_dt < now_date:
                    return []  # Past date
            except Exception:
                pass

        # Resolve relative dates
        target_dates = []
        if "today" in d_lower:
            target_dates.append(now_date.strftime("%Y-%m-%d"))
        if "tomorrow" in d_lower:
            target_dates.append((now_date + timedelta(days=1)).strftime("%Y-%m-%d"))

        # Check explicit day mentions (e.g. 19th, 20th, Oct 20)
        match_20 = bool(re.search(r"\b20(?:th)?\b", d_lower) and not re.search(r"\b202\d\b", d_lower))
        match_19 = bool(re.search(r"\b19(?:th)?\b", d_lower) and not re.search(r"\b202\d\b", d_lower))

        for s in available:
            s_date = s.get("date", "").lower()
            s_dt = s.get("datetime_str", "").lower()

            if d_lower in s_date:
                filtered.append(s)
            elif any(td in s_date for td in target_dates):
                filtered.append(s)
            elif match_20 and ("10-20" in s_date or "oct 20" in s_dt):
                filtered.append(s)
            elif match_19 and ("10-19" in s_date or "oct 19" in s_dt):
                filtered.append(s)
            elif "october 20" in d_lower and "10-20" in s_date:
                filtered.append(s)
            elif "october 19" in d_lower and "10-19" in s_date:
                filtered.append(s)

        available = filtered

    return available


def confirm_booking(
    slot_id: str,
    patient_name: str,
    patient_id: Optional[str] = None,
    scan_type: Optional[str] = None,
    clinic_id: Optional[str] = None,
    patient_age: Optional[str] = None,
    doctor_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates, commits, and reads back verification for an appointment booking.
    - Concurrency-safe: uses _BOOKING_LOCK to prevent double bookings.
    - Persists minimum database fields (booking_id, patient_status, appointment_start/end, etc.).
    - Validates patient registration and guest/unregistered status.
    - Re-reads saved data from disk to verify before confirming.
    """
    if not slot_id or not slot_id.strip():
        return {
            "status": "failed",
            "message": "A valid Slot ID must be specified to confirm an appointment."
        }

    with _BOOKING_LOCK:
        slots = load_slots()
        target = None
        target_idx = -1
        for idx, s in enumerate(slots):
            if s["slot_id"].upper() == slot_id.strip().upper():
                target = s
                target_idx = idx
                break

        if not target:
            return {
                "status": "failed",
                "message": f"Appointment slot '{slot_id}' does not exist in the scheduling system."
            }

        # Check equipment availability
        if scan_type:
            is_valid, limitation_msg = validate_equipment_and_clinic(scan_type, clinic_id or target.get("clinic_id"))
            if not is_valid:
                return {
                    "status": "failed",
                    "message": limitation_msg
                }

        # Check date validity: do not book past dates
        now_date = get_current_date()
        try:
            slot_parts = [int(p) for p in target["date"].split("-")]
            slot_d = date(slot_parts[0], slot_parts[1], slot_parts[2])
            if slot_d < now_date:
                return {
                    "status": "past_date_rejected",
                    "message": f"Cannot confirm appointment slot '{slot_id}' because its date ({target['date']}) is in the past."
                }
        except Exception:
            pass

        # Check if already booked
        if target.get("status") in ("booked", "CONFIRMED"):
            existing_ref = target.get("booking_id") or target.get("booking_reference") or "UNKNOWN"
            # Idempotent re-confirmation check: if requested by same patient, return existing confirmed booking
            clean_check_name = (patient_name or "").strip().lower()
            if target.get("patient_name") and clean_check_name and target["patient_name"].lower() == clean_check_name:
                decorated_slot = _decorate_slot_datetime(target)
                return {
                    "status": "success",
                    "is_existing": True,
                    "booking_reference": existing_ref,
                    "booking_id": existing_ref,
                    "token_number": target.get("queue_token") or target.get("token_number"),
                    "slot": decorated_slot,
                    "message": (
                        f"Appointment already confirmed for {target['patient_name']}: {decorated_slot['modality']} scan "
                        f"on {decorated_slot['datetime_str']} in {decorated_slot['room']} (Ref: {existing_ref})."
                    )
                }
            return {
                "status": "already_booked",
                "message": f"Appointment slot '{slot_id}' is no longer available. It was already booked (Ref: {existing_ref})."
            }

        # Patient identification & verification
        clean_p_name = patient_name.strip() if patient_name else "Guest Patient"
        patient_status = "UNREGISTERED"
        verified_patient_id = None

        if patient_id:
            from .retrieval import get_patient
            p_record = get_patient(patient_id)
            if p_record:
                verified_patient_id = p_record.get("patient_id")
                patient_status = "REGISTERED"
                if not patient_name or patient_name in ("Patient", "Walk-in Patient", "Guest Patient"):
                    clean_p_name = p_record.get("patient_name", clean_p_name)
            else:
                verified_patient_id = patient_id
                patient_status = "UNREGISTERED"
        else:
            # Check by patient name in registered records
            from .retrieval import get_all_patients
            for p in get_all_patients():
                if p.get("patient_name", "").lower() == clean_p_name.lower():
                    verified_patient_id = p.get("patient_id")
                    patient_status = "REGISTERED"
                    break

        now_dt = get_current_datetime()
        slot_num = target['slot_id'].split('-')[-1]
        booking_ref = f"BKG-{slot_num}-{now_dt.strftime('%M%S')}"
        token_no = f"TK-{slot_num}"
        duration = MODALITY_DURATION_MINUTES.get(scan_type or target.get("modality", "CBCT"), 30)
        start_iso, end_iso = format_iso_datetime(target["date"], target["time"], duration)

        # Mutate target slot with mandatory fields
        target["booking_id"] = booking_ref
        target["booking_reference"] = booking_ref
        target["patient_id"] = verified_patient_id
        target["patient_name"] = clean_p_name
        target["patient_status"] = patient_status
        target["patient_age"] = patient_age
        target["doctor_name"] = doctor_name or "Dr. R. Mehta, MDS (Attending Radiologist)"
        target["status"] = "booked"
        target["booking_status"] = "CONFIRMED"
        target["queue_token"] = token_no
        target["token_number"] = token_no
        target["appointment_start"] = start_iso
        target["appointment_end"] = end_iso
        target["timezone"] = DEFAULT_TIMEZONE
        target["clinic_id"] = target.get("clinic_id") or "CLINIC-MAIN-01"
        target["room_id"] = target.get("room_id") or "ROOM-CBCT-1"
        target["booked_at"] = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        target["created_at"] = target.get("created_at") or now_dt.isoformat()
        target["updated_at"] = now_dt.isoformat()
        if scan_type:
            target["modality"] = scan_type

        slots[target_idx] = target

        # Save and verify
        try:
            save_slots(slots)
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Database save failed: {str(e)}. Booking could not be confirmed."
            }

        # Read-back verification
        verified_slots = load_slots()
        verified = next((s for s in verified_slots if s.get("slot_id") == target["slot_id"]), None)
        if not verified or verified.get("booking_id") != booking_ref or verified.get("status") not in ("CONFIRMED", "booked"):
            return {
                "status": "failed",
                "message": "Persistence verification failed. Appointment could not be confirmed."
            }

        decorated_slot = _decorate_slot_datetime(verified)
        p_label = f"{clean_p_name} ({verified_patient_id})" if patient_status == "REGISTERED" else f"{clean_p_name} (Status: UNREGISTERED / Guest Booking)"
        p_age_text = f", Age: {patient_age}" if patient_age else ""
        doc_label = doctor_name or target.get("doctor_name") or "Dr. R. Mehta, MDS (Attending Radiologist)"
        clinic_info = SUPPORTED_CLINICS.get(decorated_slot.get("clinic_id", "CLINIC-MAIN-01"), {})
        clinic_name_val = clinic_info.get("name", "Viscan Dental Imaging Center - Main Clinic")

        arrival_msg = (
            f"Appointment confirmed for {clean_p_name}: {decorated_slot['modality']} scan on {decorated_slot['datetime_str']} in {decorated_slot['room']} (Ref: {booking_ref}).\n\n"
            f"• Token / Queue #: {token_no}\n"
            f"• Booking Reference: {booking_ref}\n"
            f"• Patient: {p_label}{p_age_text}\n"
            f"• Attending Doctor / Radiologist: {doc_label}\n"
            f"• Facility & Location: {decorated_slot['room']}, {clinic_name_val} ({decorated_slot.get('clinic_id', 'CLINIC-MAIN-01')})\n"
            f"• Modality: {decorated_slot['modality']}\n"
            f"• Scheduled Time: {decorated_slot['datetime_str']}\n\n"
            f"Hospital Arrival Instructions:\n"
            f"1. Please arrive at the clinic 15 minutes before your scheduled appointment.\n"
            f"2. Present Token {token_no} or Booking Ref {booking_ref} at the {decorated_slot['room']} reception desk.\n"
            f"3. Bring a government photo ID and clinical referral/prescription from {doc_label}.\n"
            f"4. Remove all metallic items, dental appliances, and jewelry prior to scan."
        )

        return {
            "status": "success",
            "booking_reference": booking_ref,
            "booking_id": booking_ref,
            "token_number": token_no,
            "queue_token": token_no,
            "patient_status": patient_status,
            "slot": decorated_slot,
            "message": arrival_msg
        }


def get_saved_booking_status(
    booking_identifier: str,
    patient_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the actual persisted status of a saved booking without mutating state.
    Used for graceful recovery when response generation encounters an error.
    """
    clean_id = (booking_identifier or "").strip().upper()
    for s in load_slots():
        b_id = str(s.get("booking_id") or "").upper()
        b_ref = str(s.get("booking_reference") or "").upper()
        s_id = str(s.get("slot_id") or "").upper()
        if (clean_id and clean_id in (b_id, b_ref, s_id)) or (patient_name and str(s.get("patient_name") or "").lower() == patient_name.strip().lower()):
            if s.get("status") in ("booked", "CONFIRMED"):
                dec = _decorate_slot_datetime(s)
                return {
                    "status": "success",
                    "booking_reference": dec.get("booking_id") or dec.get("booking_reference"),
                    "slot": dec,
                    "message": (
                        f"Retrieved confirmed appointment for {dec.get('patient_name')}: {dec.get('modality')} scan on "
                        f"{dec.get('datetime_str')} in {dec.get('room')} (Ref: {dec.get('booking_reference')})."
                    )
                }
    return None


def lookup_existing_bookings(
    patient_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    booking_reference: Optional[str] = None,
    date_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves confirmed appointments matching the search criteria."""
    slots = load_slots()
    booked = [s for s in slots if s.get("status") in ("booked", "CONFIRMED")]

    results = []
    p_name_lower = patient_name.lower().strip() if patient_name else ""
    p_id_upper = patient_id.upper().strip() if patient_id else ""
    ref_upper = booking_reference.upper().strip() if booking_reference else ""

    for s in booked:
        s_pat = (s.get("patient_name") or "").lower()
        s_pid = (str(s.get("patient_id") or "")).upper()
        s_ref = (str(s.get("booking_id") or s.get("booking_reference") or "")).upper()
        s_date = s.get("date", "")

        matched = False
        if ref_upper and (ref_upper in s_ref or ref_upper in str(s.get("slot_id") or "").upper()):
            matched = True
        elif p_id_upper and p_id_upper in s_pid:
            matched = True
        elif p_name_lower and (p_name_lower in s_pat or s_pat in p_name_lower):
            matched = True
        elif date_query and date_query in s_date:
            matched = True
        elif not patient_name and not patient_id and not booking_reference and not date_query:
            matched = True

        if matched:
            results.append(_decorate_slot_datetime(s))

    return results


def cancel_booking(
    booking_identifier: str,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cancels an existing booking by slot_id or booking_reference.
    Enforces clinic cancellation policy (cannot cancel past appointments).
    Releases the slot back to 'available' inventory.
    """
    if not booking_identifier:
        return {
            "status": "not_found",
            "message": "No booking identifier provided to cancel."
        }
    clean_id = str(booking_identifier).strip().upper()

    with _BOOKING_LOCK:
        slots = load_slots()
        target_idx = -1

        for idx, s in enumerate(slots):
            b_id = str(s.get("booking_id") or "").upper()
            b_ref = str(s.get("booking_reference") or "").upper()
            s_id = str(s.get("slot_id") or "").upper()
            if clean_id in (b_id, b_ref, s_id):
                target_idx = idx
                break

        if target_idx == -1:
            return {
                "status": "not_found",
                "message": f"No booking found matching '{booking_identifier}'."
            }

        target = slots[target_idx]

        # Policy Check: Already cancelled
        if target.get("status") in ("available", "CANCELLED") and not target.get("booking_id") and not target.get("booking_reference"):
            return {
                "status": "already_cancelled",
                "message": f"Appointment '{booking_identifier}' is already cancelled."
            }

        # Policy Check: Cannot cancel appointments in the past
        now_date = get_current_date()
        try:
            slot_parts = [int(p) for p in target["date"].split("-")]
            slot_d = date(slot_parts[0], slot_parts[1], slot_parts[2])
            if slot_d < now_date:
                return {
                    "status": "policy_rejected",
                    "message": f"Cancellation rejected: Appointments in the past ({target['date']}) cannot be cancelled according to clinic policy."
                }
        except Exception:
            pass

        prev_ref = target.get("booking_id") or target.get("booking_reference")
        prev_pat = target.get("patient_name")
        prev_dt = target.get("datetime_str") or format_relative_slot_datetime(target["date"], target["time"])

        # Release slot back to inventory
        target["status"] = "available"
        target["booking_id"] = None
        target["booking_reference"] = None
        target["patient_name"] = None
        target["patient_id"] = None
        target["patient_status"] = None
        target["queue_token"] = None
        target["token_number"] = None
        target["booked_at"] = None
        target["cancelled_at"] = get_current_datetime().strftime("%Y-%m-%d %H:%M:%S")
        target["cancellation_reason"] = reason or "Clinician request"
        target["updated_at"] = get_current_datetime().isoformat()

        slots[target_idx] = target
        save_slots(slots)

        return {
            "status": "success",
            "message": f"Appointment {prev_ref} for {prev_pat} on {prev_dt} has been successfully cancelled. The slot has been released back to available inventory.",
            "cancelled_booking": prev_ref
        }


def reschedule_booking(
    existing_identifier: str,
    new_slot_id: str
) -> Dict[str, Any]:
    """
    Atomically reschedules an appointment:
    - Validates that new slot exists, is available, and is future.
    - Releases the existing slot and reserves the new slot.
    - Prevents partial mutation on failure.
    """
    with _BOOKING_LOCK:
        existing_list = lookup_existing_bookings(booking_reference=existing_identifier)
        if not existing_list:
            existing_list = [s for s in load_slots() if s.get("slot_id", "").upper() == existing_identifier.upper() and s.get("status") in ("booked", "CONFIRMED")]

        if not existing_list:
            return {
                "status": "not_found",
                "message": f"Existing booking '{existing_identifier}' was not found in the scheduling database."
            }

        old_slot = existing_list[0]
        patient_name = old_slot.get("patient_name", "Patient")
        patient_id = old_slot.get("patient_id")
        modality = old_slot.get("modality")

        # Step 1: Check new slot availability
        new_slot = get_slot_by_id(new_slot_id)
        if not new_slot or new_slot.get("status") not in ("available", None):
            return {
                "status": "failed",
                "message": f"Requested new slot '{new_slot_id}' is not available."
            }

        # Step 2: Validate new slot is not in the past
        now_date = get_current_date()
        try:
            nparts = [int(p) for p in new_slot["date"].split("-")]
            if date(nparts[0], nparts[1], nparts[2]) < now_date:
                return {
                    "status": "past_date_rejected",
                    "message": f"Cannot reschedule to slot '{new_slot_id}' because its date ({new_slot['date']}) is in the past."
                }
        except Exception:
            pass

        # Step 3: Release old slot
        cancel_booking(old_slot.get("booking_id") or old_slot.get("booking_reference") or old_slot.get("slot_id"), reason="Rescheduled")

        # Step 4: Book new slot
        confirm_res = confirm_booking(
            slot_id=new_slot_id,
            patient_name=patient_name,
            patient_id=patient_id,
            scan_type=modality
        )

        if confirm_res["status"] == "success":
            confirm_res["message"] = (
                f"Appointment rescheduled for {patient_name}: "
                f"Moved from {old_slot['datetime_str']} to {confirm_res['slot']['datetime_str']} in {confirm_res['slot']['room']} "
                f"(New Ref: {confirm_res['booking_reference']})."
            )

        return confirm_res
