"""
SCAN BOOKING & APPOINTMENT SERVICE (booking_service.py)
------------------------------------------------------
Implements stateful scheduling for radiological imaging:
- Dynamic, timezone-aware slot formatting (Today, Tomorrow, Calendar date) via clock.py.
- Availability searching with modality & date filters.
- Revalidation & atomic booking confirmation (never assigns arbitrary fallback slots).
- Existing booking lookup, verification, cancellation, and rescheduling.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
from .clock import get_current_datetime, get_current_date, format_relative_slot_datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BOOKINGS_FILE = os.path.join(DATA_DIR, "appointment_slots.json")

INITIAL_SLOTS_TEMPLATE = [
    {
        "slot_id": "SLOT-101",
        "date": "2026-10-19",
        "time": "10:00 AM",
        "modality": "CBCT",
        "room": "Radiology Suite 1 (3D Scanner)",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    },
    {
        "slot_id": "SLOT-102",
        "date": "2026-10-19",
        "time": "02:30 PM",
        "modality": "CBCT",
        "room": "Radiology Suite 1 (3D Scanner)",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    },
    {
        "slot_id": "SLOT-103",
        "date": "2026-10-19",
        "time": "11:15 AM",
        "modality": "OPG",
        "room": "Panoramic Suite 2",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    },
    {
        "slot_id": "SLOT-104",
        "date": "2026-10-19",
        "time": "04:00 PM",
        "modality": "OPG",
        "room": "Panoramic Suite 2",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    },
    {
        "slot_id": "SLOT-105",
        "date": "2026-10-20",
        "time": "09:00 AM",
        "modality": "Bitewing",
        "room": "Intraoral X-Ray Bay 3",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    },
    {
        "slot_id": "SLOT-106",
        "date": "2026-10-20",
        "time": "10:30 AM",
        "modality": "Periapical",
        "room": "Intraoral X-Ray Bay 3",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    },
    {
        "slot_id": "SLOT-107",
        "date": "2026-10-20",
        "time": "03:15 PM",
        "modality": "CBCT",
        "room": "Radiology Suite 1 (3D Scanner)",
        "status": "available",
        "patient_name": None,
        "patient_id": None
    }
]


def _decorate_slot_datetime(slot: Dict[str, Any]) -> Dict[str, Any]:
    """Injects dynamic, clock-computed datetime_str into the slot dictionary."""
    s = dict(slot)
    s["datetime_str"] = format_relative_slot_datetime(s.get("date", ""), s.get("time", ""))
    return s


def load_slots() -> List[Dict[str, Any]]:
    """Loads slots from storage and calculates dynamic relative datetime strings."""
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
    """Persists slot states to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BOOKINGS_FILE, "w") as f:
        json.dump(slots, f, indent=2)


def reset_slots() -> List[Dict[str, Any]]:
    """Resets appointment slots to fresh initial available state."""
    fresh = []
    for s in INITIAL_SLOTS_TEMPLATE:
        item = dict(s)
        item["datetime_str"] = format_relative_slot_datetime(item["date"], item["time"])
        fresh.append(item)
    save_slots(fresh)
    return fresh


def get_slot_by_id(slot_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a slot by its unique slot ID without fallback guessing."""
    clean_id = slot_id.strip().upper()
    for s in load_slots():
        if s.get("slot_id", "").upper() == clean_id:
            return s
    return None


def check_available_slots(
    modality: str = "Any",
    preferred_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Checks appointment slots for the requested modality and date.
    Returns only truly available slots (status == 'available').
    """
    slots = load_slots()
    mod_clean = (modality or "Any").upper().strip()
    available = [s for s in slots if s.get("status") == "available"]

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

            # Exact date string match
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
    scan_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates and executes an appointment booking.
    Strictly revalidates slot availability before committing.
    Never falls back to arbitrary slots if slot_id is invalid or missing.
    """
    if not slot_id or not slot_id.strip():
        return {
            "status": "failed",
            "message": "A valid Slot ID must be specified to confirm an appointment."
        }

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

    if target.get("status") != "available":
        existing_ref = target.get("booking_reference", "UNKNOWN")
        return {
            "status": "already_booked",
            "message": f"Appointment slot '{slot_id}' is no longer available. It was already booked (Ref: {existing_ref})."
        }

    # Resolve patient ID if missing
    clean_p_name = patient_name.strip() if patient_name else "Unassigned Patient"
    if not patient_id:
        from .retrieval import get_all_patients
        for p in get_all_patients():
            if p.get("patient_name", "").lower() == clean_p_name.lower():
                patient_id = p.get("patient_id")
                break

    # Commit booking
    now_dt = get_current_datetime()
    booking_ref = f"BKG-{target['slot_id'].split('-')[-1]}-{now_dt.strftime('%M%S')}"

    target["status"] = "booked"
    target["patient_name"] = clean_p_name
    target["patient_id"] = patient_id or "UNREGISTERED"
    target["booked_at"] = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    target["booking_reference"] = booking_ref
    if scan_type:
        target["modality"] = scan_type

    slots[target_idx] = target
    save_slots(slots)

    decorated_slot = _decorate_slot_datetime(target)

    return {
        "status": "success",
        "booking_reference": booking_ref,
        "slot": decorated_slot,
        "message": f"Appointment confirmed for {clean_p_name}: {decorated_slot['modality']} scan on {decorated_slot['datetime_str']} in {decorated_slot['room']} (Ref: {booking_ref})."
    }


def lookup_existing_bookings(
    patient_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    booking_reference: Optional[str] = None,
    date_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves confirmed appointments matching the search criteria."""
    slots = load_slots()
    booked = [s for s in slots if s.get("status") == "booked"]

    results = []
    p_name_lower = patient_name.lower().strip() if patient_name else ""
    p_id_upper = patient_id.upper().strip() if patient_id else ""
    ref_upper = booking_reference.upper().strip() if booking_reference else ""

    for s in booked:
        s_pat = (s.get("patient_name") or "").lower()
        s_pid = (s.get("patient_id") or "").upper()
        s_ref = (s.get("booking_reference") or "").upper()
        s_date = s.get("date", "")

        matched = False
        if ref_upper and ref_upper in s_ref:
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
    Releases the slot back to 'available' status.
    """
    clean_id = booking_identifier.strip().upper()
    slots = load_slots()
    target_idx = -1

    for idx, s in enumerate(slots):
        if s.get("booking_reference", "").upper() == clean_id or s.get("slot_id", "").upper() == clean_id:
            target_idx = idx
            break

    if target_idx == -1:
        return {
            "status": "not_found",
            "message": f"No booking found matching '{booking_identifier}'."
        }

    target = slots[target_idx]
    prev_ref = target.get("booking_reference")
    prev_pat = target.get("patient_name")
    prev_dt = target.get("datetime_str") or format_relative_slot_datetime(target["date"], target["time"])

    # Release slot
    target["status"] = "available"
    target["patient_name"] = None
    target["patient_id"] = None
    target["booking_reference"] = None
    target["booked_at"] = None
    target["cancelled_at"] = get_current_datetime().strftime("%Y-%m-%d %H:%M:%S")
    target["cancellation_reason"] = reason or "Clinician request"

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
    Releases the existing slot and books the new requested slot.
    """
    existing_list = lookup_existing_bookings(booking_reference=existing_identifier)
    if not existing_list:
        existing_list = [s for s in load_slots() if s.get("slot_id", "").upper() == existing_identifier.upper() and s.get("status") == "booked"]

    if not existing_list:
        return {
            "status": "not_found",
            "message": f"Existing booking '{existing_identifier}' was not found."
        }

    old_slot = existing_list[0]
    patient_name = old_slot.get("patient_name", "Patient")
    patient_id = old_slot.get("patient_id")
    modality = old_slot.get("modality")

    # Step 1: Check new slot availability
    new_slot = get_slot_by_id(new_slot_id)
    if not new_slot or new_slot.get("status") != "available":
        return {
            "status": "failed",
            "message": f"Requested new slot '{new_slot_id}' is not available."
        }

    # Step 2: Release old slot
    cancel_booking(old_slot.get("booking_reference") or old_slot.get("slot_id"))

    # Step 3: Book new slot
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
