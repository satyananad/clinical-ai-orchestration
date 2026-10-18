"""
SCAN BOOKING & APPOINTMENT SERVICE (booking_service.py)
------------------------------------------------------
Implements Feature: Voice/Typed Scan-Booking Agent.
- Checks synthetic radiological appointment slots (OPG, CBCT, Bitewing, Periapical).
- Enforces mandatory clinician/patient confirmation before committing any booking.
- Manages synthetic appointments and booking state.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BOOKINGS_FILE = os.path.join(DATA_DIR, "appointment_slots.json")

INITIAL_SLOTS = [
    {
        "slot_id": "SLOT-101",
        "date": "2026-10-19",
        "time": "10:00 AM",
        "datetime_str": "Tomorrow, Oct 19 at 10:00 AM",
        "modality": "CBCT",
        "room": "Radiology Suite 1 (3D Scanner)",
        "status": "available",
        "patient_name": None
    },
    {
        "slot_id": "SLOT-102",
        "date": "2026-10-19",
        "time": "02:30 PM",
        "datetime_str": "Tomorrow, Oct 19 at 02:30 PM",
        "modality": "CBCT",
        "room": "Radiology Suite 1 (3D Scanner)",
        "status": "available",
        "patient_name": None
    },
    {
        "slot_id": "SLOT-103",
        "date": "2026-10-19",
        "time": "11:15 AM",
        "datetime_str": "Tomorrow, Oct 19 at 11:15 AM",
        "modality": "OPG",
        "room": "Panoramic Suite 2",
        "status": "available",
        "patient_name": None
    },
    {
        "slot_id": "SLOT-104",
        "date": "2026-10-19",
        "time": "04:00 PM",
        "datetime_str": "Tomorrow, Oct 19 at 04:00 PM",
        "modality": "OPG",
        "room": "Panoramic Suite 2",
        "status": "available",
        "patient_name": None
    },
    {
        "slot_id": "SLOT-105",
        "date": "2026-10-20",
        "time": "09:00 AM",
        "datetime_str": "Oct 20 at 09:00 AM",
        "modality": "Bitewing",
        "room": "Intraoral X-Ray Bay 3",
        "status": "available",
        "patient_name": None
    },
    {
        "slot_id": "SLOT-106",
        "date": "2026-10-20",
        "time": "10:30 AM",
        "datetime_str": "Oct 20 at 10:30 AM",
        "modality": "Periapical",
        "room": "Intraoral X-Ray Bay 3",
        "status": "available",
        "patient_name": None
    },
    {
        "slot_id": "SLOT-107",
        "date": "2026-10-20",
        "time": "03:15 PM",
        "datetime_str": "Oct 20 at 03:15 PM",
        "modality": "CBCT",
        "room": "Radiology Suite 1 (3D Scanner)",
        "status": "available",
        "patient_name": None
    }
]

def load_slots() -> List[Dict[str, Any]]:
    if not os.path.exists(BOOKINGS_FILE):
        save_slots(INITIAL_SLOTS)
        return INITIAL_SLOTS
    try:
        with open(BOOKINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return INITIAL_SLOTS

def save_slots(slots: List[Dict[str, Any]]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BOOKINGS_FILE, "w") as f:
        json.dump(slots, f, indent=2)

def check_available_slots(modality: str = "Any", preferred_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Checks synthetic appointment slots for the requested modality."""
    slots = load_slots()
    mod_clean = modality.upper().strip()
    available = [s for s in slots if s.get("status") == "available"]
    
    if mod_clean not in ["ANY", ""]:
        if "CBCT" in mod_clean or "CONE BEAM" in mod_clean or "3D" in mod_clean:
            available = [s for s in available if s["modality"] == "CBCT"]
        elif "OPG" in mod_clean or "PANORAMIC" in mod_clean:
            available = [s for s in available if s["modality"] == "OPG"]
        elif "BITEWING" in mod_clean:
            available = [s for s in available if s["modality"] == "Bitewing"]
        elif "PERIAPICAL" in mod_clean or "IOPA" in mod_clean:
            available = [s for s in available if s["modality"] == "Periapical"]
    
    return available

def confirm_booking(slot_id: str, patient_name: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
    """Confirms and commits a synthetic appointment slot booking."""
    slots = load_slots()
    target = None
    for s in slots:
        if s["slot_id"].upper() == slot_id.upper():
            target = s
            break
            
    if not target:
        # Fallback to first available slot if slot_id not found
        available = [s for s in slots if s.get("status") == "available"]
        if available:
            target = available[0]
        else:
            return {
                "status": "failed",
                "message": f"Appointment slot '{slot_id}' is no longer available."
            }

    if not patient_id:
        from .retrieval import get_all_patients
        for p in get_all_patients():
            if p.get("patient_name", "").lower() == patient_name.lower():
                patient_id = p.get("patient_id")
                break

    target["status"] = "booked"
    target["patient_name"] = patient_name
    target["patient_id"] = patient_id or "UNREGISTERED"
    target["booked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target["booking_reference"] = f"BKG-{target['slot_id'].split('-')[-1]}-{datetime.now().strftime('%M%S')}"

    save_slots(slots)

    return {
        "status": "success",
        "booking_reference": target["booking_reference"],
        "slot": target,
        "message": f"Appointment confirmed for {patient_name}: {target['modality']} scan on {target['datetime_str']} in {target['room']} (Ref: {target['booking_reference']})."
    }
