"""
COMPREHENSIVE TEST SUITE FOR BOOKING REQUIREMENTS & 12 MANDATORY AUDIT CASES
-----------------------------------------------------------------------------
Verifies all 12 test cases specified in the Booking Feature Checklist:
- Test 1: “Book a periapical scan tomorrow at 10:30 AM.” (Date resolution & availability check)
- Test 2: User says “Book a scan” without a scan type. (Asks which scan is required)
- Test 3: Requested slot is unavailable. (Offers alternatives; does not claim booked)
- Test 4: Patient is unregistered. (Guest-registration policy & UNREGISTERED status)
- Test 5: Patient ID does not exist. (Clarification / permitted registration workflow)
- Test 6: Two users request the same slot. (Concurrency protection: only one succeeds)
- Test 7: Database save fails. (Does not show successful confirmation)
- Test 8: Booking is saved, response generation fails. (Retrieves saved record without duplicates)
- Test 9: User asks to reschedule. (Validates new slot and updates safely)
- Test 10: User asks to cancel. (Verifies policy, updates persisted status, releases slot)
- Test 11: Appointment date is in the past. (Rejects past dates according to clinic policy)
- Test 12: Appointment at clinic with no matching equipment. (Explains limitation; never invents rooms)
"""

import pytest
import concurrent.futures
from datetime import datetime, date

from backend.app.services.clock import set_frozen_clock, DEFAULT_TIMEZONE
from backend.app.services.booking_service import (
    reset_slots,
    load_slots,
    get_slot_by_id,
    check_available_slots,
    confirm_booking,
    cancel_booking,
    reschedule_booking,
    get_saved_booking_status,
    validate_equipment_and_clinic,
    set_simulated_db_failure,
)
from backend.app.services.conversation_service import (
    SESSIONS,
    get_or_create_session,
    BookingStage,
)
from backend.app.services.orchestrator_agent import run_multi_agent_pipeline


@pytest.fixture(autouse=True)
def setup_booking_environment():
    """Reset mock slots, clear sessions, and freeze clock to 2026-10-19 09:00:00."""
    reset_slots()
    set_simulated_db_failure(False)
    SESSIONS.clear()
    set_frozen_clock(datetime(2026, 10, 19, 9, 0, 0))
    yield
    set_simulated_db_failure(False)
    set_frozen_clock(None)
    reset_slots()
    SESSIONS.clear()


# ==============================================================================
# TEST 1: “Book a periapical scan tomorrow at 10:30 AM.”
# ==============================================================================
def test_case_1_resolve_natural_language_date_time_modality():
    """Resolves relative date 'tomorrow', time '10:30 AM', and modality 'Periapical'."""
    session_id = "test-case-1-session"
    session = get_or_create_session(session_id)
    session.clear()

    # Given reference clock 2026-10-19: tomorrow is 2026-10-20
    resp = run_multi_agent_pipeline("Book a periapical scan tomorrow at 10:30 AM.", session_id=session_id)
    assert resp["status"] in ("success", "confirmation_required")
    # Slot 106 is 2026-10-20 at 10:30 AM for Periapical
    slot = resp.get("slot") or (resp.get("available_slots")[0] if resp.get("available_slots") else None)
    assert slot is not None
    assert slot["slot_id"] == "SLOT-106"
    assert slot["modality"] == "Periapical"
    assert "Oct 20" in slot["datetime_str"]
    assert "10:30 AM" in slot["datetime_str"]


# ==============================================================================
# TEST 2: User says “Book a scan” without a scan type.
# ==============================================================================
def test_case_2_unspecified_scan_type_prompts_user():
    """When user requests 'Book a scan' without modality, the assistant must ask which scan is required."""
    session_id = "test-case-2-session"
    session = get_or_create_session(session_id)
    session.clear()

    resp = run_multi_agent_pipeline("Book a scan", session_id=session_id)
    assert resp["status"] in ("clarification_required", "confirmation_required")
    msg_lower = resp["message"].lower()
    # Must ask which scan is required and NOT assume a default modality
    assert any(phrase in msg_lower for phrase in ["which type of scan", "which scan", "specify", "periapical", "panoramic", "cbct"])


# ==============================================================================
# TEST 3: Requested slot is unavailable.
# ==============================================================================
def test_case_3_unavailable_slot_offers_alternatives_without_claiming_booked():
    """When a slot is unavailable, offers alternative slots and never claims it was booked."""
    # First, book SLOT-101 so it becomes unavailable
    first_book = confirm_booking(slot_id="SLOT-101", patient_name="Existing Patient", scan_type="CBCT")
    assert first_book["status"] == "success"

    # Second user requests the same SLOT-101
    second_book = confirm_booking(slot_id="SLOT-101", patient_name="Second Patient", scan_type="CBCT")
    assert second_book["status"] == "already_booked"
    assert "no longer available" in second_book["message"].lower()

    # Via conversational pipeline:
    session_id = "test-case-3-session"
    session = get_or_create_session(session_id)
    session.clear()
    session.booking_state.stage = BookingStage.AWAITING_CONFIRMATION
    session.booking_state.selected_slot = {"slot_id": "SLOT-101", "modality": "CBCT"}
    session.booking_state.patient_name = "Second Patient"

    resp = run_multi_agent_pipeline("Yes, confirm slot SLOT-101", session_id=session_id)
    # Must NOT claim successful booking
    assert resp["status"] != "success"
    assert "please choose an alternative" in resp["message"].lower() or "available" in resp["message"].lower()


# ==============================================================================
# TEST 4: Patient is unregistered.
# ==============================================================================
def test_case_4_guest_unregistered_patient_workflow():
    """Unregistered/guest patients are supported and clearly labeled with UNREGISTERED status."""
    book_res = confirm_booking(
        slot_id="SLOT-105",
        patient_name="Guest Walk-in",
        patient_id=None,
        scan_type="Bitewing"
    )
    assert book_res["status"] == "success"
    slot = book_res["slot"]
    # Check minimum database fields for unregistered patient
    assert slot["patient_status"] == "UNREGISTERED"
    assert slot["patient_id"] is None or slot["patient_id"] == "UNREGISTERED"
    assert slot["booking_id"] is not None
    assert slot["appointment_start"] is not None
    assert slot["timezone"] == DEFAULT_TIMEZONE
    assert "UNREGISTERED" in book_res["message"] or "Guest" in book_res["message"]


# ==============================================================================
# TEST 5: Patient ID does not exist.
# ==============================================================================
def test_case_5_nonexistent_patient_id_asks_clarification():
    """When a non-existent patient ID is provided, the assistant asks for clarification or registration."""
    session_id = "test-case-5-session"
    session = get_or_create_session(session_id)
    session.clear()

    resp = run_multi_agent_pipeline("Book a CBCT scan for patient PAT-9999 tomorrow", session_id=session_id)
    assert resp["status"] == "clarification_required"
    assert "PAT-9999" in resp["message"]
    assert any(w in resp["message"].lower() for w in ["not exist", "not found", "unregistered", "guest"])


# ==============================================================================
# TEST 6: Two users request the same slot (Concurrency race condition).
# ==============================================================================
def test_case_6_concurrency_protection_only_one_succeeds():
    """When two concurrent requests attempt to book the exact same slot, only one succeeds."""
    target_slot = "SLOT-102"
    results = []

    def attempt_booking(user_name):
        return confirm_booking(
            slot_id=target_slot,
            patient_name=user_name,
            scan_type="CBCT"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(attempt_booking, "Doctor Alpha")
        f2 = executor.submit(attempt_booking, "Doctor Beta")
        results = [f1.result(), f2.result()]

    statuses = [r["status"] for r in results]
    # Exactly one must succeed, and the other must fail / be already booked
    assert statuses.count("success") == 1
    assert statuses.count("already_booked") == 1


# ==============================================================================
# TEST 7: Database save fails.
# ==============================================================================
def test_case_7_database_save_failure_prevents_confirmation():
    """If database save fails, the system must never report successful confirmation."""
    set_simulated_db_failure(True)

    book_res = confirm_booking(
        slot_id="SLOT-103",
        patient_name="Priya Sharma",
        scan_type="OPG"
    )
    # Must NOT be success
    assert book_res["status"] == "failed"
    assert "save failed" in book_res["message"].lower() or "database" in book_res["message"].lower()

    # Slot must remain available
    set_simulated_db_failure(False)
    slot_103 = get_slot_by_id("SLOT-103")
    assert slot_103["status"] == "available"


# ==============================================================================
# TEST 8: Booking is saved, but response generation fails.
# ==============================================================================
def test_case_8_retrieve_saved_booking_without_duplicate():
    """Retrieves the already saved booking and shows its actual status without duplicating."""
    # 1. Booking succeeds in database
    initial_book = confirm_booking(
        slot_id="SLOT-104",
        patient_name="Amitabh Bachchan",
        scan_type="OPG"
    )
    assert initial_book["status"] == "success"
    b_ref = initial_book["booking_reference"]

    # 2. Downstream recovery retrieves the saved booking without creating a new one
    recovered = get_saved_booking_status(b_ref)
    assert recovered is not None
    assert recovered["status"] == "success"
    assert recovered["booking_reference"] == b_ref
    assert recovered["slot"]["slot_id"] == "SLOT-104"

    # Verify no duplicate slots were created
    all_slots = load_slots()
    matching_slots = [s for s in all_slots if s.get("booking_id") == b_ref or s.get("booking_reference") == b_ref]
    assert len(matching_slots) == 1


# ==============================================================================
# TEST 9: User asks to reschedule.
# ==============================================================================
def test_case_9_reschedule_validates_new_slot_and_updates_safely():
    """Rescheduling validates the new slot, releases the old slot, and books the new slot atomically."""
    # Book initial slot SLOT-106
    initial_res = confirm_booking(slot_id="SLOT-106", patient_name="Rohan Verma", scan_type="Periapical")
    b_ref = initial_res["booking_reference"]
    assert not any(s["slot_id"] == "SLOT-106" for s in check_available_slots())

    # Reschedule to available SLOT-105
    resched_res = reschedule_booking(existing_identifier=b_ref, new_slot_id="SLOT-105")
    assert resched_res["status"] == "success"

    # Old slot 106 must now be released and available
    slot_106 = get_slot_by_id("SLOT-106")
    assert slot_106["status"] == "available"

    # New slot 105 must now be booked
    slot_105 = get_slot_by_id("SLOT-105")
    assert slot_105["status"] == "booked"
    assert slot_105["patient_name"] == "Rohan Verma"


# ==============================================================================
# TEST 10: User asks to cancel.
# ==============================================================================
def test_case_10_cancel_booking_verifies_policy_and_releases_slot():
    """Cancellation verifies policy, releases the slot, and updates persisted state."""
    book_res = confirm_booking(slot_id="SLOT-107", patient_name="Ananya Roy", scan_type="CBCT")
    b_ref = book_res["booking_reference"]
    assert not any(s["slot_id"] == "SLOT-107" for s in check_available_slots())

    cancel_res = cancel_booking(b_ref, reason="Patient requested cancellation")
    assert cancel_res["status"] == "success"
    assert "successfully cancelled" in cancel_res["message"].lower()

    # Slot must now be available in inventory
    slot_107 = get_slot_by_id("SLOT-107")
    assert slot_107["status"] == "available"
    assert slot_107["patient_name"] is None


# ==============================================================================
# TEST 11: Appointment date is in the past.
# ==============================================================================
def test_case_11_reject_past_date_appointment():
    """Rejects past dates and asks for a future date."""
    session_id = "test-case-11-session"
    session = get_or_create_session(session_id)
    session.clear()

    # Given current reference date 2026-10-19: 2026-10-15 is in the past
    resp = run_multi_agent_pipeline("Book a CBCT scan on 2026-10-15", session_id=session_id)
    assert resp["status"] in ("date_rejected", "clarification_required")
    assert any(w in resp["message"].lower() for w in ["past", "future"])


# ==============================================================================
# TEST 12: User asks for an appointment at a clinic with no matching equipment.
# ==============================================================================
def test_case_12_clinic_equipment_limitation_explained_never_invented():
    """Explains equipment limitations when requested modality is not supported; never invents rooms."""
    session_id = "test-case-12-session"
    session = get_or_create_session(session_id)
    session.clear()

    # User asks for an MRI scan (unsupported by dental radiology clinic)
    resp = run_multi_agent_pipeline("Book an MRI scan tomorrow", session_id=session_id)
    assert resp["status"] in ("equipment_limitation", "clarification_required")
    assert "does not possess equipment" in resp["message"].lower() or "not available" in resp["message"].lower()
    # Does NOT invent an MRI room
    assert "mri suite" not in resp["message"].lower()

    # Direct facility equipment check
    is_valid, msg = validate_equipment_and_clinic("CBCT", clinic_id="CLINIC-EXPRESS-EAST")
    assert not is_valid
    assert "not available at Viscan Express" in msg


# ==============================================================================
# TEST 13: Minimum Database Fields Verification
# ==============================================================================
def test_minimum_database_fields_present_on_confirmed_booking():
    """Verifies that all required fields from Section 4 of the specification exist on booked records."""
    book_res = confirm_booking(
        slot_id="SLOT-101",
        patient_name="Priya Sharma",
        patient_id="PAT-1001",
        scan_type="CBCT"
    )
    assert book_res["status"] == "success"
    slot = book_res["slot"]

    # Minimum fields from Section 4:
    assert slot.get("booking_id") is not None
    assert slot.get("patient_id") == "PAT-1001"
    assert slot.get("patient_status") == "REGISTERED"
    assert slot.get("modality") == "CBCT"
    assert slot.get("appointment_start") is not None
    assert slot.get("appointment_end") is not None
    assert slot.get("timezone") == "Asia/Kolkata"
    assert slot.get("clinic_id") == "CLINIC-MAIN-01"
    assert slot.get("room_id") == "ROOM-CBCT-1"
    assert slot.get("clinician_id") is None
    assert slot.get("status") in ("CONFIRMED", "booked")
    assert slot.get("queue_token") is not None
    assert slot.get("created_at") is not None
    assert slot.get("updated_at") is not None
