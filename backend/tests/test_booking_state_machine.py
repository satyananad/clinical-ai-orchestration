"""Integration tests for the deterministic, stateful appointment-booking state machine.

Covers the complete 7-stage workflow:
1. Intent understanding (availability, booking, lookup, cancel, reschedule)
2. Patient information collection
3. Real schedule querying & dynamic date calculation (via injectable clock)
4. Slot selection & stage tracking
5. Booking summary presentation before confirmation
6. Explicit confirmation gating (no premature booking)
7. Final booking execution, verification, and idempotency protection
8. Booking lookup & slot release on cancellation
"""

import pytest
from datetime import datetime

from app.services.clock import set_frozen_clock, DEFAULT_TIMEZONE
from app.services.booking_service import (
    reset_slots,
    check_available_slots,
    confirm_booking,
    lookup_existing_bookings,
    cancel_booking,
    get_slot_by_id,
)
from app.services.conversation_service import (
    SESSIONS,
    get_or_create_session,
    BookingStage,
)
from app.services.orchestrator_agent import run_multi_agent_pipeline


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Reset mock slots, clear sessions, and freeze clock to a deterministic reference point."""
    reset_slots()
    SESSIONS.clear()
    # Freeze clock to 2026-10-18 09:00:00
    set_frozen_clock(datetime(2026, 10, 18, 9, 0, 0))
    yield
    # Unfreeze clock
    set_frozen_clock(None)
    reset_slots()
    SESSIONS.clear()


def test_full_7_stage_booking_workflow():
    """Test full multi-turn flow: query -> collect info -> present slots -> select slot -> summary -> explicit confirm."""
    session_id = "test-session-full-flow"

    # Turn 1: Clinician says they want to book an appointment
    resp1 = run_multi_agent_pipeline("I want to book an appointment", session_id=session_id)
    state1 = get_or_create_session(session_id).booking_state
    assert state1.stage in (BookingStage.COLLECTING_DETAILS, BookingStage.SLOTS_PRESENTED)
    assert resp1.get("message") is not None

    # Turn 2: Clinician specifies patient and modality
    resp2 = run_multi_agent_pipeline("Book CBCT for patient Rahul Sharma tomorrow", session_id=session_id)
    session2 = get_or_create_session(session_id)
    state2 = session2.booking_state

    # Rahul Sharma and CBCT should be extracted, and slots for tomorrow (2026-10-19) presented
    assert state2.patient_name == "Rahul Sharma"
    assert state2.modality == "CBCT"
    assert state2.stage == BookingStage.SLOTS_PRESENTED
    assert len(state2.displayed_slots) > 0
    # Must list slots containing Tomorrow / Oct 19
    assert any("Oct 19" in s.get("datetime_str", "") or "Tomorrow" in s.get("datetime_str", "") for s in state2.displayed_slots)

    # Turn 3: Clinician selects a slot time (e.g. "10:00 AM" or "SLOT-101")
    resp3 = run_multi_agent_pipeline("Let's select the 10:00 AM slot", session_id=session_id)
    session3 = get_or_create_session(session_id)
    state3 = session3.booking_state

    # Crucial architectural requirement: Selecting a slot MUST NOT immediately commit the booking!
    # Stage must transition to AWAITING_CONFIRMATION and show a summary
    assert state3.stage == BookingStage.AWAITING_CONFIRMATION
    assert state3.selected_slot is not None
    assert state3.selected_slot.get("slot_id") == "SLOT-101"
    assert state3.summary_displayed is True
    assert "Rahul Sharma" in resp3["message"]
    assert "SLOT-101" in resp3["message"]
    assert "confirm" in resp3["message"].lower()

    # Slot must still be available in inventory (not yet mutated)
    available_before = check_available_slots()
    slot_101 = next(s for s in available_before if s["slot_id"] == "SLOT-101")
    assert slot_101["status"] == "available"

    # Turn 4: Clinician provides explicit confirmation
    resp4 = run_multi_agent_pipeline("Yes, please confirm and book it", session_id=session_id)
    session4 = get_or_create_session(session_id)
    state4 = session4.booking_state

    # Stage must now be CONFIRMED
    assert state4.stage == BookingStage.CONFIRMED
    assert state4.completed_booking is not None
    booking_ref = state4.completed_booking["booking_reference"]
    assert booking_ref.startswith("VISCAN-") or booking_ref.startswith("BKG-")
    assert booking_ref in resp4["message"]

    # Slot must now be marked as booked in inventory and no longer in available slots
    assert not any(s["slot_id"] == "SLOT-101" for s in check_available_slots())
    slot_101_after = get_slot_by_id("SLOT-101")
    assert slot_101_after is not None
    assert slot_101_after["status"] == "booked"


def test_decline_confirmation_prevents_booking():
    """Verify that declining during AWAITING_CONFIRMATION cancels the action with zero database mutations."""
    session_id = "test-session-decline"

    # Set up to AWAITING_CONFIRMATION stage
    session = get_or_create_session(session_id)
    session.booking_state.stage = BookingStage.AWAITING_CONFIRMATION
    session.booking_state.patient_name = "Priya Patel"
    session.booking_state.patient_id = "PAT-002"
    session.booking_state.patient_type = "registered"
    session.booking_state.modality = "OPG"
    session.booking_state.selected_slot = {
        "slot_id": "SLOT-103",
        "datetime_str": "Tomorrow, Oct 19 at 11:15 AM",
        "modality": "OPG",
        "room": "Panoramic Suite 2",
    }
    session.booking_state.summary_displayed = True

    # User declines
    resp = run_multi_agent_pipeline("No, cancel that, don't book it", session_id=session_id)

    # State must be cancelled / idle
    state = get_or_create_session(session_id).booking_state
    assert state.stage == BookingStage.CANCELLED or state.stage == BookingStage.IDLE
    assert "cancelled" in resp["message"].lower()

    # SLOT-103 must remain available
    available = check_available_slots()
    slot_103 = next(s for s in available if s["slot_id"] == "SLOT-103")
    assert slot_103["status"] == "available"


def test_slot_switching_before_confirmation():
    """Verify that user can change their selected slot while in AWAITING_CONFIRMATION."""
    session_id = "test-session-switch-slot"

    # Present CBCT slots
    run_multi_agent_pipeline("Show CBCT slots for patient Vikram Malhotra tomorrow", session_id=session_id)
    # Select slot 1 (SLOT-101 at 10:00 AM)
    run_multi_agent_pipeline("Select SLOT-101", session_id=session_id)

    state = get_or_create_session(session_id).booking_state
    assert state.stage == BookingStage.AWAITING_CONFIRMATION
    assert state.selected_slot["slot_id"] == "SLOT-101"

    # Clinician changes mind: switch to SLOT-102 (02:30 PM)
    resp = run_multi_agent_pipeline("Actually change that to SLOT-102 at 2:30 PM", session_id=session_id)
    state = get_or_create_session(session_id).booking_state

    # Selected slot must be updated to SLOT-102
    assert state.stage == BookingStage.AWAITING_CONFIRMATION
    assert state.selected_slot["slot_id"] == "SLOT-102"
    assert "SLOT-102" in resp["message"]

    # Neither slot should be booked yet
    available = check_available_slots()
    assert next(s for s in available if s["slot_id"] == "SLOT-101")["status"] == "available"
    assert next(s for s in available if s["slot_id"] == "SLOT-102")["status"] == "available"


def test_idempotency_prevents_duplicate_booking():
    """Verify that repeating confirmation on an already confirmed booking returns idempotency notice."""
    session_id = "test-session-idempotency"

    # Fast forward: simulate confirmed booking
    session = get_or_create_session(session_id)
    session.booking_state.stage = BookingStage.CONFIRMED
    session.booking_state.patient_name = "Sunita Rao"
    session.booking_state.completed_booking = {
        "booking_reference": "VISCAN-9999",
        "patient_name": "Sunita Rao",
        "slot": {"slot_id": "SLOT-105", "datetime_str": "Oct 20 at 09:00 AM", "modality": "Bitewing"},
    }

    # Clinician inadvertently repeats "yes confirm"
    resp = run_multi_agent_pipeline("Yes please confirm it", session_id=session_id)

    assert "already confirmed" in resp["message"].lower() or "viscan-9999" in resp["message"].lower()
    assert "VISCAN-9999" in resp["message"]


def test_lookup_existing_booking():
    """Verify that 'I already booked' or 'Show my appointment' queries real confirmed bookings."""
    # First, book a slot directly via booking_service
    book_res = confirm_booking(
        slot_id="SLOT-106",
        patient_name="Amitav Ghosh",
        patient_id="PAT-GHOSH",
        scan_type="Periapical",
    )
    assert book_res["status"] in ("success", "confirmed")
    booking_ref = book_res["booking_reference"]

    session_id = "test-session-lookup"
    # Clinician asks to lookup
    resp = run_multi_agent_pipeline(f"Show appointment details for {booking_ref}", session_id=session_id)

    assert booking_ref in resp["message"]
    assert "Amitav Ghosh" in resp["message"]
    assert "SLOT-106" in resp["message"]

    # Also lookup by patient name
    resp2 = run_multi_agent_pipeline("Look up existing appointment for Amitav Ghosh", session_id=session_id)
    assert booking_ref in resp2["message"]


def test_cancel_booking_releases_slot():
    """Verify that cancelling a booking releases the slot back to inventory."""
    # Book a slot
    book_res = confirm_booking(
        slot_id="SLOT-107",
        patient_name="Meera Sen",
        scan_type="CBCT",
    )
    booking_ref = book_res["booking_reference"]

    # Slot 107 is booked (not in available slots)
    assert not any(s["slot_id"] == "SLOT-107" for s in check_available_slots())

    session_id = "test-session-cancel"
    # Cancel via orchestrator
    resp = run_multi_agent_pipeline(f"Please cancel booking {booking_ref}", session_id=session_id)

    assert "cancelled" in resp["message"].lower()
    assert booking_ref in resp["message"]

    # Verify slot is back to available
    assert any(s["slot_id"] == "SLOT-107" for s in check_available_slots())


def test_dynamic_clock_date_calculation():
    """Verify that slots calculate dynamic relative dates ('Today', 'Tomorrow', 'Oct 20') relative to frozen clock."""
    # Frozen clock is set to 2026-10-18 in fixture
    slots_day18 = check_available_slots(preferred_date="2026-10-18")
    slots_day19 = check_available_slots(preferred_date="2026-10-19")
    slots_day20 = check_available_slots(preferred_date="2026-10-20")

    # 2026-10-19 is tomorrow relative to 2026-10-18
    assert len(slots_day19) > 0
    for s in slots_day19:
        assert s["datetime_str"].startswith("Tomorrow, Oct 19")

    # 2026-10-20 is 2 days later, so it should display as "Oct 20 at ..."
    assert len(slots_day20) > 0
    for s in slots_day20:
        assert s["datetime_str"].startswith("Oct 20 at")
        assert "Tomorrow" not in s["datetime_str"]

    # Now shift frozen clock to 2026-10-19
    set_frozen_clock(datetime(2026, 10, 19, 8, 0, 0))

    # Re-check 2026-10-19 slots: now they must say "Today, Oct 19 at ..."
    slots_rechecked = check_available_slots(preferred_date="2026-10-19")
    for s in slots_rechecked:
        assert s["datetime_str"].startswith("Today, Oct 19")


def test_booking_confirmation_token_and_hospital_instructions():
    """Verify that confirmed bookings provide a queue token and realistic hospital arrival instructions."""
    session_id = "test-session-token-instructions"
    reset_slots()

    # Step 1: Request booking for registered patient
    res1 = run_multi_agent_pipeline("Book CBCT for patient Arjun Mehta tomorrow", session_id=session_id)
    assert res1["status"] == "confirmation_required"

    # Step 2: Select slot
    res2 = run_multi_agent_pipeline("Let's select the 10:00 AM slot", session_id=session_id)
    assert res2["status"] == "confirmation_required"
    assert "Arjun Mehta" in res2["message"]

    # Step 3: Explicit confirmation
    res3 = run_multi_agent_pipeline("Yes, confirm and book it", session_id=session_id)
    assert res3["status"] == "success"

    # Verify Token number and Hospital Arrival Instructions
    msg = res3["message"]
    assert "Token / Queue #" in msg
    assert "TK-101" in msg
    assert "Hospital Arrival Instructions" in msg
    assert "15 minutes" in msg
    assert "reception" in msg.lower()
    assert "metal" in msg.lower()

    # Verify lookup also presents the token
    res_lookup = run_multi_agent_pipeline("Show appointment details for Arjun Mehta", session_id="test-session-lookup-token")
    assert "TK-101" in res_lookup["message"]
    assert "Arjun Mehta" in res_lookup["message"]
