"""
TEST SUITE FOR REAL-LIFE APPOINTMENT BOOKING WORKFLOW & 14 BACKEND MCP TOOLS
-----------------------------------------------------------------------------
Verifies the complete real-life appointment booking workflow requested by the user:
- Step 1: Identify patient (new vs existing inquiry, search, deduplication)
- Step 2: Collect patient details (automatic age calculation from DOB, no redundant prompts)
- Step 3: Understand visit reason (imaging procedure vs doctor consultation)
- Step 4: Select specialization (verified clinic catalog)
- Step 5: Choose doctor (verified doctor directory, never invented)
- Step 6: Collect date, time, and branch preferences (check real scheduling availability)
- Step 7: Final review summary (exact Section 4 markdown card)
- Step 8: Book, verify, and notify (atomic commit, disk verification, notification dispatch)
- Comprehensive unit & integration testing of all 14 MCP backend tools.
"""

import pytest
from datetime import datetime

from backend.app.services.clock import set_frozen_clock
from backend.app.services.booking_service import (
    reset_slots,
    load_slots,
    check_available_slots,
    check_appointment_availability,
    get_existing_appointments,
    book_appointment,
    reschedule_appointment,
    cancel_appointment,
    get_appointment_status,
    send_appointment_notification,
    NOTIFICATION_DISPATCH_LOG,
)
from backend.app.services.clinic_catalog import (
    list_specializations,
    list_services,
    search_doctors,
    get_doctor_details,
    get_service_details,
    find_matching_specialization,
    CLINIC_SPECIALIZATIONS,
    DOCTORS_DIRECTORY,
)
from backend.app.services.patient_registration import (
    register_patient,
    get_patient_details,
    calculate_age_from_dob,
    normalize_dob,
)
from backend.app.services.mcp_tools import execute_tool
from backend.app.services.conversation_service import (
    SESSIONS,
    get_or_create_session,
    BookingStage,
)
from backend.app.services.orchestrator_agent import run_multi_agent_pipeline


from backend.app.services.retrieval import DATA_FILE, reset_patients

@pytest.fixture(autouse=True)
def setup_workflow_environment():
    """Reset slots, clear sessions, and freeze clock to 2026-10-18 09:00:00."""
    reset_slots()
    SESSIONS.clear()
    NOTIFICATION_DISPATCH_LOG.clear()
    set_frozen_clock(datetime(2026, 10, 18, 9, 0, 0))
    with open(DATA_FILE, "r") as f:
        baseline_raw = f.read()
    yield
    set_frozen_clock(None)
    reset_slots()
    SESSIONS.clear()
    NOTIFICATION_DISPATCH_LOG.clear()
    with open(DATA_FILE, "w") as f:
        f.write(baseline_raw)
    reset_patients()


# ==============================================================================
# 1. TEST THE COMPLETE 7-TURN REAL-LIFE CONVERSATIONAL WORKFLOW (Section 3)
# ==============================================================================
def test_full_7_turn_real_life_specialist_booking_flow():
    """
    Verifies the multi-turn conversational appointment-booking dialogue:
    Turn 1: 'I want to book an appointment with a dental specialist.' -> asks existing vs first visit
    Turn 2: 'This is my first visit.' -> prompts for required registration details
    Turn 3: Registration text provided -> registers patient, calculates age, asks visit reason
    Turn 4: 'I need a CBCT scan for implant planning in the lower jaw.' -> identifies specialization & doctors, asks preferences
    Turn 5: 'Tomorrow morning at the main clinic please.' -> queries availability, presents morning slots
    Turn 6: 'Let's go with the 10:30 AM slot.' -> renders exact Section 4 markdown review card
    Turn 7: 'Yes, please confirm.' -> commits atomically, verifies persistence, dispatches notification
    """
    session_id = "session-real-life-flow-001"
    session = get_or_create_session(session_id)
    session.clear()

    # --------------------------------------------------------------------------
    # Turn 1: Inquire about appointment with a dental specialist
    # --------------------------------------------------------------------------
    r1 = run_multi_agent_pipeline("I want to book an appointment with a dental specialist.", session_id=session_id)
    assert r1["status"] in ("clarification_required", "confirmation_required")
    msg1 = r1["message"].lower()
    assert "existing patient" in msg1 or "first visit" in msg1
    assert session.booking_state.stage == BookingStage.COLLECTING_DETAILS

    # --------------------------------------------------------------------------
    # Turn 2: User indicates first visit / new patient
    # --------------------------------------------------------------------------
    r2 = run_multi_agent_pipeline("This is my first visit.", session_id=session_id)
    assert r2["status"] == "clarification_required"
    msg2 = r2["message"].lower()
    assert "full name" in msg2
    assert "date of birth" in msg2
    assert "phone" in msg2
    assert session.booking_state.is_new_patient is True

    # --------------------------------------------------------------------------
    # Turn 3: User provides complete registration details
    # --------------------------------------------------------------------------
    r3 = run_multi_agent_pipeline(
        "My name is Ananya Rao, born on 14 February 1996. My phone is +91-98765-43210 and email is ananya.rao@example.com, female.",
        session_id=session_id
    )
    assert r3["status"] == "clarification_required"
    msg3 = r3["message"]
    assert "Ananya Rao" in msg3
    assert "PAT-" in msg3
    assert "30" in msg3  # Age calculated: 2026 - 1996 = 30
    assert "reason for your visit" in msg3.lower() or "consultation" in msg3.lower()
    assert session.booking_state.stage == BookingStage.SELECTING_SERVICE
    assert session.booking_state.patient_name == "Ananya Rao"

    # --------------------------------------------------------------------------
    # Turn 4: User specifies visit reason (CBCT scan for implant planning)
    # --------------------------------------------------------------------------
    r4 = run_multi_agent_pipeline("I need a CBCT scan for implant planning in the lower jaw.", session_id=session_id)
    assert r4["status"] == "clarification_required"
    msg4 = r4["message"]
    assert "Oral and Maxillofacial Radiology" in msg4 or "Radiology" in msg4
    assert "Dr. Rajesh Mehta" in msg4
    assert "preferred date" in msg4.lower() or "morning" in msg4.lower()
    assert session.booking_state.modality == "CBCT"
    assert "implant" in (session.booking_state.notes or session.booking_state.service or "")

    # --------------------------------------------------------------------------
    # Turn 5: User specifies timing and clinic location preferences
    # --------------------------------------------------------------------------
    r5 = run_multi_agent_pipeline("Tomorrow morning at the main clinic please.", session_id=session_id)
    assert r5["status"] == "confirmation_required"
    assert "available_slots" in r5
    slots_5 = r5["available_slots"]
    assert len(slots_5) > 0
    # Must contain tomorrow morning slot
    assert any("10:00 AM" in s["datetime_str"] for s in slots_5)
    assert session.booking_state.stage == BookingStage.SLOTS_PRESENTED

    # --------------------------------------------------------------------------
    # Turn 6: User selects the 10:00 AM slot (SLOT-101)
    # --------------------------------------------------------------------------
    r6 = run_multi_agent_pipeline("Let's go with the 10:00 AM slot.", session_id=session_id)
    assert r6["status"] == "confirmation_required"
    msg6 = r6["message"]
    # Verify exact Section 4 Markdown Review Card formatting
    assert "## Review appointment" in msg6
    assert "Pending confirmation" in msg6
    assert "Patient\nAnanya Rao" in msg6
    assert "Patient ID" in msg6
    assert "Service" in msg6
    assert "Specialization\nOral and Maxillofacial Radiology" in msg6
    assert "Doctor" in msg6
    assert "Date and time" in msg6
    assert "Location" in msg6
    assert "Slot ID\nSLOT-101" in msg6
    assert "Would you like me to book this exact appointment?" in msg6
    assert session.booking_state.stage == BookingStage.AWAITING_CONFIRMATION

    # Slot must still be available before explicit confirmation
    slot_before = next(s for s in load_slots() if s["slot_id"] == "SLOT-101")
    assert slot_before.get("status") == "available"

    # --------------------------------------------------------------------------
    # Turn 7: User provides explicit confirmation
    # --------------------------------------------------------------------------
    r7 = run_multi_agent_pipeline("Yes, please confirm.", session_id=session_id)
    assert r7["status"] == "success"
    msg7 = r7["message"]
    assert "Appointment confirmed for Ananya Rao" in msg7
    assert "BKG-101" in msg7
    assert "TK-101" in msg7
    assert "Hospital Arrival Instructions" in msg7
    assert session.booking_state.stage == BookingStage.CONFIRMED

    # Verify slot is persisted as booked in inventory
    slot_after = next(s for s in load_slots() if s["slot_id"] == "SLOT-101")
    assert slot_after["status"] == "booked"
    assert slot_after["patient_name"] == "Ananya Rao"

    # Verify notification dispatch log
    assert len(NOTIFICATION_DISPATCH_LOG) >= 1
    notif = NOTIFICATION_DISPATCH_LOG[-1]
    assert notif["recipient"] == "Ananya Rao"
    assert notif["token_number"] == "TK-101"
    assert notif["status"] == "delivered"


# ==============================================================================
# 2. TEST PATIENT REGISTRATION DEDUPLICATION & AGE CALCULATION
# ==============================================================================
def test_patient_registration_deduplication_and_age():
    """Verify deduplication avoids duplicate registrations and age is calculated from DOB."""
    # 1. Normalization and calculation
    assert normalize_dob("14 February 1996") == "1996-02-14"
    assert normalize_dob("1996-02-14") == "1996-02-14"
    assert calculate_age_from_dob("1996-02-14") == 30

    # 2. Register first time
    unique_suffix = datetime.now().strftime("%f")
    p_name = f"Karan Malhotra {unique_suffix}"
    p_phone = f"+91-98111-{unique_suffix[:5]}"
    reg1 = register_patient(
        full_name=p_name,
        date_of_birth="1992-08-25",
        contact_number=p_phone,
        email="karan.m@example.com",
        sex="Male"
    )
    assert reg1["status"] == "success"
    assert reg1["is_new"] is True
    assert reg1["patient"]["patient_name"] == p_name
    p_id = reg1["patient"]["patient_id"]

    # 3. Deduplication: register second time with same name & DOB
    reg2 = register_patient(
        full_name=p_name,
        date_of_birth="1992-08-25",
        contact_number="+91-98999-88888",
    )
    assert reg2["status"] == "existing_patient"
    assert reg2["is_new"] is False
    assert reg2["patient"]["patient_id"] == p_id

    # 4. Deduplication: register with same phone number
    reg3 = register_patient(
        full_name="Karan Different",
        date_of_birth="1995-01-01",
        contact_number=p_phone
    )
    assert reg3["status"] == "existing_patient"
    assert reg3["patient"]["patient_id"] == p_id


# ==============================================================================
# 3. TEST CLINIC CATALOG & DOCTOR DIRECTORY TOOLS
# ==============================================================================
def test_clinic_catalog_and_doctors():
    """Verify list_specializations, list_services, and search_doctors functions."""
    # 1. Specializations
    specs = list_specializations()
    assert len(specs) == 6
    spec_names = [s["name"] for s in specs]
    assert "Oral and Maxillofacial Radiology" in spec_names
    assert "Oral and Maxillofacial Surgery" in spec_names
    assert "Orthodontics" in spec_names
    assert "Endodontics" in spec_names
    assert "Periodontics" in spec_names
    assert "General Dentistry" in spec_names

    # 2. Services
    services = list_services()
    assert len(services) >= 6
    cbct_srv = next(s for s in services if s["service_id"] == "SRV-CBCT")
    assert cbct_srv["category"] == "imaging"
    assert cbct_srv["requires_referral"] is True

    # 3. Doctors search
    docs = search_doctors(specialization="Oral and Maxillofacial Radiology")
    assert len(docs) >= 1
    dr_mehta = next(d for d in docs if "Mehta" in d["name"])
    assert dr_mehta["specialization"] == "Oral and Maxillofacial Radiology"

    # 4. Doctor details
    doc_det = get_doctor_details("DOC-101")
    assert doc_det is not None
    assert "Mehta" in doc_det["name"]

    # 5. Service details
    srv_det = get_service_details("SRV-CBCT")
    assert srv_det is not None
    assert srv_det["modality"] == "CBCT"


# ==============================================================================
# 4. TEST ALL 14 MCP TOOLS VIA MCP EXECUTOR
# ==============================================================================
def test_all_14_mcp_tools():
    """Executes all 14 backend MCP tools through execute_tool to ensure contract adherence."""
    # Tool 1: search_patients
    t1 = execute_tool("search_patients", {"query": "Rahul"})
    assert t1["status"] == "success"
    assert len(t1["patients"]) >= 1

    # Tool 2: get_patient_details
    t2 = execute_tool("get_patient_details", {"patient_id": "PAT-1001"})
    assert t2["status"] == "success"
    assert t2["patient"]["patient_name"] == "Priya Sharma"

    # Tool 3: register_patient
    t3 = execute_tool("register_patient", {
        "full_name": "Deepa Nair",
        "date_of_birth": "1994-03-20",
        "contact_number": "+91-98777-66554",
        "email": "deepa@example.com",
        "sex": "Female"
    })
    assert t3["status"] in ("success", "existing_patient")
    deepa_id = t3["patient"]["patient_id"]

    # Tool 4: list_services
    t4 = execute_tool("list_services", {})
    assert t4["status"] == "success"
    assert len(t4["services"]) >= 6

    # Tool 5: list_specializations
    t5 = execute_tool("list_specializations", {})
    assert t5["status"] == "success"
    assert len(t5["specializations"]) == 6

    # Tool 6: search_doctors
    t6 = execute_tool("search_doctors", {"specialization": "Orthodontics"})
    assert t6["status"] == "success"
    assert any("Singhal" in d["name"] for d in t6["doctors"])

    # Tool 7: get_doctor_details
    t7 = execute_tool("get_doctor_details", {"doctor_id": "DOC-102"})
    assert t7["status"] == "success"
    assert "Deshmukh" in t7["doctor"]["name"]

    # Tool 8: check_appointment_availability
    t8 = execute_tool("check_appointment_availability", {
        "service": "CBCT scan for implant planning",
        "preferred_date": "2026-10-20"
    })
    assert t8["status"] == "success"
    assert len(t8["available_slots"]) > 0

    # Tool 9: get_existing_appointments
    t9 = execute_tool("get_existing_appointments", {"patient_id": "PAT-1001"})
    assert t9["status"] == "success"

    # Tool 10: book_appointment
    # Test confirmation gating: confirmed=False -> confirmation_required
    t10_gated = execute_tool("book_appointment", {
        "patient_id": deepa_id,
        "patient_name": "Deepa Nair",
        "slot_id": "SLOT-107",
        "confirmed": False
    })
    assert t10_gated["status"] == "confirmation_required"

    # Test confirmation commit: confirmed=True -> success
    t10 = execute_tool("book_appointment", {
        "patient_id": deepa_id,
        "patient_name": "Deepa Nair",
        "slot_id": "SLOT-107",
        "confirmed": True
    })
    assert t10["status"] == "success"
    booking_id = t10["booking_reference"]

    # Tool 11: reschedule_appointment
    t11 = execute_tool("reschedule_appointment", {
        "existing_booking_id": booking_id,
        "new_slot_id": "SLOT-105",
        "confirmed": True
    })
    assert t11["status"] == "success"
    resched_id = t11.get("booking_reference") or t11.get("slot", {}).get("booking_id") or booking_id

    # Tool 12: cancel_appointment
    t12 = execute_tool("cancel_appointment", {
        "booking_id": resched_id,
        "confirmed": True
    })
    assert t12["status"] == "success"

    # Tool 13: get_appointment_status
    t13 = execute_tool("get_appointment_status", {"booking_id": resched_id})
    assert t13["status"] == "success"
    assert t13["booking_status"] == "CANCELLED"

    # Tool 14: send_appointment_notification
    # Book a fresh slot to test notification
    b_fresh = execute_tool("book_appointment", {
        "patient_id": "PAT-1001",
        "patient_name": "Rahul Sharma",
        "slot_id": "SLOT-101",
        "confirmed": True
    })
    fresh_ref = b_fresh["booking_reference"]
    t14 = execute_tool("send_appointment_notification", {
        "booking_id": fresh_ref,
        "channel": "sms"
    })
    assert t14["status"] == "success"
    assert t14["notification"]["booking_id"] == fresh_ref
    assert t14["notification"]["status"] == "delivered"


def test_user_exact_phrase_real_life_booking_workflow():
    """
    Verifies that when a user says 'i want book appoint':
    1. It initiates real-life appointment booking (prompt_existing_or_new_patient) instead of auto Walk-in.
    2. Turn 2: User says 'first visit' -> prompts for registration details.
    3. Turn 3: User registers -> registers patient, calculates age, asks reason for visit.
    4. Turn 4: User specifies Bitewing scan -> presents available slots (including SLOT-105).
    5. Turn 5: User picks 'slot 105' (or 'Tomorrow, Oct 20 at 09:00 AM') ->
       MUST show Review appointment summary card awaiting confirmation, and NOT commit!
    6. Turn 6: User says 'Yes, confirm' -> commits booking, returns token and arrival instructions.
    """
    session_id = "test-user-exact-flow-105"
    session = get_or_create_session(session_id)
    session.clear()

    # Turn 1: User says exact phrase "i want book appoint"
    t1 = run_multi_agent_pipeline("i want book appoint", session_id=session_id)
    assert t1["status"] == "clarification_required"
    assert "existing patient" in t1["message"].lower() or "first visit" in t1["message"].lower()
    assert session.booking_state.stage == BookingStage.COLLECTING_DETAILS

    # Turn 2: User says "first visit"
    t2 = run_multi_agent_pipeline("This is my first visit", session_id=session_id)
    assert t2["status"] == "clarification_required"
    assert "full name" in t2["message"].lower()
    assert "date of birth" in t2["message"].lower()

    # Turn 3: User provides registration details
    t3 = run_multi_agent_pipeline(
        "My name is Divya Sen, born 10 May 1995, phone +91-98765-00000, female",
        session_id=session_id
    )
    assert t3["status"] == "clarification_required"
    assert "Divya Sen" in t3["message"]
    assert "PAT-" in t3["message"]
    assert "31" in t3["message"]  # 2026 - 1995 = 31 years
    assert "reason for your visit" in t3["message"].lower() or "service" in t3["message"].lower()
    assert session.booking_state.patient_name == "Divya Sen"

    # Turn 4: User specifies visit reason
    t4 = run_multi_agent_pipeline("I need a Bitewing scan", session_id=session_id)
    assert t4["status"] == "clarification_required"
    assert "General Dentistry" in t4["message"]
    assert "preferred date" in t4["message"].lower()

    # Turn 5: User specifies timing preference
    t5 = run_multi_agent_pipeline("October 20 morning", session_id=session_id)
    assert t5["status"] == "confirmation_required"
    assert "available_slots" in t5
    assert any(s["slot_id"] == "SLOT-105" for s in t5["available_slots"])
    assert session.booking_state.stage == BookingStage.SLOTS_PRESENTED

    # Turn 6: User picks 'slot 105'
    # Architectural requirement: MUST NOT auto-confirm! Must present Review Card!
    t6 = run_multi_agent_pipeline("slot 105", session_id=session_id)
    assert t6["status"] == "confirmation_required"
    assert "Review appointment" in t6["message"]
    assert "Divya Sen" in t6["message"]
    assert "SLOT-105" in t6["message"]
    assert "Confirmation Required" in t6["message"]
    assert session.booking_state.stage == BookingStage.AWAITING_CONFIRMATION
    assert session.booking_state.selected_slot["slot_id"] == "SLOT-105"

    # Verify inventory is not mutated yet
    avail_before = check_available_slots()
    assert any(s["slot_id"] == "SLOT-105" for s in avail_before)

    # Turn 7: Explicit affirmative confirmation
    t7 = run_multi_agent_pipeline("Yes, confirm", session_id=session_id)
    assert t7["status"] == "success"
    assert "Appointment confirmed for Divya Sen" in t7["message"]
    assert "Divya Sen" in t7["message"]
    assert "BKG-105-" in t7["message"] or "BKG-" in t7["message"]
    assert "TK-105" in t7["message"] or "Token" in t7["message"]
    assert session.booking_state.stage == BookingStage.CONFIRMED

    # Verify inventory is updated
    avail_after = check_available_slots()
    assert not any(s["slot_id"] == "SLOT-105" for s in avail_after)


# ==============================================================================
# 6. TEST INCREMENTAL NEW PATIENT REGISTRATION (NAME-ONLY THEN DOB/PHONE)
# ==============================================================================
def test_incremental_new_patient_registration_flow():
    """
    Verifies that when a clinician/patient replies to new patient registration with just their name
    (e.g., 'satyanand Shukla'), the system:
    1. Does NOT mistakenly route to Clinical Database Query
    2. Correctly captures the name and invokes prompt_patient_dob_and_contact
    3. Successfully completes registration when DOB / contact is subsequently provided.
    """
    import json
    from backend.app.services.retrieval import DATA_FILE
    with open(DATA_FILE, "r") as f:
        original_data = f.read()

    try:
        session_id = "test-incremental-registration-flow"
        sess = get_or_create_session(session_id)
        sess.clear()

        # Turn 1: User asks to book
        r1 = run_multi_agent_pipeline("i want to book the appointment", session_id=session_id)
        assert r1["status"] == "clarification_required"
        assert "existing patient" in r1["answer"].lower() or "first visit" in r1["answer"].lower()

        # Turn 2: User says first time
        r2 = run_multi_agent_pipeline("first time", session_id=session_id)
        assert r2["status"] == "clarification_required"
        assert "welcome to viscan dental" in r2["answer"].lower()
        assert sess.booking_state.is_new_patient is True

        # Turn 3: User replies with just their name
        r3 = run_multi_agent_pipeline("satyanand Shukla", session_id=session_id)
        assert r3["status"] == "clarification_required"
        assert "Satyanand Shukla" in r3["answer"]
        assert "date of birth" in r3["answer"].lower() or "dob" in r3["answer"].lower()
        assert r3.get("agent_trace", {}).get("tool_called") == "prompt_patient_dob_and_contact"
        assert "database" not in r3.get("agent_trace", {}).get("tool_called", "")
        assert sess.booking_state.patient_name == "Satyanand Shukla"

        # Turn 4: User provides DOB and phone number
        r4 = run_multi_agent_pipeline("14/02/1996, 9876543210", session_id=session_id)
        assert r4["status"] == "clarification_required"
        assert "Satyanand Shukla" in r4["answer"]
        assert "registered as a new patient" in r4["answer"].lower()
        assert r4.get("agent_trace", {}).get("tool_called") == "registration_completed_ask_service"
    finally:
        with open(DATA_FILE, "w") as f:
            f.write(original_data)

def test_single_word_name_incremental_registration_flow():
    """
    Verifies that when a clinician/patient replies to new patient registration with just a single-word name
    (e.g., 'satya'), the system:
    1. Correctly identifies it as patient name and invokes prompt_patient_dob_and_contact
    2. Does NOT loop back to prompt_existing_or_new_patient
    3. Completes registration when DOB and phone number are supplied.
    """
    import json
    from backend.app.services.retrieval import DATA_FILE
    with open(DATA_FILE, "r") as f:
        original_data = f.read()

    try:
        session_id = "test-single-name-satya"
        sess = get_or_create_session(session_id)
        sess.clear()

        # Turn 1: User asks to book
        r1 = run_multi_agent_pipeline("i want to book appointment", session_id=session_id)
        assert r1["status"] == "clarification_required"
        assert r1.get("agent_trace", {}).get("tool_called") == "prompt_existing_or_new_patient"

        # Turn 2: User says first time
        r2 = run_multi_agent_pipeline("first time", session_id=session_id)
        assert r2["status"] == "clarification_required"
        assert r2.get("agent_trace", {}).get("tool_called") == "prompt_new_patient_registration"
        assert sess.booking_state.is_new_patient is True

        # Turn 3: User replies with single name 'satya'
        r3 = run_multi_agent_pipeline("satya", session_id=session_id)
        assert r3["status"] == "clarification_required"
        assert r3.get("agent_trace", {}).get("tool_called") == "prompt_patient_dob_and_contact"
        assert "Satya" in r3["answer"]
        assert "date of birth" in r3["answer"].lower() or "dob" in r3["answer"].lower()
        assert sess.booking_state.patient_name == "Satya"

        # Turn 4: User provides DOB and phone number
        r4 = run_multi_agent_pipeline("14/02/1996, 9876543210", session_id=session_id)
        assert r4["status"] == "clarification_required"
        assert "Satya" in r4["answer"]
        assert "registered as a new patient" in r4["answer"].lower()
        assert r4.get("agent_trace", {}).get("tool_called") == "registration_completed_ask_service"
    finally:
        with open(DATA_FILE, "w") as f:
            f.write(original_data)


def test_satya_with_standalone_dob_17_08_2001():
    """
    Tests the exact sequence:
    1. 'first time'
    2. 'satya'
    3. '17/08/2001'
    Ensures that DD/MM/YYYY date format registers the patient, calculates age 25,
    and advances to service inquiry without repeating the DOB prompt.
    """
    import json
    from backend.app.services.retrieval import DATA_FILE
    with open(DATA_FILE, "r") as f:
        original_data = f.read()

    try:
        session_id = "test-satya-dob-2001"
        sess = get_or_create_session(session_id)
        sess.clear()

        # Turn 1: User says first time
        r1 = run_multi_agent_pipeline("first time", session_id=session_id)
        assert r1["status"] == "clarification_required"
        assert sess.booking_state.is_new_patient is True

        # Turn 2: User provides name 'satya'
        r2 = run_multi_agent_pipeline("satya", session_id=session_id)
        assert r2["status"] == "clarification_required"
        assert "Satya" in r2["answer"]
        assert sess.booking_state.patient_name == "Satya"

        # Turn 3: User provides standalone DOB '17/08/2001'
        r3 = run_multi_agent_pipeline("17/08/2001", session_id=session_id)
        assert r3["status"] == "clarification_required"
        assert "Satya" in r3["answer"]
        assert "registered as a new patient" in r3["answer"].lower()
        assert "25" in r3["answer"]  # 2026 - 2001 = 25
        assert r3.get("agent_trace", {}).get("tool_called") == "registration_completed_ask_service"
    finally:
        with open(DATA_FILE, "w") as f:
            f.write(original_data)



