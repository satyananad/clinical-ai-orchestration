"""
Automated Evaluation Suite for:
1. Voice/Typed Scan-Booking Agent (Synthetic Appointment Slots & Confirmation)
2. Report-Aware Chatbot (Grounded Q&A over selected reports)
3. Advanced Multi-Tooth CBCT Dictation Parsing (Tooth 6, 5, 7, 28, Tooth C)
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
from backend.app.services.conversation_service import get_or_create_session
from backend.app.services.booking_service import check_available_slots, confirm_booking
from backend.app.services.llm_service import parse_dictation_with_llm

client = TestClient(app)

import os
import json
from backend.app.services.booking_service import BOOKINGS_FILE

@pytest.fixture(autouse=True)
def reset_slots_state():
    # Save original
    with open(BOOKINGS_FILE, "r") as f:
        original = f.read()
    yield
    # Restore
    with open(BOOKINGS_FILE, "w") as f:
        f.write(original)

def test_check_synthetic_appointment_slots():
    """Verifies that the scan-booking agent returns available slots for radiological modalities."""
    cbct_slots = check_available_slots("CBCT")
    assert len(cbct_slots) >= 1
    assert all(s["modality"] == "CBCT" for s in cbct_slots)
    assert cbct_slots[0]["status"] == "available"

    opg_slots = check_available_slots("OPG")
    assert len(opg_slots) >= 1
    assert all(s["modality"] == "OPG" for s in opg_slots)

def test_scan_booking_multi_turn_requires_confirmation():
    """
    Verifies that the Scan-Booking Agent:
    Turn 1: Checks slots and asks for confirmation before booking (status: confirmation_required).
    Turn 2: Confirms booking upon clinician approval.
    """
    session_id = "test_booking_turn_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Turn 1: Inquire about booking
    resp1 = run_multi_agent_pipeline("Book a CBCT scan appointment", session_id=session_id)
    assert resp1["status"] == "confirmation_required"
    assert "available_slots" in resp1
    assert len(resp1["available_slots"]) > 0
    assert "Confirmation Required" in resp1["message"]
    first_slot_id = resp1["available_slots"][0]["slot_id"]

    # Turn 2: Confirm booking
    resp2 = run_multi_agent_pipeline(f"Yes, please confirm slot {first_slot_id}", session_id=session_id)
    assert resp2["status"] == "success"
    assert "booking_reference" in resp2
    assert resp2["booking_reference"].startswith("BKG-")
    assert resp2["agent_trace"]["active_agent"] == "ScanBookingAgent"

def test_report_aware_chatbot_grounded_qa():
    """
    Verifies that the Report-Aware Chatbot:
    1. Answers questions strictly grounded in the loaded report.
    2. Provides clinical citations from the report findings/impressions.
    """
    session_id = "test_qa_grounded_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Retrieve Vikram Malhotra's CBCT report
    retrieval_res = run_multi_agent_pipeline("Open Vikram Malhotra's CBCT", session_id=session_id)
    assert retrieval_res["status"] == "success"
    assert session.active_report_id is not None

    # Ask grounded clinical question about tooth 48
    qa_res = run_multi_agent_pipeline("What are the findings for tooth 48?", session_id=session_id)
    assert qa_res["status"] == "success"
    assert qa_res["agent_trace"]["active_agent"] == "ConsultationAgent"
    assert len(qa_res.get("citations", [])) > 0
    # Vikram Malhotra's report mentions tooth 48 impaction / relation to nerve canal
    answer_text = qa_res.get("answer", "").lower()
    assert any(term in answer_text for term in ["48", "third molar", "mandibular", "impacted", "canal"])

def test_cbct_multi_tooth_dictation_structuring():
    """
    Verifies that the CBCT dictation containing Universal notation (Tooth 6, 5, 7, 28, Tooth C)
    is properly parsed and mapped to standardized FDI tooth notations.
    """
    dictation_text = (
        "A medium field-of-view CBCT examination of the maxillofacial region was performed. "
        "The study is limited by noise artifacts. The area of interest is tooth 6. "
        "Tooth 6 is mesio-palatally angulated and fully bony impacted, with its crown positioned "
        "at the level of the apical third of the roots of adjacent teeth 5 and 7. "
        "A well-defined, corticated pericoronal radiolucency with two loculations and a large "
        "internal calcification inferiorly is associated with tooth 6. "
        "Teeth 1–32 are present. The primary canine (tooth C) is retained and shows moderate "
        "physiologic root resorption. Idiopathic osteosclerosis is noted apical to tooth 28. "
        "The findings for tooth 6 are suggestive of an adenomatoid odontogenic tumor (AOT)."
    )

    parsed = parse_dictation_with_llm(dictation_text)
    assert "anatomical_sites" in parsed
    sites = parsed["anatomical_sites"]
    fdi_numbers = [s.get("fdi_number") for s in sites if s.get("fdi_number")]

    # Universal 6 is FDI 13 (Maxillary right permanent canine)
    assert 13 in fdi_numbers
    # Universal 5 is FDI 14 (Maxillary right first premolar)
    assert 14 in fdi_numbers
    # Universal 7 is FDI 12 (Maxillary right lateral incisor)
    assert 12 in fdi_numbers
    # Universal 28 is FDI 44 (Mandibular right first premolar)
    assert 44 in fdi_numbers
    # Tooth C is FDI 53 (Deciduous maxillary right canine)
    assert 53 in fdi_numbers

    # Verify findings contain impaction and pericoronal radiolucency
    findings = parsed["findings"]
    obs_text = " ".join(f.get("observation", "") for f in findings).lower()
    assert "impacted" in obs_text or "angulated" in obs_text
    assert "radiolucency" in obs_text or "calcification" in obs_text


def test_scan_booking_time_based_confirmation():
    """
    Verifies that when a clinician enters a natural time string like 'October 10:00 a.m.'
    or '10:00 AM' after appointment slots are displayed, the system:
    1. Does NOT mistake the time for a patient name retrieval.
    2. Correctly matches the corresponding slot (SLOT-101 at 10:00 AM).
    3. Successfully books and confirms the slot.
    """
    session_id = "test_time_booking_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Turn 1: Clinician asks to check appointment slots
    resp1 = run_multi_agent_pipeline("Check available appointment slots", session_id=session_id)
    assert resp1["status"] == "confirmation_required"
    assert "available_slots" in resp1
    assert "None scan" not in resp1["message"]

    # Turn 2: Clinician specifies the time "October 10:00 a.m." -> presents Review card
    resp2 = run_multi_agent_pipeline("October 10:00 a.m.", session_id=session_id)
    assert resp2["status"] == "confirmation_required"
    assert "Review appointment" in resp2["message"]

    # Turn 3: Clinician confirms booking
    resp3 = run_multi_agent_pipeline("Yes, please confirm", session_id=session_id)
    assert resp3["status"] == "success"
    assert resp3["agent_trace"]["active_agent"] == "ScanBookingAgent"
    assert "booking_reference" in resp3
    assert "10:00 AM" in resp3["message"]


def test_scan_booking_date_based_slot_query():
    """
    Verifies that when a clinician asks for slots on a specific date:
    'how many appoint ment slot have 20 october'
    The agent filters slots exclusively to October 20 (3 slots: SLOT-105, 106, 107).
    """
    session_id = "test_date_slots_session"
    session = get_or_create_session(session_id)
    session.clear()

    resp = run_multi_agent_pipeline("how many appoint ment slot have 20 october", session_id=session_id)
    assert resp["status"] == "confirmation_required"
    assert "available_slots" in resp
    assert len(resp["available_slots"]) == 3
    assert all("Oct 20" in s["datetime_str"] for s in resp["available_slots"])
    assert "October 20" in resp["message"]


def test_report_aware_chatbot_pdf_and_explanation():
    """
    Verifies that the Report-Aware Chatbot:
    1. Returns a direct PDF link when requested ('Show me the PDF').
    2. Explains the report comprehensively when requested ('Explain this report').
    """
    session_id = "test_pdf_and_explain_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Step 1: Open Vikram Malhotra's CBCT
    r1 = run_multi_agent_pipeline("Open Vikram Malhotra's CBCT report", session_id=session_id)
    assert r1["status"] == "success"

    # Step 2: Ask for PDF
    r2 = run_multi_agent_pipeline("Show me the PDF", session_id=session_id)
    assert r2["status"] == "success"
    assert "pdf_url" in r2
    assert "/patient-pdf/RPT-2026_Vikram_Malhotra_CBCT.pdf" in r2["answer"]

    # Step 3: Ask to explain report
    r3 = run_multi_agent_pipeline("Explain this report", session_id=session_id)
    assert r3["status"] == "success"
    assert "Summary for Vikram Malhotra" in r3["answer"]
    assert "Mandibular Canal & Tooth 48" in r3["answer"]


def test_general_question_dynamic_answering_no_predefined_repetition():
    """
    Verifies that the chatbot answers general clinical questions, workstation capability questions,
    and conversational inquiries dynamically and does NOT repeat a predefined static greeting.
    """
    session_id = "test_general_dynamic_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Query 1: Clinical concept: CBCT
    r1 = run_multi_agent_pipeline("what is a cbct scan?", session_id=session_id)
    assert r1["status"] == "success"
    assert r1["agent_trace"]["active_agent"] == "GeneralAssistantAgent"
    assert any(term in r1["answer"].lower() for term in ["cone beam", "3d", "volumetric", "tomography"])
    assert "Hello! I am your Dental Clinical Workstation Assistant." not in r1["answer"]

    # Query 2: Clinical concept: Caries
    r2 = run_multi_agent_pipeline("explain dental caries", session_id=session_id)
    assert r2["status"] == "success"
    assert r2["agent_trace"]["active_agent"] == "GeneralAssistantAgent"
    assert any(term in r2["answer"].lower() for term in ["caries", "demineralization", "decay", "enamel", "dentin"])
    assert "Hello! I am your Dental Clinical Workstation Assistant." not in r2["answer"]

    # Query 3: Workstation capabilities
    r3 = run_multi_agent_pipeline("what are you doing?", session_id=session_id)
    assert r3["status"] == "success"
    assert any(term in r3["answer"].lower() for term in ["assistant", "retrieval", "imaging", "dictation", "workstation"])

    # Query 4: Clinical concept: OPG
    r4 = run_multi_agent_pipeline("what is an opg?", session_id=session_id)
    assert r4["status"] == "success"
    assert any(term in r4["answer"].lower() for term in ["panoramic", "radiograph", "orthopantomogram", "dentition"])
    assert "Hello! I am your Dental Clinical Workstation Assistant." not in r4["answer"]


def test_direct_time_slot_confirmation_without_prior_pending_slots():
    """
    Verifies that if a clinician inputs a time like 'October 10:00 a.m.' directly
    even when session pending_booking_slots was None, it never searches for a patient
    named 'October 10:00 A.M.' and successfully resolves to the matching appointment slot.
    """
    session_id = "test_cold_slot_session"
    session = get_or_create_session(session_id)
    session.clear()
    assert session.pending_booking_slots is None

    res = run_multi_agent_pipeline("Confirm October 10:00 a.m.", session_id=session_id)
    assert res["status"] == "success"
    assert res["agent_trace"]["active_agent"] == "ScanBookingAgent"
    assert "booking_reference" in res
    assert "10:00 AM" in res["message"]


def test_report_retrieval_includes_pdf_url_and_chat_link():
    """
    Verifies that retrieving a patient report provides direct pdf_url and formatted
    clickable link in the chat response.
    """
    session_id = "test_retrieval_pdf_session"
    session = get_or_create_session(session_id)
    session.clear()

    res = run_multi_agent_pipeline("Open Vikram Malhotra's CBCT report", session_id=session_id)
    assert res["status"] == "success"
    assert res["pdf_url"] == "/patient-pdf/RPT-2026_Vikram_Malhotra_CBCT.pdf"
    assert "[View Diagnostic PDF](/patient-pdf/RPT-2026_Vikram_Malhotra_CBCT.pdf)" in res["answer"]


def test_doctor_and_hospital_inquiry():
    """
    Verifies that when a clinician asks for doctor and hospital name (e.g. 'docoter and hospital name'),
    the assistant does not output placeholder '[Clinic Name]' or an intro greeting,
    but instead responds dynamically explaining the dental radiology clinic suites, doctors, and capabilities.
    """
    session_id = "test_doc_hosp_session"
    session = get_or_create_session(session_id)
    session.clear()

    res = run_multi_agent_pipeline("docoter and hospital name", session_id=session_id)
    assert res["status"] == "success"
    assert res["agent_trace"]["active_agent"] == "GeneralAssistantAgent"
    assert "[Clinic Name]" not in res["answer"]
    assert any(term in res["answer"].lower() for term in ["hospital", "doctor", "radiology", "suite", "radiologist"])


def test_doctor_availability_query_and_booking_workflow():
    """
    Verifies that an inquiry like 'is there any dr have 23rd october':
    1. Routes directly to ScanBookingAgent.
    2. Identifies October 23.
    3. Informs that no slots exist for Oct 23 and lists upcoming open slots (Oct 19 & 20).
    4. Displays Review Card when clinician picks 'October 10:00 a.m.'.
    5. Confirms booking upon clinician confirmation.
    """
    session_id = "test_oct_23_booking_workflow"
    session = get_or_create_session(session_id)
    session.clear()

    # Step 1: Doctor availability inquiry for 23rd October
    res1 = run_multi_agent_pipeline("is there any dr have 23rd october", session_id=session_id)
    assert res1["status"] == "confirmation_required"
    assert res1["agent_trace"]["active_agent"] == "ScanBookingAgent"
    assert "October 23" in res1["message"]
    assert "No appointment slots" in res1["message"]
    assert "SLOT-101" in res1["message"]
    assert "Confirmation Required" in res1["message"]

    # Step 2: Clinician picks 'October 10:00 a.m.' -> displays review card
    res2 = run_multi_agent_pipeline("October 10:00 a.m.", session_id=session_id)
    assert res2["status"] == "confirmation_required"
    assert "Review appointment" in res2["message"]

    # Step 3: Clinician confirms
    res3 = run_multi_agent_pipeline("Yes, confirm", session_id=session_id)
    assert res3["status"] == "success"
    assert res3["agent_trace"]["active_agent"] == "ScanBookingAgent"
    assert "booking_reference" in res3
    assert "10:00 AM" in res3["message"]


def test_tooth_48_findings_chip_removed_from_ui():
    """Verify that 'Tooth 48 Findings' chip is completely removed from the UI dashboard."""
    from backend.app.main import get_dashboard_ui
    html = get_dashboard_ui()
    assert "Tooth 48 Findings" not in html
    assert "Check Scan Slots" in html


def test_booking_workflow_captures_doctor_age_location_and_patient():
    """
    Verify that booking workflow collects and displays:
    1. Patient Name and Age
    2. Referring/Attending Doctor
    3. Room & Clinic Location
    4. Slot & Modality details
    """
    from backend.app.services.booking_service import reset_slots
    reset_slots()
    session_id = "test_doc_age_location_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Step 1: Clinician specifies patient, age, referring doctor, and modality
    r1 = run_multi_agent_pipeline(
        "Book a Periapical scan tomorrow at 10:30 AM for patient Anita Verma, age 34, referred by Dr. Sharma",
        session_id=session_id
    )
    assert r1["status"] in ("success", "confirmation_required")
    assert "Anita Verma" in r1["message"]
    assert "34 years" in r1["message"] or "Age: 34" in r1["message"]
    assert "Dr. Sharma" in r1["message"]
    assert "Intraoral X-Ray Bay 3" in r1["message"]
    assert "Viscan Dental Imaging Center - Main Clinic" in r1["message"]


def test_booking_with_active_patient_calculates_age_and_location():
    """
    Verify that when an active patient is selected in session,
    booking automatically derives patient name, ID, calculated age from DOB,
    and formats room and clinic location.
    """
    from backend.app.services.booking_service import reset_slots
    reset_slots()
    session_id = "test_active_patient_booking_session"
    session = get_or_create_session(session_id)
    session.clear()

    # Retrieve Priya Sharma (PAT-1001, DOB: 1998-04-12 -> 28 years)
    r_retrieval = run_multi_agent_pipeline("Open Priya Sharma PAT-1001's OPG report", session_id=session_id)
    assert r_retrieval["status"] == "success"
    assert session.active_patient_id == "PAT-1001"

    # Select a slot
    r_select = run_multi_agent_pipeline("Select slot SLOT-106", session_id=session_id)
    assert r_select["status"] == "confirmation_required"
    assert "Priya Sharma" in r_select["message"]
    assert "PAT-1001" in r_select["message"]
    assert "28 years" in r_select["message"] or "1998" in r_select["message"]
    assert "Dr. R. Mehta" in r_select["message"]
    assert "Intraoral X-Ray Bay 3" in r_select["message"]
    assert "Viscan Dental Imaging Center - Main Clinic" in r_select["message"]
    assert "Confirmation Required" in r_select["message"]

    # Confirm booking
    r_confirm = run_multi_agent_pipeline("Yes, confirm", session_id=session_id)
    assert r_confirm["status"] == "success"
    assert "Priya Sharma" in r_confirm["message"]
    assert "Dr. R. Mehta" in r_confirm["message"]
    assert "Intraoral X-Ray Bay 3" in r_confirm["message"]
    assert "Viscan Dental Imaging Center - Main Clinic" in r_confirm["message"]
    assert "Token / Queue #" in r_confirm["message"]


def test_multiturn_booking_preserves_patient_age_doctor_and_facility():
    """
    Verifies that multi-entity booking queries preserve patient details across turns:
    Turn 1: 'Book a CBCT scan for patient Rahul Verma age 48 Dr. Mehta Main Clinic'
    Turn 2: 'Confirm slot SLOT-107'
    Asserting that slot confirmation attributes patient to Rahul Verma (PAT-1003),
    age 48 years, Dr. Mehta, and registered patient status instead of defaulting to Walk-in Patient.
    """
    from backend.app.services.booking_service import reset_slots
    reset_slots()
    session_id = "test_multiturn_booking_rahul"
    session = get_or_create_session(session_id)
    session.clear()

    # Turn 1: Initial query with patient, age, doctor, and clinic location
    r1 = run_multi_agent_pipeline(
        "Book a CBCT scan for patient Rahul Verma age 48 Dr. Mehta Main Clinic",
        session_id=session_id
    )
    assert r1["status"] in ("confirmation_required", "slots_presented")
    assert "Rahul Verma" in r1["message"]
    assert "CBCT" in r1["message"]
    trace_args = r1.get("agent_trace", {}).get("arguments", {})
    assert trace_args.get("patient_name") == "Rahul Verma"
    assert trace_args.get("patient_id") == "PAT-1003"
    assert trace_args.get("patient_age") == "48 years"
    assert trace_args.get("doctor_name") == "Dr. Mehta"

    # Turn 2: Follow-up slot confirmation
    r2 = run_multi_agent_pipeline("Confirm slot SLOT-107", session_id=session_id)
    assert r2["status"] == "success"
    slot = r2.get("slot", {})
    assert slot.get("patient_name") == "Rahul Verma"
    assert slot.get("patient_id") == "PAT-1003"
    assert slot.get("patient_age") == "48 years"
    assert slot.get("doctor_name") == "Dr. Mehta"
    assert slot.get("patient_status") == "REGISTERED"
    assert "Rahul Verma (PAT-1003)" in r2["message"]
    assert "Age: 48 years" in r2["message"]
    assert "Dr. Mehta" in r2["message"]
    assert "Viscan Dental Imaging Center - Main Clinic" in r2["message"]







