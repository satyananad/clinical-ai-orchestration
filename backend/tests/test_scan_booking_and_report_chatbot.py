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

    # Turn 1: Clinician asks to book an appointment
    resp1 = run_multi_agent_pipeline("I want to book appointment", session_id=session_id)
    assert resp1["status"] == "confirmation_required"
    assert "available_slots" in resp1
    assert "None scan" not in resp1["message"]

    # Turn 2: Clinician specifies the time "October 10:00 a.m."
    resp2 = run_multi_agent_pipeline("October 10:00 a.m.", session_id=session_id)
    assert resp2["status"] == "success"
    assert resp2["agent_trace"]["active_agent"] == "ScanBookingAgent"
    assert "booking_reference" in resp2
    assert "10:00 AM" in resp2["message"]


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


