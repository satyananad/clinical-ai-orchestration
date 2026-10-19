from fastapi import APIRouter
from ..schemas.report import AssistantChatRequest, AssistantChatResponse
from ..services.conversation_service import get_or_create_session
from ..services.orchestrator_agent import run_multi_agent_pipeline

router = APIRouter(tags=["Assistant"])

@router.post("/assistant/chat", response_model=AssistantChatResponse)
def assistant_chat_endpoint(payload: AssistantChatRequest):
    """
    Multi-Agent conversational interface for voice/text requests:
    - Triage Orchestrator: Dispatches to specialized sub-agent via MCP Tool Calling.
    - Retrieval Agent: Locates patient records and enforces safety stops on ambiguous names.
    - Scribe Agent: Converts spoken dictation into validated JSON drafts.
    - Consultation Agent: Answers report questions grounded with verbatim citations.
    """
    session_id = payload.session_id or "default_session"
    session = get_or_create_session(session_id)
    if payload.active_patient_id and not session.active_patient_id:
        session.set_active_patient(payload.active_patient_id)
    if payload.active_report_id and not session.active_report_id:
        session.set_active(payload.active_patient_id or session.active_patient_id, payload.active_report_id)
    result = run_multi_agent_pipeline(payload.query, session_id=session_id)
    return result

from pydantic import BaseModel
from typing import Optional

class SelectPatientRequest(BaseModel):
    patient_id: str

class SelectReportRequest(BaseModel):
    patient_id: str
    report_id: str

@router.post("/sessions/{session_id}/select-patient")
def select_patient_endpoint(session_id: str, payload: SelectPatientRequest):
    """Explicitly select active patient in server-side session state without opening a report."""
    from ..services.retrieval import get_patient
    patient = get_patient(payload.patient_id)
    if not patient:
        return {"status": "not_found", "message": f"Patient '{payload.patient_id}' not found."}
    session = get_or_create_session(session_id)
    session.set_active_patient(patient["patient_id"])
    return {
        "status": "success",
        "message": f"Patient {patient['patient_name']} ({patient['patient_id']}) selected.",
        "patient": patient,
        "reports": patient.get("reports", [])
    }

@router.post("/sessions/{session_id}/select-report")
def select_report_endpoint(session_id: str, payload: SelectReportRequest):
    """Explicitly select active report in server-side session state."""
    from ..services.retrieval import get_patient, get_report_by_id
    patient = get_patient(payload.patient_id)
    report = get_report_by_id(payload.report_id)
    if not patient or not report:
        return {"status": "not_found", "message": "Patient or report not found."}
    session = get_or_create_session(session_id)
    session.set_active(patient["patient_id"], report["report_id"])
    return {
        "status": "success",
        "message": f"Opened {report['report_type']} report ({report['report_id']}) for {patient['patient_name']}.",
        "patient": patient,
        "report": report
    }

@router.get("/sessions/{session_id}/context")
def get_session_context_endpoint(session_id: str):
    """Fetch current active patient and report from session state."""
    from ..services.retrieval import get_patient, get_report_by_id
    session = get_or_create_session(session_id)
    patient = get_patient(session.active_patient_id) if session.active_patient_id else None
    report = get_report_by_id(session.active_report_id) if session.active_report_id else None
    return {
        "session_id": session_id,
        "active_patient_id": session.active_patient_id,
        "active_report_id": session.active_report_id,
        "active_patient": patient,
        "active_report": report
    }

@router.post("/sessions/{session_id}/reset")
def reset_session_endpoint(session_id: str):
    """Clear conversation context and active patient/report selection."""
    session = get_or_create_session(session_id)
    session.clear()
    return {
        "status": "success",
        "message": f"Session '{session_id}' has been reset.",
        "session_id": session_id
    }

