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
    result = run_multi_agent_pipeline(payload.query, session_id=session_id)
    return result

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
