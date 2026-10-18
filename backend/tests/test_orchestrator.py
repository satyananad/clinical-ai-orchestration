"""
TESTS FOR MULTI-AGENT ORCHESTRATOR & MCP TOOL DISPATCHING
---------------------------------------------------------
Verifies:
1. Triage Agent routing to RetrievalAgent via search_patient_reports.
2. Triage Agent routing to ClinicalScribeAgent via structure_clinical_dictation.
3. Triage Agent routing to ConsultationAgent via summarize_active_report.
4. Correct agent_trace and MCP JSON-RPC payload generation.
"""

import pytest
from backend.app.services.orchestrator_agent import run_multi_agent_pipeline, triage_clinician_request
from backend.app.services.conversation_service import get_or_create_session

def test_triage_routing_to_retrieval_agent():
    """Verify that a request to open a report routes to RetrievalAgent and search_patient_reports."""
    triage = triage_clinician_request("Open Priya Sharma's latest OPG report", session_id="test_triage_1")
    assert triage["selected_agent"] == "RetrievalAgent"
    assert triage["tool_name"] == "search_patient_reports"
    assert "Priya" in triage["arguments"]["patient_name"]

def test_triage_routing_to_clinical_scribe_agent():
    """Verify that clinical findings dictation routes to ClinicalScribeAgent and structure_clinical_dictation."""
    text = "Lower right third molar is horizontally impacted with apical radiolucency"
    triage = triage_clinician_request(text, session_id="test_triage_2")
    assert triage["selected_agent"] == "ClinicalScribeAgent"
    assert triage["tool_name"] == "structure_clinical_dictation"
    assert triage["arguments"]["dictation_text"] == text

def test_triage_routing_to_consultation_agent():
    """Verify that contextual follow-up routes to ConsultationAgent."""
    session_id = "test_triage_3"
    s = get_or_create_session(session_id)
    s.set_active("PAT-1001", "RPT-2001")
    
    triage = triage_clinician_request("Summarize her report", session_id=session_id)
    assert triage["selected_agent"] == "ConsultationAgent"
    assert triage["tool_name"] == "summarize_active_report"

def test_run_multi_agent_pipeline_execution_trace():
    """Verify end-to-end multi-agent execution returns agent_trace with MCP JSON-RPC payload."""
    res = run_multi_agent_pipeline("Open Vikram Malhotra's latest CBCT report", session_id="test_trace_session")
    assert res["status"] == "success"
    assert "agent_trace" in res
    trace = res["agent_trace"]
    assert trace["orchestrator"] == "TriageAgent"
    assert trace["active_agent"] == "RetrievalAgent"
    assert trace["tool_called"] == "search_patient_reports"
    assert trace["mcp_jsonrpc"]["method"] == "tools/call"
    assert trace["mcp_jsonrpc"]["params"]["name"] == "search_patient_reports"


def test_greeting_pipeline_returns_clean_welcome_without_db_stats():
    """Verify conversational greetings ('hey', 'hello') do not dump database statistics or citations."""
    for greeting in ["hey", "hello", "hi", "good morning", "help"]:
        res = run_multi_agent_pipeline(greeting, session_id="test_greeting_session")
        assert res["status"] == "success"
        assert res.get("agent_trace", {}).get("active_agent") == "GeneralAssistantAgent"
        assert "Hello! I am your Dental Clinical Workstation Assistant." in res.get("answer", "")
        assert "35 radiological reports" not in res.get("answer", "")
        assert res.get("citations") == []

