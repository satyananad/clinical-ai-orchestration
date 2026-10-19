"""
TEST SUITE FOR META LLAMA 3.1 INTEGRATION & GENERATIVE WORKSTATION TASKS
------------------------------------------------------------------------
Verifies that:
1. Llama 3.1 is configured as the active LLM engine across the workstation.
2. Clinical dictation structuring utilizes Llama 3.1 to generate Pydantic-validated findings.
3. LLM tool-calling & triage dispatches specialized agents based on natural-language reasoning.
4. Report-grounded QA leverages Llama 3.1 to synthesize factual answers with verbatim citations.
5. Standard MCP function schemas conform to Ollama / Llama 3.1 tool-calling format.
6. Graceful degradation: If the Ollama daemon is offline, deterministic clinical safety gates protect execution.
"""

import os
import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.llm_service import (
    OLLAMA_MODEL,
    OLLAMA_ENDPOINT,
    OLLAMA_CHAT_ENDPOINT,
    query_llm,
    query_ollama,
    query_ollama_chat_tools,
    parse_dictation_with_llm
)
from backend.app.services.orchestrator_agent import (
    select_tool_with_llm,
    triage_clinician_request,
    generate_dynamic_conversational_reply
)
from backend.app.services.mcp_tools import get_ollama_tool_definitions, DENTAL_ASSISTANT_TOOLS
from backend.app.services.conversation_service import get_or_create_session, SESSIONS
from backend.app.services.report_qa_service import answer_report_question

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_session_state():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_llama31_model_env_configuration():
    """Verify that Llama 3.1 is configured as the default model."""
    assert OLLAMA_MODEL == "llama3.1"
    assert "generate" in OLLAMA_ENDPOINT
    assert "chat" in OLLAMA_CHAT_ENDPOINT


def test_llama31_native_function_calling_schema():
    """Verify that MCP tool definitions match Ollama / Llama 3.1 tool-calling format."""
    ollama_tools = get_ollama_tool_definitions()
    assert len(ollama_tools) == len(DENTAL_ASSISTANT_TOOLS)
    for tool in ollama_tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_llama31_dictation_structuring_generative():
    """Verify that clinical dictation structuring executes LLM prompt and parses into Pydantic model."""
    dictation_text = (
        "Patient is a 52-year-old male. CBCT shows vertical bone loss on distal aspect of tooth 47, "
        "measuring 6.5 mm. Tooth 48 is horizontally impacted in close approximation to mandibular nerve canal."
    )

    mock_llm_json = json.dumps({
        "patient_details": {
            "age": 52,
            "gender": "Male",
            "demographics_summary": "52-year-old male",
            "study_type": "CBCT"
        },
        "anatomical_sites": [
            {
                "site_type": "Tooth",
                "anatomical_name": "Mandibular right second molar",
                "fdi_number": 47,
                "quadrant": "Mandibular Right"
            },
            {
                "site_type": "Tooth",
                "anatomical_name": "Mandibular right third molar",
                "fdi_number": 48,
                "quadrant": "Mandibular Right"
            }
        ],
        "findings": [
            {
                "anatomical_site": "Tooth 47",
                "tooth_number": "47",
                "category": "Periodontal",
                "observation": "Vertical bone loss on distal aspect of tooth 47",
                "polarity": "present",
                "certainty": "definite observation",
                "measurements": ["6.5 mm"],
                "source_quote": "vertical bone loss on distal aspect of tooth 47, measuring 6.5 mm"
            },
            {
                "anatomical_site": "Tooth 48",
                "tooth_number": "48",
                "category": "Impaction",
                "observation": "Horizontally impacted tooth 48 in close proximity to nerve canal",
                "polarity": "present",
                "certainty": "definite observation",
                "measurements": [],
                "source_quote": "Tooth 48 is horizontally impacted in close approximation to mandibular nerve canal"
            }
        ],
        "impressions": [
            "Severe localized vertical bone loss at tooth 47 (6.5 mm).",
            "Complete horizontal impaction of tooth 48 approximating inferior alveolar canal."
        ],
        "recommendations": [
            "Periodontal evaluation and surgical assessment for tooth 48 retrieval."
        ],
        "extraction_status": "COMPLETE"
    })

    with patch("backend.app.services.llm_service.query_llm", return_value=mock_llm_json):
        result = parse_dictation_with_llm(dictation_text)
        assert result["extraction_status"] == "COMPLETE"
        assert len(result["findings"]) == 2
        assert result["patient_details"]["age"] == 52
        assert result["patient_details"]["gender"] == "Male"
        assert any(f["tooth_number"] == "47" for f in result["findings"])
        assert any(f["tooth_number"] == "48" for f in result["findings"])
        assert len(result["impressions"]) == 2


def test_llama31_dynamic_tool_triage():
    """Verify that Llama 3.1 dynamically routes unseen clinical queries to the appropriate agent."""
    session = get_or_create_session("test_llama31_triage_session")

    mock_triage_response = json.dumps({
        "selected_agent": "ScanBookingAgent",
        "tool_name": "check_appointment_slots",
        "arguments": {
            "modality": "CBCT",
            "preferred_date": "2026-10-20"
        },
        "rationale": "Clinician requests available 3D CBCT imaging slots for tomorrow."
    })

    with patch("backend.app.services.orchestrator_agent.query_llm", return_value=mock_triage_response):
        decision = select_tool_with_llm("Can we check the available schedule for a 3D CBCT tomorrow morning?", session)
        assert decision is not None
        assert decision["selected_agent"] == "ScanBookingAgent"
        assert decision["tool_name"] == "check_appointment_slots"
        assert decision["arguments"]["modality"] == "CBCT"


def test_llama31_report_grounded_qa_synthesis():
    """Verify that clinical questions prompt Llama 3.1 with report grounding and return citations."""
    mock_llm_answer = (
        "Based on OPG report RPT-2001, tooth 48 demonstrates mesioangular impaction "
        "with roots in close proximity to the right inferior alveolar nerve canal. "
        "Attending radiologist recommends surgical extraction under local anesthesia."
    )

    with patch("backend.app.services.report_qa_service.query_llm", return_value=mock_llm_answer):
        qa_result = answer_report_question("RPT-2001", "What did the radiologist say about wisdom tooth 48?")
        assert qa_result["status"] == "success"
        assert "mesioangular impaction" in qa_result["answer"].lower()
        assert len(qa_result.get("citations", [])) > 0


def test_llama31_dynamic_conversational_reply():
    """Verify that general workstation queries invoke Llama 3.1 to generate helpful conversational responses."""
    session = get_or_create_session("test_llama31_conv_session")

    mock_assistant_reply = (
        "I am your Clinical Workstation Assistant powered by Meta Llama 3.1. "
        "You can ask me to retrieve patient imaging (e.g. 'Open Priya Sharma's CBCT'), "
        "dictate diagnostic findings, or book imaging appointments."
    )

    with patch("backend.app.services.orchestrator_agent.query_llm", return_value=mock_assistant_reply):
        reply = generate_dynamic_conversational_reply("Hello, what capabilities do you have?", session)
        assert "Meta Llama 3.1" in reply or "Clinical Workstation Assistant" in reply


def test_llama31_offline_graceful_degradation():
    """Verify that when Ollama is offline (returns None), the workstation gracefully degrades to deterministic safety rules."""
    with patch("backend.app.services.llm_service.query_llm", return_value=None):
        # 1. Dictation structuring degrades to dynamic semantic parse with review notice
        dictation = "Tooth 16 has occlusal caries with normal periapical tissues."
        result = parse_dictation_with_llm(dictation)
        assert result is not None
        assert len(result["findings"]) >= 1
        assert result["extraction_status"] == "FALLBACK_PARSED_REQUIRES_REVIEW"

        # 2. End-to-end API call succeeds with safety banner
        response = client.post("/dictation/structure", json={"text": dictation})
        assert response.status_code == 200
        data = response.json()
        assert data["report_metadata"]["clinician_review_required"] is True
        assert len(data["findings"]) >= 1
