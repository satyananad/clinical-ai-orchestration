"""
Tests for Model Context Protocol (MCP) and Tool Calling Integrations
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.mcp_tools import (
    DENTAL_ASSISTANT_TOOLS,
    get_ollama_tool_definitions,
    execute_tool,
    handle_mcp_jsonrpc_request
)

client = TestClient(app)


def test_mcp_tool_definitions():
    """Verify tool definitions conform to MCP and OpenAI function schema."""
    assert len(DENTAL_ASSISTANT_TOOLS) >= 3
    tool_names = [t["name"] for t in DENTAL_ASSISTANT_TOOLS]
    assert "search_patient_reports" in tool_names
    assert "structure_clinical_dictation" in tool_names
    assert "summarize_active_report" in tool_names

    ollama_tools = get_ollama_tool_definitions()
    assert len(ollama_tools) == len(DENTAL_ASSISTANT_TOOLS)
    assert ollama_tools[0]["type"] == "function"
    assert "name" in ollama_tools[0]["function"]


def test_mcp_jsonrpc_tools_list():
    """Test standard MCP JSON-RPC 2.0 tools/list request."""
    rpc_payload = {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/list",
        "params": {}
    }
    response = handle_mcp_jsonrpc_request(rpc_payload)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 42
    assert "result" in response
    assert "tools" in response["result"]
    assert len(response["result"]["tools"]) >= 3


def test_mcp_jsonrpc_tools_call_retrieval():
    """Test standard MCP JSON-RPC 2.0 tools/call for report search with disambiguation check."""
    rpc_payload = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "search_patient_reports",
            "arguments": {
                "patient_name": "Vikram Malhotra",
                "modality": "OPG"
            }
        }
    }
    response = handle_mcp_jsonrpc_request(rpc_payload)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 101
    assert "content" in response["result"]
    assert len(response["result"]["content"]) > 0
    
    import json
    parsed_res = json.loads(response["result"]["content"][0]["text"])
    assert parsed_res["status"] == "success"
    assert parsed_res["patient"]["patient_name"] == "Vikram Malhotra"
    assert parsed_res["report"]["report_type"] == "OPG"


def test_mcp_jsonrpc_tools_call_dictation():
    """Test standard MCP JSON-RPC 2.0 tools/call for clinical dictation structuring."""
    rpc_payload = {
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": {
            "name": "structure_clinical_dictation",
            "arguments": {
                "dictation_text": "Tooth 25 shows no definite periapical radiolucency with moderate alveolar bone loss."
            }
        }
    }
    response = handle_mcp_jsonrpc_request(rpc_payload)
    assert response["jsonrpc"] == "2.0"
    import json
    parsed_res = json.loads(response["result"]["content"][0]["text"])
    assert parsed_res["status"] == "success"
    assert len(parsed_res["findings"]) > 0
    # Tooth 25 is recognized either in anatomical_sites or in findings
    site_names = [s.get("anatomical_name", "") for s in parsed_res.get("anatomical_sites", [])]
    fdi_nums = [s.get("fdi_number") for s in parsed_res.get("anatomical_sites", [])]
    assert 25 in fdi_nums or any("25" in str(name) for name in site_names)


def test_mcp_api_endpoints():
    """Test FastAPI /api/mcp HTTP endpoints."""
    # GET /api/mcp/tools
    res = client.get("/api/mcp/tools")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["mcp_tools"]) >= 3

    # POST /api/mcp (JSON-RPC)
    rpc_res = client.post("/api/mcp", json={
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/list",
        "params": {}
    })
    assert rpc_res.status_code == 200
    rpc_data = rpc_res.json()
    assert rpc_data["jsonrpc"] == "2.0"
    assert len(rpc_data["result"]["tools"]) >= 3
