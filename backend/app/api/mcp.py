"""
MCP (Model Context Protocol) API Router
---------------------------------------
Exposes MCP endpoints following Anthropic / Model Context Protocol JSON-RPC 2.0 standard
and Ollama / OpenAI Tool Calling format for AI agent integration.
"""

from fastapi import APIRouter, Body
from typing import Dict, Any, List
from ..services.mcp_tools import (
    DENTAL_ASSISTANT_TOOLS,
    get_ollama_tool_definitions,
    execute_tool,
    handle_mcp_jsonrpc_request
)

router = APIRouter(prefix="/api/mcp", tags=["Model Context Protocol (MCP) & Tools"])


@router.get("/tools", summary="List Available Tools (Ollama / OpenAI / MCP)")
def list_tools():
    """
    Returns the schema definitions of all dental assistant tools in both
    standard MCP format and OpenAI / Ollama function calling format.
    """
    return {
        "status": "success",
        "mcp_tools": DENTAL_ASSISTANT_TOOLS,
        "openai_tools": get_ollama_tool_definitions()
    }


@router.post("", summary="MCP JSON-RPC 2.0 Endpoint")
def mcp_jsonrpc_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Standard Model Context Protocol (MCP) JSON-RPC 2.0 endpoint.
    Handles 'tools/list' and 'tools/call' requests from external AI agents or IDE clients.
    """
    return handle_mcp_jsonrpc_request(payload)


@router.post("/call", summary="Execute Tool Directly")
def execute_tool_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Convenience REST endpoint for invoking a tool directly:
    {
        "name": "search_patient_reports",
        "arguments": {"patient_name": "Priya Sharma", "modality": "OPG"}
    }
    """
    tool_name = payload.get("name")
    arguments = payload.get("arguments", {})
    result = execute_tool(tool_name, arguments)
    return {
        "tool": tool_name,
        "result": result
    }
