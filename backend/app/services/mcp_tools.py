"""
MODEL CONTEXT PROTOCOL (MCP) & FUNCTION / TOOL CALLING SERVICE
-------------------------------------------------------------
Implements standardized Tool Calling & MCP (Model Context Protocol) specifications.
Decouples LLM natural language interpretation from deterministic clinical execution.

Exposed Tools:
1. search_patient_reports: Locates diagnostic imaging reports with identity disambiguation.
2. structure_clinical_dictation: Ingests clinical transcripts into structured findings & certainty.
3. summarize_active_report: Grounds conversational follow-up questions with source citations.
"""

import json
from typing import Dict, Any, List, Optional
from .retrieval import resolve_report_retrieval, get_patient, get_report_by_id
from .llm_service import parse_dictation_with_llm
from .validation import verify_anti_hallucination

# =====================================================================
# 1. TOOL CALLING DEFINITIONS (OpenAI / Ollama / MCP Compatible)
# =====================================================================
DENTAL_ASSISTANT_TOOLS = [
    {
        "name": "search_patient_reports",
        "description": "Searches for a patient's diagnostic radiological imaging reports (OPG, CBCT, Bitewing, Periapical). Enforces safety disambiguation if multiple patients share similar names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_name": {
                    "type": "string",
                    "description": "Full or partial name of the patient (e.g. 'Priya Sharma' or 'Vikram Malhotra')."
                },
                "modality": {
                    "type": "string",
                    "enum": ["OPG", "CBCT", "Bitewing", "Periapical", "Cephalogram", "Any"],
                    "description": "The radiological scan modality requested."
                }
            },
            "required": ["patient_name"]
        }
    },
    {
        "name": "structure_clinical_dictation",
        "description": "Converts spoken clinical observations and audio transcripts into validated structured JSON findings with dental tooth numbering and diagnostic certainty classification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dictation_text": {
                    "type": "string",
                    "description": "Verbatim speech-to-text transcript or clinical dictation audio text."
                },
                "patient_id": {
                    "type": "string",
                    "description": "Optional patient identifier to associate with the dictation draft."
                }
            },
            "required": ["dictation_text"]
        }
    },
    {
        "name": "summarize_active_report",
        "description": "Answers follow-up questions and summarizes findings grounded in the currently active patient report with verbatim citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "report_id": {
                    "type": "string",
                    "description": "The report ID to summarize or query."
                },
                "query": {
                    "type": "string",
                    "description": "The specific clinician question (e.g. 'Summarize her report' or 'What about tooth 48?')."
                }
            },
            "required": ["report_id"]
        }
    },
    {
        "name": "verify_clinical_claim",
        "description": "Verifies an AI-generated clinical claim against the source radiological report across tooth numbers, laterality, measurements, negation, and diagnostic certainty.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "report_id": {
                    "type": "string",
                    "description": "The source report ID to verify against."
                },
                "claim_text": {
                    "type": "string",
                    "description": "The clinical claim or statement to verify."
                }
            },
            "required": ["report_id", "claim_text"]
        }
    }
]

# Standard Ollama / OpenAI Tool Spec
def get_ollama_tool_definitions() -> List[Dict[str, Any]]:
    """Converts MCP schemas into Ollama / OpenAI function calling format."""
    ollama_tools = []
    for tool in DENTAL_ASSISTANT_TOOLS:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"]
            }
        })
    return ollama_tools

# =====================================================================
# 2. DETERMINISTIC TOOL EXECUTION ENGINE
# =====================================================================
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes business logic in response to an LLM Tool Call.
    Separates probabilistic LLM reasoning from deterministic business-rule execution.
    """
    if tool_name == "search_patient_reports":
        patient_name = arguments.get("patient_name", "").strip()
        patient_id = arguments.get("patient_id", "").strip()
        modality = arguments.get("modality", "Any").strip()
        if modality.upper() in ["PANORAMIC", "ORTHOPANTOMOGRAM"]:
            modality = "OPG"
        elif modality.upper() in ["CONE BEAM", "3D", "3D VOLUME"]:
            modality = "CBCT"
        return resolve_report_retrieval(query=patient_id or patient_name, patient_name=patient_name, modality=modality)

    elif tool_name == "structure_clinical_dictation":
        dictation_text = arguments.get("dictation_text", "")
        parsed = parse_dictation_with_llm(dictation_text)
        guardrails = verify_anti_hallucination(dictation_text, parsed.get("findings", []))
        return {
            "status": "success",
            "raw_dictation": dictation_text,
            "anatomical_sites": parsed.get("anatomical_sites", []),
            "findings": parsed.get("findings", []),
            "impressions": parsed.get("impressions", []),
            "recommendations": parsed.get("recommendations", []),
            "safety_guardrails": guardrails
        }

    elif tool_name == "summarize_active_report":
        report_id = arguments.get("report_id", "")
        query = arguments.get("query", "Summarize her report")
        report = get_report_by_id(report_id)
        if not report:
            return {"status": "not_found", "message": f"Report '{report_id}' not found."}

        citations = [
            f"Findings: {'; '.join(f.get('observation', '') for f in report.get('findings', []))}",
            f"Impression: {'; '.join(report.get('impression', []))}"
        ]
        return {
            "status": "success",
            "report_id": report_id,
            "report_type": report.get("report_type"),
            "clinical_indication": report.get("clinical_indication"),
            "findings": report.get("findings"),
            "impression": report.get("impression"),
            "citations": citations
        }

    elif tool_name == "verify_clinical_claim":
        from .claim_verification import verify_report_claim
        report_id = arguments.get("report_id", "")
        claim_text = arguments.get("claim_text", "")
        report = get_report_by_id(report_id)
        if not report:
            return {"status": "not_found", "message": f"Report '{report_id}' not found."}

        verif_res = verify_report_claim(claim_text, report, patient_id=report.get("patient_id"))
        return {
            "status": "success",
            "verification_status": verif_res.status.value,
            "passed": verif_res.passed,
            "violations": verif_res.violations,
            "citations": verif_res.evidence_citations
        }

    else:
        return {
            "status": "error",
            "message": f"Unknown tool name: '{tool_name}'"
        }

# =====================================================================
# 3. MODEL CONTEXT PROTOCOL (MCP) JSON-RPC DISPATCHER
# =====================================================================
def handle_mcp_jsonrpc_request(rpc_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard MCP (Model Context Protocol) JSON-RPC 2.0 Handler.
    Supports:
    - 'tools/list': Returns all available clinical tools.
    - 'tools/call': Executes a specific tool call with arguments.
    """
    method = rpc_payload.get("method")
    req_id = rpc_payload.get("id", 1)
    params = rpc_payload.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": DENTAL_ASSISTANT_TOOLS
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        exec_result = execute_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(exec_result, indent=2)
                    }
                ]
            }
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found."
            }
        }
