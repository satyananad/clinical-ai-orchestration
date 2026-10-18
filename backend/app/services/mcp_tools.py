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
import re
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
    },
    {
        "name": "query_clinic_database",
        "description": "Answers natural language questions about the clinic database, including total scan counts, modality breakdowns (OPG, CBCT, Bitewing), report counts for a patient, or lists of registered patients.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The natural language question asked by the clinician."
                },
                "patient_name": {
                    "type": "string",
                    "description": "Optional patient name if the question inquires about a specific patient."
                },
                "modality": {
                    "type": "string",
                    "description": "Optional scan modality (e.g. 'OPG', 'CBCT', 'Bitewing') if querying about a scan type."
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "check_appointment_slots",
        "description": "Checks available synthetic radiological appointment slots (CBCT, OPG, Bitewing, Periapical) and presents options for clinician/patient confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "description": "The requested imaging modality (e.g. 'CBCT', 'OPG', 'Bitewing')."
                },
                "patient_name": {
                    "type": "string",
                    "description": "The patient's name to book for."
                },
                "preferred_date": {
                    "type": "string",
                    "description": "Optional preferred date or day (e.g. 'tomorrow', '2026-10-19')."
                }
            },
            "required": []
        }
    },
    {
        "name": "confirm_scan_booking",
        "description": "Confirms and books a specific radiological appointment slot after explicit clinician or patient confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slot_id": {
                    "type": "string",
                    "description": "The selected slot ID to confirm (e.g. 'SLOT-101')."
                },
                "patient_name": {
                    "type": "string",
                    "description": "The patient name for the confirmed appointment."
                },
                "patient_id": {
                    "type": "string",
                    "description": "Optional patient ID."
                }
            },
            "required": ["patient_name"]
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

def execute_clinic_database_query(question: str, patient_name: Optional[str] = None, modality: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic clinical database query executor.
    Calculates exact facts, scan counts, modality breakdowns, and patient report lists.
    """
    from .retrieval import get_all_patients
    patients = get_all_patients()
    total_patients = len(patients)
    total_reports = sum(len(p.get("reports", [])) for p in patients)
    patient_names = [
        f"{p.get('patient_name')} ({p.get('patient_id')})"
        if any(other['patient_name'] == p['patient_name'] and other['patient_id'] != p['patient_id'] for other in patients)
        else p.get("patient_name")
        for p in patients
    ]

    # Modality aggregation
    modality_to_patients = {}
    for p in patients:
        p_name = p.get("patient_name")
        for r in p.get("reports", []):
            m_type = r.get("report_type", "Other")
            if m_type not in modality_to_patients:
                modality_to_patients[m_type] = []
            modality_to_patients[m_type].append({
                "patient_name": p_name,
                "patient_id": p.get("patient_id"),
                "report_id": r.get("report_id")
            })

    # Patient matching if name provided or mentioned in question
    target_name = patient_name or ""
    matched_patients = []
    if target_name:
        for p in patients:
            if target_name.lower() in p.get("patient_name", "").lower():
                matched_patients.append(p)
    if not matched_patients:
        for p in patients:
            p_full = p.get("patient_name", "")
            if p_full.lower() in question.lower():
                matched_patients.append(p)
        if not matched_patients:
            for p in patients:
                first_name = p.get("patient_name", "").split()[0].lower()
                if len(first_name) > 3 and first_name in question.lower():
                    matched_patients.append(p)

    # Modality matching if specified
    target_mod = modality or ""
    matched_modality = None
    if target_mod:
        for real_m in modality_to_patients.keys():
            if target_mod.lower() in real_m.lower():
                matched_modality = real_m
                break
    if not matched_modality:
        for m_key in ["cbct", "opg", "bitewing", "periapical", "cephalogram", "dental examination", "iopa"]:
            if m_key in question.lower():
                for real_m in modality_to_patients.keys():
                    if m_key in real_m.lower():
                        matched_modality = real_m
                        break
                if matched_modality:
                    break

    modality_summary = ", ".join([f"{k}: {len(v)}" for k, v in modality_to_patients.items()])

    # Formulate answer
    q_low = question.lower().strip()
    is_greeting = any(re.search(r"\b" + re.escape(g) + r"\b", q_low) for g in ["hey", "hello", "hi", "good morning", "good afternoon", "greetings", "help", "who are you"]) and not any(w in q_low for w in ["database", "report", "patient", "scan", "how many", "list", "count", "record"])
    if is_greeting:
        return {
            "status": "success",
            "answer": "Hello! I am your Dental Clinical Workstation Assistant. How can I help you today? You can ask me to open a patient scan (e.g. 'Open Vikram Malhotra's CBCT'), dictate a clinical report, summarize active findings, or check appointment slots.",
            "citations": []
        }

    if matched_patients:
        if len(matched_patients) == 1:
            p = matched_patients[0]
            p_cnt = len(p.get("reports", []))
            r_desc = ", ".join([f"{r.get('report_type')} ({r.get('report_id')})" for r in p.get("reports", [])])
            answer = f"{p['patient_name']} has {p_cnt} reports in the database: {r_desc}."
        else:
            lines = [f"• {p['patient_name']} ({p.get('patient_id')}): {len(p.get('reports', []))} reports ({', '.join(r.get('report_type') for r in p.get('reports', []))})" for p in matched_patients]
            answer = f"I found {len(matched_patients)} matching patients in the database:\n" + "\n".join(lines)
    elif matched_modality:
        records = modality_to_patients.get(matched_modality, [])
        pts = list(dict.fromkeys([f"{r['patient_name']} ({r['report_id']})" for r in records]))
        answer = f"There is {len(records)} {matched_modality} scan in the database across: {', '.join(pts)}." if len(records) == 1 else f"There are {len(records)} {matched_modality} scans in the database across: {', '.join(pts)}."
    elif "patient" in question.lower() and any(w in question.lower() for w in ["how many", "list", "show", "who", "all"]):
        answer = f"There are {total_patients} registered patients in the clinic database: {', '.join(patient_names)}."
    else:
        answer = f"There are currently {total_reports} radiological reports across {total_patients} registered patients in the clinic database ({modality_summary}). You can ask me to open any patient's report (e.g. 'Open Vikram Malhotra's CBCT')."

    return {
        "status": "success",
        "answer": answer,
        "citations": [f"Clinic Database: {total_reports} total scans across {total_patients} registered patients"]
    }

# =====================================================================
# 2. DETERMINISTIC TOOL EXECUTION ENGINE
# =====================================================================
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes business logic in response to an LLM Tool Call.
    Separates probabilistic LLM reasoning from deterministic business-rule execution.
    """
    try:
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
            guardrails = verify_anti_hallucination(dictation_text, parsed.get("findings", []), impressions=parsed.get("impressions", []))
            return {
                "status": "success",
                "patient_details": parsed.get("patient_details"),
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

        elif tool_name == "query_clinic_database":
            question = arguments.get("question", "")
            p_name = arguments.get("patient_name")
            mod = arguments.get("modality")
            return execute_clinic_database_query(question=question, patient_name=p_name, modality=mod)

        elif tool_name == "check_appointment_slots":
            from .booking_service import check_available_slots
            raw_mod = arguments.get("modality")
            modality = raw_mod if (raw_mod and str(raw_mod).lower() not in ["none", "any"]) else "Radiology"
            preferred_date = arguments.get("preferred_date")
            slots = check_available_slots(modality=raw_mod or "Any", preferred_date=preferred_date)
            return {
                "status": "success",
                "modality": modality,
                "available_slots": slots,
                "count": len(slots),
                "message": f"Found {len(slots)} available appointment slot(s) for {modality} scan. Please confirm which slot you would like to book."
            }

        elif tool_name == "confirm_scan_booking":
            from .booking_service import confirm_booking
            slot_id = arguments.get("slot_id", "")
            patient_name = arguments.get("patient_name", "")
            patient_id = arguments.get("patient_id")
            return confirm_booking(slot_id=slot_id, patient_name=patient_name, patient_id=patient_id)

        else:
            return {
                "status": "error",
                "message": f"Unknown tool name: '{tool_name}'",
                "retry_allowed": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Clinical tool '{tool_name}' failed to execute: {str(e)}",
            "retry_allowed": True
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
        is_error = exec_result.get("status") in ["error", "not_found"]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(exec_result, indent=2)
                    }
                ],
                "isError": is_error
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
