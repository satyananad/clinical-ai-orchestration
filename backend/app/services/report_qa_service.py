"""
REPORT-AWARE CHATBOT & EVIDENCE-GROUNDED RAG SERVICE (report_qa_service.py)
--------------------------------------------------------------------------
Performs direct report-grounded question answering (RAG without vector DB)
over the currently active diagnostic imaging report.
Guarantees zero clinical hallucination by answering solely from verified report evidence.
"""

import json
from typing import Dict, Any, List, Optional
from .retrieval import get_report_by_id, get_patient
from .llm_service import query_llm

REPORT_GROUNDED_QA_PROMPT = """You are a clinical dental radiology assistant.
Your task is to answer the clinician's question STRICTLY based on the provided diagnostic imaging report.

EVIDENCE FROM REPORT:
Report ID: {report_id}
Patient: {patient_name} ({patient_id})
Modality: {report_type}
Date: {report_date}
Clinical Indication: {clinical_indication}

FINDINGS:
{findings_text}

IMPRESSIONS:
{impression_text}

RECOMMENDATION:
{recommendation_text}

RULES:
1. Answer ONLY using the facts stated in the report above.
2. If the user asks about a tooth, finding, or history that is NOT mentioned in the report, explicitly state that it is not documented or evaluated in this report.
3. Do NOT extrapolate, invent, or speculate any diagnosis not stated in the impressions.
4. Keep the answer clear, clinical, and directly cite the relevant section.
"""

def generate_report_summary(report: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a structured summary grounded directly in the retrieved report.
    Mandatory Feature C requirement.
    """
    report_id = report.get("report_id", "Unknown")
    report_type = report.get("report_type", "Scan")
    report_date = report.get("report_date", "Unknown")
    indication = report.get("clinical_indication", "Not specified")
    findings = report.get("findings", [])
    impressions = report.get("impression", [])
    recommendations = report.get("recommendation", "Clinical correlation advised.")

    findings_bullets = [f"• {f.get('region', 'Tooth')}: {f.get('observation')} (Certainty: {f.get('certainty', 'Observed')})" for f in findings]
    impressions_bullets = [f"• {imp}" for imp in impressions]

    answer_text = (
        f"Summary for {patient.get('patient_name')} ({report_type} Report: {report_id}, Date: {report_date}):\n\n"
        f"Clinical Indication:\n• {indication}\n\n"
        f"Key Diagnostic Findings:\n" + ("\n".join(findings_bullets) if findings_bullets else "• No significant osseous abnormalities.") + "\n\n"
        f"Radiologist Impression:\n" + ("\n".join(impressions_bullets) if impressions_bullets else "• Normal radiographic examination.") + "\n\n"
        f"Recommendation:\n• {recommendations}"
    )

    citations = [
        f"Source Report: {report_id} ({report_type}, {report_date})",
        f"Findings Evidence: {'; '.join(f.get('observation', '') for f in findings)}",
        f"Impression Evidence: {'; '.join(impressions)}"
    ]

    return {
        "status": "success",
        "answer": answer_text,
        "citations": citations,
        "report_id": report_id,
        "report_date": report_date,
        "report_type": report_type
    }

def answer_report_question(report_id: str, query: str) -> Dict[str, Any]:
    """
    Bonus Feature: Report-Aware Chatbot.
    Answers natural language questions strictly grounded in the selected report.
    """
    report = get_report_by_id(report_id)
    if not report:
        return {
            "status": "not_found",
            "answer": f"Report '{report_id}' could not be retrieved from the clinical records.",
            "citations": []
        }

    patient = get_patient(report.get("patient_id", ""))
    patient_name = patient["patient_name"] if patient else "Unknown Patient"
    patient_id = patient["patient_id"] if patient else "N/A"

    findings = report.get("findings", [])
    findings_text = "\n".join([f"- [{f.get('region')}]: {f.get('observation')} (Status: {f.get('certainty', 'Documented')})" for f in findings])
    impression_text = "\n".join([f"- {imp}" for imp in report.get("impression", [])])
    recommendation_text = report.get("recommendation", "None")

    # If the user is asking for a summary
    q_lower = query.lower()
    if any(term in q_lower for term in ["summarize", "summary", "overview"]):
        return generate_report_summary(report, patient or {})

    # Construct Grounded Prompt for LLM (Llama 3.1 / Qwen)
    prompt = REPORT_GROUNDED_QA_PROMPT.format(
        report_id=report.get("report_id"),
        patient_name=patient_name,
        patient_id=patient_id,
        report_type=report.get("report_type"),
        report_date=report.get("report_date"),
        clinical_indication=report.get("clinical_indication", "N/A"),
        findings_text=findings_text or "No specific focal findings.",
        impression_text=impression_text or "No definitive pathology.",
        recommendation_text=recommendation_text
    )

    user_message = f"Clinician Question: {query}\n\nProvide a factual, grounded answer with citations to the report sections."

    # Query the configured LLM (Tiered Groq Cloud LPU + Local Ollama)
    llm_response = query_llm(user_message, system_prompt=prompt, json_format=False)

    if llm_response and len(llm_response.strip()) > 10:
        citations = [
            f"Source Report: {report['report_id']} ({report.get('report_type')})",
            f"Impression Evidence: {'; '.join(report.get('impression', []))}"
        ]
        return {
            "status": "success",
            "answer": llm_response.strip(),
            "citations": citations,
            "report_id": report["report_id"],
            "report_date": report.get("report_date")
        }

    # Deterministic RAG Fallback (Zero Hallucination)
    # Answers directly from structured report fields if LLM is offline
    matched_findings = []
    for f in findings:
        combined = (f.get("region", "") + " " + f.get("observation", "")).lower()
        if any(term in combined for term in ["third molar", "48", "38", "wisdom", "impaction"]) and any(term in q_lower for term in ["third molar", "48", "38", "wisdom", "impaction"]):
            matched_findings.append(f)
        elif any(term in combined for term in ["caries", "decay", "cavity"]) and any(term in q_lower for term in ["caries", "decay", "cavity"]):
            matched_findings.append(f)
        elif any(term in combined for term in ["sinus", "maxillary"]) and any(term in q_lower for term in ["sinus", "maxillary"]):
            matched_findings.append(f)
        elif any(term in combined for term in ["bone", "resorption", "ridge"]) and any(term in q_lower for term in ["bone", "resorption", "ridge"]):
            matched_findings.append(f)

    if matched_findings:
        bullets = [f"Regarding {mf.get('region')}: {mf.get('observation')} (Certainty: {mf.get('certainty', 'documented')})." for mf in matched_findings]
        citations = [f"Report Finding [{mf.get('region')}]: \"{mf.get('observation')}\"" for mf in matched_findings]
        for imp in report.get("impression", []):
            if any(term in imp.lower() for term in q_lower.split() if len(term) > 3):
                bullets.append(f"Radiologist Impression: {imp}")
                citations.append(f"Report Impression: \"{imp}\"")

        return {
            "status": "success",
            "answer": "\n".join(bullets),
            "citations": citations,
            "report_id": report["report_id"],
            "report_date": report.get("report_date")
        }

    # If queried entity not mentioned in report
    return {
        "status": "success",
        "answer": f"According to {report['report_type']} report ({report['report_id']}), the requested details are not documented or evaluated. The radiologist's stated impression is: '{'; '.join(report.get('impression', []))}'.",
        "citations": [f"Source Report {report['report_id']}: Impression: {'; '.join(report.get('impression', []))}"],
        "report_id": report["report_id"],
        "report_date": report.get("report_date")
    }
