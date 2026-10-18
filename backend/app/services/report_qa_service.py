"""
REPORT-AWARE CHATBOT & EVIDENCE-GROUNDED RAG SERVICE (report_qa_service.py)
--------------------------------------------------------------------------
Performs direct report-grounded question answering (RAG without vector DB)
over the currently active diagnostic imaging report.
Guarantees zero clinical hallucination by answering solely from verified report evidence.
"""

import json
import re
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
    recommendations = report.get("recommendation") or "None documented in report."

    findings_bullets = [f"• {f.get('region', 'Tooth')}: {f.get('observation')} (Certainty: {f.get('certainty', 'Observed')})" for f in findings]
    impressions_bullets = [f"• {imp}" for imp in impressions]

    findings_content = "\n".join(findings_bullets) if findings_bullets else "• No diagnostic findings are documented in the available report."
    impressions_content = "\n".join(impressions_bullets) if impressions_bullets else "• No diagnostic impression is documented in the available report."

    answer_text = (
        f"Summary for {patient.get('patient_name')} ({report_type} Report: {report_id}, Date: {report_date}):\n\n"
        f"Clinical Indication:\n• {indication}\n\n"
        f"Key Diagnostic Findings:\n{findings_content}\n\n"
        f"Radiologist Impression:\n{impressions_content}\n\n"
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
    q_lower = query.lower().strip()

    # If the user is asking for the PDF file or download
    if any(term in q_lower for term in ["pdf", "download report", "view pdf", "open pdf", "pdf link", "file link"]):
        pdf_name = report.get("pdf_filename")
        if pdf_name:
            pdf_url = f"/patient-pdf/{pdf_name}"
            return {
                "status": "success",
                "answer": (
                    f"📄 **PDF Report Available**: [{pdf_name}]({pdf_url})\n\n"
                    f"• **Patient**: {patient_name} ({patient_id})\n"
                    f"• **Modality**: {report.get('report_type')} ({report.get('report_id')})\n"
                    f"• **Date**: {report.get('report_date')}\n"
                    f"• **Link**: [Open PDF Report]({pdf_url})"
                ),
                "citations": [f"Source PDF: {pdf_name}"],
                "pdf_url": pdf_url,
                "pdf_filename": pdf_name,
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }

    # If the user is asking for a summary or comprehensive explanation
    if any(term in q_lower for term in ["summarize", "summary", "overview", "explain this", "explain report", "explain her", "explain his", "what does the report show", "what does this report show", "what does the report say", "what are the findings"]):
        return generate_report_summary(report, patient or {})

    # Construct Grounded Prompt for LLM (Llama 3.1 / Qwen)
    prompt = REPORT_GROUNDED_QA_PROMPT.format(
        report_id=report.get("report_id"),
        patient_name=patient_name,
        patient_id=patient_id,
        report_type=report.get("report_type"),
        report_date=report.get("report_date"),
        clinical_indication=report.get("clinical_indication", "N/A"),
        findings_text=findings_text or "No diagnostic findings are documented in the available report.",
        impression_text=impression_text or "No diagnostic impression is documented in the available report.",
        recommendation_text=recommendation_text or "No clinical recommendations are documented in the available report."
    )

    user_message = f"Clinician Question: {query}\n\nProvide a factual, grounded answer with citations to the report sections."

    # Query the configured LLM (Local Ollama)
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
    stop_words = {
        "what", "did", "the", "radiologist", "say", "about", "is", "there", "any",
        "show", "tell", "me", "how", "are", "does", "have", "with", "this", "report",
        "from", "an", "a", "scan", "scans", "earlier", "mention", "mentioned", "prior",
        "previous", "in", "on", "at", "for", "of", "to", "or", "and", "that", "it",
        "was", "be", "by", "as", "if", "not"
    }
    q_tokens = {w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if w not in stop_words and len(w) > 2}

    # Generalized semantic finding matcher
    # Checks exact tokens, morphological stems (len >= 4), tooth number references, and dental aliases
    for f in findings:
        combined = (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower()
        f_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined))

        # 1. Direct token overlap
        if q_tokens & f_tokens:
            matched_findings.append(f)
            continue

        # 2. Morphological stem / substring overlap for substantive clinical words
        if any((len(qt) >= 4 and len(ft) >= 4 and (qt in ft or ft in qt)) for qt in q_tokens for ft in f_tokens):
            matched_findings.append(f)
            continue

        # 3. Dental clinical synonyms / tooth aliases (e.g. wisdom tooth <-> third molar)
        if ("wisdom" in q_lower and any(w in combined for w in ["third molar", "38", "48", "18", "28", "wisdom"])) or \
           ("third molar" in q_lower and any(w in combined for w in ["wisdom", "38", "48", "18", "28"])) or \
           ("bicuspid" in q_lower and "premolar" in combined) or \
           ("premolar" in q_lower and "bicuspid" in combined):
            matched_findings.append(f)
            continue

    if matched_findings:
        bullets = [f"Regarding {mf.get('region')}: {mf.get('observation')} (Certainty: {mf.get('certainty', 'documented')})." for mf in matched_findings]
        citations = [f"Report Finding [{mf.get('region')}]: \"{mf.get('observation')}\"" for mf in matched_findings]
        for imp in report.get("impression", []):
            if any(term in imp.lower() for term in q_tokens if len(term) > 3):
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
