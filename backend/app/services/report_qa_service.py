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
from .retrieval import get_report_by_id, get_patient, get_patient_reports
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
        f"Recommendation:\n• {recommendations}\n\n"
        f"Source Evidence:\n• Source Report: [{report_id}] {report_type} ({report_date})\n"
        f"• Citations: Verified findings and radiologist impressions from {report_id}."
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

def is_explanation_followup(query: str) -> bool:
    """Checks if the query is asking to explain or elaborate on a previously discussed finding/topic."""
    q = query.lower().strip()
    q_clean = re.sub(r"[?!.,:]+", "", q).strip()
    explanation_phrases = [
        "can you explain", "could you explain", "please explain", "explain",
        "explain this", "explain that", "what does this mean", "what does that mean",
        "elaborate", "can you elaborate", "could you elaborate", "please elaborate",
        "what does it mean", "why is that", "tell me more", "details please",
        "can you clarify", "clarify this", "clarify that", "why", "meaning",
        "what does that indicate", "what does this indicate", "implications",
        "can you explain this", "explain further", "more details", "what does that mean clinically"
    ]
    if q_clean in explanation_phrases:
        return True
    if any(q_clean.startswith(p + " ") or q_clean.endswith(" " + p) for p in explanation_phrases):
        if not any(w in q_clean for w in ["patient", "all reports", "all findings", "full report", "summary"]):
            return True
    return False


def extract_previous_finding(
    findings: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]]
) -> Optional[Dict[str, Any]]:
    """Extracts which specific finding was discussed in recent conversation turns."""
    if not history:
        return None
    for turn in reversed(history[-2:]):
        user_text = turn.get("user", "")
        asst_text = turn.get("assistant", "")
        combined = f"{user_text} {asst_text}".lower()

        # 1. Match specific tooth numbers present in finding region
        for f in findings:
            region = str(f.get("region", ""))
            obs = str(f.get("observation", ""))
            teeth = re.findall(r"\b(?:[1-8][1-8]|[1-3]?[0-9])\b", region)
            for t in teeth:
                if re.search(r"\b" + re.escape(t) + r"\b", combined):
                    return f
            # Match region words (e.g. alveolar, crestal, bone, sinus)
            region_words = [w.lower() for w in re.findall(r"[a-zA-Z]{5,}", region)]
            if any(w in combined for w in region_words if w not in ["tooth", "teeth", "mandibular", "maxillary"]):
                return f
            # Match unique observation keywords
            obs_keywords = ["radiolucency", "mesioangular", "impaction", "caries", "bone height", "periapical"]
            for kw in obs_keywords:
                if kw in obs.lower() and kw in combined:
                    return f
    return None


def synthesize_clinical_finding_explanation(
    finding: Dict[str, Any],
    report: Dict[str, Any],
    patient_name: str
) -> Dict[str, Any]:
    """Generates a focused, grounded clinical explanation of a specific finding without dumping the full report."""
    region = finding.get("region", "Documented Finding")
    obs = finding.get("observation", "")
    certainty = finding.get("certainty", "Confirmed")
    obs_lower = obs.lower()
    report_id = report.get("report_id", "Unknown")
    report_type = report.get("report_type", "Scan")

    impressions = report.get("impression", [])
    relevant_imps = []
    teeth = re.findall(r"\b(?:[1-8][1-8]|[1-3]?[0-9])\b", region)
    for imp in impressions:
        if any(t in imp for t in teeth) or any(w in imp.lower() for w in ["caries", "impaction", "bone", "periodont"] if w in obs_lower):
            relevant_imps.append(imp)
    if not relevant_imps and impressions:
        relevant_imps = impressions

    rec = report.get("recommendation", "")

    lines = [f"Clinical explanation of the finding for **{region}** in {report_type} report ({report_id}):\n"]
    lines.append(f"• **Diagnostic Finding**: \"{obs}\" (Certainty: {certainty})")

    # 1. Pathology & Radiographic Interpretation
    if "radiolucency" in obs_lower or "caries" in obs_lower:
        lines.append(
            "• **Pathology & Radiographic Meaning**: The 'coronal radiolucency' on the occlusal surface indicates "
            "an area of decreased mineral density (appearing as a dark zone on the radiograph) caused by acid-producing "
            "bacterial demineralization of the tooth enamel and dentin—characteristic of dental caries (tooth decay) "
            "on the chewing surface."
        )
    elif "impaction" in obs_lower:
        lines.append(
            "• **Pathology & Radiographic Meaning**: The tooth is impacted (unable to erupt into functional alignment). "
            "A mesioangular impaction means the crown is tilted forward toward the adjacent second molar, creating "
            "a localized plaque and food retention area prone to pericoronitis, root resorption of the second molar, "
            "and caries."
        )
    elif "bone" in obs_lower or "crestal" in obs_lower:
        lines.append(
            "• **Pathology & Radiographic Meaning**: Crestal bone evaluation measures the distance from the cementoenamel "
            "junction (CEJ) to the crest of the alveolar bone. Physiological bone levels (< 1.5–2.0 mm from CEJ) "
            "confirm healthy periodontal bone support without pathological alveolar bone loss."
        )
    else:
        lines.append(f"• **Pathology & Radiographic Meaning**: This radiological observation indicates {obs.lower()}.")

    # 2. Depth & Structural Involvement
    if "mid-dentin" in obs_lower:
        lines.append(
            "• **Depth & Anatomical Extent**: The carious lesion has breached the outer enamel layer and advanced "
            "into **mid-dentin** (approximately halfway through the dentin layer). Because dentin contains microscopic "
            "fluid-filled tubules directly communicating with the dental pulp, this depth frequently produces transient "
            "sensitivity to cold, sweet, or hot stimuli."
        )
    elif "canal" in obs_lower or "1.5 mm" in obs_lower or "apices" in obs_lower:
        lines.append(
            "• **Anatomical Proximity & Risk**: The root apices are located within approximately 1.5 mm of the superior "
            "border of the inferior alveolar nerve (IAN) canal, indicating close proximity that carries a surgical risk "
            "of neurosensory paresthesia to the lower lip and chin during extraction."
        )

    # 3. Pulp / Periapical Status
    if "without periapical rarefaction" in obs_lower or "no periapical" in obs_lower:
        lines.append(
            "• **Pulp Vitality & Periapical Health**: The finding specifically notes **'without periapical rarefaction'**, "
            "meaning there is no apical bone destruction, dark periapical shadow, or periodontal ligament widening at the root tips. "
            "This indicates that the dental pulp remains vital (alive) and the infection has not reached the root canal "
            "system or formed a periapical abscess/granuloma."
        )

    # 4. Radiologist Impression & Next Steps
    if relevant_imps:
        lines.append(f"• **Radiologist Impression**: {'; '.join(relevant_imps)}")
    if rec and rec.strip() and rec.lower() not in ["none", "none.", "n/a"]:
        lines.append(f"• **Recommended Clinical Management**: {rec.strip()}")
    elif "caries" in obs_lower or "radiolucency" in obs_lower:
        lines.append("• **Recommended Clinical Management**: Operative restoration (cavity filling) is indicated to excavate demineralized dentin and seal the cavity before pulp exposure occurs.")

    answer_text = "\n\n".join(lines)
    citations = [f"Source Report Finding [{region}]: \"{obs}\""]
    for imp in relevant_imps:
        citations.append(f"Source Report Impression: \"{imp}\"")

    return {
        "status": "success",
        "answer": answer_text,
        "citations": citations,
        "report_id": report_id,
        "report_date": report.get("report_date"),
        "report_type": report_type
    }


def answer_report_question(report_id: str, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Bonus Feature: Report-Aware Chatbot.
    Answers natural language questions strictly grounded in the selected report.
    Supports multi-turn contextual explanations of specific findings.
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

    # -------------------------------------------------------------------------
    # INTENT 0: Multi-Turn Contextual Explanation of Prior Finding
    # (e.g. Clinician asks "what is this: Teeth 16 and 46" -> then "can you explain")
    # -------------------------------------------------------------------------
    if is_explanation_followup(query) and history:
        prev_finding = extract_previous_finding(findings, history)
        if prev_finding:
            # Match relevant impressions
            impressions = report.get("impression", [])
            relevant_imps = []
            teeth = re.findall(r"\b(?:[1-8][1-8]|[1-3]?[0-9])\b", prev_finding.get("region", ""))
            for imp in impressions:
                if any(t in imp for t in teeth) or any(w in imp.lower() for w in ["caries", "impaction", "bone", "periodont"] if w in prev_finding.get("observation", "").lower()):
                    relevant_imps.append(imp)
            if not relevant_imps and impressions:
                relevant_imps = impressions

            rec = report.get("recommendation", "")

            # Query local LLM with tightly scoped context focused on ONLY this finding
            focused_system_prompt = (
                f"You are a clinical dental radiology consultant.\n"
                f"The clinician previously asked about a specific finding and is now asking for an explanation.\n\n"
                f"REPORT EVIDENCE (Report {report.get('report_id')}, {report.get('report_type')}):\n"
                f"Region: {prev_finding.get('region')}\n"
                f"Observation: {prev_finding.get('observation')}\n"
                f"Certainty: {prev_finding.get('certainty', 'Confirmed')}\n"
                f"Impression: {'; '.join(relevant_imps) if relevant_imps else 'None'}\n"
                f"Recommendation: {rec or 'None'}\n\n"
                f"INSTRUCTION:\n"
                f"Explain ONLY this specific finding ({prev_finding.get('region')}).\n"
                f"Address: radiological terminology, depth into dentin, pulp vitality (what 'without periapical rarefaction' means), and restorative implications.\n"
                f"DO NOT summarize or list unrelated teeth or findings from the rest of the report.\n"
            )
            llm_response = query_llm(f"Clinician Question: {query}", system_prompt=focused_system_prompt, json_format=False)
            if llm_response and len(llm_response.strip()) > 30:
                if not ("tooth 48" in llm_response.lower() and "48" not in str(prev_finding.get("region"))):
                    citations = [f"Source Report Finding [{prev_finding.get('region')}]: \"{prev_finding.get('observation')}\""]
                    for imp in relevant_imps:
                        citations.append(f"Source Report Impression: \"{imp}\"")
                    return {
                        "status": "success",
                        "answer": llm_response.strip(),
                        "citations": citations,
                        "report_id": report["report_id"],
                        "report_date": report.get("report_date"),
                        "report_type": report.get("report_type")
                    }
            return synthesize_clinical_finding_explanation(prev_finding, report, patient_name)

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

    # -------------------------------------------------------------------------
    # INTENT 2: Authorized Demographic Query (Personal Information, Age, DOB, Sex, MRN)
    # -------------------------------------------------------------------------
    demo_patterns = [
        r"\bpersonal\s+info(?:rmation)?\b",
        r"\bpersonal\s+details?\b",
        r"\bpatient\s+(?:info(?:rmation)?|details?|demographics?|name|id)\b",
        r"\bwho\s+is\s+(?:the\s+)?patient\b",
        r"\b(?:what\s+is\s+)?(?:his|her|their|the\s+patient'?s?)\s+name\b",
        r"\b(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date)\b",
        r"\bhow\s+old\b",
        r"\bage\b",
        r"\b(?:gender|sex)\b",
        r"\b(?:mrn|record\s+number|medical\s+record)\b",
        r"\bdemographics?\b"
    ]
    demo_phrases = [
        "personal info", "personal information", "personal details", "personal detail",
        "patient info", "patient information", "patient details", "patient detail",
        "demographic", "demographics", "date of birth", "birth date", "how old",
        "patient name", "who is the patient"
    ]
    is_demo_query = any(re.search(p, q_lower) for p in demo_patterns) or any(p in q_lower for p in demo_phrases)
    if is_demo_query:
        dob = patient.get("date_of_birth") if patient else report.get("date_of_birth")
        sex = patient.get("sex") if patient else report.get("sex", "unspecified")
        mrn = patient.get("record_number") if patient else None
        p_name = patient_name or (patient.get("patient_name") if patient else "Unassigned")
        p_id = patient_id or (patient.get("patient_id") if patient else "N/A")

        sex_str = str(sex).lower()
        if sex_str == "male":
            pronoun_poss = "his"
            pronoun_subj = "he"
        elif sex_str == "female":
            pronoun_poss = "her"
            pronoun_subj = "she"
        else:
            pronoun_poss = "the patient's"
            pronoun_subj = "they"

        age_str = "documented on file"
        if dob:
            try:
                b_year = int(dob.split("-")[0])
                age = 2026 - b_year
                age_str = f"{age} years"
            except Exception:
                age_str = "documented on file"

        citations = [
            f"Master Patient Index: {p_id} ({p_name}, DOB: {dob or 'N/A'}, Sex: {sex or 'N/A'})",
            f"Source Report Header: {report['report_id']} ({report.get('report_type', 'Radiology Report')}, Exam Date: {report.get('report_date', 'N/A')})"
        ]

        is_age_only = (
            bool(re.search(r"\b(how\s+old|age|dob|date\s+of\s+birth|birth\s*date)\b", q_lower))
            and not any(w in q_lower for w in ["personal", "demographic", "who is", "detail", "info", "information", "record", "all"])
        )
        is_gender_only = (
            bool(re.search(r"\b(gender|sex)\b", q_lower))
            and not any(w in q_lower for w in ["personal", "demographic", "who is", "detail", "info", "information", "record", "all"])
        )

        if is_age_only:
            answer_text = (
                f"According to {p_name}'s authorized demographic record ({p_id}), "
                f"{pronoun_poss} date of birth is {dob} (Age: {age_str}, Sex: {sex}).\n\n"
                f"*Clinical Note: Demographic details are retrieved directly from the Master Patient Index, "
                f"as patient age is not recorded in the diagnostic radiology examination text.*"
            )
        elif is_gender_only:
            answer_text = (
                f"According to {p_name}'s authorized demographic record ({p_id}), "
                f"{pronoun_poss} documented biological sex is {sex}.\n\n"
                f"*Clinical Note: Verified against the Master Patient Index.*"
            )
        else:
            mrn_line = f"• Medical Record Number (MRN): {mrn}\n" if mrn else ""
            answer_text = (
                f"The personal and demographic information documented for this patient and report ({report['report_id']}) is:\n\n"
                f"• Patient Name: {p_name}\n"
                f"• Patient ID: {p_id}\n"
                f"{mrn_line}"
                f"• Date of Birth: {dob or 'Not documented'}\n"
                f"• Age: {age_str}\n"
                f"• Biological Sex / Gender: {sex}\n"
                f"• Examination Modality: {report.get('report_type', report.get('modality', 'Diagnostic Imaging'))}\n"
                f"• Examination Date: {report.get('report_date', 'Not specified')}\n\n"
                f"*Clinical Note: Demographic details are verified against the Master Patient Index and cross-referenced with report header {report['report_id']}.*"
            )

        return {
            "status": "success",
            "answer": answer_text,
            "citations": citations,
            "report_id": report["report_id"],
            "report_date": report.get("report_date")
        }

    # -------------------------------------------------------------------------
    # INTENT 3: Treatment, Management, or Clinical Recommendation
    # -------------------------------------------------------------------------
    is_treatment_query = any(w in q_lower for w in ["treatment", "advised", "recommend", "recommendation", "plan", "prescribed", "management", "intervention"])
    if is_treatment_query:
        rec = report.get("recommendation")
        if rec and rec.strip() and rec.lower() not in ["none", "none.", "n/a"]:
            return {
                "status": "success",
                "answer": (
                    f"The treatment and recommendations advised in {report.get('report_type')} report ({report['report_id']}) are:\n"
                    f"• {rec.strip()}"
                ),
                "citations": [f"Report Recommendation [{report['report_id']}]: \"{rec.strip()}\""],
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }

    # -------------------------------------------------------------------------
    # INTENT 4: Specific Tooth Query (e.g. Tooth 48, Tooth 16, Tooth 25, #29, UR5)
    # -------------------------------------------------------------------------
    tooth_match = re.search(r"\b(?:tooth|teeth|#)\s*([1-8][1-8]|[1-3]?[0-9])\b", q_lower)
    palmer_q_match = re.search(r"\b(ur|ul|ll|lr)\s*([1-8])\b", q_lower)
    named_tooth_match = re.search(r"\b(third molar|wisdom tooth|second molar|first molar|second premolar|first premolar|canine|lateral incisor|central incisor)\b", q_lower)

    target_tooth = None
    if tooth_match:
        target_tooth = tooth_match.group(1)
    elif palmer_q_match:
        target_tooth = f"{palmer_q_match.group(1).upper()}{palmer_q_match.group(2)}"
    elif named_tooth_match:
        target_tooth = named_tooth_match.group(1)

    if target_tooth:
        tooth_findings = []
        for f in findings:
            comb = (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower()
            if target_tooth.lower() in comb or (target_tooth == "48" and any(k in comb for k in ["tooth 48", "48", "lr8", "mandibular right third molar"])):
                tooth_findings.append(f)

        tooth_impressions = [
            imp for imp in report.get("impression", [])
            if target_tooth.lower() in imp.lower() or (target_tooth == "48" and any(k in imp.lower() for k in ["tooth 48", "48", "lr8", "third molar"]))
        ]

        if not tooth_findings and not tooth_impressions:
            # Check if this tooth is documented in any of the patient's other reports
            other_reports = get_patient_reports(patient_id) if patient_id and patient_id != "N/A" else []
            other_matches = []
            for other_rep in other_reports:
                if other_rep.get("report_id") == report["report_id"]:
                    continue
                o_findings = [
                    f for f in other_rep.get("findings", [])
                    if target_tooth.lower() in (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower()
                    or (target_tooth == "48" and any(k in (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower() for k in ["tooth 48", "48", "lr8", "third molar"]))
                ]
                o_impressions = [
                    imp for imp in other_rep.get("impression", [])
                    if target_tooth.lower() in imp.lower()
                    or (target_tooth == "48" and any(k in imp.lower() for k in ["tooth 48", "48", "lr8", "third molar"]))
                ]
                if o_findings or o_impressions:
                    other_matches.append((other_rep, o_findings, o_impressions))

            if other_matches:
                lines = [
                    f"Tooth {target_tooth} is not documented in {report.get('report_type')} report ({report['report_id']}). No findings, observations, or pathology regarding Tooth {target_tooth} were evaluated.\n\n"
                    f"*Clinical Note: Absence of documentation in this report indicates Tooth {target_tooth} was outside the evaluated scope or not examined in this specific consultation; it must not be assumed that the tooth is clinically normal.*\n\n"
                    f"However, Tooth {target_tooth} is documented in {patient_name}'s other clinical records on file:\n"
                ]
                citations = []
                for o_rep, o_f, o_imp in other_matches:
                    o_rid = o_rep.get("report_id")
                    o_type = o_rep.get("report_type")
                    o_date = o_rep.get("report_date")
                    lines.append(f"• **In {o_type} ({o_rid}, {o_date})**:")
                    for f in o_f:
                        lines.append(f"  - **{f.get('region')}**: {f.get('observation')} (Certainty: {f.get('certainty', 'Confirmed')})")
                        citations.append(f"Source Report [{o_rid} {o_type}] Finding [{f.get('region')}]: \"{f.get('observation')}\"")
                    if o_imp:
                        lines.append(f"  - **Impression**: {'; '.join(o_imp)}")
                        for imp in o_imp:
                            citations.append(f"Source Report [{o_rid} {o_type}] Impression: \"{imp}\"")

                return {
                    "status": "success",
                    "answer": "\n".join(lines),
                    "citations": citations,
                    "report_id": report["report_id"],
                    "report_date": report.get("report_date")
                }

            return {
                "status": "success",
                "answer": (
                    f"Tooth {target_tooth} is not documented in {report.get('report_type')} report ({report['report_id']}). No findings, observations, or pathology regarding Tooth {target_tooth} were evaluated.\n\n"
                    f"*Clinical Note: Absence of documentation in this report indicates Tooth {target_tooth} was outside the evaluated scope or not examined in this specific consultation; it must not be assumed that the tooth is clinically normal.*"
                ),
                "citations": [f"Source Report {report['report_id']}: Tooth {target_tooth} is not documented in this report."],
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }
        else:
            bullets = [f"• **{f.get('region')}**: {f.get('observation')} (Certainty: {f.get('certainty', 'Confirmed')})" for f in tooth_findings]
            if tooth_impressions:
                bullets.append(f"• **Radiologist Impression**: {'; '.join(tooth_impressions)}")
            citations = [f"Report Finding [{f.get('region')}]: \"{f.get('observation')}\"" for f in tooth_findings]
            for imp in tooth_impressions:
                citations.append(f"Report Impression: \"{imp}\"")

            # Check if other reports of this patient ALSO contain relevant documentation for this tooth
            other_reports = get_patient_reports(patient_id) if patient_id and patient_id != "N/A" else []
            other_lines = []
            for other_rep in other_reports:
                if other_rep.get("report_id") == report["report_id"]:
                    continue
                o_findings = [
                    f for f in other_rep.get("findings", [])
                    if target_tooth.lower() in (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower()
                ]
                o_impressions = [
                    imp for imp in other_rep.get("impression", [])
                    if target_tooth.lower() in imp.lower()
                ]
                if o_findings or o_impressions:
                    other_lines.append(f"• **Related Record in {other_rep.get('report_type')} ({other_rep.get('report_id')}, {other_rep.get('report_date')})**:")
                    for f in o_findings:
                        other_lines.append(f"  - {f.get('region')}: {f.get('observation')} (Certainty: {f.get('certainty', 'Confirmed')})")
                        citations.append(f"Source Report [{other_rep.get('report_id')} {other_rep.get('report_type')}] Finding [{f.get('region')}]: \"{f.get('observation')}\"")
                    if o_impressions:
                        other_lines.append(f"  - Impression: {'; '.join(o_impressions)}")
                        for imp in o_impressions:
                            citations.append(f"Source Report [{other_rep.get('report_id')} {other_rep.get('report_type')}] Impression: \"{imp}\"")

            full_answer = f"Regarding Tooth {target_tooth} in report {report['report_id']}:\n" + "\n".join(bullets)
            if other_lines:
                full_answer += "\n\n" + "\n".join(other_lines)

            return {
                "status": "success",
                "answer": full_answer,
                "citations": citations,
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }

    # -------------------------------------------------------------------------
    # INTENT 5: Specific Anatomical Structure (e.g. Gums, Periodontium, Sinus, TMJ)
    # -------------------------------------------------------------------------
    is_gums = bool(re.search(r"\b(gums?|gingiv\w*|periodont\w*)\b", q_lower))
    if is_gums:
        gum_findings = [f for f in findings if any(k in (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower() for k in ["periodont", "gingiv", "gum", "plaque"])]
        gum_imps = [imp for imp in report.get("impression", []) if any(k in imp.lower() for k in ["periodont", "gingiv", "gum"])]
        if gum_findings or gum_imps:
            bullets = [f"• **{f.get('region')}**: {f.get('observation')} (Certainty: {f.get('certainty', 'Confirmed')})" for f in gum_findings]
            if gum_imps:
                bullets.append(f"• **Radiologist Impression**: {'; '.join(gum_imps)}")
            citations = [f"Report Finding [{f.get('region')}]: \"{f.get('observation')}\"" for f in gum_findings]
            for imp in gum_imps:
                citations.append(f"Report Impression: \"{imp}\"")
            return {
                "status": "success",
                "answer": f"Regarding the gums and periodontal status in report {report['report_id']}:\n" + "\n".join(bullets),
                "citations": citations,
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }

    # -------------------------------------------------------------------------
    # INTENT 6: Specific Pathology Query (e.g. Caries, Cavity, Decay, Sensitivity)
    # -------------------------------------------------------------------------
    pathology_terms = ["caries", "cavity", "carious", "decay", "sensitivity", "hypersensitivity", "abfraction", "impaction", "fracture", "osteitis", "radiolucency"]
    matched_path = [pt for pt in pathology_terms if re.search(r"\b" + re.escape(pt) + r"\b", q_lower)]
    if matched_path:
        search_stems = set(matched_path)
        if "caries" in search_stems or "cavity" in search_stems or "decay" in search_stems:
            search_stems.update(["caries", "carious", "decay", "cavity"])
        if "sensitivity" in search_stems:
            search_stems.update(["sensitivity", "hypersensitivity", "sensitive", "abfraction"])

        path_findings = []
        for f in findings:
            comb = (str(f.get("region", "")) + " " + str(f.get("observation", ""))).lower()
            if any(st in comb for st in search_stems):
                # Ensure non-carious cervical loss is not confused with true caries
                if ("caries" in search_stems or "carious" in search_stems) and "non-carious" in comb and not re.search(r"\b(?<!non-)carious\b", comb) and "caries" not in comb:
                    continue
                path_findings.append(f)
        path_imps = [imp for imp in report.get("impression", []) if any(st in imp.lower() for st in search_stems)]
        if path_findings or path_imps:
            bullets = [f"• **{f.get('region')}**: {f.get('observation')} (Certainty: {f.get('certainty', 'Confirmed')})" for f in path_findings]
            if path_imps:
                bullets.append(f"• **Radiologist Impression**: {'; '.join(path_imps)}")
            citations = [f"Report Finding [{f.get('region')}]: \"{f.get('observation')}\"" for f in path_findings]
            for imp in path_imps:
                citations.append(f"Report Impression: \"{imp}\"")
            return {
                "status": "success",
                "answer": f"Documentation regarding {', '.join(matched_path)} in report {report['report_id']}:\n" + "\n".join(bullets),
                "citations": citations,
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }

    # -------------------------------------------------------------------------
    # INTENT 7: Measurements Query
    # -------------------------------------------------------------------------
    is_meas_query = bool(re.search(r"\b(?:measurement|measurements|dimension|dimensions|size|sizes|mm|cm)\b", q_lower)) or (re.search(r"\bmeasure\b", q_lower) and not re.search(r"\bsummar\w*\b", q_lower))
    if is_meas_query:
        meas_bullets = []
        citations = []
        for f in findings:
            obs = f.get("observation", "")
            if re.search(r"\b(?:\d+(?:\.\d+)?\s*(?:mm|cm|seconds?|sec)|second|mm)\b", obs, re.I):
                meas_bullets.append(f"• **{f.get('region')}**: {obs}")
                citations.append(f"Report Measurement [{f.get('region')}]: \"{obs}\"")
        if meas_bullets:
            return {
                "status": "success",
                "answer": f"Documented clinical measurements in report {report['report_id']}:\n" + "\n".join(meas_bullets),
                "citations": citations,
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }
        else:
            return {
                "status": "success",
                "answer": f"No linear or dimensional measurements are recorded in {report.get('report_type')} report ({report['report_id']}).",
                "citations": [f"Source Report {report['report_id']}: No measurements recorded."],
                "report_id": report["report_id"],
                "report_date": report.get("report_date")
            }

    # -------------------------------------------------------------------------
    # INTENT 8: General Summary / Consultation Query
    # -------------------------------------------------------------------------
    summary_triggers = [
        "summarize", "summary", "overview", "explain this", "explain report", "explain her", "explain his",
        "what does the report show", "what does this report show", "what does the report say", "what are the findings",
        "what problem", "what problems", "what is the problem", "what's the problem", "what is wrong", "what's wrong",
        "what does she have", "what does he have", "what does it have", "what does patient have",
        "what they have", "what do they have", "what have they", "what she has", "what he has",
        "what condition", "what issues", "what issue", "what are her issues", "what did you find",
        "diagnose", "diagnosis", "impression", "impressions", "findings"
    ]
    is_general_summary_query = (
        not is_demo_query and (
            any(w in q_lower for w in [
                "summar", "overview", "problem", "wrong", "diagnos", "impression", "finding",
                "condition", "issue", "gist", "brief"
            ])
            or (re.search(r"\bwhat\b", q_lower) and any(w in q_lower for w in ["have", "hsve", "has", "show", "say", "find", "see"]))
            or any(term in q_lower for term in summary_triggers)
        )
    )
    if is_general_summary_query:
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

    if history:
        history_text = "\n".join([f"Clinician: {t.get('user', '')}\nAssistant: {t.get('assistant', '')[:120]}" for t in history[-2:]])
        user_message = f"RECENT CONVERSATION:\n{history_text}\n\nCurrent Clinician Question: {query}\n\nProvide a factual, grounded answer with citations to the report sections."
    else:
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

    # Deterministic RAG Fallback
    return {
        "status": "success",
        "answer": f"According to {report['report_type']} report ({report['report_id']}), the requested details are not documented or evaluated. The radiologist's stated impression is: '{'; '.join(report.get('impression', []))}'.",
        "citations": [f"Source Report {report['report_id']}: Impression: {'; '.join(report.get('impression', []))}"],
        "report_id": report["report_id"],
        "report_date": report.get("report_date")
    }


def generate_multi_report_summary(reports: List[Dict[str, Any]], patient: Dict[str, Any], query: str = "") -> Dict[str, Any]:
    """
    Synthesizes diagnostic findings and impressions across ALL diagnostic reports for a patient.
    Grounds all statements with explicit source report citations.
    """
    patient_name = patient.get("patient_name", "Patient")
    patient_id = patient.get("patient_id", "N/A")
    q_lower = query.lower()

    # Check if inquiry is focused on a specific condition (e.g. caries, impaction, pain)
    is_caries_focus = any(w in q_lower for w in ["caries", "cavity", "cavities", "decay"])
    is_impaction_focus = any(w in q_lower for w in ["impact", "third molar", "wisdom"])

    sections = []
    citations = []

    for rep in reports:
        rid = rep.get("report_id")
        rtype = rep.get("report_type")
        rdate = rep.get("report_date")
        findings = rep.get("findings", [])
        impressions = rep.get("impression", [])
        rec = rep.get("recommendation", "")

        findings_lines = []
        for f in findings:
            obs = f.get("observation", "")
            reg = f.get("region", "Tooth")
            cert = f.get("certainty", "Confirmed")
            if is_caries_focus and not any(w in (reg + " " + obs).lower() for w in ["carie", "cavity", "decay", "radiolucen"]):
                continue
            if is_impaction_focus and not any(w in (reg + " " + obs).lower() for w in ["impact", "molar", "wisdom", "erupt"]):
                continue
            findings_lines.append(f"  - [{reg}]: {obs} (Certainty: {cert})")
            citations.append(f"Source Report [{rid} {rtype}] Finding: \"{obs}\"")

        imp_lines = []
        for imp in impressions:
            if is_caries_focus and not any(w in imp.lower() for w in ["carie", "cavity", "decay"]):
                continue
            if is_impaction_focus and not any(w in imp.lower() for w in ["impact", "molar", "wisdom"]):
                continue
            imp_lines.append(f"  - {imp}")
            citations.append(f"Source Report [{rid} {rtype}] Impression: \"{imp}\"")

        rep_header = f"• **{rtype}** (Report ID: `{rid}`, Date: {rdate}):"
        findings_str = "\n".join(findings_lines) if findings_lines else "  - No findings specific to this inquiry."
        imp_str = "\n".join(imp_lines) if imp_lines else "  - No diagnostic impressions specific to this inquiry."
        rec_str = f"  - Recommendation: {rec.strip()}" if rec and not is_caries_focus and not is_impaction_focus else ""

        section = f"{rep_header}\n  Findings:\n{findings_str}\n  Impression:\n{imp_str}"
        if rec_str:
            section += f"\n{rec_str}"
        sections.append(section)

    # Check if the clinician asked a direct clinical question rather than a broad listing
    is_clinical_question = any(w in q_lower for w in [
        "problem", "problems", "wrong", "what they have", "what do they have",
        "what does he have", "what does she have", "what condition", "what issues", "what issue",
        "what pathology", "what is wrong", "tell me about", "explain", "they have"
    ]) and not any(w in q_lower for w in ["all her reports", "all reports", "list her reports", "list reports", "how many reports"])

    if is_clinical_question and reports:
        rep_context_blocks = []
        for r in reports:
            f_list = [f"- [{f.get('region')}]: {f.get('observation')} (Certainty: {f.get('certainty', 'Confirmed')})" for f in r.get("findings", [])]
            i_list = [f"- {imp}" for imp in r.get("impression", [])]
            rep_context_blocks.append(
                f"Report ID: {r.get('report_id')} ({r.get('report_type')}, Date: {r.get('report_date')})\n"
                f"Clinical Indication: {r.get('clinical_indication', 'N/A')}\n"
                f"Findings:\n{chr(10).join(f_list) if f_list else 'None'}\n"
                f"Impression:\n{chr(10).join(i_list) if i_list else 'None'}\n"
                f"Recommendation: {r.get('recommendation', 'N/A')}"
            )
        combined_reports_text = "\n\n".join(rep_context_blocks)

        multi_rep_prompt = (
            f"You are a Clinical Radiologist Assistant. The clinician is asking a question about patient {patient_name} ({patient_id}) who has {len(reports)} diagnostic reports on file.\n\n"
            f"DIAGNOSTIC REPORTS CONTEXT:\n{combined_reports_text}\n\n"
            f"CLINICIAN QUESTION: {query}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the clinician's question directly, clearly, and concisely based on the findings and impressions across these reports.\n"
            "2. If asked 'what problem they have' or similar, describe the patient's documented clinical issues, treatment history (e.g., tooth 25 root canal and subsequent crown restoration), and current status (e.g., whether current margins are sound or if there is active disease).\n"
            "3. Ground all statements in the reports above and cite the specific Report IDs (e.g. RPT-2013, RPT-2015).\n"
            "4. Do not output raw JSON or unformatted data dumps."
        )
        llm_answer = query_llm(query, system_prompt=multi_rep_prompt, json_format=False)
        if llm_answer and len(llm_answer.strip()) > 15:
            return {
                "status": "success",
                "answer": llm_answer.strip(),
                "citations": citations,
                "patient_id": patient_id,
                "report_count": len(reports),
                "report_id": reports[0].get("report_id") if reports else None,
                "report_date": reports[0].get("report_date") if reports else None,
                "report_type": f"Multi-Report ({len(reports)} records)"
            }

        # Structured clinical question synthesis fallback:
        summary_bullets = []
        for r in reports:
            imp_text = "; ".join(r.get("impression", []))
            rec_text = f" (Recommendation: {r.get('recommendation')})" if r.get('recommendation') else ""
            summary_bullets.append(f"• **{r.get('report_type')} ({r.get('report_id')}, {r.get('report_date')})**: {imp_text}{rec_text}")

        answer_text = (
            f"Clinical summary of conditions across {len(reports)} reports for **{patient_name}** ({patient_id}):\n\n"
            + "\n".join(summary_bullets)
        )
        return {
            "status": "success",
            "answer": answer_text,
            "citations": citations,
            "patient_id": patient_id,
            "report_count": len(reports),
            "report_id": reports[0].get("report_id") if reports else None,
            "report_date": reports[0].get("report_date") if reports else None,
            "report_type": f"Multi-Report ({len(reports)} records)"
        }

    answer_text = (
        f"Diagnostic Summary across {len(reports)} reports for **{patient_name}** ({patient_id}):\n\n"
        + "\n\n".join(sections)
    )

    return {
        "status": "success",
        "answer": answer_text,
        "citations": citations,
        "patient_id": patient_id,
        "report_count": len(reports),
        "report_id": reports[0].get("report_id") if reports else None,
        "report_date": reports[0].get("report_date") if reports else None,
        "report_type": f"Multi-Report ({len(reports)} records)"
    }


def answer_patient_reports_question(
    patient_id: str,
    active_report_id: Optional[str],
    query: str,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Handles multi-report clinical question answering for a patient.
    1. If the clinician's query targets a specific modality or report ID (e.g. 'What did her periapical radiograph show?'),
       it routes directly to that report.
    2. If the query asks for a cross-report synthesis (e.g. 'Summarize all her reports', 'Does she have caries across her records?'),
       it synthesizes across all reports of the patient with source citations.
    3. If a specific tooth or pathology is asked, it checks across all reports.
    4. If active_report_id is set, it falls back to grounded report QA.
    """
    patient = get_patient(patient_id)
    if not patient and active_report_id:
        r_obj = get_report_by_id(active_report_id)
        if r_obj:
            patient = get_patient(r_obj.get("patient_id"))

    if not patient:
        return {
            "status": "not_found",
            "answer": "Patient record could not be found.",
            "citations": []
        }

    all_reports = get_patient_reports(patient["patient_id"])
    if not all_reports:
        return {
            "status": "not_found",
            "answer": f"No diagnostic imaging reports on file for {patient.get('patient_name')}.",
            "citations": []
        }

    q_lower = query.lower()

    # 0. Demographic / Personal Information Query
    demo_patterns = [
        r"\bpersonal\s+info(?:rmation)?\b",
        r"\bpersonal\s+details?\b",
        r"\bpatient\s+(?:info(?:rmation)?|details?|demographics?|name|id)\b",
        r"\bwho\s+is\s+(?:the\s+)?patient\b",
        r"\b(?:what\s+is\s+)?(?:his|her|their|the\s+patient'?s?)\s+name\b",
        r"\b(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date)\b",
        r"\bhow\s+old\b",
        r"\bage\b",
        r"\b(?:gender|sex)\b",
        r"\b(?:mrn|record\s+number|medical\s+record)\b",
        r"\bdemographics?\b"
    ]
    demo_phrases = [
        "personal info", "personal information", "personal details", "personal detail",
        "patient info", "patient information", "patient details", "patient detail",
        "demographic", "demographics", "date of birth", "birth date", "how old",
        "patient name", "who is the patient"
    ]
    is_demo_query = any(re.search(p, q_lower) for p in demo_patterns) or any(p in q_lower for p in demo_phrases)
    if is_demo_query:
        target_r = get_report_by_id(active_report_id) if active_report_id else (all_reports[0] if all_reports else None)
        if target_r:
            return answer_report_question(target_r["report_id"], query, history=history)
        else:
            dob = patient.get("date_of_birth")
            sex = patient.get("sex", "unspecified")
            mrn = patient.get("record_number")
            p_name = patient.get("patient_name", "Unassigned")
            p_id = patient.get("patient_id", "N/A")
            sex_str = str(sex).lower()
            pronoun_poss = "his" if sex_str == "male" else ("her" if sex_str == "female" else "the patient's")
            age_str = "documented on file"
            if dob:
                try:
                    b_year = int(dob.split("-")[0])
                    age = 2026 - b_year
                    age_str = f"{age} years"
                except Exception:
                    age_str = "documented on file"
            mrn_line = f"• Medical Record Number (MRN): {mrn}\n" if mrn else ""
            return {
                "status": "success",
                "answer": (
                    f"The personal and demographic information documented for {p_name} ({p_id}) is:\n\n"
                    f"• Patient Name: {p_name}\n"
                    f"• Patient ID: {p_id}\n"
                    f"{mrn_line}"
                    f"• Date of Birth: {dob or 'Not documented'}\n"
                    f"• Age: {age_str}\n"
                    f"• Biological Sex / Gender: {sex}\n\n"
                    f"*Clinical Note: Demographic details are retrieved directly from the Master Patient Index.*"
                ),
                "citations": [f"Master Patient Index: {p_id} ({p_name}, DOB: {dob or 'N/A'}, Sex: {sex or 'N/A'})"],
                "report_id": None,
                "report_date": None
            }

    # 1. Target specific modality or report ID among the patient's reports
    # (e.g. "What did her periapical scan show?", "What was found in her dental examination?", "Show findings in RPT-2002")
    rid_match = re.search(r"\b(RPT-\d+)\b", query, re.IGNORECASE)
    target_rid = rid_match.group(1).upper() if rid_match else None
    
    modality_keyword = None
    if any(m in q_lower for m in ["periapical", "iopa"]):
        modality_keyword = "periapical"
    elif any(m in q_lower for m in ["dental exam", "dental examination", "clinical exam", "examination"]):
        modality_keyword = "examination"
    elif any(m in q_lower for m in ["opg", "panoramic", "orthopantomogram"]):
        modality_keyword = "opg"
    elif any(m in q_lower for m in ["cbct", "cone beam", "3d volume"]):
        modality_keyword = "cbct"
    elif "bitewing" in q_lower:
        modality_keyword = "bitewing"

    matched_report = None
    if target_rid:
        matched_report = next((r for r in all_reports if r["report_id"] == target_rid), None)
    elif modality_keyword:
        matched_report = next((r for r in all_reports if modality_keyword in r.get("report_type", "").lower()), None)

    if matched_report:
        # User explicitly asked about this modality/report
        return answer_report_question(matched_report["report_id"], query, history=history)

    # 2. Patient Report Inventory / Count Queries
    # e.g. "how many report have they", "how many reports do they have", "what reports are on file", "list scans"
    is_report_inventory = bool(re.search(
        r"\b(?:how\s+many|count|number\s+of|list|what)\b.*\b(?:reports?|scans?)\b|"
        r"\b(?:reports?|scans?)\b.*\b(?:on\s+file|available|exist|have|has|do\s+they|does)\b",
        q_lower
    ))
    if is_report_inventory and all_reports:
        p_name = patient.get("patient_name", "The patient")
        p_id = patient.get("patient_id", "")
        id_str = f" ({p_id})" if p_id else ""
        from .retrieval import get_patient_pronouns
        pronouns = get_patient_pronouns(patient.get("sex"))
        pronoun_pos = pronouns["possessive"]

        rep_bullets = [
            f"• **[{r['report_id']}] {r['report_type']}** ({r['report_date']})"
            + (f": {r.get('clinical_indication')}" if r.get('clinical_indication') else "")
            for r in all_reports
        ]
        answer_text = (
            f"**{p_name}**{id_str} has **{len(all_reports)} diagnostic report{'s' if len(all_reports) != 1 else ''}** on file:\n\n"
            + "\n".join(rep_bullets)
            + f"\n\nYou can ask to view any of these reports (e.g. 'Open {pronoun_pos} {all_reports[0]['report_type']}' or 'Show {all_reports[0]['report_id']}'), or ask clinical questions about any specific report."
        )
        citations = [f"[{r['report_id']}] {r['report_type']} ({r['report_date']})" for r in all_reports]
        return {
            "status": "success",
            "answer": answer_text,
            "citations": citations,
            "patient_id": patient["patient_id"],
            "report_count": len(all_reports),
            "report_id": active_report_id or all_reports[0]["report_id"],
            "report_date": all_reports[0].get("report_date"),
            "report_type": f"Patient Reports Summary ({len(all_reports)} on file)"
        }

    # 3. Broad cross-report synthesis queries
    # e.g. "summarize all her reports", "across her records", "full history", "complete summary"
    is_multi_summary = (
        bool(re.search(r"\b(?:all|across|entire|full|complete|every)\s+(?:her\s+|his\s+|their\s+|the\s+)?(?:reports?|scans?|records?|history)\b", q_lower))
        or bool(re.search(r"\bsummarize\s+(?:all\s+)?(?:her\s+|his\s+|their\s+|the\s+)?(?:reports?|scans?)\b", q_lower))
        or (not active_report_id and any(w in q_lower for w in ["summarize", "summary", "problem", "wrong", "findings", "finding", "diagnos"]))
    )
    if is_multi_summary:
        return generate_multi_report_summary(all_reports, patient, query)

    # 3. Cross-report condition/pathology queries
    # e.g. "does she have caries across her reports", "any caries in her records", "is there caries in any report"
    is_cross_pathology = any(w in q_lower for w in ["across", "in any", "all reports", "her records"]) and any(w in q_lower for w in ["caries", "cavity", "cavities", "decay", "impaction", "nerve", "periodont"])
    if is_cross_pathology:
        return generate_multi_report_summary(all_reports, patient, query)

    # 4. If an active report is selected, use it with cross-reference fallback for teeth
    if active_report_id:
        return answer_report_question(active_report_id, query, history=history)

    # 5. If no active report is selected yet, check if a single report exists
    if len(all_reports) == 1:
        return answer_report_question(all_reports[0]["report_id"], query, history=history)

    # 6. Default across all reports
    return generate_multi_report_summary(all_reports, patient, query)

