import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from pydantic import ValidationError
from .validation import label_diagnostic_certainty
from .dental_anatomy import normalize_dental_site, classify_observation_category
from .report_service import synthesize_impression_from_findings
from ..schemas.report import (
    PatientDetailsModel,
    AnatomicalSiteModel,
    FindingModel,
    DictationDraftReportModel,
    RetrievalIntentModel,
)

# AI Endpoints & Credentials
def _load_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(var_name)
    if val:
        return val
    # Check project root .env
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{var_name}="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return default

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_CHAT_ENDPOINT = os.getenv("OLLAMA_CHAT_ENDPOINT", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


# =====================================================================
# GENERAL CLINICAL DICTATION STRUCTURING PROMPT
# =====================================================================
DICTATION_SYSTEM_PROMPT = """You are a clinical dental dictation structuring assistant.
Convert the supplied dental dictation into structured report data.

Rules:
1. Extract all clinically relevant findings explicitly stated in the input.
2. Support any dental anatomy, tooth, region, modality, finding, measurement, impression, differential, and recommendation.
3. Preserve laterality, tooth identifiers, measurements, negation, and uncertainty.
4. Do not invent findings, diagnoses, measurements, or recommendations.
5. Do not turn a possible finding into a confirmed finding.
6. Keep separate findings at separate anatomical sites separate.
7. Do not assign a finding to a tooth or region unless supported by the source.
8. Preserve negative findings when explicitly stated (polarity: "absent", certainty: "negative / normal finding").
9. Distinguish observations from impressions and recommendations.
10. If wording is ambiguous or corrupted by speech recognition, set requires_clarification: true and record clarification_reason instead of guessing.
11. Include source_quote with exact verbatim transcript evidence for each finding.
12. Missing information is unknown, not normal.
14. Separate patient metadata (patient name, patient ID, MRN) and study/examination modality into 'patient_details'. NEVER include administrative metadata or study headers as clinical findings or in impressions.
15. If a finding discusses adjacent teeth or general surrounding anatomy, do NOT force a specific tooth number onto it.
16. NEVER extract prompt instructions, user commands, meta-directives (such as 'Generate findings...', 'Retrieve patient demographics...', 'Flag missing examination...') or filler meta statements (e.g. 'No additional clinical findings were dictated') as clinical findings or impressions.
17. Return only valid JSON matching the supplied schema.

JSON Schema:
{
  "patient_details": {
    "age": integer or null,
    "gender": "Female" | "Male" or null,
    "demographics_summary": "string" or null,
    "patient_name": "string" or null,
    "patient_id": "string" or null,
    "study_type": "string" or null
  },
  "anatomical_sites": [
    {
      "site_type": "Tooth" | "Regional Anatomy" | "Osseous Structure",
      "anatomical_name": "string",
      "quadrant": "string" or null,
      "fdi_number": integer or null,
      "palmer_notation": "string" or null,
      "universal_number": integer or null
    }
  ],
  "findings": [
    {
      "anatomical_site": "string",
      "tooth_number": "string" or null,
      "category": "string",
      "observation": "string",
      "polarity": "present" | "absent" | "uncertain",
      "certainty": "definite observation" | "definite observation (etiology uncertain)" | "suspected / differential diagnosis" | "negative / normal finding" | "unclear / clinician review required",
      "measurements": ["string"],
      "source_quote": "string",
      "related_sites": ["string"],
      "requires_clarification": false,
      "clarification_reason": "string" or null
    }
  ],
  "impressions": ["string"],
  "recommendations": ["string"]
}"""

RETRIEVAL_SYSTEM_PROMPT = """You are a dental clinic voice assistant.
Extract the patient name and requested imaging modality from the clinician's request.
Return strictly valid JSON:
{
  "patient_name": "extracted patient name",
  "report_type": "OPG" | "CBCT" | "Bitewing" | "Periapical" | "Cephalogram" | "Any"
}"""


# =====================================================================
# LOCAL OLLAMA LLM CLIENT (Local Private Execution)
# =====================================================================
def query_ollama(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """Queries the local Ollama LLM endpoint (llama3.1)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    if system_prompt:
        payload["system"] = system_prompt
    if json_format:
        payload["format"] = "json"

    timeout = int(os.getenv("OLLAMA_TIMEOUT", "45"))
    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            if response_text:
                print(f"\n [LOCAL OLLAMA ACTIVE] Received response from {OLLAMA_MODEL} ({len(response_text)} chars)!")
            return response_text
    except Exception as e:
        if os.getenv("DEBUG_LLM", "0") == "1":
            print(f"[Ollama LLM Notice] {OLLAMA_MODEL} query did not respond: {e}")
        return None


def query_llm(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """
    Direct Local Ollama LLM Inference (llama3.1).
    Private, local execution with deterministic fallback if daemon is offline.
    """
    return query_ollama(prompt, system_prompt=system_prompt, json_format=json_format)



def query_ollama_chat_tools(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Direct function / tool calling via local Ollama /api/chat.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False
    }
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "45"))
    req = urllib.request.Request(
        OLLAMA_CHAT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0].get("function", {})
                tool_name = tc.get("name")
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                print(f"\n✅ [OLLAMA TOOL CALL ACTIVE] Model {OLLAMA_MODEL} called tool '{tool_name}'!")
                return {"tool_name": tool_name, "arguments": args}
            return None
    except Exception as e:
        if os.getenv("DEBUG_LLM", "0") == "1":
            print(f"[Ollama Tool Notice] {OLLAMA_MODEL} chat tool call did not respond: {e}")
        return None


def extract_retrieval_intent_with_llm(user_input: str) -> Dict[str, str]:
    """Uses LLM reasoning to extract patient name and scan modality from voice or text, validated with Pydantic."""
    prompt = f'Clinician request: "{user_input}"'
    raw_llm = query_llm(prompt, system_prompt=RETRIEVAL_SYSTEM_PROMPT, json_format=True)
    if raw_llm:
        try:
            parsed = json.loads(raw_llm)
            validated = RetrievalIntentModel.model_validate(parsed)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError):
            pass

    # Dynamic semantic extraction (without hardcoded stop tokens)
    lower_input = user_input.lower().strip()

    modality = "Any"
    for m, m_name in [
        ("opg", "OPG"), ("panoramic", "OPG"), ("orthopantomogram", "OPG"),
        ("cbct", "CBCT"), ("cone beam", "CBCT"), ("3d volume", "CBCT"),
        ("bitewing", "Bitewing"), ("periapical", "Periapical"), ("iopa", "Periapical"),
        ("cephalogram", "Cephalogram"), ("ceph", "Cephalogram"),
        ("examination", "Dental Examination"), ("exam", "Dental Examination")
    ]:
        if m in lower_input:
            modality = m_name
            break

    # Explicit Patient ID (e.g. PAT-1001, PT-1001)
    id_match = re.search(r"\b(?:pat|pt)[- ]?0*(\d+)\b", user_input, re.IGNORECASE)
    if id_match:
        return {"patient_name": f"PAT-{id_match.group(1)}", "report_type": modality}

    pronoun_refs = {
        "he", "she", "his", "her", "hers", "hir", "their", "theirs", "its", "him", "them",
        "this", "that", "these", "those", "it", "hiss", "herr"
    }

    # Possessive: "Open Priya's latest OPG report"
    pos_match = re.search(r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)(?:'s|’s)\b", user_input)
    if pos_match:
        cand = pos_match.group(1).strip()
        if not any(w in pronoun_refs for w in cand.lower().split()):
            return {"patient_name": cand, "report_type": modality}

    # Prepositional: "for Priya Sharma", "patient Vikram"
    prep_match = re.search(r"\b(?:for\s+patient|patient|pt|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b", user_input, re.IGNORECASE)
    if prep_match:
        cand = prep_match.group(1).strip()
        if not any(w in pronoun_refs for w in cand.lower().split()):
            return {"patient_name": cand, "report_type": modality}

    # Direct imperative: "open Amit Kumar"
    imp_match = re.search(r"^(?:open|show|view|find|load|retrieve|fetch)\s+(?:report\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)?)$", user_input, re.IGNORECASE)
    if imp_match:
        cand = imp_match.group(1).strip()
        if not any(w in pronoun_refs for w in cand.lower().split()):
            # Also check if cand is just modality or report word
            cand_words = set(cand.lower().split())
            if not cand_words.issubset({"opg", "cbct", "panoramic", "bitewing", "periapical", "iopa", "report", "scan", "exam", "examination", "latest", "recent"}):
                return {"patient_name": cand, "report_type": modality}

    return {"patient_name": "", "report_type": modality}


def clean_conversational_dictation(text: str) -> str:
    """Strips conversational preambles like 'hey can you generate the report for the...' or 'Dictate: ...'"""
    cleaned = re.sub(
        r"^(?:dictate\s*[:\-]?\s*|dictation\s*[:\-]?\s*|please\s+dictate\s*[:\-]?\s*)",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"^(?:hey\s+|hello\s+|hi\s+)?(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:generate|create|write|draft|make|give\s+me)?\s*(?:the|a)?\s*(?:report|clinical\s+report|dictation)?\s*(?:for\s+(?:the)?)?\s*",
        "",
        cleaned.strip(),
        flags=re.IGNORECASE
    )
    return cleaned.strip()

def detect_transcription_artifacts(text: str) -> List[Dict[str, str]]:
    """Identifies speech-to-text corruptions, phonetic slips, and unclear non-medical phrases."""
    artifacts = []
    # Pattern 1: Acoustic / phonetic speech-to-text slips (e.g. 'root Bridge option' / 'bridge option' / 'root bridging' for resorption)
    m_slip = re.search(r"\b(root\s+bridge\s+option|bridge\s+option|root\s+bridging)\b", text, re.IGNORECASE)
    if m_slip:
        artifacts.append({
            "matched_phrase": m_slip.group(0),
            "flag": f"Unclear phrase: '{m_slip.group(0)}' (flagged as likely speech transcription error; requires clinician review)",
            "category": "Root Pathology & Resorption"
        })

    # Pattern 2: Explicit audio / transcription quality markers
    m_unclear = re.search(r"\b(inaudible|unintelligible|unclear\s+speech|muffled\s+audio|audio\s+gap|garbled|phoneme\s+slip)\b", text, re.IGNORECASE)
    if m_unclear:
        artifacts.append({
            "matched_phrase": m_unclear.group(0),
            "flag": f"Acoustic quality marker: '{m_unclear.group(0)}' (requires clinician review)",
            "category": "Speech Recognition Clarification"
        })

    return artifacts

def parse_dictation_with_llm(dictation_text: str) -> Dict[str, Any]:
    """
    Feature B: Clinical Dictation Structuring Engine.
    Uses LLM instruction reasoning with deterministic dental anatomy normalization
    for standardized FDI, Universal, and Palmer notation, diagnostic certainty,
    and anti-hallucination verification.
    """
    cleaned_text = clean_conversational_dictation(dictation_text)
    from .claim_verification import is_clinically_relevant_dictation, extract_patient_demographics
    clinical_body = extract_clinical_body(cleaned_text)
    if not is_clinically_relevant_dictation(cleaned_text) and not is_clinically_relevant_dictation(clinical_body):
        patient_details = extract_patient_demographics(cleaned_text)
        return {
            "anatomical_sites": [],
            "findings": [],
            "impressions": [],
            "recommendations": [],
            "extraction_status": "NON_CLINICAL_INPUT_REJECTED",
            "findings_count": 0,
            "patient_details": patient_details,
            "clinical_warning": "Input text does not contain dental or maxillofacial clinical dictation. No clinical findings were extracted."
        }

    prompt = f'Clinical dictation transcript:\n"{cleaned_text}"'
    raw_llm = query_llm(prompt, system_prompt=DICTATION_SYSTEM_PROMPT, json_format=True)
    result = None

    if raw_llm:
        try:
            parsed = json.loads(raw_llm)
            validated = DictationDraftReportModel.model_validate(parsed)
            result = validated.model_dump()
        except (json.JSONDecodeError, ValidationError):
            # One retry with explicit JSON format instruction
            retry_prompt = f"{prompt}\n\nIMPORTANT: Return ONLY valid, parseable JSON matching the schema. No markdown formatting or extra text."
            retry_llm = query_llm(retry_prompt, system_prompt=DICTATION_SYSTEM_PROMPT, json_format=True)
            if retry_llm:
                try:
                    parsed = json.loads(retry_llm)
                    validated = DictationDraftReportModel.model_validate(parsed)
                    result = validated.model_dump()
                except (json.JSONDecodeError, ValidationError):
                    result = None

    is_fallback = False
    if not result or not result.get("findings"):
        fallback = _dynamic_semantic_parse(cleaned_text)
        is_fallback = True
        if not result:
            result = fallback
        else:
            if not result.get("findings"):
                result["findings"] = fallback.get("findings", [])
            if not result.get("impressions") and fallback.get("impressions"):
                result["impressions"] = fallback.get("impressions", [])
            if not result.get("recommendations") and fallback.get("recommendations"):
                result["recommendations"] = fallback.get("recommendations", [])
            if not result.get("anatomical_sites") and fallback.get("anatomical_sites"):
                result["anatomical_sites"] = fallback.get("anatomical_sites", [])

    if not result.get("findings"):
        result["extraction_status"] = "INCOMPLETE_REQUIRES_REVIEW"
    elif is_fallback:
        result["extraction_status"] = "FALLBACK_PARSED_REQUIRES_REVIEW"
    else:
        result["extraction_status"] = "COMPLETE"
    result["findings_count"] = len(result.get("findings", []))

    # Ensure patient_details is populated
    from .claim_verification import extract_patient_demographics, extract_measurements
    if not result.get("patient_details"):
        result["patient_details"] = extract_patient_demographics(dictation_text)

    # 1. Enforce No Invented Recommendations
    has_explicit_rec = bool(re.search(r"\b(recommend|recommended|recommendation|advise|advised|routine follow-up)\b", dictation_text, re.IGNORECASE))
    if not has_explicit_rec:
        result["recommendations"] = []

    # 2. Flag Transcription Artifacts (e.g. 'root Bridge option')
    artifacts = detect_transcription_artifacts(dictation_text)
    if artifacts:
        clean_findings = []
        replaced = False
        for f in result.get("findings", []):
            obs_lower = f.get("observation", "").lower()
            if any(term in obs_lower for term in ["bridging", "bridge option", "root bridge"]):
                clean_findings.append({
                    "anatomical_site": f.get("anatomical_site", "Dentition"),
                    "tooth_number": f.get("tooth_number"),
                    "category": "Root Pathology & Resorption",
                    "observation": f"Unclear phrase: '{artifacts[0]['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                    "polarity": "uncertain",
                    "certainty": "unclear / clinician review required",
                    "measurements": [],
                    "source_quote": f.get("source_quote"),
                    "related_sites": [],
                    "requires_clarification": True,
                    "clarification_reason": f"Acoustic artifact: '{artifacts[0]['matched_phrase']}'"
                })
                replaced = True
            else:
                clean_findings.append(f)
        if not replaced and clean_findings:
            clean_findings.insert(len(clean_findings)//2, {
                "anatomical_site": clean_findings[0].get("anatomical_site", "Dentition"),
                "tooth_number": clean_findings[0].get("tooth_number"),
                "category": "Root Pathology & Resorption",
                "observation": f"Unclear phrase: '{artifacts[0]['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                "polarity": "uncertain",
                "certainty": "unclear / clinician review required",
                "measurements": [],
                "source_quote": clean_findings[0].get("source_quote"),
                "related_sites": [],
                "requires_clarification": True,
                "clarification_reason": f"Acoustic artifact: '{artifacts[0]['matched_phrase']}'"
            })
        result["findings"] = clean_findings

        # Clean impression so it doesn't assert verified bridging
        clean_impressions = []
        for imp in result.get("impressions", []):
            clean_imp = re.sub(r"(?:with\s+no\s+|no\s+)?root\s+bridg(?:ing|e\s+option)[^;,\.]*", "unclear phrase 'root Bridge option' flagged for clinician review", imp, flags=re.I)
            clean_impressions.append(clean_imp)
        result["impressions"] = clean_impressions

    
    # DETERMINISTIC CLINICAL NORMALIZATION LAYER 
    
    normalized_sites = []
    seen_site_keys = set()
    for s in result.get("anatomical_sites", []):
        norm_s = normalize_dental_site(s)
        key = norm_s.get("fdi_number") or norm_s.get("anatomical_name")
        if key not in seen_site_keys:
            seen_site_keys.add(key)
            normalized_sites.append(norm_s)
    result["anatomical_sites"] = normalized_sites

    # Categorize findings: respect LLM classification when specific, fallback to deterministic categorization only if generic
    from .claim_verification import classify_clinical_finding_status
    valid_findings = []
    for f in result.get("findings", []):
        obs = f.get("observation", "")
        obs_l = obs.lower().strip()
        # Filter out administrative metadata, demographics, study headers, or author details erroneously placed into findings
        if (
            re.search(r"\b(?:patient\s*id|mrn|pt\s*id|serial\s*#?|serial\s*number)\b", obs_l) or
            re.search(r"\b(?:patient\s*name|patient:|dob:|date\s+of\s+birth:)\b", obs_l) or
            re.search(r"\b(?:study\s*date|report\s*date|report\s*type)\b", obs_l) or
            re.search(r"\b(?:radiology\s+report\s+details|conducted\s+by|gdc\s*[:#]|gdc\s+registration|suite\s+\d+|harley\s+st|dental\s+scan\s+ltd)\b", obs_l) or
            re.search(r"^(?:panoramic\s+radiograph|cone\s+beam\s+ct|cbct|opg|bitewings?|periapical(?:\s+radiograph)?)$", obs_l) or
            re.search(r"^(?:dentalscan\s*office|clinical\s*notes)\s*[:\-]", obs_l)
        ):
            continue

        # Extract pure recommendations into recommendations list
        if re.search(r"\b(further\s+evaluation\s+is\s+recommended|routine\s+follow-up\s+is\s+advised)\b", obs_l) or (re.search(r"\b(recommend|recommended|advise|advised)\b", obs_l) and not re.search(r"\b(caries|lesion|root|canal|fracture|ridge|cortex|bone\s+loss)\b", obs_l)):
            clean_rec = re.sub(r"^(?:recommendations?:\s*)?", "", obs, flags=re.I).strip()
            clean_rec = clean_rec[0].upper() + clean_rec[1:]
            if clean_rec not in result.get("recommendations", []):
                result.setdefault("recommendations", []).append(clean_rec)
            continue

        site = f.get("anatomical_site", "")
        # Ensure adjacent teeth, multi-tooth, or edentulous findings do not retain an unsupported specific tooth assignment
        palmer_matches = re.findall(r"\b(UR|UL|LL|LR)[\s_-]*([1-8]|[A-E])\b", obs, re.IGNORECASE)
        num_teeth_in_obs = len(set(f"{m[0].upper()}{m[1].upper()}" for m in palmer_matches))
        is_adjacent = bool(re.search(r"\b(adjacent\s+teeth|surrounding\s+teeth|adjacent\s+structures|adjacent\s+dentition|surrounding\s+dentition)\b", obs_l))
        is_edentulous = bool(re.search(r"\b(?:edentulous|missing\s+teeth|missing\s+crowns|extraction\s+socket|edentulous\s+space|edentulous\s+ridge)\b", obs_l))
        is_multi_tooth = num_teeth_in_obs > 1 or bool(re.search(r"\b(?:posterior\s+teeth|two\s+teeth|three\s+teeth|multiple\s+teeth|teeth\s+\d+)\b", obs_l))

        if is_adjacent:
            f["tooth_number"] = None
            if not site or site in ("Dentition", "General Observation"):
                f["anatomical_site"] = "Adjacent Teeth"
        elif is_edentulous or is_multi_tooth:
            f["tooth_number"] = None

        if "unclear phrase" in obs.lower() or f.get("requires_clarification"):
            f["category"] = "Root Pathology & Resorption"
            f["certainty"] = "unclear / clinician review required"
            f["polarity"] = "uncertain"
        else:
            current_cat = f.get("category")
            if not current_cat or current_cat in ("General Observation", "unspecified", "Unknown"):
                f["category"] = classify_observation_category(obs, site)

            # Evaluate clinical status and certainty
            finding_eval = classify_clinical_finding_status(obs)
            eval_status = finding_eval.get("finding_status", "positive")
            eval_legacy = finding_eval.get("legacy_label", "definite observation")
            if eval_status == "negative":
                f["polarity"] = "absent"
                if not f.get("certainty") or f.get("certainty") == "definite observation":
                    f["certainty"] = eval_legacy
            else:
                f["polarity"] = "present"
                if f.get("certainty") == "negative / normal finding":
                    f["certainty"] = eval_legacy

        if not f.get("measurements"):
            f["measurements"] = extract_measurements(obs)
        if not f.get("source_quote"):
            f["source_quote"] = obs

        valid_findings.append(f)
    result["findings"] = valid_findings

    # Clean impressions: eliminate metadata contamination
    clean_imps = []
    for imp in result.get("impressions", []):
        parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", imp) if p.strip()]
        for p in parts:
            if not re.search(r"\b(?:patient\s*id|patient\s*name|study\s*type|mrn|dob:|scan\s*type:|serial\s*#?|radiology\s+report\s+details|conducted\s+by)\b", p, re.I) and len(p) > 3:
                clean_imps.append(p)
    if not clean_imps and result["findings"]:
        clean_imps = synthesize_impression_from_findings(result["findings"], artifacts)
    result["impressions"] = list(dict.fromkeys(clean_imps))

    return result



def extract_clinical_body(text: str) -> str:
    """
    Extracts the clinical observations section from dictation text,
    stripping document headers (patient demographics, study metadata, clinic headers),
    document footers (author details, credentials, facility addresses, abbreviations),
    and separating clinical dictation from surrounding prompt instructions, directives,
    and report formatting requests.
    """
    from .claim_verification import is_meta_or_instruction_sentence, is_clinical_observation_sentence, is_clinically_relevant_dictation

    raw_text = text.strip()

    # 1. Check for quoted transcript boundary: e.g. 'transcript: "..."' or '"Lower right third molar..."'
    quoted_matches = re.findall(r'"([^"]{15,})"', raw_text)
    if quoted_matches:
        for qm in quoted_matches:
            if is_clinically_relevant_dictation(qm):
                raw_text = qm.strip()
                break

    # 2. Section Header Splitting
    m_findings = re.search(r"\b(?:DENTAL\s+FINDINGS|CLINICAL\s+FINDINGS|RADIOLOGICAL\s+FINDINGS|IMAGING\s+FINDINGS|FINDINGS)\s*[:\-]\s*", raw_text, re.IGNORECASE)
    if m_findings:
        clinical_text = raw_text[m_findings.end():]
    else:
        m_notes = re.search(r"\bCLINICAL\s+NOTES\s*[:\-].*?(?:\.\s+|\n+)", raw_text, re.IGNORECASE)
        if m_notes:
            clinical_text = raw_text[m_notes.end():]
        else:
            m_header_end = re.search(r"(?:SERIAL\s*#?:?\s*[A-Za-z0-9_-]+.*?)(?:\.\s+|\n+)", raw_text, re.IGNORECASE)
            if m_header_end:
                clinical_text = raw_text[m_header_end.end():]
            else:
                clinical_text = raw_text

    # 3. Strip Document Footer
    m_footer = re.search(r"\b(?:RADIOLOGY\s+REPORT\s+DETAILS|REPORT\s+DETAILS|CONDUCTED\s+BY\s+DR|REPORTED\s+BY\s+DR|ABBREVIATIONS|REPORT\s+WAS\s+VERIFIED)\b", clinical_text, re.IGNORECASE)
    if m_footer:
        clinical_text = clinical_text[:m_footer.start()]

    # 4. Truncate at Instruction / Prompt Meta Boundary:
    # When clinical dictation is followed by report generation directives (e.g. "Generate a clean, editable report..."),
    # detect the boundary where non-clinical meta instructions begin and discard everything from that boundary onwards.
    sentences = split_clinical_sentences(clinical_text)
    has_seen_clinical = False
    valid_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if is_meta_or_instruction_sentence(s_clean):
            # If we already have clinical dictation, an instruction sentence marks the end of the dictation body!
            if has_seen_clinical:
                break
            else:
                # Instruction sentence before clinical content: skip it
                continue
        if is_clinical_observation_sentence(s_clean):
            has_seen_clinical = True
            valid_sentences.append(s_clean)
        else:
            # Check if ambiguous or non-instruction
            if has_seen_clinical:
                # If following clinical content but not clearly clinical, check if it's an imperative instruction
                if re.match(r"^(?:generate|create|produce|keep|separate|preserve|ensure|require|make|render|write|format|do\s+not|if\s+the|the\s+impression)\b", s_clean.lower()):
                    break
                valid_sentences.append(s_clean)

    if valid_sentences:
        clinical_text = ". ".join(s.rstrip(".") for s in valid_sentences) + "."

    return clinical_text.strip()


def split_clinical_sentences(clinical_text: str) -> List[str]:
    """
    Splits clinical text into distinct observation sentences without prematurely truncating
    decimal numbers (0.5 mm) or breaking on abbreviations (Dr., Dipl., Ltd.).
    """
    cleaned = re.sub(r"\b(?:IMPRESSIONS?\s*(?:&|and)\s*SPECIFIC\s+FINDINGS|SPECIFIC\s+FINDINGS|DENTAL\s+FINDINGS)\s*[:\-]?\s*", " ", clinical_text, flags=re.IGNORECASE)
    protected = re.sub(r"\b(Dr|Dipl|Ltd|vs|eg|ie)\.", r"\1__DOT__", cleaned, flags=re.IGNORECASE)
    raw_sents = re.split(r"\.(?!\d)(?:\s+(?=[A-Z0-9])|\s*$)|;\s*(?=(?:further\s+evaluation|recommend|advise))|[\n?!]+", protected)
    sentences = []
    for s in raw_sents:
        s_clean = s.replace("__DOT__", ".").strip()
        if len(s_clean) > 3:
            sentences.append(s_clean)
    if len(sentences) <= 1 and len(cleaned.split()) > 7:
        sentences = [
            s.strip() for s in re.split(
                r"\s+(?=(?:it\s+is\s+|the\s+surrounding\s+|no\s+other\s+))",
                cleaned,
                flags=re.IGNORECASE
            ) if len(s.strip()) > 3
        ]
    return sentences


def analyze_sentence_anatomy(sentence: str, anatomical_sites: List[Dict[str, Any]], active_quad: Optional[str]) -> Tuple[str, Optional[str], List[str]]:
    """
    Determines anatomical attribution for a single clinical observation:
    - Scope single-tooth observations only when explicit evidence exists for exactly one tooth.
    - Preserves multi-tooth, edentulous, adjacent teeth, and regional findings with tooth_number = None.
    """
    s_l = sentence.lower()
    detected_teeth = []

    # 0. Anterior / Central Incisors regional anatomy
    if re.search(r"\b(?:lower|mandibular)\s+central\s+incisors?\b", s_l):
        norm31 = normalize_dental_site({"palmer_notation": "LL1", "anatomical_name": "LL1"})
        norm41 = normalize_dental_site({"palmer_notation": "LR1", "anatomical_name": "LR1"})
        if norm31.get("fdi_number") and not any(dt.get("fdi_number") == 31 for dt in detected_teeth):
            detected_teeth.append(norm31)
        if norm41.get("fdi_number") and not any(dt.get("fdi_number") == 41 for dt in detected_teeth):
            detected_teeth.append(norm41)
    elif re.search(r"\b(?:upper|maxillary)\s+central\s+incisors?\b", s_l):
        norm11 = normalize_dental_site({"palmer_notation": "UR1", "anatomical_name": "UR1"})
        norm21 = normalize_dental_site({"palmer_notation": "UL1", "anatomical_name": "UL1"})
        if norm11.get("fdi_number") and not any(dt.get("fdi_number") == 11 for dt in detected_teeth):
            detected_teeth.append(norm11)
        if norm21.get("fdi_number") and not any(dt.get("fdi_number") == 21 for dt in detected_teeth):
            detected_teeth.append(norm21)

    if re.search(r"\b(?:lower\s+front\s+region|mandibular\s+anterior\s+region|mandibular\s+anterior)\b", s_l):
        return "Mandibular Anterior Region", None, []
    if re.search(r"\b(?:upper\s+front\s+region|maxillary\s+anterior\s+region|maxillary\s+anterior)\b", s_l):
        return "Maxillary Anterior Region", None, []

    # 1. Edentulous / missing tooth site
    is_edentulous = bool(re.search(r"\b(?:edentulous|missing\s+teeth|missing\s+crowns|extraction\s+socket|edentulous\s+space|edentulous\s+ridge)\b", s_l)) and not re.search(r"\bdentate\s+adult\b", s_l)

    # 2. Extract tooth tokens (Palmer notation and Tooth #)
    palmer_matches = re.findall(r"\b(UR|UL|LL|LR)[\s_-]*([1-8]|[A-E])\b", sentence, re.IGNORECASE)
    tooth_num_matches = re.findall(r"\b(?:teeth|tooth)\s*#?(\d{1,2})\b", sentence, re.IGNORECASE)

    for pm in palmer_matches:
        raw = f"{pm[0].upper()}{pm[1].upper()}"
        norm = normalize_dental_site({"anatomical_name": raw})
        if norm.get("fdi_number"):
            detected_teeth.append(norm)
    for tm in tooth_num_matches:
        t_val = int(tm)
        norm = normalize_dental_site({"fdi_number": t_val, "anatomical_name": f"Tooth {t_val}"})
        if norm.get("fdi_number"):
            detected_teeth.append(norm)

    named_map = [
        (r"third\s+molars?", 8), (r"wisdom\s+teeth?", 8),
        (r"second\s+molars?", 7), (r"first\s+molars?", 6),
        (r"second\s+premolars?", 5), (r"first\s+premolars?", 4),
        (r"canines?", 3), (r"lateral\s+incisors?", 2), (r"central\s+incisors?", 1)
    ]
    has_resolved_central = bool(re.search(r"\b(?:lower|mandibular|upper|maxillary)\s+central\s+incisors?\b", s_l))
    for n_pat, pos_idx in named_map:
        if pos_idx == 1 and has_resolved_central:
            continue
        if re.search(r"\b" + n_pat + r"\b", s_l):
            s_quad = None
            if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul)\b", s_l):
                s_quad = "UL"
            elif re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur)\b", s_l):
                s_quad = "UR"
            elif re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll)\b", s_l):
                s_quad = "LL"
            elif re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr)\b", s_l):
                s_quad = "LR"
            elif active_quad:
                s_quad = active_quad

            if s_quad:
                p_code = f"{s_quad}{pos_idx}"
                norm = normalize_dental_site({"anatomical_name": p_code})
                if norm.get("fdi_number") and not any(dt.get("fdi_number") == norm["fdi_number"] for dt in detected_teeth):
                    detected_teeth.append(norm)

    unique_fdi = {}
    for dt in detected_teeth:
        fdi = dt.get("fdi_number")
        if fdi and fdi not in unique_fdi:
            unique_fdi[fdi] = dt

    if is_edentulous:
        if len(unique_fdi) == 1:
            st = list(unique_fdi.values())[0]
            site_label = f"{st.get('anatomical_name')} Region (Edentulous)"
            return site_label, None, [st.get("anatomical_name")]
        return "Edentulous Region", None, [st.get("anatomical_name") for st in unique_fdi.values()]

    fdi_set = set(unique_fdi.keys())
    if fdi_set == {31, 41}:
        rel = [st.get("anatomical_name") for st in unique_fdi.values()]
        return "Mandibular Central Incisors (Tooth 31, 41)", None, rel
    if fdi_set == {11, 21}:
        rel = [st.get("anatomical_name") for st in unique_fdi.values()]
        return "Maxillary Central Incisors (Tooth 11, 21)", None, rel

    if len(unique_fdi) > 1 or re.search(r"\b(?:posterior\s+teeth|two\s+teeth|three\s+teeth|multiple\s+teeth|teeth\s+\d+)\b", s_l):
        rel = [st.get("anatomical_name") for st in unique_fdi.values()]
        return "Multiple Teeth", None, rel

    if len(unique_fdi) == 1:
        st = list(unique_fdi.values())[0]
        return st.get("anatomical_name"), str(st.get("fdi_number")), [st.get("anatomical_name")]

    # 3. If no specific tooth detected, evaluate regional non-dental structures
    if re.search(r"\b(?:inferior\s+alveolar\s+(?:nerve\s+)?canal|ianc|mandibular\s+canal)\b", s_l):
        return "Inferior Alveolar Canal & Alveolar Crest", None, []
    if re.search(r"\b(?:temporomandibular|tmj)\b", s_l) and re.search(r"\b(?:sinus|sinuses)\b", s_l):
        return "Temporomandibular Joints & Paranasal Sinuses", None, []
    if re.search(r"\b(?:temporomandibular|tmj|condyle)\b", s_l):
        return "Temporomandibular Joints", None, []
    if re.search(r"\b(?:paranasal\s+sinuses?|maxillary\s+sinuses?|sinuses?)\b", s_l):
        return "Maxillary & Paranasal Sinuses", None, []
    if re.search(r"\b(?:airway|cervical\s+spine)\b", s_l):
        return "Regional Anatomy", None, []
    if re.search(r"\b(?:calcifications?|tonsilloliths?)\b", s_l):
        return "Soft Tissue", None, []

    # 4. Adjacent teeth / surrounding dentition
    if re.search(r"\b(adjacent\s+teeth|surrounding\s+teeth|adjacent\s+structures|adjacent\s+dentition|surrounding\s+dentition)\b", s_l):
        return "Adjacent Teeth", None, []

    if re.search(r"\b(?:alveolar\s+ridge|buccal\s+cortical\s+plate|cortical\s+plate)\b", s_l):
        for st in anatomical_sites:
            if "first molar" in st.get("anatomical_name", "").lower() or "alveolar" in st.get("anatomical_name", "").lower():
                return st.get("anatomical_name"), None, []
        return "Alveolar Ridge", None, []

    # Check for quadrant/regional mention
    if re.search(r"\b(?:lower\s+right|mandibular\s+right|quadrant\s+4|lr)\b", s_l):
        return "Mandibular Right Quadrant", None, []
    if re.search(r"\b(?:lower\s+left|mandibular\s+left|quadrant\s+3|ll)\b", s_l):
        return "Mandibular Left Quadrant", None, []
    if re.search(r"\b(?:upper\s+right|maxillary\s+right|quadrant\s+1|ur)\b", s_l):
        return "Maxillary Right Quadrant", None, []
    if re.search(r"\b(?:upper\s+left|maxillary\s+left|quadrant\s+2|ul)\b", s_l):
        return "Maxillary Left Quadrant", None, []

    return "Dentition", None, []


def _dynamic_semantic_parse(dictation_text: str) -> Dict[str, Any]:
    """
    Algorithmic clinical entity extraction when external LLM endpoints are offline.
    Infers dental quadrants, Palmer/FDI/Universal numbering, and categories deterministically.
    Strictly separates administrative metadata, study metadata, technical FOV limitations,
    and genuine clinical observations.
    """
    from .claim_verification import (
        extract_patient_demographics,
        extract_measurements,
        classify_clinical_finding_status,
        is_clinically_relevant_dictation
    )
    cleaned_text = clean_conversational_dictation(dictation_text)
    patient_details = extract_patient_demographics(cleaned_text)

    # 1. Clean and partition text into clinical observation body
    clinical_body = extract_clinical_body(cleaned_text)
    if not is_clinically_relevant_dictation(clinical_body):
        return {
            "patient_details": patient_details,
            "anatomical_sites": [],
            "findings": [],
            "impressions": [],
            "recommendations": [],
            "extraction_status": "NON_CLINICAL_INPUT_REJECTED",
            "findings_count": 0,
            "clinical_warning": "Input text does not contain dental or maxillofacial clinical dictation. No clinical findings were extracted."
        }

    # Strip age, measurements, and dates before extracting teeth to prevent misidentifying age as FDI tooth
    clean_no_age = re.sub(r"\b\d{1,2}\s*(?:-|–|\s)?(?:year|yr)s?(?:-|–|\s)?old\b|\bage\s*(?:is\s*)?\d{1,2}\b", " ", clinical_body, flags=re.IGNORECASE)
    clean_no_age = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|mg|ml|%)\b", " ", clean_no_age, flags=re.IGNORECASE)
    clean_no_age = re.sub(r"\b(?:19|20)\d{2}\b", " ", clean_no_age)

    # Segment clinical text into distinct observation sentences
    sentences = split_clinical_sentences(clinical_body)
    if not sentences:
        sentences = [clinical_body]

    anatomical_sites = []
    findings = []
    impressions = []
    recommendations = []
    seen_sites = set()

    # Helper to detect quadrant context from text
    def detect_quad_from_text(txt: str) -> Optional[str]:
        tl = txt.lower()
        if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul\b)", tl):
            return "UL"
        if re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur\b)", tl):
            return "UR"
        if re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll\b)", tl):
            return "LL"
        if re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr\b)", tl):
            return "LR"
        return None

    doc_quad = detect_quad_from_text(cleaned_text)
    active_quad = doc_quad

    # Quadrant-relative named teeth
    quad_tooth_patterns = [
        (r"third\s+molars?", 8), (r"wisdom\s+teeth?", 8),
        (r"second\s+molars?", 7),
        (r"first\s+molars?", 6),
        (r"second\s+premolars?", 5), (r"second\s+bicuspids?", 5),
        (r"first\s+premolars?", 4), (r"first\s+bicuspids?", 4),
        (r"canines?", 3),
        (r"lateral\s+incisors?", 2),
        (r"central\s+incisors?", 1)
    ]
    for s in sentences:
        s_lower = s.lower()
        has_central_pattern = False
        if re.search(r"\b(?:lower|mandibular)\s+central\s+incisors?\b", s_lower):
            has_central_pattern = True
            for p_code in ["LL1", "LR1"]:
                norm = normalize_dental_site({"palmer_notation": p_code, "anatomical_name": p_code})
                if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                    seen_sites.add(norm["fdi_number"])
                    anatomical_sites.append(norm)
        elif re.search(r"\b(?:upper|maxillary)\s+central\s+incisors?\b", s_lower):
            has_central_pattern = True
            for p_code in ["UL1", "UR1"]:
                norm = normalize_dental_site({"palmer_notation": p_code, "anatomical_name": p_code})
                if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                    seen_sites.add(norm["fdi_number"])
                    anatomical_sites.append(norm)

        s_quad = detect_quad_from_text(s) or doc_quad
        if s_quad:
            for name_pat, pos_idx in quad_tooth_patterns:
                if pos_idx == 1 and has_central_pattern:
                    continue
                if re.search(r"\b" + name_pat + r"\b", s_lower):
                    palmer_code = f"{s_quad}{pos_idx}"
                    norm = normalize_dental_site({"palmer_notation": palmer_code, "anatomical_name": palmer_code})
                    if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                        seen_sites.add(norm["fdi_number"])
                        anatomical_sites.append(norm)

    # Detect Palmer Teeth
    deciduous_matches = re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([A-E])\b", clean_no_age, re.IGNORECASE)
    for m in deciduous_matches:
        raw_token = f"{m.group(1).upper()} {m.group(2).upper()}"
        norm = normalize_dental_site({"anatomical_name": raw_token})
        if norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    palmer_matches = re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([1-8])\b", clean_no_age, re.IGNORECASE)
    for m in palmer_matches:
        raw_token = f"{m.group(1).upper()}{m.group(2)}"
        norm = normalize_dental_site({"anatomical_name": raw_token})
        if norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # Numbered teeth
    compound_matches = re.finditer(r"\b(?:teeth|tooth)\s*#?(\d{1,2})(?:\s*(?:,|and|&)\s*#?(\d{1,2}))?\b", clean_no_age, re.IGNORECASE)
    has_universal_study = bool(re.search(r"\b(?:teeth|tooth)\s*1\s*(?:-|–|to)\s*32\b", clean_no_age, re.IGNORECASE))

    for m in compound_matches:
        for grp in [m.group(1), m.group(2)]:
            if not grp:
                continue
            num = int(grp)
            is_hash = "#" in m.group(0)

            is_fdi_quad_match = False
            if (11 <= num <= 18 and re.search(r"\b(upper\s+right|maxillary\s+right|ur)\b", clean_no_age, re.I)) or \
               (21 <= num <= 28 and re.search(r"\b(upper\s+left|maxillary\s+left|ul)\b", clean_no_age, re.I)) or \
               (31 <= num <= 38 and re.search(r"\b(lower\s+left|mandibular\s+left|ll)\b", clean_no_age, re.I)) or \
               (41 <= num <= 48 and re.search(r"\b(lower\s+right|mandibular\s+right|lr)\b", clean_no_age, re.I)):
                is_fdi_quad_match = True

            if is_hash:
                norm = normalize_dental_site({"universal_number": num, "anatomical_name": f"Tooth {num}"})
            elif num > 32 or is_fdi_quad_match or (num in [12, 16, 25, 36, 48] and not has_universal_study):
                norm = normalize_dental_site({"fdi_number": num, "anatomical_name": f"Tooth {num}"})
            else:
                norm = normalize_dental_site({"universal_number": num, "anatomical_name": f"Tooth {num}"})

            if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                seen_sites.add(norm["fdi_number"])
                anatomical_sites.append(norm)

    # Deciduous Universal Letters
    decid_letter_matches = re.finditer(r"\btooth\s*#?([A-T])\b", clean_no_age, re.IGNORECASE)
    for m in decid_letter_matches:
        letter = m.group(1).upper()
        norm = normalize_dental_site({"universal_letter": letter, "anatomical_name": f"Tooth {letter}"})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # Regional extra-dental anatomy
    extra_sites_map = [
        ([r"palatine\s+tonsilloliths?", r"tonsilloliths?", r"tonsils?"], "Palatine Tonsils", "Soft Tissue"),
        ([r"maxillary\s+sinuses?", r"sinuses?", r"sinus\s+floor", r"sinus\s+communication"], "Maxillary Sinuses", "Regional Anatomy"),
        ([r"marginal\s+alveolar\s+bone", r"alveolar\s+bone", r"alveolar\s+crest", r"alveolar\s+ridge"], "Marginal Alveolar Bone", "Osseous Structure"),
        ([r"cervical\s+spine", r"vertebrae"], "Cervical Spine", "Regional Anatomy"),
        ([r"tmj", r"temporomandibular", r"condyle"], "Temporomandibular Joints", "Regional Anatomy"),
        ([r"inferior\s+alveolar\s+nerve", r"mandibular\s+canal", r"ianc"], "Inferior Alveolar Canal & Alveolar Crest", "Regional Anatomy"),
    ]
    for patterns, site_title, s_type in extra_sites_map:
        if any(re.search(r"\b" + p + r"\b", clean_no_age, re.IGNORECASE) for p in patterns):
            norm = normalize_dental_site({"anatomical_name": site_title, "site_type": s_type})
            if norm["anatomical_name"] not in seen_sites:
                seen_sites.add(norm["anatomical_name"])
                anatomical_sites.append(norm)

    # Primary default anatomical context
    primary_site_label = anatomical_sites[0].get("anatomical_name", "Dentition") if anatomical_sites else "Dentition"

    # Detect transcription artifacts in raw text
    artifacts = detect_transcription_artifacts(dictation_text)

    # Parse clinical sentences into structured findings
    for s in sentences:
        s_lower = s.lower().strip()

        # Skip administrative boilerplate, demographics, or prompt meta-instructions
        from .claim_verification import is_meta_or_instruction_sentence, is_clinical_observation_sentence
        if is_meta_or_instruction_sentence(s):
            continue
        if not is_clinical_observation_sentence(s):
            continue

        if (
            re.search(r"^patient\s+is\s+a\b", s_lower) or
            re.search(r"\breport\s+was\s+verified\b", s_lower) or
            re.search(r"\bmeasurements\s+should\s+not\s+be\s+made\b", s_lower) or
            re.search(r"^(?:dentalscan|office:|report\s+type:|report\s+date:|date\s+of\s+birth:|dob:|study\s+date:|scan\s+type:)\b", s_lower) or
            re.search(r"^(?:patient:|name:)\s+[a-z\s]+$", s_lower) or
            re.search(r"^(?:age:|gender:)\s+[0-9a-z\s]+$", s_lower) or
            re.search(r"^(?:abbreviations:|dental\s+scan\s+ltd|email:|website:|telephone:|gdc\s+registration|conducted\s+by)\b", s_lower) or
            re.search(r"^(?:patient\s*id|mrn|pt\s*id|id)\s*[:\-]", s_lower) or
            re.search(r"^(?:patient\s*name|name)\s*[:\-]", s_lower) or
            re.search(r"^(?:study|study\s*type|scan\s*type|modality|examination)\s*[:\-]", s_lower) or
            re.search(r"^(?:panoramic\s+radiograph|cone\s+beam\s+ct|cbct|opg|bitewings?|periapical(?:\s+radiograph)?)$", s_lower)
        ):
            continue

        # Clean section header prefix
        s = re.sub(r"^(?:IMPRESSIONS?\s*(?:&|and)\s*SPECIFIC\s+FINDINGS|SPECIFIC\s+FINDINGS|DENTAL\s+FINDINGS|RADIOLOGICAL\s+FINDINGS|CLINICAL\s+FINDINGS|FINDINGS):\s*", "", s, flags=re.IGNORECASE).strip()
        s_lower = s.lower()
        if len(s_lower) < 5:
            continue

        # Check for pure technique/indication boilerplate
        if re.search(r"\b(?:cbct|examination|study|scan)\s+(?:of\s+the\s+[a-z\s]+\s+)?was\s+performed\s+for\b", s_lower):
            continue

        # Check for pure recommendations (move to recommendations list)
        if (
            re.search(r"\b(further\s+evaluation\s+is\s+recommended|routine\s+follow-up\s+is\s+advised)\b", s_lower) or
            (re.search(r"\b(recommend|recommended|advise|advised)\b", s_lower) and not re.search(r"\b(caries|lesion|root|canal|fracture|ridge|cortex|bone\s+loss)\b", s_lower))
        ):
            clean_rec = re.sub(r"^(?:recommendations?:\s*)?", "", s, flags=re.I).strip()
            clean_rec = clean_rec[0].upper() + clean_rec[1:]
            if clean_rec not in recommendations:
                recommendations.append(clean_rec)
            continue

        # Check for general negative findings statement
        if "no other abnormal finding" in s_lower:
            findings.append({
                "anatomical_site": primary_site_label,
                "tooth_number": None,
                "category": "General Observation",
                "observation": f"{s} (dictated clinician statement; not independently verified)",
                "polarity": "absent",
                "certainty": "negative / normal finding",
                "measurements": [],
                "source_quote": s,
                "related_sites": [],
                "requires_clarification": False,
                "clarification_reason": None
            })
            continue

        # Check for unclear speech transcription artifact
        art_match = next((a for a in artifacts if a["matched_phrase"].lower() in s_lower), None)
        if art_match:
            findings.append({
                "anatomical_site": primary_site_label,
                "tooth_number": None,
                "category": art_match.get("category", "Root Pathology & Resorption"),
                "observation": f"Unclear phrase: '{art_match['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                "polarity": "uncertain",
                "certainty": "unclear / clinician review required",
                "measurements": [],
                "source_quote": s,
                "related_sites": [],
                "requires_clarification": True,
                "clarification_reason": f"Acoustic artifact: '{art_match['matched_phrase']}'"
            })
            continue

        # Evaluate clinical status, certainty, and polarity
        finding_eval = classify_clinical_finding_status(s)
        eval_status = finding_eval.get("finding_status", "positive")
        certainty = finding_eval.get("legacy_label", "definite observation")
        if eval_status == "negative":
            polarity = "absent"
        elif "unclear" in certainty:
            polarity = "uncertain"
        else:
            polarity = "present"

        # Determine target anatomical site, tooth number, and related sites
        target_site, tooth_num, rel_sites = analyze_sentence_anatomy(s, anatomical_sites, active_quad)
        category = classify_observation_category(s, target_site)
        meas = extract_measurements(s)

        findings.append({
            "anatomical_site": target_site,
            "tooth_number": tooth_num,
            "category": category,
            "observation": s,
            "polarity": polarity,
            "certainty": certainty,
            "measurements": meas,
            "source_quote": s,
            "related_sites": rel_sites,
            "requires_clarification": False,
            "clarification_reason": None
        })

    # Synthesize clean, comprehensive Impressions dynamically
    if not impressions:
        impressions = synthesize_impression_from_findings(findings, artifacts)

    # Collect any explicit recommendations in full text
    has_explicit_rec = bool(re.search(r"\b(recommend|recommended|recommendation|advise|advised|routine follow-up)\b", dictation_text, re.IGNORECASE))
    if has_explicit_rec:
        for s in sentences:
            if re.search(r"\b(recommend|recommended|recommendation|advise|advised|follow-up)\b", s, re.IGNORECASE) and "performed for" not in s.lower():
                clean_rec = re.sub(r"^(?:recommendations?:\s*)?", "", s, flags=re.I).strip()
                clean_rec = clean_rec[0].upper() + clean_rec[1:]
                if clean_rec not in recommendations:
                    recommendations.append(clean_rec)

    return {
        "patient_details": patient_details,
        "anatomical_sites": anatomical_sites,
        "findings": findings,
        "impressions": list(dict.fromkeys(impressions)),
        "recommendations": list(dict.fromkeys(recommendations))
    }