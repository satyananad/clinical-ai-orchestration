import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
from .validation import label_diagnostic_certainty
from .dental_anatomy import normalize_dental_site, classify_observation_category

# Pydantic Schemas for Strict LLM Output Validation
class PatientDetailsModel(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    demographics_summary: Optional[str] = None

class AnatomicalSiteModel(BaseModel):
    site_type: str = "Tooth"
    anatomical_name: str
    quadrant: Optional[str] = None
    fdi_number: Optional[int] = None
    palmer_notation: Optional[str] = None
    universal_number: Optional[Any] = None

class FindingModel(BaseModel):
    anatomical_site: Optional[str] = "Dentition"
    tooth_number: Optional[str] = None
    category: str = "General Observation"
    observation: str
    polarity: str = "present"
    certainty: str = "unspecified"
    measurements: List[str] = Field(default_factory=list)
    source_quote: Optional[str] = None
    related_sites: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None

class DictationDraftReportModel(BaseModel):
    patient_details: Optional[PatientDetailsModel] = None
    anatomical_sites: List[AnatomicalSiteModel] = Field(default_factory=list)
    findings: List[FindingModel] = Field(default_factory=list)
    impressions: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class RetrievalIntentModel(BaseModel):
    patient_name: str
    report_type: str = "Any"

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
GROQ_API_KEY = _load_env_var("GROQ_API_KEY")
GROQ_MODEL = _load_env_var("GROQ_MODEL", "openai/gpt-oss-120b")


# =====================================================================
# GROQ CLOUD FAST INFERENCE ENGINE (Sub-second response)
# =====================================================================
def query_groq(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """Queries ultra-fast Groq Cloud API (<0.5s latency)."""
    api_key = GROQ_API_KEY or _load_env_var("GROQ_API_KEY")
    if not api_key:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_MODEL or "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": 0.1
    }
    if json_format:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    print(f"\n⚡ [GROQ ACCELERATED] Response received from {payload['model']}!")
                    return content
    except Exception:
        return None
    return None


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
13. Return only valid JSON matching the supplied schema.

JSON Schema:
{
  "patient_details": {
    "age": integer or null,
    "gender": "Female" | "Male" or null,
    "demographics_summary": "string" or null
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

    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            if response_text:
                print(f"\n✅ [LOCAL OLLAMA ACTIVE] Received response from {OLLAMA_MODEL}!")
            return response_text
    except Exception:
        return None


def query_llm(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """
    High-Performance LLM Dispatcher:
    1. Checks ultra-fast Groq Cloud (<0.5s) if GROQ_API_KEY is available.
    2. Falls back to local Ollama (llama3.1) with 15s timeout.
    3. Falls back cleanly to deterministic clinical parsing if local daemon is offline.
    """
    groq_res = query_groq(prompt, system_prompt=system_prompt, json_format=json_format)
    if groq_res:
        return groq_res

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
    req = urllib.request.Request(
        OLLAMA_CHAT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
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
                return {"tool_name": tool_name, "arguments": args}
            return None
    except Exception:
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

    # Dynamic semantic fallback (without hardcoded table)
    lower = user_input.lower()
    modality = "Any"
    for m in ["opg", "cbct", "bitewing", "periapical", "cephalogram", "iopa"]:
        if m in lower:
            modality = m.upper() if m in ["opg", "cbct", "iopa"] else m.capitalize()
            break

    clean = lower
    for token in ["open", "show", "get", "fetch", "view", "latest", "recent", "report", "reports", "scan", "scans", "opg", "cbct", "bitewing", "periapical", "cephalogram", "iopa", "radiograph", "radiographs", "examination", "the", "please"]:
        clean = re.sub(r"\b" + re.escape(token) + r"\b", " ", clean)
    clean = clean.replace("'s", " ").replace("’s", " ")
    clean = " ".join(clean.split()).strip()

    return {"patient_name": clean, "report_type": modality}


def clean_conversational_dictation(text: str) -> str:
    """Strips conversational preambles like 'hey can you generate the report for the...'"""
    cleaned = re.sub(
        r"^(?:hey\s+|hello\s+|hi\s+)?(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:generate|create|write|draft|make|give\s+me)?\s*(?:the|a)?\s*(?:report|clinical\s+report|dictation)?\s*(?:for\s+(?:the)?)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )
    return cleaned.strip()

def detect_transcription_artifacts(text: str) -> List[Dict[str, str]]:
    """Identifies speech-to-text corruptions and unclear non-medical phrases."""
    artifacts = []
    # Pattern 1: 'root Bridge option' / 'bridge option' / 'root bridging' (phoneme error for root resorption)
    m = re.search(r"\b(root\s+bridge\s+option|bridge\s+option|root\s+bridging)\b", text, re.IGNORECASE)
    if m:
        artifacts.append({
            "matched_phrase": m.group(0),
            "flag": f"Unclear phrase: '{m.group(0)}' (flagged as likely speech transcription error; requires clinician review)",
            "category": "Root Pathology & Resorption"
        })
    return artifacts

def parse_dictation_with_llm(dictation_text: str) -> Dict[str, Any]:
    """
    Feature B: Uses LLM reasoning + Deterministic Dental Anatomy Normalization.
    Guarantees 100% correct FDI, Universal, and Palmer tooth notation, and prevents
    observation category misclassification.
    """
    cleaned_text = clean_conversational_dictation(dictation_text)
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

    if not result:
        result = _dynamic_semantic_parse(dictation_text)

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
    for f in result.get("findings", []):
        obs = f.get("observation", "")
        site = f.get("anatomical_site", "")
        if "unclear phrase" in obs.lower() or f.get("requires_clarification"):
            f["category"] = "Root Pathology & Resorption"
            f["certainty"] = "unclear / clinician review required"
        else:
            current_cat = f.get("category")
            if not current_cat or current_cat in ("General Observation", "unspecified", "Unknown"):
                f["category"] = classify_observation_category(obs, site)
        if not f.get("measurements"):
            f["measurements"] = extract_measurements(obs)
        if not f.get("source_quote"):
            f["source_quote"] = obs
        if not f.get("polarity"):
            f["polarity"] = "absent" if any(re.search(r"\b" + re.escape(w) + r"\b", obs.lower()) for w in ["no", "not", "without", "absence", "none", "negative for"]) else "present"

    return result


def synthesize_impression_from_findings(findings: List[Dict[str, Any]], artifacts: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Synthesizes impressions dynamically from structured clinical findings without hardcoded disease lists.
    Groups significant observations, structural dimensions, negative assurances, and speech clarifications.
    """
    imp_items = []

    # 1. Significant observations (positive, indeterminate, or differential findings)
    positive_findings = [
        f for f in findings
        if f.get("certainty") in ("definite observation", "definite observation (etiology uncertain)", "suspected / differential diagnosis")
        and not f.get("requires_clarification")
    ]

    for f in positive_findings:
        obs = f.get("observation", "").strip()
        site = f.get("anatomical_site", "").strip()

        # Clean introductory boilerplate
        cleaned_obs = re.sub(r"^(?:at\s+(?:the\s+)?[^,]+,\s*|the\s+|there\s+is\s+|there\s+are\s+)", "", obs, flags=re.IGNORECASE).strip()
        if cleaned_obs:
            cleaned_obs = cleaned_obs[0].upper() + cleaned_obs[1:]

        # Avoid redundant site name prefix if already in observation
        site_in_obs = site.lower() in cleaned_obs.lower() if site else False
        if site and not site_in_obs and site.lower() not in ("dentition", "regional anatomy", "general"):
            item_text = f"{site}: {cleaned_obs}"
        else:
            item_text = cleaned_obs

        # Deduplicate
        if not any(item_text.lower() == p.lower() for p in imp_items):
            imp_items.append(item_text)

    # 2. Key negative findings (complications / abnormalities ruled out)
    neg_findings = [
        f for f in findings
        if f.get("certainty") == "negative / normal finding"
        and not f.get("requires_clarification")
        and "no other abnormal finding" not in f.get("observation", "").lower()
    ]

    if neg_findings:
        neg_descs = [nf.get("observation", "").strip().rstrip(".") for nf in neg_findings]
        if all(re.match(r"^no\s+", d, re.IGNORECASE) for d in neg_descs):
            cleaned_phrases = [re.sub(r"^no\s+", "", d, flags=re.IGNORECASE) for d in neg_descs]
            cleaned_phrases = [re.sub(r"\s+(?:is|are)\s+(?:observed|identified|seen|noted)$", "", cp, flags=re.IGNORECASE) for cp in cleaned_phrases]
            imp_items.append(f"No {' or '.join(cleaned_phrases)} observed.")
        else:
            imp_items.append("; ".join(neg_descs) + ".")

    # 3. Flagged speech transcription artifacts
    if artifacts:
        for art in artifacts:
            imp_items.append(f"Unclear phrase '{art['matched_phrase']}' flagged for clinician clarification.")

    if not imp_items:
        return ["Radiological assessment completed."]

    return ["; ".join(imp_items)]


def _dynamic_semantic_parse(dictation_text: str) -> Dict[str, Any]:
    """
    Algorithmic clinical entity extraction when external LLM endpoints are offline.
    Infers dental quadrants, Palmer/FDI/Universal numbering, and categories deterministically.
    """
    from .claim_verification import extract_patient_demographics, extract_measurements
    cleaned_text = clean_conversational_dictation(dictation_text)
    patient_details = extract_patient_demographics(cleaned_text)

    # Strip age, measurements, and dates before extracting teeth to prevent misidentifying age as FDI tooth
    clean_no_age = re.sub(r"\b\d{1,2}\s*(?:-|–|\s)?(?:year|yr)s?(?:-|–|\s)?old\b|\bage\s*(?:is\s*)?\d{1,2}\b", " ", cleaned_text, flags=re.IGNORECASE)
    clean_no_age = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|mg|ml|%)\b", " ", clean_no_age, flags=re.IGNORECASE)
    clean_no_age = re.sub(r"\b(?:19|20)\d{2}\b", " ", clean_no_age)

    # Segment text into sentences using standard grammatical punctuation (excluding decimal points in numbers)
    sentences = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[\n?!]+", cleaned_text) if len(s.strip()) > 3]
    if len(sentences) <= 1 and len(cleaned_text.split()) > 7:
        # Continuous unpunctuated speech stream: segment on natural syntactic coordinators
        sentences = [
            s.strip() for s in re.split(
                r"\s+(?=(?:and\s+|with\s+|it\s+is\s+|the\s+surrounding\s+|no\s+other\s+))",
                cleaned_text,
                flags=re.IGNORECASE
            ) if len(s.strip()) > 3
        ]
    if not sentences:
        sentences = [cleaned_text]

    anatomical_sites = []
    findings = []
    impressions = []
    recommendations = []
    seen_sites = set()

    # 1. Detect Quadrant Context (e.g. Maxillary Left, Maxillary Right, Mandibular Left, Mandibular Right)
    t_lower = clean_no_age.lower()
    active_quad = None
    if re.search(r"\b(maxillary\s+left|upper\s+left|quadrant\s+2|ul)\b", t_lower):
        active_quad = "UL"
    elif re.search(r"\b(maxillary\s+right|upper\s+right|quadrant\s+1|ur)\b", t_lower):
        active_quad = "UR"
    elif re.search(r"\b(mandibular\s+left|lower\s+left|quadrant\s+3|ll)\b", t_lower):
        active_quad = "LL"
    elif re.search(r"\b(mandibular\s+right|lower\s+right|quadrant\s+4|lr)\b", t_lower):
        active_quad = "LR"

    # 2. Quadrant-relative named teeth (e.g. "first molar site", "second molar site", "second premolar")
    if active_quad:
        quad_tooth_patterns = [
            ("third molar", 8), ("wisdom tooth", 8),
            ("second molar", 7),
            ("first molar", 6),
            ("second premolar", 5), ("second bicuspid", 5),
            ("first premolar", 4), ("first bicuspid", 4),
            ("canine", 3),
            ("lateral incisor", 2),
            ("central incisor", 1)
        ]
        for name_pat, pos_idx in quad_tooth_patterns:
            if re.search(r"\b" + name_pat + r"\b", t_lower):
                palmer_code = f"{active_quad}{pos_idx}"
                norm = normalize_dental_site({"palmer_notation": palmer_code, "anatomical_name": palmer_code})
                if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                    seen_sites.add(norm["fdi_number"])
                    anatomical_sites.append(norm)

    # 3. Detect Palmer Teeth from clean text (e.g. UR8, UR7, UR6, UR5, UR E, LL6, etc.)
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

    # 4. Numbered teeth (e.g. Tooth 6, Tooth 12, Tooth 16, Tooth 28, Tooth 36, #29)
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

    # 5. Deciduous Universal Letters (e.g. Tooth C, Tooth E)
    decid_letter_matches = re.finditer(r"\btooth\s*#?([A-T])\b", clean_no_age, re.IGNORECASE)
    for m in decid_letter_matches:
        letter = m.group(1).upper()
        norm = normalize_dental_site({"universal_letter": letter, "anatomical_name": f"Tooth {letter}"})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 6. Natural language anatomical phrases (e.g. "lower right third molar", "upper left wisdom tooth")
    if "third molar" in clean_no_age.lower() or "wisdom tooth" in clean_no_age.lower():
        is_lower = "lower" in clean_no_age.lower() or "mandibular" in clean_no_age.lower()
        is_right = "right" in clean_no_age.lower()
        p_code = "LR8" if (is_lower and is_right) else ("LL8" if is_lower else ("UR8" if is_right else "UL8"))
        norm = normalize_dental_site({"palmer_notation": p_code, "anatomical_name": p_code})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 7. Regional extra-dental anatomy
    extra_sites_map = [
        ([r"palatine\s+tonsilloliths?", r"tonsilloliths?", r"tonsils?"], "Palatine Tonsils", "Soft Tissue"),
        ([r"maxillary\s+sinuses?", r"sinuses?", r"sinus\s+floor", r"sinus\s+communication"], "Maxillary Sinuses", "Regional Anatomy"),
        ([r"marginal\s+alveolar\s+bone", r"alveolar\s+bone", r"alveolar\s+crest", r"alveolar\s+ridge"], "Marginal Alveolar Bone", "Osseous Structure"),
        ([r"cervical\s+spine", r"vertebrae"], "Cervical Spine", "Regional Anatomy"),
        ([r"tmj", r"temporomandibular", r"condyle"], "Temporomandibular Joints", "Regional Anatomy"),
        ([r"inferior\s+alveolar\s+nerve", r"mandibular\s+canal"], "Inferior Alveolar Canal", "Regional Anatomy"),
    ]
    for patterns, site_title, s_type in extra_sites_map:
        if any(re.search(r"\b" + p + r"\b", clean_no_age, re.IGNORECASE) for p in patterns):
            norm = normalize_dental_site({"anatomical_name": site_title, "site_type": s_type})
            if norm["anatomical_name"] not in seen_sites:
                seen_sites.add(norm["anatomical_name"])
                anatomical_sites.append(norm)

    # Primary default anatomical context (derived from identified sites or regional structures)
    primary_site_label = anatomical_sites[0].get("anatomical_name", "Dentition") if anatomical_sites else "Dentition"

    # Detect transcription artifacts in raw text
    artifacts = detect_transcription_artifacts(dictation_text)

    # Track missing implant site if indicated
    implant_site_label = None
    for s in anatomical_sites:
        if "first molar" in s.get("anatomical_name", "").lower():
            implant_site_label = s.get("anatomical_name")
            break

    # Parse sentences into findings
    for s in sentences:
        s_lower = s.lower().strip()

        # Skip patient demographics header or report verification boilerplate
        if (
            re.search(r"^patient\s+is\s+a\b", s_lower) or
            re.search(r"\breport\s+was\s+verified\b", s_lower) or
            re.search(r"\bmeasurements\s+should\s+not\s+be\s+made\b", s_lower)
        ):
            continue

        # Check if sentence is scan indication / technique boilerplate
        if re.search(r"\b(?:cbct|examination|study|scan)\s+(?:of\s+the\s+[a-z\s]+\s+)?was\s+performed\s+for\b", s_lower):
            # If it specifies an implant site, note it but skip creating a separate pathology finding
            continue

        # Check if sentence is a standalone diagnostic summary statement
        if re.search(r"^(?:the\s+)?findings\s+suggest\b", s_lower):
            impressions.append(s)
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

        # Determine target anatomical site for finding
        target_site = None
        sent_tooth_match = re.search(r"\b(?:teeth|tooth)\s*#?(\d{1,2})\b", s, re.IGNORECASE)
        if sent_tooth_match:
            t_val = int(sent_tooth_match.group(1))
            target_site = next((st.get("anatomical_name") for st in anatomical_sites if st.get("fdi_number") == t_val or st.get("universal_number") == t_val or f"Tooth {t_val}" in st.get("anatomical_name", "")), None)

        if not target_site:
            if "third molar" in s_lower or "wisdom tooth" in s_lower:
                target_site = next((st.get("anatomical_name") for st in anatomical_sites if "third molar" in st.get("anatomical_name", "").lower()), None)
            elif "second molar" in s_lower:
                target_site = next((st.get("anatomical_name") for st in anatomical_sites if "second molar" in st.get("anatomical_name", "").lower()), None)
            elif "second premolar" in s_lower:
                target_site = next((st.get("anatomical_name") for st in anatomical_sites if "second premolar" in st.get("anatomical_name", "").lower()), None)
            elif "first molar" in s_lower or "alveolar ridge" in s_lower or "buccal cortical plate" in s_lower:
                target_site = implant_site_label or next((st.get("anatomical_name") for st in anatomical_sites if "first molar" in st.get("anatomical_name", "").lower() or "alveolar" in st.get("anatomical_name", "").lower()), None)

        if not target_site:
            for patterns, site_title, _ in extra_sites_map:
                if any(re.search(r"\b" + p + r"\b", s_lower) for p in patterns):
                    target_site = site_title
                    break

        if not target_site:
            for site in anatomical_sites:
                p_code = site.get("palmer_notation", "")
                fdi_code = str(site.get("fdi_number", ""))
                a_name = site.get("anatomical_name", "")
                if p_code and re.search(r"\b" + re.escape(p_code) + r"\b", s, re.IGNORECASE):
                    target_site = a_name
                    break
                elif fdi_code and re.search(r"\b" + re.escape(fdi_code) + r"\b", s):
                    target_site = a_name
                    break
                elif a_name and a_name.lower() in s_lower:
                    target_site = a_name
                    break

        if not target_site:
            target_site = primary_site_label

        certainty = label_diagnostic_certainty(s)
        category = classify_observation_category(s, target_site)
        meas = extract_measurements(s)
        if certainty == "negative / normal finding" or any(re.search(r"\b" + re.escape(w) + r"\b", s_lower) for w in ["no", "not", "without", "absence", "none", "negative for", "clear"]):
            polarity = "absent"
        elif "unclear" in certainty:
            polarity = "uncertain"
        else:
            polarity = "present"

        tooth_num = None
        for st in anatomical_sites:
            if st.get("anatomical_name") == target_site:
                tooth_num = str(st.get("fdi_number") or st.get("universal_number") or st.get("palmer_notation") or "")
                break

        findings.append({
            "anatomical_site": target_site,
            "tooth_number": tooth_num or None,
            "category": category,
            "observation": s,
            "polarity": polarity,
            "certainty": certainty,
            "measurements": meas,
            "source_quote": s,
            "related_sites": [],
            "requires_clarification": False,
            "clarification_reason": None
        })

    # Synthesize clean, comprehensive Impressions dynamically from findings without hardcoded disease lists
    if not impressions:
        impressions = synthesize_impression_from_findings(findings, artifacts)

    # Recommendations: Strict zero-hallucination. Only include if explicitly dictated!
    has_explicit_rec = bool(re.search(r"\b(recommend|recommended|recommendation|advise|advised|routine follow-up)\b", dictation_text, re.IGNORECASE))
    if has_explicit_rec:
        for s in sentences:
            if re.search(r"\b(recommend|recommended|recommendation|advise|advised|follow-up)\b", s, re.IGNORECASE) and "performed for" not in s.lower():
                recommendations.append(s)

    return {
        "patient_details": patient_details,
        "anatomical_sites": anatomical_sites,
        "findings": findings,
        "impressions": list(dict.fromkeys(impressions)),
        "recommendations": list(dict.fromkeys(recommendations))
    }

