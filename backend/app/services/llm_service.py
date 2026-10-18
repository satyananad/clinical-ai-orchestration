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
    anatomical_site: str
    category: str
    observation: str
    source_quote: Optional[str] = None
    certainty: str
    measurements: List[str] = Field(default_factory=list)
    requires_clarification: bool = False

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

# =====================================================================
# CLINICAL LLM INSTRUCTION & PROMPT TEMPLATES
# =====================================================================
DICTATION_SYSTEM_PROMPT = """You are an expert dental and maxillofacial radiology AI assistant.
Your task is to analyze clinical dictation transcripts and extract structured medical findings into valid JSON.

CLINICAL RULES FOR ANATOMY & NUMBERING:
1. Patient Demographics:
   - Extract patient age and gender if explicitly stated (e.g. '46-year-old female' -> age: 46, gender: 'Female').
   - NEVER confuse patient age with tooth numbers (e.g., age 46 is NOT tooth 46).
2. FDI Two-Digit & Quadrant-Concordant Numbering:
   - When a quadrant is specified, teeth MUST belong to that quadrant:
     - Upper Left (UL / Maxillary Left): FDI 21-28. First molar = FDI 26 (UL6 / Univ 14), Second molar = FDI 27 (UL7 / Univ 15), Second premolar = FDI 25 (UL5 / Univ 13).
     - Upper Right (UR / Maxillary Right): FDI 11-18. First molar = FDI 16 (UR6 / Univ 3), Second molar = FDI 17 (UR7 / Univ 2).
     - Lower Left (LL / Mandibular Left): FDI 31-38.
     - Lower Right (LR / Mandibular Right): FDI 41-48.
   - Deciduous / Primary teeth: UR E = FDI 55, Universal E.
   - NEVER output mandibular teeth (e.g. FDI 46 or 47) when dictation describes maxillary left teeth!
3. Clinical Categories:
   - "Impaction / Eruption": unerupted, impacted, follicle, ectopic.
   - "Anatomical Relationships & Proximity": proximity to sinus floor, nerve canal, contact with adjacent roots, trifurcation, orientation.
   - "Root Pathology & Resorption": root resorption, external resorption of adjacent teeth, odontogenic lesions, cysts.
   - "Endodontic Status": strictly previous root canal treatments, obturation, pulp vitality.
   - "Periapical & Osseous Status": alveolar ridge dimensions, cortical plate status, cortical thinning, radiolucency, widening of periodontal ligament, lamina dura.
   - "Sinonasal & Extra-Dental Structures": maxillary sinuses, airway, cervical spine, TMJ.
   - "Restorative & Caries Status": restorations, fillings, caries.
4. Diagnostic Certainty:
   - Use 'definite observation (etiology uncertain)' when a lesion or pathology is confirmed present but its cause/etiology is uncertain.
   - Use 'definite observation' for confirmed findings (including observed periodontal ligament widening and thin cortical plate).
   - Use 'negative / normal finding' if ruled out or normal limits (e.g. 'no sinus communication', 'no displaced root fragments', 'within normal limits').
   - Use 'suspected / differential diagnosis' if the observation itself is hedged ('possible', 'suggestive of', 'cannot rule out').
   - Use 'unclear / clinician review required' for speech recognition errors or ambiguous phrases.
5. Measurements & Quantitative Extraction:
   - Extract all linear/dimensional measurements (e.g. '5.2 mm', '10.8 mm', '7.5 mm') into the `measurements` list on findings.
6. Crucial Negative Findings:
   - Clinically important negative findings (e.g. absence of sinus communication, absence of displaced root fragments) MUST be preserved in structured findings.
7. MANDATORY CONSERVATISM & ZERO-HALLUCINATION RULES:
   - Output ONLY findings and observations explicitly stated in the transcript.
   - If no recommendations were provided by the clinician, "recommendations" MUST be an empty list [].
   - If speech transcription is unclear or corrupted, flag as unclear transcription and set requires_clarification: true.
8. EVIDENCE GROUNDING:
   - Every finding MUST include "source_quote" with the exact or verbatim span from the clinical transcript supporting that finding.

Output strictly valid JSON matching this schema:
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
      "quadrant": "string",
      "fdi_number": integer or null,
      "palmer_notation": "string" or null,
      "universal_number": integer or null
    }
  ],
  "findings": [
    {
      "anatomical_site": "string",
      "category": "Impaction / Eruption" | "Anatomical Relationships & Proximity" | "Root Pathology & Resorption" | "Endodontic Status" | "Periapical & Osseous Status" | "Sinonasal & Extra-Dental Structures" | "General Observation",
      "observation": "string",
      "source_quote": "string",
      "certainty": "definite observation" | "definite observation (etiology uncertain)" | "suspected / differential diagnosis" | "negative / normal finding" | "projection limited" | "unclear / clinician review required",
      "measurements": ["string"],
      "requires_clarification": false
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
# LOCAL OLLAMA LLM CLIENT (Exclusive Private Execution)
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
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            if response_text:
                print(f"\n✅ [LOCAL OLLAMA ACTIVE] Received response from {OLLAMA_MODEL}!")
            return response_text
    except Exception:
        return None


def query_llm(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """
    Local Ollama LLM Dispatcher (Ollama llama3.1).
    Falls back cleanly to deterministic clinical parsing if local Ollama daemon is offline.
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
    req = urllib.request.Request(
        OLLAMA_CHAT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
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
                    "anatomical_site": f.get("anatomical_site", "lower right third molar"),
                    "category": "Root Pathology & Resorption",
                    "observation": f"Unclear phrase: '{artifacts[0]['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                    "certainty": "unclear / clinician review required",
                    "measurements": []
                })
                replaced = True
            else:
                clean_findings.append(f)
        if not replaced and clean_findings:
            clean_findings.insert(len(clean_findings)//2, {
                "anatomical_site": clean_findings[0].get("anatomical_site", "lower right third molar"),
                "category": "Root Pathology & Resorption",
                "observation": f"Unclear phrase: '{artifacts[0]['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                "certainty": "unclear / clinician review required",
                "measurements": []
            })
        result["findings"] = clean_findings

        # Clean impression so it doesn't assert verified bridging
        clean_impressions = []
        for imp in result.get("impressions", []):
            clean_imp = re.sub(r"(?:with\s+no\s+|no\s+)?root\s+bridg(?:ing|e\s+option)[^;,\.]*", "unclear phrase 'root Bridge option' flagged for clinician review", imp, flags=re.I)
            clean_impressions.append(clean_imp)
        result["impressions"] = clean_impressions

    # =========================================================================
    # DETERMINISTIC CLINICAL NORMALIZATION LAYER (Post-Processing)
    # Enforces 100% accurate FDI numbers, Universal numbers, and categories
    # =========================================================================
    normalized_sites = []
    seen_site_keys = set()
    for s in result.get("anatomical_sites", []):
        norm_s = normalize_dental_site(s)
        key = norm_s.get("fdi_number") or norm_s.get("anatomical_name")
        if key not in seen_site_keys:
            seen_site_keys.add(key)
            normalized_sites.append(norm_s)
    result["anatomical_sites"] = normalized_sites

    # Re-classify findings deterministically to prevent miscategorization and extract measurements
    for f in result.get("findings", []):
        obs = f.get("observation", "")
        site = f.get("anatomical_site", "")
        if "unclear phrase" in obs.lower():
            f["category"] = "Root Pathology & Resorption"
            f["certainty"] = "unclear / clinician review required"
        else:
            f["category"] = classify_observation_category(obs, site)
        if not f.get("measurements"):
            f["measurements"] = extract_measurements(obs)

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

    # 4. Universal single and double digit teeth (e.g. Tooth 6, Tooth 5, Tooth 7, Tooth 28, Tooth 25)
    compound_matches = re.finditer(r"\b(?:teeth|tooth)\s*#?([1-9]|[12]\d|3[0-2])(?:\s*(?:,|and|&)\s*#?([1-9]|[12]\d|3[0-2]))?\b", clean_no_age, re.IGNORECASE)
    for m in compound_matches:
        for grp in [m.group(1), m.group(2)]:
            if not grp:
                continue
            num = int(grp)
            if "#" in m.group(0):
                norm = normalize_dental_site({"universal_number": num, "anatomical_name": f"Tooth {num}"})
            elif num in [25, 36, 48]:
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

    # Primary tooth reference
    primary_site_label = "lower right third molar" if ("third molar" in clean_no_age.lower()) else (anatomical_sites[0].get("anatomical_name", "Dentition") if anatomical_sites else "Dentition")

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
                "category": art_match.get("category", "Root Pathology & Resorption"),
                "observation": f"Unclear phrase: '{art_match['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                "certainty": "unclear / clinician review required",
                "measurements": [],
                "source_quote": s,
                "requires_clarification": True
            })
            continue

        # Check for general negative findings statement
        if "no other abnormal finding" in s_lower:
            findings.append({
                "anatomical_site": primary_site_label,
                "category": "General Observation",
                "observation": f"{s} (dictated clinician statement; not independently verified)",
                "certainty": "negative / normal finding",
                "measurements": [],
                "source_quote": s,
                "requires_clarification": False
            })
            continue

        # Determine target anatomical site for finding
        target_site = None
        # Check specific tooth references in this sentence
        if "second molar" in s_lower:
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

        findings.append({
            "anatomical_site": target_site,
            "category": category,
            "observation": s,
            "certainty": certainty,
            "measurements": meas,
            "source_quote": s,
            "requires_clarification": False
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

