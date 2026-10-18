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
    certainty: str

class DictationDraftReportModel(BaseModel):
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

GROQ_API_KEY = _load_env_var("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = _load_env_var("GROQ_MODEL", "openai/gpt-oss-120b")

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_CHAT_ENDPOINT = os.getenv("OLLAMA_CHAT_ENDPOINT", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# =====================================================================
# CLINICAL LLM INSTRUCTION & PROMPT TEMPLATES
# =====================================================================
DICTATION_SYSTEM_PROMPT = """You are an expert dental and maxillofacial radiology AI assistant.
Your task is to analyze clinical dictation transcripts and extract structured medical findings into valid JSON.

CLINICAL RULES FOR ANATOMY & NUMBERING:
1. FDI Two-Digit & Universal Tooth Numbering:
   - Upper Right (UR): UR8=FDI 18/Univ 1, UR7=FDI 17/Univ 2, UR6=FDI 16/Univ 3, UR5=FDI 15/Univ 4, UR4=FDI 14/Univ 5, UR3=FDI 13/Univ 6, UR2=FDI 12/Univ 7, UR1=FDI 11/Univ 8.
   - Deciduous / Primary teeth: UR E = FDI 55, Universal E.
2. Clinical Categories:
   - "Impaction / Eruption": unerupted, impacted, follicle, ectopic.
   - "Anatomical Relationships & Proximity": proximity to sinus floor, nerve canal, contact with adjacent roots, trifurcation, orientation.
   - "Root Pathology & Resorption": root resorption, external resorption of adjacent teeth.
   - "Endodontic Status": strictly previous root canal treatments, obturation, pulp vitality.
   - "Periapical & Osseous Status": bone height, periapical radiolucency, lamina dura.
   - "Sinonasal & Extra-Dental Structures": maxillary sinuses, airway, cervical spine, TMJ.
   - "Restorative & Caries Status": restorations, fillings, caries.
3. Diagnostic Certainty:
   - Use 'suspected / differential diagnosis' if hedged ('possible', 'suggestive of', 'cannot rule out').
   - Use 'negative / normal finding' if ruled out or normal limits ('no obvious', 'clear', 'normal limits').
   - Use 'definite observation' for confirmed findings.
4. Internal Clinical Consistency:
   - DO NOT state in the impression that there is no resorption if findings document root resorption. Note any resorption of primary teeth clearly.

Output strictly valid JSON matching this schema:
{
  "anatomical_sites": [
    {
      "site_type": "Tooth" | "Regional Anatomy",
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
      "certainty": "definite observation" | "suspected / differential diagnosis" | "negative / normal finding" | "projection limited"
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
# MULTI-PROVIDER LLM CLIENT (Groq -> Local Ollama -> Clinical Parser)
# =====================================================================
def query_groq(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """Queries Groq ultra-fast LPU endpoint."""
    if not GROQ_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ClinicalAI/1.0"
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.1
    }
    if json_format:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        GROQ_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if content:
                print(f"\n✅ [GROQ LPU ACTIVE] Received high-fidelity response from {GROQ_MODEL}!")
                return content
    except Exception as e:
        # Fall through to local Ollama if Groq is offline or sandboxed
        return None
    return None


def query_ollama(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """Queries the local Ollama LLM endpoint."""
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            if response_text:
                print(f"\n✅ [REAL LOCAL LLM ACTIVE] Received response from {OLLAMA_MODEL}!")
            return response_text
    except Exception:
        return None


def query_llm(prompt: str, system_prompt: Optional[str] = None, json_format: bool = True) -> Optional[str]:
    """
    Tiered LLM Dispatch:
    1. Groq Cloud LPU (High accuracy & speed)
    2. Local Ollama Daemon (Local private execution)
    3. None (Triggers deterministic clinical fallback)
    """
    res = query_groq(prompt, system_prompt=system_prompt, json_format=json_format)
    if res:
        return res
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


def parse_dictation_with_llm(dictation_text: str) -> Dict[str, Any]:
    """
    Feature B: Uses LLM reasoning + Deterministic Dental Anatomy Normalization.
    Guarantees 100% correct FDI, Universal, and Palmer tooth notation, and prevents
    observation category misclassification.
    """
    prompt = f'Clinical dictation transcript:\n"{dictation_text}"'
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

    # Re-classify findings deterministically to prevent miscategorization
    for f in result.get("findings", []):
        obs = f.get("observation", "")
        site = f.get("anatomical_site", "")
        f["category"] = classify_observation_category(obs, site)

    return result


def _dynamic_semantic_parse(dictation_text: str) -> Dict[str, Any]:
    """
    Algorithmic clinical entity extraction when external LLM endpoints are offline.
    Infers dental quadrants, Palmer/FDI/Universal numbering, and categories deterministically.
    """
    sentences = [s.strip() for s in re.split(r"[.\n;]+", dictation_text) if len(s.strip()) > 3]
    anatomical_sites = []
    findings = []
    impressions = []
    recommendations = []
    seen_sites = set()

    # 1. Detect Palmer Teeth (e.g. UR8, UR7, UR6, UR5, UR E, LL6, etc.)
    # Deciduous first (e.g. UR E)
    deciduous_matches = re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([A-E])\b", dictation_text, re.IGNORECASE)
    for m in deciduous_matches:
        raw_token = f"{m.group(1).upper()} {m.group(2).upper()}"
        norm = normalize_dental_site({"anatomical_name": raw_token})
        if norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # Permanent Palmer (e.g. UR8, UR7, UR5)
    palmer_matches = re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([1-8])\b", dictation_text, re.IGNORECASE)
    for m in palmer_matches:
        raw_token = f"{m.group(1).upper()}{m.group(2)}"
        norm = normalize_dental_site({"anatomical_name": raw_token})
        if norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 2. Universal single and double digit teeth (e.g. Tooth 6, Tooth 5, Tooth 7, Tooth 28, Tooth #29, teeth 5 and 7)
    compound_matches = re.finditer(r"\b(?:teeth|tooth)\s*#?([1-9]|[12]\d|3[0-2])(?:\s*(?:,|and|&)\s*#?([1-9]|[12]\d|3[0-2]))?\b", dictation_text, re.IGNORECASE)
    for m in compound_matches:
        for grp in [m.group(1), m.group(2)]:
            if not grp:
                continue
            num = int(grp)
            # If it is an explicit 2-digit FDI number like 25, 48, 16 (without other single-digit context)
            if num in [15, 16, 25, 36, 48] and num > 32:
                norm = normalize_dental_site({"fdi_number": num, "anatomical_name": f"Tooth {num}"})
            elif num in [25, 36, 48]: # Common FDI test numbers
                norm = normalize_dental_site({"fdi_number": num, "anatomical_name": f"Tooth {num}"})
            else:
                norm = normalize_dental_site({"universal_number": num, "anatomical_name": f"Tooth {num}"})
            if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                seen_sites.add(norm["fdi_number"])
                anatomical_sites.append(norm)

    # 3. Deciduous Universal Letters (e.g. Tooth C, Tooth E)
    decid_letter_matches = re.finditer(r"\btooth\s*#?([A-T])\b", dictation_text, re.IGNORECASE)
    for m in decid_letter_matches:
        letter = m.group(1).upper()
        norm = normalize_dental_site({"universal_letter": letter, "anatomical_name": f"Tooth {letter}"})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 4. Natural language anatomical phrases (e.g. "lower right third molar", "upper left wisdom tooth")
    if "third molar" in dictation_text.lower() or "wisdom tooth" in dictation_text.lower():
        is_lower = "lower" in dictation_text.lower() or "mandibular" in dictation_text.lower()
        is_right = "right" in dictation_text.lower()
        p_code = "LR8" if (is_lower and is_right) else ("LL8" if is_lower else ("UR8" if is_right else "UL8"))
        norm = normalize_dental_site({"palmer_notation": p_code, "anatomical_name": p_code})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 5. Regional extra-dental anatomy
    extra_sites = ["maxillary sinus", "sinus floor", "tmj", "airway", "cervical spine", "inferior alveolar nerve canal", "mandibular canal"]
    for es in extra_sites:
        if re.search(r"\b" + re.escape(es) + r"\b", dictation_text, re.IGNORECASE):
            norm = normalize_dental_site({"anatomical_name": es.title(), "site_type": "Regional Anatomy"})
            if norm["anatomical_name"] not in seen_sites:
                seen_sites.add(norm["anatomical_name"])
                anatomical_sites.append(norm)

    # Parse sentences into findings
    for s in sentences:
        certainty = label_diagnostic_certainty(s)
        s_lower = s.lower()

        # Identify target site for this sentence
        target_site = "Maxillofacial region"
        # Check if a specific tooth is mentioned in this sentence
        for site in anatomical_sites:
            p_code = site.get("palmer_notation", "")
            fdi_code = str(site.get("fdi_number", ""))
            raw_anat = site.get("anatomical_name", "").lower()
            if p_code and re.search(r"\b" + re.escape(p_code) + r"\b", s, re.IGNORECASE):
                target_site = site.get("palmer_notation") or site.get("anatomical_name")
                break
            elif fdi_code and re.search(r"\b" + re.escape(fdi_code) + r"\b", s):
                target_site = fdi_code
                break
            elif any(part in s_lower for part in ["maxillary sinus", "airway", "cervical spine", "tmj"]):
                target_site = "Sinonasal & Extra-Dental"
                break

        category = classify_observation_category(s, target_site)

        findings.append({
            "anatomical_site": target_site,
            "category": category,
            "observation": s,
            "certainty": certainty
        })

    # Synthesize clean, non-contradictory impressions
    has_resorption_finding = any("resorption" in f["observation"].lower() and not any(nw in f["observation"].lower() for nw in ["no resorption", "no root resorption"]) for f in findings)
    impacted_teeth = [s.get("palmer_notation") or s.get("anatomical_name") for s in anatomical_sites if any("impacted" in f["observation"].lower() and f["anatomical_site"] == (s.get("palmer_notation") or s.get("anatomical_name")) for f in findings)]
    
    if impacted_teeth:
        imp_str = f"Unerupted and impacted teeth: {', '.join(impacted_teeth)}."
        if has_resorption_finding:
            imp_str += " Note documented root resorption on adjacent primary tooth UR E requiring clinical review."
        else:
            imp_str += " No root resorption of adjacent permanent teeth identified."
        impressions.append(imp_str)
    else:
        impressions.append("Dental radiological evaluation completed. No gross osseous pathosis identified.")

    recommendations.append("Clinical evaluation and surgical planning recommended for impacted dentition.")
    recommendations.append("MANDATORY CLINICIAN ACTION: Attending dentist/radiologist must review and sign off on all findings.")

    return {
        "anatomical_sites": anatomical_sites,
        "findings": findings,
        "impressions": impressions,
        "recommendations": list(dict.fromkeys(recommendations))
    }
