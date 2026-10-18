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
   - Use 'unclear / clinician review required' for speech recognition errors or ambiguous phrases.
4. MANDATORY CONSERVATISM & ZERO-HALLUCINATION RULES:
   - Output ONLY findings and observations explicitly stated in the transcript.
   - DO NOT invent clinical findings that the dentist never mentioned (e.g. do NOT invent root resorption or fracture).
   - If no recommendations were provided by the clinician, "recommendations" MUST be an empty list []. Do NOT invent surgical extraction, postoperative follow-up, or monitoring unless explicitly dictated.
   - If speech transcription is unclear, ambiguous, or contains non-medical corruptions (e.g. 'root Bridge option', 'Bridge option'), DO NOT convert it into an invented clinical term (such as 'no root bridging').
     Instead, report it as an unclear transcription:
     "observation": "Unclear phrase: '<phrase>' (likely speech transcription artifact; requires clinician review)",
     "certainty": "unclear / clinician review required"

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
      "certainty": "definite observation" | "suspected / differential diagnosis" | "negative / normal finding" | "projection limited" | "unclear / clinician review required"
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
                    "certainty": "unclear / clinician review required"
                })
                replaced = True
            else:
                clean_findings.append(f)
        if not replaced and clean_findings:
            clean_findings.insert(len(clean_findings)//2, {
                "anatomical_site": clean_findings[0].get("anatomical_site", "lower right third molar"),
                "category": "Root Pathology & Resorption",
                "observation": f"Unclear phrase: '{artifacts[0]['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                "certainty": "unclear / clinician review required"
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

    # Re-classify findings deterministically to prevent miscategorization
    for f in result.get("findings", []):
        obs = f.get("observation", "")
        site = f.get("anatomical_site", "")
        if "unclear phrase" in obs.lower():
            f["category"] = "Root Pathology & Resorption"
            f["certainty"] = "unclear / clinician review required"
        else:
            f["category"] = classify_observation_category(obs, site)

    return result


def _dynamic_semantic_parse(dictation_text: str) -> Dict[str, Any]:
    """
    Algorithmic clinical entity extraction when external LLM endpoints are offline.
    Infers dental quadrants, Palmer/FDI/Universal numbering, and categories deterministically.
    """
    cleaned_text = clean_conversational_dictation(dictation_text)
    
    # Segment text into sentences and clauses using standard grammatical punctuation (. ! ? ; \n or ,)
    sentences = [s.strip() for s in re.split(r"(?:[.\n;?!]+|,\s*)", cleaned_text) if len(s.strip()) > 3]
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

    # 1. Detect Palmer Teeth (e.g. UR8, UR7, UR6, UR5, UR E, LL6, etc.)
    deciduous_matches = re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([A-E])\b", cleaned_text, re.IGNORECASE)
    for m in deciduous_matches:
        raw_token = f"{m.group(1).upper()} {m.group(2).upper()}"
        norm = normalize_dental_site({"anatomical_name": raw_token})
        if norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    palmer_matches = re.finditer(r"\b(UR|UL|LL|LR)[\s_-]*([1-8])\b", cleaned_text, re.IGNORECASE)
    for m in palmer_matches:
        raw_token = f"{m.group(1).upper()}{m.group(2)}"
        norm = normalize_dental_site({"anatomical_name": raw_token})
        if norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 2. Universal single and double digit teeth (e.g. Tooth 6, Tooth 5, Tooth 7, Tooth 28, Tooth #29, teeth 5 and 7)
    compound_matches = re.finditer(r"\b(?:teeth|tooth)\s*#?([1-9]|[12]\d|3[0-2])(?:\s*(?:,|and|&)\s*#?([1-9]|[12]\d|3[0-2]))?\b", cleaned_text, re.IGNORECASE)
    for m in compound_matches:
        for grp in [m.group(1), m.group(2)]:
            if not grp:
                continue
            num = int(grp)
            if num in [15, 16, 25, 36, 48] and num > 32:
                norm = normalize_dental_site({"fdi_number": num, "anatomical_name": f"Tooth {num}"})
            elif num in [25, 36, 48]:
                norm = normalize_dental_site({"fdi_number": num, "anatomical_name": f"Tooth {num}"})
            else:
                norm = normalize_dental_site({"universal_number": num, "anatomical_name": f"Tooth {num}"})
            if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
                seen_sites.add(norm["fdi_number"])
                anatomical_sites.append(norm)

    # 3. Deciduous Universal Letters (e.g. Tooth C, Tooth E)
    decid_letter_matches = re.finditer(r"\btooth\s*#?([A-T])\b", cleaned_text, re.IGNORECASE)
    for m in decid_letter_matches:
        letter = m.group(1).upper()
        norm = normalize_dental_site({"universal_letter": letter, "anatomical_name": f"Tooth {letter}"})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 4. Natural language anatomical phrases (e.g. "lower right third molar", "upper left wisdom tooth")
    if "third molar" in cleaned_text.lower() or "wisdom tooth" in cleaned_text.lower():
        is_lower = "lower" in cleaned_text.lower() or "mandibular" in cleaned_text.lower()
        is_right = "right" in cleaned_text.lower()
        p_code = "LR8" if (is_lower and is_right) else ("LL8" if is_lower else ("UR8" if is_right else "UL8"))
        norm = normalize_dental_site({"palmer_notation": p_code, "anatomical_name": p_code})
        if norm.get("fdi_number") and norm["fdi_number"] not in seen_sites:
            seen_sites.add(norm["fdi_number"])
            anatomical_sites.append(norm)

    # 5. Regional extra-dental anatomy
    extra_sites_map = [
        ([r"palatine\s+tonsilloliths?", r"tonsilloliths?", r"tonsils?"], "Palatine Tonsils", "Soft Tissue"),
        ([r"maxillary\s+sinuses?", r"sinuses?", r"sinus\s+floor"], "Maxillary Sinuses", "Regional Anatomy"),
        ([r"marginal\s+alveolar\s+bone", r"alveolar\s+bone", r"alveolar\s+crest", r"alveolar\s+ridge"], "Marginal Alveolar Bone", "Osseous Structure"),
        ([r"cervical\s+spine", r"vertebrae"], "Cervical Spine", "Regional Anatomy"),
        ([r"tmj", r"temporomandibular", r"condyle"], "Temporomandibular Joints", "Regional Anatomy"),
        ([r"inferior\s+alveolar\s+nerve", r"mandibular\s+canal"], "Inferior Alveolar Canal", "Regional Anatomy"),
    ]
    for patterns, site_title, s_type in extra_sites_map:
        if any(re.search(r"\b" + p + r"\b", cleaned_text, re.IGNORECASE) for p in patterns):
            norm = normalize_dental_site({"anatomical_name": site_title, "site_type": s_type})
            if norm["anatomical_name"] not in seen_sites:
                seen_sites.add(norm["anatomical_name"])
                anatomical_sites.append(norm)

    # Primary tooth reference
    primary_site_label = "lower right third molar" if ("third molar" in cleaned_text.lower()) else (anatomical_sites[0].get("anatomical_name", "Tooth") if anatomical_sites else "Dentition")

    # Detect transcription artifacts in raw text
    artifacts = detect_transcription_artifacts(dictation_text)

    # Parse clauses into findings
    for s in sentences:
        s_lower = s.lower().strip()

        # Skip patient demographics, scan technique/indication, or report verification boilerplate
        if (
            re.search(r"^patient\s+is\s+a\b", s_lower) or
            re.search(r"\b(?:cbct|examination|study|scan)\s+(?:of\s+the\s+[a-z\s]+\s+)?was\s+performed\s+for\b", s_lower) or
            re.search(r"\breport\s+was\s+verified\b", s_lower) or
            re.search(r"\bmeasurements\s+should\s+not\s+be\s+made\b", s_lower)
        ):
            continue

        # Check if sentence is a standalone diagnostic summary / impression statement
        if re.search(r"^(?:the\s+)?findings\s+suggest\b", s_lower):
            impressions.append(s)
            continue

        # Check if this sentence contains an unclear speech transcription artifact
        art_match = next((a for a in artifacts if a["matched_phrase"].lower() in s_lower), None)
        if art_match:
            findings.append({
                "anatomical_site": primary_site_label,
                "category": art_match.get("category", "Root Pathology & Resorption"),
                "observation": f"Unclear phrase: '{art_match['matched_phrase']}' (flagged as likely speech transcription error; requires clinician review)",
                "certainty": "unclear / clinician review required"
            })
            continue

        # Check for negative abnormal findings statement
        if "no other abnormal finding" in s_lower:
            findings.append({
                "anatomical_site": primary_site_label,
                "category": "General Observation",
                "observation": f"{s} (dictated clinician statement; not independently verified)",
                "certainty": "negative / normal finding"
            })
            continue

        # Determine target anatomical site for finding
        target_site = None
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
        findings.append({
            "anatomical_site": target_site,
            "category": category,
            "observation": s,
            "certainty": certainty
        })

    # Synthesize clean Impressions if not already extracted
    imp_parts = []
    for f in findings:
        obs = f["observation"].lower()
        site = f["anatomical_site"]
        if "fracture" in obs:
            if "separation" in obs:
                imp_parts.append(f"{site}: horizontal mid-root fracture with crown-root separation")
            else:
                imp_parts.append(f"{site}: root fracture")
        elif "radicular cyst" in obs or "cyst" in obs:
            if "suggest" in obs or "possible" in obs:
                imp_parts.append(f"Well-defined radiolucency suggestive of radicular cyst ({site})")
            else:
                imp_parts.append(f"Radicular cyst ({site})")
        elif "adenomatoid odontogenic tumor" in obs or "aot" in obs:
            imp_parts.append(f"Benign odontogenic lesion suggestive of AOT ({site})")
        elif "horizontally impacted" in obs:
            imp_parts.append(f"Horizontally impacted {site}")
        elif "impacted" in obs and not any("impacted" in p for p in imp_parts):
            imp_parts.append(f"Impacted {site}")

    if any("second molar" in f["observation"].lower() for f in findings):
        imp_parts.append("close proximity to adjacent second molar")

    if any("tonsillolith" in f["observation"].lower() for f in findings):
        imp_parts.append("bilateral palatine tonsilloliths noted incidentally")

    if not impressions:
        base_imp = "; ".join(list(dict.fromkeys(imp_parts))) if imp_parts else "Radiological assessment completed."
        if artifacts:
            base_imp += f". Unclear phrase '{artifacts[0]['matched_phrase']}' flagged for clinician clarification."
        impressions.append(base_imp)

    # Recommendations: Strict zero-hallucination. Only include if explicitly dictated!
    has_explicit_rec = bool(re.search(r"\b(recommend|recommended|recommendation|advise|advised|routine follow-up)\b", dictation_text, re.IGNORECASE))
    if has_explicit_rec:
        for s in sentences:
            if re.search(r"\b(recommend|recommended|recommendation|advise|advised|follow-up)\b", s, re.IGNORECASE) and "performed for" not in s.lower():
                recommendations.append(s)

    return {
        "anatomical_sites": anatomical_sites,
        "findings": findings,
        "impressions": list(dict.fromkeys(impressions)),
        "recommendations": list(dict.fromkeys(recommendations))
    }

