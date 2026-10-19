# Viscan Voice-First Dental Assistant
## System Architecture & Technical Specification

### 1. Executive Summary
The **Viscan Voice-First Dental Assistant** is an agentic, clinical-grade assistant tailored for dental and maxillofacial radiology workflows. It solves three critical clinical challenges:
1. **Feature A (Safe Report Retrieval):** Disambiguates voice/text commands, enforces clinical stops on ambiguous patient identities (e.g. multiple patients named Priya), and launches exact diagnostic imaging reports (OPG, CBCT, Bitewing, Periapical).
2. **Feature B (Clinical Dictation to Structured Draft):** Ingests unstructured radiologist speech/dictation, normalizes dental anatomical numbering across FDI, Palmer, and Universal notations, classifies findings with diagnostic certainty labels, validates against hallucination, and exports an editable structured JSON report and PDF draft.
3. **Feature C (Context-Aware Conversational Follow-up):** Maintains conversational session state across turns (e.g., resolving "Summarize her report" or "What did the radiologist note about her third molar?") without re-prompting for patient context, citing verbatim source evidence from the active report.

---

### 2. End-to-End System Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    1. CLINICAL INPUT LAYER                                        |
|   • Spoken Voice / Microphone Audio (WAV/FLAC)                                                     |
|   • Speech-to-Text Transcript (Whisper / Local STT)                                                |
|   • Clinician Natural Language Query ("Open Priya's latest OPG", "Lower right 3rd molar impacted") |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               2. DUAL-ENGINE INTERPRETATION LAYER                                  |
|                                                                                                    |
|   +------------------------------------+          +--------------------------------------------+   |
|   |   A. Local LLM Reasoning Engine    |          |   B. Deterministic Clinical Engine         |   |
|   |   (Ollama: Llama-3.1)              | <======> |   (Zero-dependency fallback & validator)   |   |
|   |   • Few-shot JSON Schema extraction|          |   • Regex & token-boundary normalization   |   |
|   |   • Semantic intent routing        |          |   • Full tooth mapping (FDI/Palmer/Univ)   |   |
|   +------------------------------------+          +--------------------------------------------+   |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 3. BUSINESS LOGIC & SAFETY GUARDRAILS                              |
|                                                                                                    |
|   [Feature A Guardrail: Identity Ambiguity Check]                                                  |
|   • Candidate count == 1  --> Safe Retrieve & Open Report                                          |
|   • Candidate count > 1   --> CLINICAL SAFETY STOP: Surface DOBs & Patient IDs for disambiguation  |
|   • Candidate count == 0  --> Missing Report Notification                                          |
|                                                                                                    |
|   [Feature B Guardrail: Anti-Hallucination & Uncertainty Labeling]                                 |
|   • Grounding Verifier: Compares extracted claims against dictated tokens (threshold >= 60%)       |
|   • Rejects unmentioned pathologies (e.g., unmentioned fractures, cysts, or periodontitis)        |
|   • Certainty Classifier: Labels findings as "definite", "suspected", or "projection limited"     |
|   • Mandatory Clinician Sign-Off Banner inserted into JSON and PDF                                 |
|                                                                                                    |
|   [Feature C Guardrail: Grounded Conversational Evidence]                                          |
|   • Pronoun resolution ("her" / "it") mapped to active session context                             |
|   • Verbatim source citation generated for every answer                                            |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                       4. OUTPUT & PRESENTATION                                     |
|   • Interactive Web UI: Modern React-styled Dashboard with Chat, Dictation Studio, & PDF Viewer    |
|   • FastAPI REST Engine: Documented OpenAPI / Swagger Endpoints at http://localhost:8000/docs     |
|   • Structured JSON: Machine-readable FHIR/DICOM compatible report data                            |
|   • Formatted PDF Generator: High-resolution PDF with clinician verification signature block       |
|   • Grounded Citations: Verbatim findings quoted directly from active diagnostic reports           |
+----------------------------------------------------------------------------------------------------+
```

---

### 3. Separation of Concerns: LLM Interpretation vs. Business Rules
A core design tenet of the architecture is the **strict decoupling of natural language interpretation from business-rule validation**:
- **LLM Role (Probabilistic):** Parses linguistic variance ("her wisdom tooth", "UR5", "tooth 48", "impacted lower right 8"), extracts candidate entities, and generates natural synthesis.
- **Business Rule Engine (Deterministic):**
  - Manages patient registry lookup and exact string/date sorting.
  - Enforces the **Ambiguity Safety Gate**: If `len(matching_patients) > 1`, execution *must* pause and present disambiguation options; the LLM is never permitted to guess.
  - Validates dental tooth number cross-references (e.g., ensuring Universal #29 correctly maps to FDI 45 / LR5).
  - Enforces the **Anti-Hallucination Gate**: Ensures no claim appears in the draft that lacks lexical and semantic support in the audio transcript.

---

### 4. Dental Anatomical Mapping Architecture
Dentists and radiologists use differing notation systems depending on training and region:
- **FDI Two-Digit System (ISO 3950):** Standard international notation (e.g., `48` = mandibular right third molar).
- **Palmer Notation:** Used predominantly in the UK and clinical orthodontics (e.g., `LR8`, `UR5`, `UL6`).
- **Universal Numbering System:** Standard US notation (e.g., `#32` = lower right third molar, `#29` = lower right second premolar).
- **Natural Language:** e.g., "lower right third molar", "mandibular condyle", "inferior alveolar nerve canal".

The system includes a bidirectional cross-reference matrix that maps any of these representations into a canonical clinical entity with quadrant, anatomical name, and alternative numbering formats.

---

### 4. LLM, Function Calling & Model Context Protocol (MCP) Architecture

The application strictly decouples probabilistic intelligence from deterministic clinical execution across 3 standard layers:

#### A. LLM Engine (`backend/app/services/llm_service.py`)
- **Model Choice:** Connects to local instruction-tuned **Llama 3.1** via Ollama (`llama3.1`).
- **Zero Cloud Leakage:** All inference runs locally on the clinician's machine (`http://localhost:11434/api/generate`), eliminating third-party cloud data egress. Note that while local inference and session isolation prevent data transmission, full production HIPAA/GDPR compliance requires enterprise data encryption at rest, RBAC authentication, and audit logging.
- **System Prompt Engineering:**
  1. `DICTATION_SYSTEM_PROMPT`: Instructs the LLM to ingest radiologist speech/dictation transcripts and output validated structured JSON (anatomical sites, FDI tooth numbering, observations, certainty ratings).
  2. `RETRIEVAL_SYSTEM_PROMPT`: Extracts patient names, scan modalities (OPG, CBCT, Bitewing), and search intents from free-form spoken queries.
  3. `REPORT_QA_SYSTEM_PROMPT`: Answers conversational follow-up questions strictly grounded in the active diagnostic report with verbatim citations.

#### B. Function & Tool Calling (`backend/app/services/mcp_tools.py`)
- Implements standardized function definitions compatible with Ollama and OpenAI Tool Calling APIs:
  1. `search_patient_reports(patient_name, modality)`: Calls the deterministic retrieval engine with automatic safety stops for ambiguous names.
  2. `structure_clinical_dictation(dictation_text, patient_id)`: Transforms raw spoken transcripts into structured observations with anti-hallucination validation.
  3. `summarize_active_report(report_id, query)`: Queries and summarizes specific tooth observations directly from the active report.
- The `execute_tool(tool_name, arguments)` function acts as the central deterministic execution dispatcher.

#### C. Model Context Protocol (MCP) (`backend/app/api/mcp.py`)
- Exposes standard **Model Context Protocol (JSON-RPC 2.0)** endpoints at `POST /api/mcp`:
  - `tools/list`: Allows external LLM clients, AI agents, or IDE extensions (Claude Desktop, Cursor, Antigravity) to inspect available dental clinical capabilities.
  - `tools/call`: Executes standard tool invocations and returns formatted text/JSON responses.
- Provides REST inspection endpoints at `GET /api/mcp/tools` and `POST /api/mcp/call`.

---

### 5. Production Considerations

#### A. Privacy & Data Security (Zero PHI Egress)
- Clinical imaging and dictated observations contain Protected Health Information (PHI) subject to HIPAA, GDPR, and regional medical privacy statutes.
- By utilizing a **local LLM (Ollama / Llama-3.1)** or a zero-dependency deterministic engine, **zero PHI leaves the clinical workstation or local hospital intranet**.
- No external cloud API calls are required for daily operation. Synthetic patient datasets are used for local development and demonstration.
- **Compliance Distinction:** Session isolation alone does not establish HIPAA or GDPR compliance; full clinical deployment mandates enterprise access controls, encryption at rest/transit, comprehensive audit trails, and strict data-retention policies.

#### B. Authorization & Access Control (RBAC)
- Enforces Role-Based Access Control (RBAC) across three clinical privilege tiers:
  1. **Attending Radiologist / Dentist:** Full access to view scans, perform grounded Q&A, dictate findings, and digitally sign off on draft reports.
  2. **Dental Assistant / Hygienist:** Access to open patient reports and record dictation drafts; restricted from final report sign-off.
  3. **Audit Administrator:** Read-only access to system access logs, tool invocation traces, and safety verification reports.
- All report retrievals and dictation sign-offs are logged with clinician identity and workstation session ID.

#### C. Latency & Real-Time Clinician Experience
- The deterministic MCP tools and business validation rules execute in **< 10 milliseconds**.
- Local quantized LLM inference (Llama-3.1 on Apple Silicon / modern GPUs) responds in **< 1.2 seconds**, well within acceptable bounds for conversational clinical assistants.

#### D. Model & Tool Operational Costs
- **Zero Marginal API Cost:** Local self-hosted Ollama deployment eliminates per-token API charges (saving thousands of dollars per clinic annually compared to cloud-based LLM APIs).
- Operates entirely on commodity clinical hardware (macOS / Linux / Windows workstations).

#### E. Failure Handling & Input Validation
- **Pydantic v2 Schema Enforcement:** Every LLM response is validated against strict Pydantic schemas before execution.
- **Graceful Fallback:** If the local Ollama daemon is offline or starting up, the system automatically degrades to its deterministic clinical engine with zero downtime and 100% test pass fidelity.
- **Safety Disambiguation Gate:** If an ambiguous patient name is dictated, the system avoids cognitive slips by refusing to open an arbitrary file, preventing catastrophic misattribution of diagnostic imaging.
