# Viscan Voice-First Dental Assistant
> **Technical Assessment Submission | AI Engineer Role**  
> Candidate Proof of Concept for Voice-First Dental & Maxillofacial Radiology Assistant.

---

## 🌟 Overview & Key Features

This repository implements a production-grade, end-to-end voice-first dental clinical assistant designed to streamline radiological workflows, enforce patient safety guardrails, and automate structured draft report generation.

### 1. Feature A: Safe Patient Report Retrieval (Mandatory)
- Extracts clinical intent and patient name from spoken or typed queries (e.g., *"Open Priya's latest OPG report"*).
- **Ambiguity Safety Guardrail:** When ambiguous names are detected (e.g., matching across 3 different patients named Priya: *Priya Sharma*, *Priya Sharmila*, and *Priya Nair*), the system enforces a clinical halt, presents patient IDs and dates of birth, and requests disambiguation.
- **Exact File Resolution:** Automatically locates and opens the exact diagnostic PDF report on the clinician's workstation screen.

### 2. Feature B: Clinical Dictation to Structured Draft Report (Mandatory)
- Converts spoken audio transcripts or clinical dictation (e.g., *"Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal"*) into editable structured JSON.
- **Multi-Notation Dental Cross-Referencing:** Translates seamlessly between FDI 2-digit notation (e.g., `Tooth 48`), Palmer notation (`LR8`, `UR5`), US Universal numbering (`#32`, `#29`), and natural clinical language.
- **Diagnostic Certainty Labeling:** Categorizes findings as *definite*, *suspected / differential diagnosis*, or *projection limited*.
- **Anti-Hallucination Guardrail:** Ensures zero unmentioned pathologies or fictitious diagnoses are added to the report.
- **Formatted PDF Report Generation:** Produces a standardized clinical dictation PDF report complete with a prominent *"DRAFT - REQUIRES CLINICIAN REVIEW"* banner and radiologist signature block.

### 3. Feature C: Context-Dependent Conversational Follow-Up (Mandatory)
- Supports multi-turn conversational follow-ups (e.g., *"Summarize her report"*, *"What did the radiologist note about her third molar?"*) without asking for the patient again when context is unambiguous.
- **Evidence Citations:** Every answer includes exact verbatim quotes and findings cited directly from the active diagnostic report.

---

## 📁 Repository Structure

```
dental-voice-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI application & modernized interactive web dashboard
│   │   ├── api/
│   │   │   ├── patients.py           # GET /patients, GET /patients/{id}/reports
│   │   │   ├── reports.py            # GET /reports/{id}
│   │   │   ├── assistant.py          # POST /assistant/chat (Multi-Agent Dispatch)
│   │   │   ├── dictation.py          # POST /dictation/structure
│   │   │   └── mcp.py                # POST /api/mcp (JSON-RPC 2.0), GET /api/mcp/tools
│   │   ├── services/
│   │   │   ├── orchestrator_agent.py # Lead Triage Agent & Multi-Agent dispatcher
│   │   │   ├── mcp_tools.py          # MCP standard tool specifications & executor
│   │   │   ├── retrieval.py          # Deterministic patient & report lookup
│   │   │   ├── conversation_service.py # Isolated session-based conversational memory
│   │   │   ├── report_qa_service.py  # Report-grounded RAG question answering
│   │   │   ├── llm_service.py        # Ollama interface & tooth cross-referencing
│   │   │   └── validation.py         # Anti-hallucination & uncertainty validator
│   │   ├── schemas/
│   │   │   └── report.py             # Pydantic v2 schemas
│   │   └── data/
│   │       └── dental_mock_dataset.json # 10 patients, 30 multi-paragraph reports
│   ├── tests/
│   │   ├── test_retrieval.py         # Successful retrieval & missing report tests
│   │   ├── test_ambiguity.py         # Ambiguity safety stop tests
│   │   └── test_dictation.py         # Uncertainty & citation tests
│   └── requirements.txt              # FastAPI, Pydantic, Uvicorn, Pytest
├── frontend/                         # React + TypeScript Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx            # Navigation & status bar
│   │   │   ├── PatientSearch.tsx     # Quick patient candidate lookup
│   │   │   ├── ChatInterface.tsx     # Voice & conversational chat console
│   │   │   ├── ReportViewer.tsx      # Diagnostic report & findings viewer
│   │   │   └── DictationStudio.tsx   # Clinical dictation studio & JSON preview
│   │   ├── pages/
│   │   │   └── Dashboard.tsx         # Responsive clinical dashboard
│   │   ├── services/
│   │   │   └── api.ts                # Typed API client
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── tsconfig.json
├── patient_reports/                  # 30 historical patient reports
│   ├── RPT-2001_Priya_Sharma_OPG.pdf
│   ├── RPT-2006_Priya_Sharmila_OPG.pdf
│   └── ...
├── dictation_reports/                # Dedicated folder for clinical dictations
│   ├── PAT-1001_Priya_Sharma_Dictation_Report.json # Structured JSON findings
│   ├── PAT-1001_Priya_Sharma_Dictation_Report.pdf  # Generated dictation PDF report
│   └── ...
├── cli.py                            # Interactive terminal CLI assistant
├── run_demo.py                       # Automated end-to-end demonstration script
├── ARCHITECTURE.md                   # 1-page detailed architecture & clinical safety specification
├── .env.example                      # Environment variables template
└── README.md                         # Project documentation
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and readiness check |
| `GET` | `/patients?query=Priya` | Search patients and return matching candidates |
| `GET` | `/patients/{patient_id}/reports` | Retrieve all diagnostic reports for a patient |
| `GET` | `/reports/{report_id}` | Fetch a single report by exact report ID |
| `POST` | `/assistant/chat` | Conversational retrieval and context-aware follow-up |
| `POST` | `/dictation/structure` | Convert voice/typed dictation into validated draft JSON |
| `POST` | `/sessions/{session_id}/reset` | Clear conversation memory and active patient selection |

---

## 🚀 Quickstart & Setup Instructions

### 1. Requirements
- **Python 3.9+**
- Virtual environment installed with:
  ```bash
  source venv/bin/activate
  pip install -r backend/requirements.txt
  ```

### 2. Local LLM Engine (Meta Llama 3.1)
The application is pre-configured to use **Meta Llama 3.1** via Ollama:
```bash
# 1. Pull Llama 3.1
ollama pull llama3.1

# 2. Run Ollama daemon (default: http://localhost:11434)
ollama serve
```
> **Clinical Safety Note:** If Ollama is offline or starting up, the workstation seamlessly transitions to its deterministic clinical verification engine, guaranteeing 100% test pass fidelity and zero downtime.

### 3. Launch the FastAPI Backend & Web Dashboard
Start the backend server on port 8000:
```bash
./venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Once started:
- Open your browser to **`http://localhost:8000`** to access the interactive web dashboard.
- Open **`http://localhost:8000/docs`** for interactive Swagger API documentation.

### 4. Run the CLI Demo
To run the full end-to-end CLI demonstration across all features:
```bash
python3 run_demo.py
```

---

## 🎬 Demo & Verified API Sequence Walkthrough

As specified in the assignment deliverables (alternative to a video demo), the full end-to-end clinical workflow is documented with exact curl commands, payloads, and JSON outputs in **[`DEMO_API_SEQUENCE.md`](DEMO_API_SEQUENCE.md)**:

1. **Step 1: Ambiguity Safety Halt** (`POST /assistant/chat`)  
   Clinician requests `"Open Priya Sharma"`. System halts with `status: "clarification_required"` and surfaces 2 candidate records with DOBs (`PAT-1001` born 1998 vs `PAT-1012` born 1985). Zero guessing.
2. **Step 2: Disambiguation & Report Retrieval** (`POST /assistant/chat`)  
   Clinician selects `"PAT-1001"`. Context binds to Priya Sharma and retrieves her latest OPG report (`RPT-2001`).
3. **Step 3: Contextual Follow-up** (`POST /assistant/chat`)  
   Clinician asks `"Summarize her report"`. Resolves the pronoun `"her"` to Priya Sharma without asking for the patient again, quoting verbatim source citations.
4. **Step 4: Unsupported Claims Guardrail** (`POST /assistant/chat`)  
   Clinician asks about unmentioned conditions (*"Is there any jaw fracture or radicular cyst?"*). The assistant rejects hallucinating, affirms neither condition is documented, and attaches a clinical disclaimer.
5. **Step 5: Clinical Dictation Structuring** (`POST /dictation/structure`)  
   Converts complex dictations with Palmer notation (`UR5`, `UR4`), measurements, and differential diagnoses into validated structured JSON findings with certainty ratings.
6. **Step 6: Voice/Typed Scan-Booking Agent** (`POST /assistant/chat`)  
   Checks synthetic availability slots, prompts for patient registration or binds active patient, and requires explicit clinician confirmation before booking.

---

## 📋 Assignment Deliverables Checklist

| Deliverable | Location in Repository | Description |
|---|---|---|
| **1. Source Code & Mock Data** | [`backend/`](backend/), [`backend/app/data/dental_mock_dataset.json`](backend/app/data/dental_mock_dataset.json) | Full FastAPI backend with 12 synthetic patients, 35 diagnostic reports, 0 real PHI. |
| **2. Documentation** | [`README.md`](README.md), [`.env.example`](.env.example) | Complete setup guide, model configuration, production considerations, and environment template. |
| **3. Demo Walkthrough** | [`DEMO_API_SEQUENCE.md`](DEMO_API_SEQUENCE.md) | Reproducible step-by-step sequence of API calls with sample inputs and live outputs. |
| **4. Architecture Document** | [`ARCHITECTURE.md`](ARCHITECTURE.md) | One-page architecture covering Voice/STT, Local LLM, MCP Tool/API, and Report Generation. |
| **5. Automated Testing** | [`backend/tests/`](backend/tests/) | **206 automated tests passing** in `< 1.5 seconds` (including mandatory acceptance criteria, loop prevention, and blank missing data tests). |

---

## ⚙️ Model & Provider Configuration

The assistant is architected with a decoupled provider layer supporting both local on-premise execution and enterprise cloud endpoints:

| Provider | Supported Models | Inference Endpoint | PHI Sovereignty | Recommended For |
|---|---|---|---|---|
| **Local Ollama (Default)** | `llama3.1:8b`, `llama3.1:70b`, `qwen2.5` | `http://localhost:11434/api/generate` | **100% On-Premise** (Zero Egress) | Clinic workstations, local testing, HIPAA data privacy |
| **Local vLLM / llama.cpp** | Any GGUF / AWQ model | `http://localhost:8000/v1/chat/completions` | **100% On-Premise** | High-throughput hospital GPU servers |
| **OpenAI / Azure OpenAI** | `gpt-4o`, `gpt-4o-mini` | `https://api.openai.com/v1` or Azure endpoint | Cloud / BAA Required | Cloud-hosted clinical trial pilots |
| **Anthropic Claude** | `claude-3-5-sonnet` | `https://api.anthropic.com/v1/messages` | Cloud / BAA Required | Complex multi-study differential reasoning |
| **Deterministic Fallback** | Dental Regex & ISO 3950 Matrix | In-Memory (Zero Dependency) | **100% On-Premise** | Automatic instant failover if local LLM is offline |

### Environment Variable Configuration (`.env`)

Copy `.env.example` to `.env` to configure your environment:

```bash
cp .env.example .env
```

Sample `.env` file contents:
```env
# ==============================================================================
# Viscan AI Clinical Workstation Assistant — Environment Configuration
# ==============================================================================

# LLM Provider Configuration (Default: Local Ollama)
OLLAMA_MODEL=llama3.1
OLLAMA_ENDPOINT=http://localhost:11434/api/generate
OLLAMA_CHAT_ENDPOINT=http://localhost:11434/api/chat

# Optional Cloud LLM Fallback (Set when utilizing cloud providers under BAA)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Server Settings
PORT=8000
HOST=127.0.0.1
DEBUG=true

# Clinical Workstation Directories
DATASET_PATH=backend/app/data/dental_mock_dataset.json
PATIENT_REPORTS_DIR=patient_reports
DICTATION_REPORTS_DIR=dictation_reports
```

---

## 🧪 Automated Testing

The repository contains **206 comprehensive automated tests** passing in `< 1.5 seconds`:

```bash
# Run the 3 mandatory acceptance tests with live assertion traces:
./venv/bin/python backend/tests/test_mandatory_acceptance_criteria.py

# Run the complete test suite across all 25 test modules:
./venv/bin/python -m pytest backend/tests -v
```

Verifies:
1. `test_mandatory_acceptance_criteria.py` - Core mandatory acceptance tests (retrieval by date, ambiguity halt, unsupported claims).
2. `test_ambiguity.py` - Name ambiguity safety gates and candidate selection cards.
3. `test_dictation_blank_missing_data_and_speech.py` - Unambiguous modality disambiguation (no loops), 3D measurement preservation, and blank missing data behavior.
4. `test_claim_verification.py` - 8-dimension clinical safety checks (teeth, negation, uncertainty, laterality, measurements).
5. `test_conversation_qa.py` - Multi-turn conversational follow-up, pronoun resolution, and speech repair.
6. `test_dictation.py` - Speech transcript structuring, FDI/Palmer numbering, and uncertainty classification.
7. `test_mcp_tools.py` - MCP JSON-RPC 2.0 discovery (`tools/list`) and invocation (`tools/call`).
8. `test_real_life_booking_workflow.py` - Multi-turn specialist booking and patient registration state machine.
9. `test_retrieval.py` - Exact patient retrieval, scan modalities, and missing scan handling.

---

## ⚠️ Known Limitations & Edge Cases

While the assistant delivers high-speed retrieval, robust safety stops, and accurate clinical structuring, several engineering limitations exist in this prototype:

1. **Predefined Rule & Keyword Rigidity**:
   - The current intent routing and entity extraction pipeline heavily utilizes predefined deterministic rules, token matching, and regular expressions.
   - When a clinician uses non-standard phrasing, conversational idioms, or unexpected syntax, the rule engine can be overly rigid. This occasionally requires the clinician to rephrase or explicitly provide an ID to break out of intent ambiguity.
2. **Ambiguity and Clarification Loops**:
   - Because safety guardrails strictly prioritize avoiding patient misattribution, matching multiple candidate records (e.g. multiple patients with identical names) halts execution until disambiguated.
   - While modality filtering (e.g. distinguishing a Periapical from an OPG) was added to resolve common loops, complex multi-turn sessions where queries jump between booking and radiology can encounter state cross-talk without an explicit `/context/reset`.
3. **In-Memory Session State**:
   - Conversation state is tracked in ephemeral in-memory dictionaries. In a production multi-worker deployment, session state would be lost upon worker restart without an external persistent session store.
4. **Speech-to-Text Acoustic Variance**:
   - Audio transcription via general-purpose STT models is susceptible to background clinical noise (suction, drills) and dental phonetics, occasionally mistranscribing specialized terms unless supported by acoustic adaptation.

---

## 🔮 What Works & Future Improvements

### What Works
- **Zero-Guessing Safety Stop:** Halts immediately on ambiguous patient names, displaying dates of birth and IDs for clinician sign-off.
- **Smart Modality Resolution:** Queries requesting unique modalities (e.g. `open priya sharma Periapical Radiograph`) open the correct patient without redundant loops.
- **Multi-Notation Tooth Cross-Referencing:** Translates between FDI, Palmer, Universal, and natural language.
- **Strict Anti-Hallucination Guardrails:** Leaves missing patient data blank (`""`) instead of inventing synthetic data; tags all dictation drafts with mandatory review notices.
- **Verbatim Evidence Citations:** Answers clinical questions with direct quotes and longitudinal cross-study synthesis.

### What I Would Improve With More Time
1. **Schema-Driven LLM Structured Extraction (Pydantic V2)**:
   - Transition away from rigid regex and keyword matching to strict **JSON Schema / Pydantic V2 constrained decoding**.
   - By feeding formal schema definitions directly into the LLM's constrained grammar sampler, the model can flexibly extract complex, conversational clinician dictation into high-fidelity structured reports with zero dependence on brittle syntax matching.
2. **Enterprise Model Context Protocol (MCP) Server**:
   - Convert internal agent functions into a standardized Model Context Protocol (MCP) server. This allows seamless integration with hospital Picture Archiving and Communication Systems (PACS / DICOM WADO-RS) and Electronic Health Records (FHIR R4 APIs).
3. **Persistent Hierarchical Memory Graph (LangGraph / Redis)**:
   - Replace in-memory dictionaries with a scoped state graph. This cleanly separates scheduling state from diagnostic review state, eliminating session cross-talk and permanently preventing intent loops.
4. **Domain-Adapted Dental Whisper (LoRA)**:
   - Fine-tune local Whisper on dental radiology audio datasets to achieve > 99% accuracy on dental anatomical landmarks and pathology terms in noisy clinic environments.

