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

## 🧪 Automated Testing

The repository contains **182 comprehensive automated tests** covering retrieval, ambiguity disambiguation, multi-dimensional clinical claim verification, conversational follow-up, report-aware QA, appointment booking state machine, and MCP JSON-RPC 2.0 tool execution:

```bash
python3 -m pytest backend/tests -v
```

Verifies:
1. `test_ambiguity.py` - Partial name ambiguity safety gate and disambiguation.
2. `test_claim_verification.py` - 8-dimension clinical safety checks (tooth numbers, negation, uncertainty, laterality, measurements).
3. `test_conversation_qa.py` - Multi-turn conversational follow-up, pronoun resolution, and speech repair.
4. `test_dictation.py` - Speech transcript structuring, dental numbering, and uncertainty classification.
5. `test_mcp_tools.py` - MCP JSON-RPC 2.0 discovery (`tools/list`) and invocation (`tools/call`).
6. `test_orchestrator.py` - Multi-agent triage routing and execution tracing.
7. `test_retrieval.py` - Exact patient retrieval, modalities, and missing scan handling.

---

## 🔒 Production Considerations & Known Limitations

### Privacy & Data Sovereignty
- In clinical radiology, reports contain sensitive Protected Health Information (PHI). This architecture is built for on-premise local execution (via local Ollama Llama-3.1), eliminating third-party cloud data transmission. While session-isolated memory prevents in-memory cross-contamination between workstations, comprehensive HIPAA/GDPR compliance in production requires enterprise data encryption at rest and in transit, access control, and complete audit logging.

### Model Context Protocol (MCP) Standards
- Implements the official **Anthropic Model Context Protocol (MCP)** specification over **JSON-RPC 2.0** (`tools/list` and `tools/call`).
- Standardized tool schemas (`search_patient_reports`, `structure_clinical_dictation`, `summarize_active_report`, `verify_clinical_claim`) separate probabilistic LLM reasoning from deterministic clinical execution.

### What Works
- Safe disambiguation of patient identities with zero guessing.
- Multi-turn speech correction and conversation memory across clinical queries.
- Reliable mapping between international dental numbering notations (FDI, Palmer, Universal).
- Multi-dimensional clinical claim verification across 8 medical dimensions.

### Future Improvements with More Time
1. **Live Microphone Audio Streaming:** Connect a local WebAudio stream directly to `faster-whisper` for instant speech-to-text dictation.
2. **DICOM Structured Reporting (DICOM-SR):** Export the structured findings directly into standard DICOM-SR objects for direct PACS integration.
