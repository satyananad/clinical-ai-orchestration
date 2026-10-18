import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .api import patients, reports, assistant, dictation, mcp

app = FastAPI(
    title="Viscan Voice-First Dental Assistant API",
    description="Multi-Agent system with MCP Tool Calling for report retrieval, clinical dictation, and conversational context.",
    version="1.0.0"
)

# Enable CORS for local and frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(patients.router)
app.include_router(reports.router)
app.include_router(assistant.router)
app.include_router(dictation.router)
app.include_router(mcp.router)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "patient_reports")
DICTATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dictation_reports")

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify backend service readiness."""
    return {
        "status": "healthy",
        "service": "viscan-dental-assistant-api",
        "version": "1.0.0",
        "agents": ["TriageAgent", "RetrievalAgent", "ClinicalScribeAgent", "ConsultationAgent"],
        "mcp_protocol": "JSON-RPC 2.0"
    }

@app.get("/patient-pdf/{filename}", tags=["Reports"])
def serve_patient_pdf(filename: str):
    """Serves the generated PDF file directly in browser."""
    filepath = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/pdf", filename=filename)
    return {"error": "File not found"}

@app.get("/dictation-pdf/{filename}", tags=["Dictation"])
def serve_dictation_pdf(filename: str):
    """Serves the generated dictation draft PDF file."""
    filepath = os.path.join(DICTATION_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/pdf", filename=filename)
    return {"error": "File not found"}

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def get_dashboard_ui():
    """Serves the modernized interactive Multi-Agent Dental Assistant UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Viscan AI Dental Assistant | Multi-Agent Clinical Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .custom-scroll::-webkit-scrollbar { width: 5px; height: 5px; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        @keyframes pulse-slow { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .animate-pulse-slow { animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 font-sans antialiased min-h-screen flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="bg-slate-800/90 backdrop-blur border-b border-slate-700/80 sticky top-0 z-50 shadow-md">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-blue-500/20">
                    <i class="fa-solid fa-tooth"></i>
                </div>
                <div>
                    <div class="flex items-center space-x-2">
                        <h1 class="font-bold text-base text-white tracking-tight leading-none">Viscan Dental AI</h1>
                        <span class="text-[10px] bg-blue-500/20 text-blue-300 font-mono px-2 py-0.5 rounded border border-blue-500/30">v1.0 Multi-Agent</span>
                    </div>
                    <p class="text-[11px] text-slate-400 mt-0.5">Voice-First Radiology Assistant • Model Context Protocol (MCP)</p>
                </div>
            </div>

            <!-- Active Session Context Pill -->
            <div id="activeContextBanner" class="hidden sm:flex items-center space-x-2 bg-slate-900/80 border border-slate-700 px-3 py-1.5 rounded-full text-xs">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span class="text-slate-400">Context:</span>
                <span id="activeContextText" class="font-semibold text-emerald-300">None Selected</span>
                <button onclick="resetSession()" class="text-slate-400 hover:text-rose-400 ml-1 text-xs" title="Reset Session Context">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>

            <!-- Quick Links -->
            <div class="flex items-center space-x-2 sm:space-x-3">
                <a href="/docs" target="_blank" class="text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600 rounded-lg px-3 py-1.5 transition flex items-center">
                    <i class="fa-solid fa-book-medical mr-1.5 text-blue-400"></i> Swagger API
                </a>
                <a href="/api/mcp/tools" target="_blank" class="text-xs bg-indigo-900/60 hover:bg-indigo-800 text-indigo-200 border border-indigo-700 rounded-lg px-3 py-1.5 transition flex items-center">
                    <i class="fa-solid fa-network-wired mr-1.5 text-indigo-400"></i> MCP Tools
                </a>
            </div>
        </div>
    </header>

    <!-- Workspace Tabs Navigation -->
    <div class="bg-slate-800 border-b border-slate-700">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 sm:space-x-4">
            <button onclick="switchTab('assistant')" id="tabBtn-assistant" class="px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 border-blue-500 text-blue-400 flex items-center transition">
                <i class="fa-solid fa-comments mr-2"></i> 1. Voice & Conversational Assistant (Feature A & C)
            </button>
            <button onclick="switchTab('dictation')" id="tabBtn-dictation" class="px-4 py-3 text-xs sm:text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center transition">
                <i class="fa-solid fa-microphone-lines mr-2"></i> 2. Clinical Dictation Studio (Feature B)
            </button>
            <button onclick="switchTab('mcp')" id="tabBtn-mcp" class="px-4 py-3 text-xs sm:text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center transition">
                <i class="fa-solid fa-microchip mr-2"></i> 3. Multi-Agent & MCP Inspector
            </button>
        </div>
    </div>

    <!-- MAIN CONTAINER -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">

        <!-- =================================================================== -->
        <!-- TAB 1: CONVERSATIONAL ASSISTANT & REPORT VIEWER (FEATURES A, C & BONUS) -->
        <!-- =================================================================== -->
        <div id="tabContent-assistant" class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left 6 Columns: Interactive Chat & Fast Test Buttons -->
            <div class="lg:col-span-6 flex flex-col space-y-4">
                
                <!-- Dynamic Patient Directory & Clinical Actions Bar -->
                <div class="bg-slate-800/90 border border-slate-700 rounded-2xl p-3.5 shadow-sm space-y-2.5">
                    <div class="flex flex-wrap items-center justify-between gap-2">
                        <label class="text-[11px] font-bold uppercase tracking-wider text-slate-300 flex items-center">
                            <i class="fa-solid fa-users-medical mr-1.5 text-blue-400"></i> Clinic Patient Directory:
                        </label>
                        <span class="text-[10px] text-slate-400">Select any patient to query dynamically</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <select id="patientDirectorySelect" class="flex-1 bg-slate-900 border border-slate-700 text-slate-100 text-xs rounded-xl px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none">
                            <option value="Priya Sharma">Priya Sharma (PAT-1001 • DOB: 1998-04-12)</option>
                            <option value="John Smith">John Smith (PAT-1002 • DOB: 1985-04-12)</option>
                            <option value="John Smith">John Smith (PAT-1003 • DOB: 1992-11-20)</option>
                            <option value="Aarav Mehta">Aarav Mehta (PAT-1004 • DOB: 1979-09-03)</option>
                            <option value="Fatima Al-Mansoor">Fatima Al-Mansoor (PAT-1005 • DOB: 1995-02-18)</option>
                            <option value="Marcus Vance">Marcus Vance (PAT-1006 • DOB: 1968-11-05)</option>
                            <option value="Ananya Deshmukh">Ananya Deshmukh (PAT-1007 • DOB: 2001-07-22)</option>
                            <option value="Vikram Malhotra">Vikram Malhotra (PAT-1008 • DOB: 1983-05-30)</option>
                            <option value="Elena Rostova">Elena Rostova (PAT-1009 • DOB: 1990-12-14)</option>
                            <option value="Kavita Reddy">Kavita Reddy (PAT-1010 • DOB: 1974-08-27)</option>
                        </select>
                        <button onclick="triggerSelectedPatientRetrieval()" class="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-3 py-2 rounded-xl transition flex items-center shadow-md">
                            <i class="fa-solid fa-folder-open mr-1.5"></i> Open Scan
                        </button>
                    </div>

                    <!-- Quick Dynamic Clinical Actions -->
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-1.5 pt-1">
                        <button onclick="triggerAction('opg')" class="text-left text-[11px] bg-slate-700/60 hover:bg-slate-700 text-slate-200 border border-slate-600 rounded-lg p-2 transition">
                            <span class="font-bold block text-[10px] text-blue-400"><i class="fa-solid fa-tooth mr-1"></i> Latest OPG</span>
                            Fetch Panoramic
                        </button>
                        <button onclick="triggerAction('cbct')" class="text-left text-[11px] bg-slate-700/60 hover:bg-slate-700 text-slate-200 border border-slate-600 rounded-lg p-2 transition">
                            <span class="font-bold block text-[10px] text-indigo-400"><i class="fa-solid fa-cube mr-1"></i> Latest CBCT</span>
                            Fetch 3D Scan
                        </button>
                        <button onclick="runSamplePrompt('Open John Smith\'s latest report')" class="text-left text-[11px] bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 rounded-lg p-2 transition">
                            <span class="font-bold block text-[10px] text-amber-400"><i class="fa-solid fa-triangle-exclamation mr-1"></i> Ambiguity Stop</span>
                            Test Rule 1 (2 Patients)
                        </button>
                        <button onclick="triggerAction('summarize')" class="text-left text-[11px] bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-lg p-2 transition">
                            <span class="font-bold block text-[10px] text-emerald-400"><i class="fa-solid fa-arrows-spin mr-1"></i> Summarize</span>
                            Feature C Follow-up
                        </button>
                    </div>
                </div>

                <!-- Chat Messages Terminal Card -->
                <div class="bg-slate-800 border border-slate-700 rounded-2xl shadow-lg flex flex-col h-[520px]">
                    <div class="p-3 border-b border-slate-700 bg-slate-800/90 rounded-t-2xl flex items-center justify-between">
                        <div class="flex items-center space-x-2">
                            <span class="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse"></span>
                            <span class="font-semibold text-xs text-white">Live Multi-Agent Dialogue Stream</span>
                        </div>
                        <button onclick="resetSession()" class="text-[11px] text-slate-400 hover:text-rose-400 transition flex items-center">
                            <i class="fa-solid fa-rotate-left mr-1"></i> Clear Dialogue
                        </button>
                    </div>

                    <!-- Chat Container -->
                    <div id="chatMessages" class="flex-1 p-4 overflow-y-auto space-y-3 custom-scroll text-xs">
                        <!-- Initial Welcome Message -->
                        <div class="bg-slate-700/50 border border-slate-600/80 rounded-2xl p-3.5 text-slate-200">
                            <p class="font-bold text-blue-400 text-xs mb-1 flex items-center">
                                <i class="fa-solid fa-robot mr-1.5"></i> Multi-Agent Orchestrator Ready
                            </p>
                            Welcome Doctor. The system routes voice and text commands to specialized clinical agents:
                            <div class="mt-2 text-[11px] text-slate-300 space-y-1">
                                <div>• <strong>RetrievalAgent:</strong> Locates scans and safely pauses on ambiguous names.</div>
                                <div>• <strong>ConsultationAgent:</strong> Summarizes reports and answers grounded Q&A with citations.</div>
                            </div>
                        </div>
                    </div>

                    <!-- Input Controls -->
                    <div class="p-3 border-t border-slate-700 bg-slate-800/95 rounded-b-2xl">
                        <div class="flex items-center space-x-2">
                            <input id="chatInput" type="text" placeholder="Speak or type (e.g. 'Open Vikram Malhotra\'s CBCT', 'Summarize her report')..." 
                                   onkeydown="if(event.key==='Enter') sendChat()"
                                   class="flex-1 text-xs bg-slate-900 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-xl px-3.5 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <button onclick="toggleVoiceRecord('chatInput', 'voiceBtn')" id="voiceBtn" class="bg-slate-700 hover:bg-slate-600 text-slate-200 w-10 h-10 rounded-xl flex items-center justify-center transition shadow-sm" title="Click to Speak Voice Command">
                                <i class="fa-solid fa-microphone text-sm"></i>
                            </button>
                            <button onclick="sendChat()" class="bg-blue-600 hover:bg-blue-500 text-white w-10 h-10 rounded-xl flex items-center justify-center shadow-lg shadow-blue-600/30 transition">
                                <i class="fa-solid fa-paper-plane text-xs"></i>
                            </button>
                        </div>
                    </div>
                </div>

            </div>

            <!-- Right 6 Columns: Active Report Viewer & Findings -->
            <div class="lg:col-span-6 flex flex-col space-y-4">
                
                <!-- Report Card -->
                <div class="bg-slate-800 border border-slate-700 rounded-2xl p-5 shadow-lg flex flex-col min-h-[580px]">
                    <div class="flex items-center justify-between border-b border-slate-700 pb-3 mb-4">
                        <div>
                            <div class="flex items-center space-x-2">
                                <span id="reportModalityBadge" class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">NO SCAN</span>
                                <h2 id="reportTitle" class="font-bold text-white text-base">Radiology Report Viewer</h2>
                            </div>
                            <p id="reportPatientMeta" class="text-xs text-slate-400 mt-1">Patient: None selected • Report ID: -</p>
                        </div>
                        <div id="pdfDownloadContainer" class="hidden">
                            <a id="pdfDownloadBtn" href="#" target="_blank" class="inline-flex items-center text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white px-3.5 py-2 rounded-xl transition shadow-lg shadow-rose-600/20">
                                <i class="fa-solid fa-file-pdf mr-1.5 text-sm"></i> Open PDF Report
                            </a>
                        </div>
                    </div>

                    <!-- Dynamic Report Content -->
                    <div id="reportContent" class="flex-1 space-y-4 text-xs text-slate-200">
                        <div class="p-8 bg-slate-900/60 border border-dashed border-slate-700 rounded-2xl text-center text-slate-400">
                            <i class="fa-solid fa-file-waveform text-4xl mb-3 text-slate-600"></i>
                            <p class="font-medium text-slate-300">No diagnostic imaging report active.</p>
                            <p class="text-[11px] text-slate-500 mt-1">Select any patient from the directory above or type in the chat (e.g. <code class="text-blue-400 bg-slate-800 px-1 py-0.5 rounded">"Open Vikram Malhotra's CBCT"</code>) to load findings.</p>
                        </div>
                    </div>
                </div>

            </div>

        </div>

        <!-- =================================================================== -->
        <!-- TAB 2: CLINICAL DICTATION STUDIO (FEATURE B) -->
        <!-- =================================================================== -->
        <div id="tabContent-dictation" class="hidden flex-col space-y-6">
            
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                <div class="flex flex-wrap items-center justify-between border-b border-slate-700 pb-4 mb-4 gap-2">
                    <div class="flex items-center space-x-3">
                        <div class="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">
                            <i class="fa-solid fa-stethoscope text-base"></i>
                        </div>
                        <div>
                            <h2 class="font-bold text-white text-base">Feature B: Clinical Speech & Audio Dictation Studio</h2>
                            <p class="text-xs text-slate-400">Normalizes dental tooth numbering (FDI, Palmer, Universal), labels uncertainty, and generates signed drafts.</p>
                        </div>
                    </div>
                    <span class="text-xs font-semibold text-emerald-300 bg-emerald-500/20 border border-emerald-500/40 px-3 py-1 rounded-full flex items-center">
                        <i class="fa-solid fa-shield-halved mr-1.5"></i> Anti-Hallucination Guardrail Active
                    </span>
                </div>

                <!-- Dictation Input Section -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <label class="text-xs font-bold uppercase tracking-wider text-slate-300">Clinician Voice Dictation Audio Transcript:</label>
                        <button onclick="toggleVoiceRecord('dictationInput', 'dictationMicBtn')" id="dictationMicBtn" class="text-xs bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-3 py-1.5 rounded-xl transition flex items-center shadow-md">
                            <i class="fa-solid fa-microphone mr-1.5"></i> Click & Speak Dictation
                        </button>
                    </div>
                    <textarea id="dictationInput" rows="3" 
                              class="w-full text-xs bg-slate-900 border border-slate-700 text-slate-100 rounded-xl p-3.5 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-mono"
                              placeholder="e.g. Lower right third molar is horizontally impacted with roots approximating the canal. UR5 shows apical radiolucency 8 by 7 mm."></textarea>
                    
                    <!-- Pre-Built Audio Dictation Templates -->
                    <div class="flex flex-wrap items-center justify-between gap-2 pt-1">
                        <div class="flex flex-wrap items-center gap-1.5">
                            <span class="text-[11px] text-slate-400 mr-1">Clinical Templates:</span>
                            <button onclick="insertSampleDictation(1)" class="text-[11px] bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1.5 rounded-lg transition border border-slate-600">
                                📋 Mandibular Molar Impaction
                            </button>
                            <button onclick="insertSampleDictation(2)" class="text-[11px] bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1.5 rounded-lg transition border border-slate-600">
                                📋 Maxillary Premolar (UR5, UR4)
                            </button>
                            <button onclick="insertSampleDictation(3)" class="text-[11px] bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1.5 rounded-lg transition border border-slate-600">
                                📋 Universal Notation #29 & TMJ
                            </button>
                        </div>
                        <button onclick="processDictation()" id="synthesizeBtn" class="text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white px-5 py-2.5 rounded-xl font-bold shadow-lg shadow-emerald-600/30 transition flex items-center">
                            <i class="fa-solid fa-wand-magic-sparkles mr-2 text-sm"></i> Synthesize Structured Report & Generate PDF
                        </button>
                    </div>
                </div>

                <!-- Structured Dictation Output Container -->
                <div id="dictationOutputContainer" class="mt-6 hidden border-t border-slate-700 pt-5 space-y-4">
                    
                    <!-- Mandatory Sign-Off Banner -->
                    <div class="bg-amber-500/10 border-l-4 border-amber-500 p-3.5 rounded-r-xl flex items-center justify-between">
                        <div>
                            <p class="text-xs font-bold text-amber-300 flex items-center">
                                <i class="fa-solid fa-triangle-exclamation mr-2"></i> MANDATORY CLINICIAN REVIEW REQUIRED
                            </p>
                            <p class="text-[11px] text-amber-200/80 mt-0.5">
                                AI synthesized draft from spoken transcript. Must be reviewed, verified, and signed by attending radiologist.
                            </p>
                        </div>
                        <div id="dictationPdfDownloadBtnContainer"></div>
                    </div>

                    <!-- Findings Summary Cards -->
                    <div id="dictationBadgesContainer" class="flex flex-wrap gap-2"></div>

                    <!-- 2 Columns: JSON and Readable Report -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-slate-900 border border-slate-700 rounded-xl p-3.5">
                            <div class="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
                                <span class="text-xs font-bold text-emerald-400">Validated Structured JSON (FHIR Schema)</span>
                                <span class="text-[10px] text-slate-500 font-mono">Normalized Teeth</span>
                            </div>
                            <pre id="jsonOutput" class="text-[11px] text-emerald-300 font-mono max-h-64 overflow-y-auto custom-scroll"></pre>
                        </div>
                        <div class="bg-slate-900 border border-slate-700 rounded-xl p-3.5">
                            <div class="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
                                <span class="text-xs font-bold text-slate-200">Readable Radiology Draft Narrative</span>
                                <span class="text-[10px] text-slate-500 font-mono">Formatted Draft</span>
                            </div>
                            <div id="narrativeOutput" class="text-xs text-slate-300 whitespace-pre-wrap font-sans max-h-64 overflow-y-auto custom-scroll"></div>
                        </div>
                    </div>
                </div>

            </div>

        </div>

        <!-- =================================================================== -->
        <!-- TAB 3: MULTI-AGENT & MCP INSPECTOR -->
        <!-- =================================================================== -->
        <div id="tabContent-mcp" class="hidden flex-col space-y-6">
            
            <!-- Agent Architecture Overview -->
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                <h2 class="font-bold text-white text-base mb-2 flex items-center">
                    <i class="fa-solid fa-diagram-project mr-2 text-indigo-400"></i> Multi-Agent Architecture & Tool Calling Flow
                </h2>
                <p class="text-xs text-slate-400 mb-6">How probabilistic LLM reasoning routes commands into deterministic clinical business logic.</p>

                <!-- Grid of 4 Specialized Agents -->
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    
                    <div class="bg-slate-900/80 border border-blue-500/40 rounded-xl p-4 shadow-sm">
                        <div class="w-8 h-8 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold mb-2">
                            <i class="fa-solid fa-compass"></i>
                        </div>
                        <h3 class="font-bold text-xs text-white">Triage Agent</h3>
                        <p class="text-[10px] text-blue-400 font-semibold mb-2">Orchestrator</p>
                        <p class="text-[11px] text-slate-400">Inspects input and routes requests to the correct sub-agent via standard MCP Tool Calling.</p>
                    </div>

                    <div class="bg-slate-900/80 border border-emerald-500/40 rounded-xl p-4 shadow-sm">
                        <div class="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold mb-2">
                            <i class="fa-solid fa-magnifying-glass"></i>
                        </div>
                        <h3 class="font-bold text-xs text-white">Retrieval Agent</h3>
                        <p class="text-[10px] text-emerald-400 font-semibold mb-2">MCP: search_patient_reports</p>
                        <p class="text-[11px] text-slate-400">Queries mock database. Halts execution if multiple matching patients exist to prevent misattribution.</p>
                    </div>

                    <div class="bg-slate-900/80 border border-purple-500/40 rounded-xl p-4 shadow-sm">
                        <div class="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold mb-2">
                            <i class="fa-solid fa-pen-fancy"></i>
                        </div>
                        <h3 class="font-bold text-xs text-white">Clinical Scribe Agent</h3>
                        <p class="text-[10px] text-purple-400 font-semibold mb-2">MCP: structure_clinical_dictation</p>
                        <p class="text-[11px] text-slate-400">Transforms speech transcripts into structured JSON findings with certainty labels and PDF drafts.</p>
                    </div>

                    <div class="bg-slate-900/80 border border-amber-500/40 rounded-xl p-4 shadow-sm">
                        <div class="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold mb-2">
                            <i class="fa-solid fa-comments"></i>
                        </div>
                        <h3 class="font-bold text-xs text-white">Consultation Agent</h3>
                        <p class="text-[10px] text-amber-400 font-semibold mb-2">MCP: summarize_active_report</p>
                        <p class="text-[11px] text-slate-400">Performs evidence-grounded Q&A over the active report context with verbatim source citations.</p>
                    </div>

                </div>

                <!-- MCP Tool Endpoints Table -->
                <div class="mt-8">
                    <h3 class="font-bold text-xs text-white uppercase tracking-wider mb-3">Model Context Protocol (JSON-RPC 2.0) Active Endpoints:</h3>
                    <div class="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden text-xs">
                        <table class="w-full text-left">
                            <thead class="bg-slate-800 text-slate-400 text-[10px] uppercase font-bold">
                                <tr>
                                    <th class="p-3">Tool Name</th>
                                    <th class="p-3">MCP Method</th>
                                    <th class="p-3">Input Parameters</th>
                                    <th class="p-3">Safety Gate Guarantee</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800 text-slate-300">
                                <tr>
                                    <td class="p-3 font-mono font-bold text-blue-400">search_patient_reports</td>
                                    <td class="p-3 font-mono text-slate-400">tools/call</td>
                                    <td class="p-3 font-mono text-slate-400">patient_name, modality</td>
                                    <td class="p-3 text-emerald-400">Pauses on ambiguity; refuses to guess patient identity.</td>
                                </tr>
                                <tr>
                                    <td class="p-3 font-mono font-bold text-emerald-400">structure_clinical_dictation</td>
                                    <td class="p-3 font-mono text-slate-400">tools/call</td>
                                    <td class="p-3 font-mono text-slate-400">dictation_text, patient_id</td>
                                    <td class="p-3 text-emerald-400">Anti-hallucination verification; labels diagnostic certainty.</td>
                                </tr>
                                <tr>
                                    <td class="p-3 font-mono font-bold text-amber-400">summarize_active_report</td>
                                    <td class="p-3 font-mono text-slate-400">tools/call</td>
                                    <td class="p-3 font-mono text-slate-400">report_id, query</td>
                                    <td class="p-3 text-emerald-400">Direct evidence grounding; cites verbatim report text.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div>

    </main>

    <!-- UI JAVASCRIPT LOGIC -->
    <script>
        const API_BASE = "";
        let currentSessionId = "viscan_ui_session";

        function switchTab(tabId) {
            ['assistant', 'dictation', 'mcp'].forEach(t => {
                document.getElementById(`tabContent-${t}`).classList.add('hidden');
                document.getElementById(`tabBtn-${t}`).className = "px-4 py-3 text-xs sm:text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center transition";
            });
            document.getElementById(`tabContent-${tabId}`).classList.remove('hidden');
            document.getElementById(`tabBtn-${tabId}`).className = "px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 border-blue-500 text-blue-400 flex items-center transition";
        }

        function runSamplePrompt(text) {
            document.getElementById('chatInput').value = text;
            sendChat();
        }

        function triggerSelectedPatientRetrieval() {
            const sel = document.getElementById("patientDirectorySelect");
            const name = sel ? sel.value : "Priya Sharma";
            runSamplePrompt(`Open ${name}'s latest report`);
        }

        function triggerAction(actionType) {
            const sel = document.getElementById("patientDirectorySelect");
            const name = sel ? sel.value : "Priya Sharma";
            if (actionType === 'opg') {
                runSamplePrompt(`Open ${name}'s latest OPG report`);
            } else if (actionType === 'cbct') {
                runSamplePrompt(`Open ${name}'s latest CBCT report`);
            } else if (actionType === 'summarize') {
                runSamplePrompt(`Summarize her report`);
            }
        }

        async function sendChat(customQuery = null) {
            const input = document.getElementById("chatInput");
            const query = customQuery || input.value.trim();
            if (!query) return;

            appendMessage("user", query);
            if (!customQuery) input.value = "";

            try {
                const resp = await fetch(`${API_BASE}/assistant/chat`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: query, session_id: currentSessionId })
                });
                const data = await resp.json();

                // Display Agent Trace Badge if available
                let agentBadgeHtml = "";
                if (data.agent_trace) {
                    agentBadgeHtml = `
                        <div class="mb-2 pb-1.5 border-b border-slate-700 flex items-center justify-between text-[10px] text-blue-400 font-mono">
                            <span><i class="fa-solid fa-microchip mr-1"></i> Agent: <strong>${data.agent_trace.active_agent}</strong></span>
                            <span class="bg-blue-500/20 px-1.5 py-0.5 rounded border border-blue-500/30">MCP Tool: ${data.agent_trace.tool_called}</span>
                        </div>
                    `;
                }

                // CASE 1: Clarification Required (Feature A Ambiguity Safety Check)
                if (data.status === "clarification_required" && data.candidates) {
                    let html = `${agentBadgeHtml}<p class="font-semibold text-amber-300 text-xs mb-2"><i class="fa-solid fa-triangle-exclamation mr-1"></i> ${data.message}</p>`;
                    html += `<div class="space-y-1.5 mt-2">`;
                    data.candidates.forEach(c => {
                        html += `
                            <div class="flex items-center justify-between p-2.5 bg-slate-800 hover:bg-slate-700/80 border border-slate-700 rounded-xl transition">
                                <div>
                                    <span class="font-bold text-white text-xs">${c.patient_name}</span>
                                    <span class="text-slate-400 text-[11px] ml-1.5">(DOB: ${c.date_of_birth} | ${c.sex})</span>
                                </div>
                                <button onclick="sendChat('Open ${c.patient_name}\\'s latest report')" 
                                        class="text-[11px] bg-blue-600 hover:bg-blue-500 text-white font-semibold px-2.5 py-1 rounded-lg transition shadow-sm">
                                    Select Patient
                                </button>
                            </div>
                        `;
                    });
                    html += `</div>`;
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-amber-500/40");
                    return;
                }

                // CASE 2: Report Successfully Retrieved
                if (data.status === "success" && data.report) {
                    renderReport(data.report, data.patient);
                    updateContextBanner(data.patient, data.report);
                    let html = `${agentBadgeHtml}<p class="text-xs text-emerald-300 font-medium"><i class="fa-solid fa-circle-check mr-1 text-emerald-400"></i> ${data.message}</p>`;
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-emerald-500/40");
                    return;
                }

                // CASE 3: Conversational Answer (Feature C & Bonus Q&A)
                if (data.answer) {
                    let html = `${agentBadgeHtml}<p class="whitespace-pre-line text-xs text-slate-200">${data.answer}</p>`;
                    if (data.citations && data.citations.length > 0) {
                        html += `<div class="mt-3 pt-2.5 border-t border-slate-700 text-[11px] text-slate-400">`;
                        html += `<span class="font-bold text-slate-300 flex items-center mb-1"><i class="fa-solid fa-quote-left mr-1.5 text-blue-400"></i> Source Evidence Citations:</span>`;
                        data.citations.forEach(cit => {
                            html += `<div class="mt-1 bg-slate-800/80 p-2 rounded-lg border border-slate-700 font-mono text-[10px] text-slate-300">• ${cit}</div>`;
                        });
                        html += `</div>`;
                    }
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-slate-600");
                    return;
                }

                // Default Message
                appendMessage("assistant", data.message || "Request completed.");
            } catch (err) {
                appendMessage("assistant", "Error connecting to Viscan API.");
            }
        }

        function appendMessage(sender, text) {
            const container = document.getElementById("chatMessages");
            const div = document.createElement("div");
            if (sender === "user") {
                div.className = "bg-blue-600 text-white rounded-2xl p-3 max-w-[85%] ml-auto text-xs shadow-md";
                div.innerHTML = `<p class="font-bold text-[10px] text-blue-200 mb-0.5">Clinician</p>${text}`;
            } else {
                div.className = "bg-slate-700/80 border border-slate-600 text-slate-100 rounded-2xl p-3 max-w-[85%] text-xs shadow-md";
                div.innerHTML = `<p class="font-bold text-[10px] text-slate-400 mb-0.5">Viscan Assistant</p>${text}`;
            }
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function appendCustomMessage(sender, html, classes) {
            const container = document.getElementById("chatMessages");
            const div = document.createElement("div");
            div.className = `rounded-2xl p-3.5 max-w-[90%] text-xs shadow-md ${classes}`;
            div.innerHTML = html;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function updateContextBanner(patient, report) {
            const banner = document.getElementById("activeContextBanner");
            const text = document.getElementById("activeContextText");
            if (patient && report) {
                text.innerText = `${patient.patient_name} (${report.report_type})`;
                banner.classList.remove("hidden");
            } else {
                text.innerText = "None Selected";
            }
        }

        function renderReport(r, p) {
            document.getElementById("reportModalityBadge").innerText = r.report_type;
            document.getElementById("reportTitle").innerText = `${r.report_type} Diagnostic Report`;
            document.getElementById("reportPatientMeta").innerText = `Patient: ${p ? p.patient_name : 'Verified'} (${r.patient_id}) • Date: ${r.report_date} • Report ID: ${r.report_id}`;
            
            let html = `
                <div class="bg-blue-500/10 p-3 rounded-xl border border-blue-500/30 text-xs">
                    <span class="font-bold text-blue-300">Clinical Indication:</span> ${r.clinical_indication || 'Baseline screening'}
                </div>
                <div class="space-y-2 mt-3">
                    <h4 class="font-bold text-xs text-white uppercase tracking-wider">Key Findings:</h4>
                    <div class="space-y-2">
            `;
            r.findings.forEach(f => {
                html += `
                    <div class="bg-slate-900/80 p-2.5 rounded-xl border border-slate-700 text-xs">
                        <div class="flex justify-between font-semibold text-slate-200">
                            <span>${f.region}</span>
                            <span class="text-[10px] text-blue-300 bg-blue-500/20 px-2 py-0.5 rounded border border-blue-500/30">${f.certainty || 'Confirmed'}</span>
                        </div>
                        <p class="text-slate-400 mt-1">${f.observation}</p>
                    </div>
                `;
            });
            html += `
                </div>
                <div class="mt-4 pt-3 border-t border-slate-700">
                    <h4 class="font-bold text-xs text-white uppercase tracking-wider mb-1.5">Radiologist Impression:</h4>
                    <ul class="list-disc list-inside text-xs text-slate-300 space-y-1">
            `;
            r.impression.forEach(imp => {
                html += `<li>${imp}</li>`;
            });
            html += `
                    </ul>
                </div>
                <div class="mt-3 p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-xs text-emerald-300">
                    <span class="font-bold">Recommendation:</span> ${r.recommendation || 'Clinical correlation recommended.'}
                </div>
            `;
            document.getElementById("reportContent").innerHTML = html;

            if (r.pdf_filename) {
                document.getElementById("pdfDownloadContainer").classList.remove("hidden");
                document.getElementById("pdfDownloadBtn").href = `/patient-pdf/${r.pdf_filename}`;
            }
        }

        async function processDictation() {
            const text = document.getElementById("dictationInput").value.trim();
            if (!text) return;

            const btn = document.getElementById("synthesizeBtn");
            btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i> Structuring via Llama 3.1...`;

            try {
                const resp = await fetch(`${API_BASE}/dictation/structure`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text })
                });
                const data = await resp.json();

                document.getElementById("dictationOutputContainer").classList.remove("hidden");
                document.getElementById("jsonOutput").innerText = JSON.stringify(data, null, 2);
                document.getElementById("narrativeOutput").innerText = data.narrative_draft;

                // Render Findings Badges
                let badgesHtml = "";
                if (data.anatomical_sites) {
                    data.anatomical_sites.forEach(s => {
                        badgesHtml += `
                            <span class="inline-flex items-center text-xs bg-blue-500/20 border border-blue-500/40 text-blue-300 px-2.5 py-1 rounded-lg">
                                <i class="fa-solid fa-tooth mr-1.5"></i> ${s.anatomical_name} ${s.fdi_number ? `(FDI ${s.fdi_number})` : ''}
                            </span>
                        `;
                    });
                }
                if (data.safety_guardrails) {
                    badgesHtml += `
                        <span class="inline-flex items-center text-xs bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 px-2.5 py-1 rounded-lg">
                            <i class="fa-solid fa-shield-check mr-1.5"></i> ${data.safety_guardrails.status}: ${data.safety_guardrails.grounded_ratio} Claims Grounded
                        </span>
                    `;
                }
                document.getElementById("dictationBadgesContainer").innerHTML = badgesHtml;

                // PDF Download Button
                if (data.dictation_pdf_path) {
                    const filename = data.dictation_pdf_path.split("/").pop();
                    document.getElementById("dictationPdfDownloadBtnContainer").innerHTML = `
                        <a href="/dictation-pdf/${filename}" target="_blank" class="text-xs bg-amber-500 hover:bg-amber-400 text-slate-900 font-bold px-3 py-1.5 rounded-lg transition shadow">
                            <i class="fa-solid fa-file-pdf mr-1"></i> Download PDF Draft
                        </a>
                    `;
                }
            } catch (err) {
                alert("Error structuring dictation.");
            } finally {
                btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles mr-2 text-sm"></i> Synthesize Structured Report & Generate PDF`;
            }
        }

        function insertSampleDictation(type) {
            const input = document.getElementById("dictationInput");
            if (type === 1) {
                input.value = "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal. Mild pericoronal radiolucency noted around the crown.";
            } else if (type === 2) {
                input.value = "UR5 is root treated with a unilocular well defined apical radiolucency 8 by 7 by 6 mm. UR4 is root treated with an apical radiolucency 1 to 2 mm wide. Distal to UR8 is an unerupted supernumerary tooth with an enlarged follicle space, possibly a dentigerous cyst.";
            } else if (type === 3) {
                input.value = "Tooth #29 shows apical rarefying and sclerosing osteitis. TMJs show bilateral condylar flattening consistent with degenerative joint changes. Left palatine tonsillolith noted incidentally.";
            }
        }

        async function resetSession() {
            await fetch(`${API_BASE}/sessions/${currentSessionId}/reset`, { method: "POST" });
            updateContextBanner(null, null);
            document.getElementById("chatMessages").innerHTML = `
                <div class="bg-slate-700/50 border border-slate-600 rounded-2xl p-3 text-slate-300 text-xs">
                    <p class="font-bold text-blue-400 mb-1"><i class="fa-solid fa-rotate-left mr-1"></i> Context Cleared</p>
                    Session memory reset. You can request any new patient scan.
                </div>
            `;
            document.getElementById("reportModalityBadge").innerText = "NO SCAN";
            document.getElementById("reportTitle").innerText = "Radiology Report Viewer";
            document.getElementById("reportPatientMeta").innerText = "Patient: None selected • Report ID: -";
            document.getElementById("reportContent").innerHTML = `
                <div class="p-8 bg-slate-900/60 border border-dashed border-slate-700 rounded-2xl text-center text-slate-400">
                    <i class="fa-solid fa-file-waveform text-4xl mb-3 text-slate-600"></i>
                    <p class="font-medium text-slate-300">No diagnostic imaging report active.</p>
                </div>
            `;
            document.getElementById("pdfDownloadContainer").classList.add("hidden");
        }

        let activeRecognition = null;
        let isListening = false;

        function toggleVoiceRecord(targetInputId, btnId) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                alert("Speech Recognition is supported in Chrome, Edge, and Safari. Please use a compatible browser.");
                return;
            }

            const input = document.getElementById(targetInputId);
            const btn = document.getElementById(btnId);

            if (isListening && activeRecognition) {
                activeRecognition.stop();
                isListening = false;
                btn.classList.remove("bg-rose-600", "text-white", "animate-pulse");
                return;
            }

            try {
                const recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = "en-US";

                btn.classList.add("bg-rose-600", "text-white", "animate-pulse");
                isListening = true;
                activeRecognition = recognition;

                recognition.onresult = (event) => {
                    let transcript = "";
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        transcript += event.results[i][0].transcript;
                    }
                    if (transcript) {
                        input.value = transcript;
                    }
                };

                recognition.onerror = (event) => {
                    console.warn("Speech recognition notice:", event.error);
                    isListening = false;
                    btn.classList.remove("bg-rose-600", "text-white", "animate-pulse");
                    if (event.error === 'not-allowed') {
                        alert("Microphone permission was denied. Please allow microphone access in your browser address bar.");
                    }
                };

                recognition.onend = () => {
                    isListening = false;
                    btn.classList.remove("bg-rose-600", "text-white", "animate-pulse");
                    // If in main chat and we captured text, send it directly to the agent!
                    if (targetInputId === "chatInput" && input.value.trim()) {
                        sendChat();
                    }
                };

                recognition.start();
            } catch (err) {
                console.error("Speech recognition error:", err);
                isListening = false;
                btn.classList.remove("bg-rose-600", "text-white", "animate-pulse");
            }
        }
    </script>
</body>
</html>
"""
