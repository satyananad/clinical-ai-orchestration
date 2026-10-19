import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .api import patients, reports, assistant, dictation, mcp

app = FastAPI(
    title="Dental Radiology & Clinical Workstation API",
    description="Clinical multi-agent system with MCP tool execution for radiology retrieval, clinical dictation, and evidence-grounded consultation.",
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
        "service": "dental-clinical-workstation-api",
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
    """Serves the modernized interactive Clinical Dental Workstation UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dental Clinical Workstation | Radiology & Charting</title>
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
    <header class="bg-slate-800/95 backdrop-blur border-b border-slate-700/80 sticky top-0 z-50 shadow-md">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-blue-600 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-sky-600/20">
                    <i class="fa-solid fa-hospital-user"></i>
                </div>
                <div>
                    <div class="flex items-center space-x-2">
                        <h1 class="font-bold text-base text-white tracking-tight leading-none">Dental Clinical Workstation</h1>
                        <span class="text-[10px] bg-sky-500/10 text-sky-300 font-medium px-2 py-0.5 rounded border border-sky-500/20">Clinical EHR</span>
                    </div>
                    <p class="text-[11px] text-slate-400 mt-0.5">Radiology Records & Clinical Dictation</p>
                </div>
            </div>

            <!-- Active Session Context Pill -->
            <div id="activeContextBanner" class="hidden sm:flex items-center space-x-2 bg-slate-900/90 border border-slate-700 px-3.5 py-1.5 rounded-full text-xs shadow-sm">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span class="text-slate-400 font-medium">Active Study:</span>
                <span id="activeContextText" class="font-semibold text-emerald-300">None Selected</span>
                <button onclick="resetSession()" class="text-slate-400 hover:text-rose-400 ml-1.5 text-xs transition" title="Clear Active Study Context">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>

        </div>
    </header>

    <!-- Workspace Tabs Navigation -->
    <div class="bg-slate-800 border-b border-slate-700">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 sm:space-x-4">
            <button onclick="switchTab('assistant')" id="tabBtn-assistant" class="px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 border-sky-500 text-sky-400 flex items-center transition">
                <i class="fa-solid fa-x-ray mr-2"></i> Patient Records & Imaging
            </button>
            <button onclick="switchTab('dictation')" id="tabBtn-dictation" class="px-4 py-3 text-xs sm:text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center transition">
                <i class="fa-solid fa-microphone-lines mr-2"></i> Clinical Dictation & Notes
            </button>
        </div>
    </div>

    <!-- MAIN CONTAINER -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">

        <!-- =================================================================== -->
        <!-- TAB 1: PATIENT RECORDS & RADIOLOGY IMAGING -->
        <!-- =================================================================== -->
        <!-- =================================================================== -->
        <!-- TAB 1: PATIENT RECORDS & RADIOLOGY IMAGING -->
        <!-- =================================================================== -->
        <div id="tabContent-assistant" class="space-y-6">

            <!-- 1. CLINIC PATIENT DIRECTORY (Mandatory: Always Visible) -->
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-lg">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-3 border-b border-slate-700/80 gap-2">
                    <div class="flex items-center space-x-2.5">
                        <div class="w-8 h-8 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center text-sm font-bold">
                            <i class="fa-solid fa-users"></i>
                        </div>
                        <div>
                            <div class="flex items-center space-x-2">
                                <h2 class="font-bold text-white text-xs sm:text-sm">Clinic Patient Directory</h2>
                                <span id="patientDirectoryCount" class="text-[10px] bg-slate-700 text-sky-300 px-2 py-0.5 rounded-full font-mono">Loading...</span>
                            </div>
                            <p class="text-[11px] text-slate-400 mt-0.5">Click any registered patient to load their diagnostic record and view all associated imaging reports.</p>
                        </div>
                    </div>
                    <div class="relative w-full sm:w-72">
                        <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-xs text-slate-500"></i>
                        <input id="patientSearchInput" type="text" placeholder="Search patients by name or ID..."
                               oninput="filterPatientDirectory(this.value)"
                               class="w-full text-xs bg-slate-900 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-xl pl-8 pr-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-sky-500">
                    </div>
                </div>

                <!-- Patient Cards List Container -->
                <div id="patientDirectoryGrid" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2.5 max-h-48 overflow-y-auto custom-scroll pr-1">
                    <div class="p-3 text-center text-slate-500 text-xs col-span-full">
                        <i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Loading patient directory...
                    </div>
                </div>
            </div>

            <!-- 2. SELECTED PATIENT & ASSOCIATED REPORTS PANEL -->
            <div id="selectedPatientSection" class="hidden bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-lg space-y-4">
                <div class="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-700">
                    <div class="flex items-center space-x-3">
                        <span class="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-sm">
                            <i class="fa-solid fa-id-card"></i>
                        </span>
                        <div>
                            <div class="flex items-center space-x-2">
                                <h3 id="selectedPatientName" class="font-bold text-white text-base">Patient Name</h3>
                                <span id="selectedPatientIdBadge" class="text-[11px] bg-emerald-500/10 text-emerald-300 font-mono px-2.5 py-0.5 rounded-lg border border-emerald-500/20">PAT-XXXX</span>
                                <span id="selectedPatientRecordBadge" class="text-[10px] bg-slate-700 text-slate-300 px-2 py-0.5 rounded">MRN</span>
                            </div>
                            <p id="selectedPatientDemographics" class="text-xs text-slate-400 mt-0.5">DOB: - • Gender: - • Status: Active</p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span id="selectedPatientReportCount" class="text-xs bg-slate-900 text-slate-300 px-3 py-1.5 rounded-xl border border-slate-700">
                            0 Reports Available
                        </span>
                        <button onclick="clearPatientSelection()" class="text-xs text-slate-400 hover:text-rose-400 px-2 py-1 rounded transition" title="Deselect Patient">
                            <i class="fa-solid fa-xmark mr-1"></i> Deselect
                        </button>
                    </div>
                </div>

                <!-- Associated Reports Grid (Report Cards) -->
                <div>
                    <div class="flex items-center justify-between mb-2">
                        <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center">
                            <i class="fa-solid fa-folder-tree mr-1.5 text-sky-400"></i> Associated Diagnostic Reports (Select to Open):
                        </h4>
                        <span class="text-[10px] text-slate-500">Patient selection kept separate from report selection</span>
                    </div>
                    <div id="associatedReportsGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        <!-- Populated dynamically with Report Cards -->
                    </div>
                </div>
            </div>

            <!-- 3. SPLIT WORKSPACE: Assistant Console & Active Study Viewer -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

                <!-- Left 5 Columns: Clinical Assistant & Voice/Text Search -->
                <div class="lg:col-span-5 flex flex-col space-y-4">

                    <!-- Clinical Assistant Terminal Card -->
                    <div class="bg-slate-800 border border-slate-700 rounded-2xl shadow-lg flex flex-col h-[540px]">
                        <div class="p-3 border-b border-slate-700 bg-slate-800/95 rounded-t-2xl flex items-center justify-between">
                            <div class="flex items-center space-x-2">
                                <i class="fa-solid fa-stethoscope text-sky-400 text-sm"></i>
                                <span class="font-semibold text-xs text-white">Clinical Assistant Console</span>
                            </div>
                            <button onclick="resetSession()" class="text-[11px] text-slate-400 hover:text-rose-400 transition flex items-center">
                                <i class="fa-solid fa-rotate-left mr-1"></i> Reset
                            </button>
                        </div>

                        <!-- Chat Container -->
                        <div id="chatMessages" class="flex-1 p-3.5 overflow-y-auto space-y-2.5 custom-scroll text-xs">
                            <!-- Initial Welcome Message -->
                            <div class="bg-slate-700/50 border border-slate-600/80 rounded-2xl p-3 text-slate-200">
                                <p class="font-bold text-sky-400 text-xs mb-1 flex items-center">
                                    <i class="fa-solid fa-user-doctor mr-1.5"></i> Clinical Assistant Ready
                                </p>
                                <p class="mb-2 text-[11px]">Select a patient above, click a report card, or ask natural language clinical questions:</p>
                                <div class="flex flex-wrap gap-1 pt-1">
                                    <button onclick="sendChat('Open Vikram Malhotra\'s CBCT')" class="text-[10px] bg-slate-800 hover:bg-slate-700 text-sky-300 border border-slate-600 px-2 py-0.5 rounded transition">
                                        Open Vikram's CBCT
                                    </button>
                                    <button onclick="sendChat('Summarize her report')" class="text-[10px] bg-slate-800 hover:bg-slate-700 text-amber-300 border border-slate-600 px-2 py-0.5 rounded transition">
                                        Summarize Report
                                    </button>
                                    <button onclick="sendChat('What are the findings for tooth 48?')" class="text-[10px] bg-slate-800 hover:bg-slate-700 text-teal-300 border border-slate-600 px-2 py-0.5 rounded transition">
                                        Tooth 48 Findings
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- Input Controls -->
                        <div class="p-3 border-t border-slate-700 bg-slate-800/95 rounded-b-2xl">
                            <div class="flex items-center space-x-2">
                                <input id="chatInput" type="text" placeholder="Type query or question (e.g. 'Summarize her report')..." 
                                       onkeydown="if(event.key==='Enter') sendChat()"
                                       class="flex-1 text-xs bg-slate-900 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-sky-500">
                                <button onclick="toggleVoiceRecord('chatInput', 'voiceBtn')" id="voiceBtn" class="bg-slate-700 hover:bg-slate-600 text-slate-200 w-9 h-9 rounded-xl flex items-center justify-center transition shadow-sm" title="Click to Speak Voice Command">
                                    <i class="fa-solid fa-microphone text-xs"></i>
                                </button>
                                <button onclick="sendChat()" class="bg-sky-600 hover:bg-sky-500 text-white w-9 h-9 rounded-xl flex items-center justify-center shadow-lg shadow-sky-600/30 transition">
                                    <i class="fa-solid fa-paper-plane text-xs"></i>
                                </button>
                            </div>
                        </div>
                    </div>

                </div>

                <!-- Right 7 Columns: Active Radiology Study Viewer -->
                <div class="lg:col-span-7 flex flex-col space-y-4">
                    
                    <!-- Report Card -->
                    <div class="bg-slate-800 border border-slate-700 rounded-2xl p-5 shadow-lg flex flex-col min-h-[540px]">
                        <div class="flex items-center justify-between border-b border-slate-700 pb-3 mb-4">
                            <div>
                                <div class="flex items-center space-x-2">
                                    <span id="reportModalityBadge" class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-sky-500/20 text-sky-300 border border-sky-500/30">NO STUDY LOADED</span>
                                    <h2 id="reportTitle" class="font-bold text-white text-base">Diagnostic Radiology Report</h2>
                                </div>
                                <p id="reportPatientMeta" class="text-xs text-slate-400 mt-1">Patient: None selected • Study ID: -</p>
                            </div>
                            <div id="pdfDownloadContainer" class="hidden">
                                <a id="pdfDownloadBtn" href="#" target="_blank" class="inline-flex items-center text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white px-3.5 py-2 rounded-xl transition shadow-lg shadow-rose-600/20">
                                    <i class="fa-solid fa-file-pdf mr-1.5 text-sm"></i> View Diagnostic PDF
                                </a>
                            </div>
                        </div>

                        <!-- Dynamic Report Content -->
                        <div id="reportContent" class="flex-1 space-y-4 text-xs text-slate-200">
                            <div class="p-8 bg-slate-900/60 border border-dashed border-slate-700 rounded-2xl text-center text-slate-400">
                                <i class="fa-solid fa-file-waveform text-4xl mb-3 text-slate-600"></i>
                                <p class="font-medium text-slate-300">No diagnostic imaging study currently active.</p>
                                <p class="text-[11px] text-slate-500 mt-1">Select a patient above and click <code class="text-sky-400 bg-slate-800 px-1 py-0.5 rounded">"Open Report"</code>, or ask the assistant.</p>
                            </div>
                        </div>
                    </div>

                </div>

            </div>

        </div>

        <!-- =================================================================== -->
        <!-- TAB 2: CLINICAL DICTATION & NOTES -->
        <!-- =================================================================== -->
        <div id="tabContent-dictation" class="hidden flex-col space-y-6">
            
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                <div class="flex flex-wrap items-center justify-between border-b border-slate-700 pb-4 mb-4 gap-2">
                    <div class="flex items-center space-x-3">
                        <div class="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">
                            <i class="fa-solid fa-stethoscope text-base"></i>
                        </div>
                        <div>
                            <h2 class="font-bold text-white text-base">Clinical Speech & Dictation</h2>
                            <p class="text-xs text-slate-400">Dictate examination findings to produce verified structured reports with standardized tooth notation.</p>
                        </div>
                    </div>
                    <span class="text-xs font-semibold text-emerald-300 bg-emerald-500/20 border border-emerald-500/40 px-3 py-1 rounded-full flex items-center">
                        <i class="fa-solid fa-shield-halved mr-1.5"></i> Clinical Verification Active
                    </span>
                </div>

                <!-- Dictation Input Section -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <label class="text-xs font-bold uppercase tracking-wider text-slate-300">Clinical Voice or Written Dictation:</label>
                        <button onclick="toggleVoiceRecord('dictationInput', 'dictationMicBtn')" id="dictationMicBtn" class="text-xs bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-3 py-1.5 rounded-xl transition flex items-center shadow-md">
                            <i class="fa-solid fa-microphone mr-1.5"></i> Click & Speak Dictation
                        </button>
                    </div>
                    <textarea id="dictationInput" rows="3" 
                              class="w-full text-xs bg-slate-900 border border-slate-700 text-slate-100 rounded-xl p-3.5 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-mono"
                              placeholder="e.g. Lower right third molar is horizontally impacted with roots approximating the canal. UR5 shows apical radiolucency 8 by 7 mm."></textarea>
                    
                    <div class="flex items-center justify-end pt-2">
                        <button onclick="processDictation()" id="synthesizeBtn" class="text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white px-5 py-2.5 rounded-xl font-bold shadow-lg shadow-emerald-600/30 transition flex items-center">
                            <i class="fa-solid fa-file-circle-check mr-2 text-sm"></i> Generate Verified Clinical Report & PDF
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
                                Generated clinical draft from dictation. Must be verified and signed by attending clinician prior to permanent record entry.
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
                                <span class="text-xs font-bold text-emerald-400">Structured Clinical Findings (FHIR Schema)</span>
                                <span class="text-[10px] text-slate-500 font-mono">Standardized Notation</span>
                            </div>
                            <pre id="jsonOutput" class="text-[11px] text-emerald-300 font-mono max-h-64 overflow-y-auto custom-scroll"></pre>
                        </div>
                        <div class="bg-slate-900 border border-slate-700 rounded-xl p-3.5">
                            <div class="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
                                <span class="text-xs font-bold text-slate-200">Radiology Narrative Draft</span>
                                <span class="text-[10px] text-slate-500 font-mono">Formatted Draft</span>
                            </div>
                            <div id="narrativeOutput" class="text-xs text-slate-300 whitespace-pre-wrap font-sans max-h-64 overflow-y-auto custom-scroll"></div>
                        </div>
                    </div>
                </div>

            </div>

        </div>

    </main>

    <!-- UI JAVASCRIPT LOGIC -->
    <script>
        const API_BASE = "";
        let currentSessionId = "viscan_ui_session";

        let allClinicPatients = [];
        let activeSelectedPatient = null;
        let activeSelectedReport = null;

        async function loadPatientDirectory() {
            const grid = document.getElementById("patientDirectoryGrid");
            const countBadge = document.getElementById("patientDirectoryCount");
            try {
                const resp = await fetch(`${API_BASE}/patients`);
                allClinicPatients = await resp.json();
                if (countBadge) countBadge.innerText = `${allClinicPatients.length} Patients Registered`;
                renderPatientDirectory(allClinicPatients);
            } catch (err) {
                console.error("Failed to load patient directory:", err);
                if (grid) grid.innerHTML = `<div class="p-3 text-center text-rose-400 text-xs col-span-full">Unable to load patient directory.</div>`;
            }
        }

        function filterPatientDirectory(query) {
            const q = (query || "").trim().toLowerCase();
            if (!q) {
                renderPatientDirectory(allClinicPatients);
                return;
            }
            const filtered = allClinicPatients.filter(p => 
                p.patient_name.toLowerCase().includes(q) || 
                p.patient_id.toLowerCase().includes(q) ||
                (p.date_of_birth && p.date_of_birth.includes(q))
            );
            renderPatientDirectory(filtered);
        }

        function renderPatientDirectory(patients) {
            const grid = document.getElementById("patientDirectoryGrid");
            if (!grid) return;
            if (!patients || patients.length === 0) {
                grid.innerHTML = `<div class="p-3 text-center text-slate-500 text-xs col-span-full">No matching patients found in clinic records.</div>`;
                return;
            }
            grid.innerHTML = patients.map(p => {
                const isSelected = activeSelectedPatient && activeSelectedPatient.patient_id === p.patient_id;
                const reportCount = (p.reports || []).length;
                return `
                    <div onclick="selectPatientById('${p.patient_id}')" 
                         class="p-2.5 rounded-xl border cursor-pointer transition flex items-center justify-between ${isSelected ? 'bg-sky-950/60 border-sky-500 shadow-md shadow-sky-500/10' : 'bg-slate-900/80 hover:bg-slate-700/70 border-slate-700/80'}">
                        <div class="flex items-center space-x-2.5 overflow-hidden">
                            <div class="w-7 h-7 rounded-lg ${isSelected ? 'bg-sky-500 text-white' : 'bg-slate-800 text-slate-300'} flex-shrink-0 flex items-center justify-center font-bold text-xs">
                                ${p.patient_name.split(' ').map(n=>n[0]).join('').substring(0,2)}
                            </div>
                            <div class="truncate">
                                <div class="font-bold text-xs text-white truncate flex items-center space-x-1">
                                    <span>${p.patient_name}</span>
                                </div>
                                <div class="text-[10px] text-slate-400 font-mono mt-0.5">${p.patient_id} • ${p.date_of_birth || '-'}</div>
                            </div>
                        </div>
                        <div class="text-right flex-shrink-0 ml-2">
                            <span class="text-[10px] font-semibold ${isSelected ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' : 'bg-slate-800 text-slate-400 border border-slate-700'} px-2 py-0.5 rounded-full">
                                ${reportCount} ${reportCount === 1 ? 'report' : 'reports'}
                            </span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function selectPatientById(patientId, autoOpenReportId = null, shouldLogToChat = true) {
            try {
                const resp = await fetch(`${API_BASE}/sessions/${currentSessionId}/select-patient`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ patient_id: patientId })
                });
                const data = await resp.json();
                if (data.status === "success" && data.patient) {
                    activeSelectedPatient = data.patient;
                    renderPatientDirectory(allClinicPatients);
                    renderSelectedPatientSection(data.patient, data.reports || []);

                    if (!autoOpenReportId) {
                        updateContextBanner(data.patient, null);
                        if (shouldLogToChat) {
                            appendCustomMessage("assistant", `
                                <div class="text-xs text-slate-200">
                                    <div class="font-bold text-sky-400 flex items-center mb-1">
                                        <i class="fa-solid fa-user-check mr-1.5"></i> Patient Selected: ${data.patient.patient_name} (${data.patient.patient_id})
                                    </div>
                                    <p class="text-[11px] text-slate-300">Displaying ${data.reports.length} associated diagnostic reports on file. Click <strong>"Open Report"</strong> on any report card to load its radiological findings, or request via chat (e.g. <em>"Open her OPG"</em>).</p>
                                </div>
                            `, "bg-slate-700/60 border border-sky-500/40");
                        }
                    } else {
                        await openReportById(patientId, autoOpenReportId, false);
                    }
                }
            } catch (err) {
                console.error("Failed to select patient:", err);
            }
        }

        function clearPatientSelection() {
            activeSelectedPatient = null;
            activeSelectedReport = null;
            const sec = document.getElementById("selectedPatientSection");
            if (sec) sec.classList.add("hidden");
            renderPatientDirectory(allClinicPatients);
            updateContextBanner(null, null);
            document.getElementById("reportModalityBadge").innerText = "NO STUDY LOADED";
            document.getElementById("reportTitle").innerText = "Diagnostic Radiology Report";
            document.getElementById("reportPatientMeta").innerText = "Patient: None selected • Study ID: -";
            document.getElementById("reportContent").innerHTML = `
                <div class="p-8 bg-slate-900/60 border border-dashed border-slate-700 rounded-2xl text-center text-slate-400">
                    <i class="fa-solid fa-file-waveform text-4xl mb-3 text-slate-600"></i>
                    <p class="font-medium text-slate-300">No diagnostic imaging study currently active.</p>
                    <p class="text-[11px] text-slate-500 mt-1">Select a patient above and click <code class="text-sky-400 bg-slate-800 px-1 py-0.5 rounded">"Open Report"</code>, or ask the assistant.</p>
                </div>
            `;
            document.getElementById("pdfDownloadContainer").classList.add("hidden");
        }

        function renderSelectedPatientSection(patient, reports) {
            const section = document.getElementById("selectedPatientSection");
            if (!section) return;
            section.classList.remove("hidden");
            document.getElementById("selectedPatientName").innerText = patient.patient_name;
            document.getElementById("selectedPatientIdBadge").innerText = patient.patient_id;
            document.getElementById("selectedPatientRecordBadge").innerText = patient.record_number || "MRN-ACTIVE";
            document.getElementById("selectedPatientDemographics").innerText = `DOB: ${patient.date_of_birth || '-'} • Gender: ${patient.sex || '-'} • Status: ${patient.status || 'Active'}`;
            document.getElementById("selectedPatientReportCount").innerText = `${reports.length} ${reports.length === 1 ? 'Report Available' : 'Reports Available'}`;

            const reportsGrid = document.getElementById("associatedReportsGrid");
            if (!reports || reports.length === 0) {
                reportsGrid.innerHTML = `<div class="p-3 text-slate-500 text-xs col-span-full">No diagnostic reports on file for this patient.</div>`;
                return;
            }

            reportsGrid.innerHTML = reports.map(r => {
                const isOpen = activeSelectedReport && activeSelectedReport.report_id === r.report_id;
                const badgeColor = r.report_type === 'CBCT' ? 'bg-purple-500/20 text-purple-300 border-purple-500/30' :
                                   r.report_type === 'OPG' ? 'bg-sky-500/20 text-sky-300 border-sky-500/30' :
                                   r.report_type === 'Dental Examination' ? 'bg-teal-500/20 text-teal-300 border-teal-500/30' :
                                   'bg-amber-500/20 text-amber-300 border-amber-500/30';
                return `
                    <div class="bg-slate-900 border ${isOpen ? 'border-sky-500 ring-1 ring-sky-500/40 bg-sky-950/30' : 'border-slate-700/80 hover:border-slate-600'} rounded-xl p-3.5 flex flex-col justify-between shadow-sm transition">
                        <div>
                            <div class="flex items-center justify-between pb-1.5 mb-2 border-b border-slate-800">
                                <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border ${badgeColor}">
                                    ${r.report_type}
                                </span>
                                <span class="text-[10px] font-mono text-slate-400">
                                    <i class="fa-regular fa-calendar mr-1"></i>${r.report_date}
                                </span>
                            </div>
                            <div class="font-bold text-xs text-white mb-1 flex items-center justify-between">
                                <span>${r.report_id}</span>
                                <span class="text-[10px] text-emerald-400 font-semibold">${r.report_status || 'Final'}</span>
                            </div>
                            <p class="text-[11px] text-slate-400 line-clamp-2 leading-relaxed mb-3">
                                ${r.clinical_indication || 'Clinical radiological evaluation.'}
                            </p>
                        </div>
                        <div class="flex items-center space-x-2 pt-2 border-t border-slate-800/80">
                            <button onclick="openReportById('${patient.patient_id}', '${r.report_id}')"
                                    class="flex-1 text-[11px] font-bold ${isOpen ? 'bg-sky-500 text-white' : 'bg-sky-600 hover:bg-sky-500 text-white'} py-1.5 px-2.5 rounded-lg transition shadow-sm flex items-center justify-center">
                                <i class="fa-solid fa-folder-open mr-1.5"></i> ${isOpen ? 'Currently Open' : 'Open Report'}
                            </button>
                            ${r.pdf_filename ? `
                                <a href="/patient-pdf/${r.pdf_filename}" target="_blank"
                                   class="text-[11px] bg-slate-800 hover:bg-slate-700 text-rose-400 border border-slate-700 font-semibold py-1.5 px-2.5 rounded-lg transition flex items-center" title="View Diagnostic PDF">
                                    <i class="fa-solid fa-file-pdf"></i>
                                </a>
                            ` : ''}
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function openReportById(patientId, reportId, shouldLogToChat = true) {
            try {
                const resp = await fetch(`${API_BASE}/sessions/${currentSessionId}/select-report`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ patient_id: patientId, report_id: reportId })
                });
                const data = await resp.json();
                if (data.status === "success" && data.report) {
                    activeSelectedReport = data.report;
                    activeSelectedPatient = data.patient;
                    renderReport(data.report, data.patient);
                    updateContextBanner(data.patient, data.report);
                    renderSelectedPatientSection(data.patient, data.patient.reports || []);
                    renderPatientDirectory(allClinicPatients);

                    if (shouldLogToChat) {
                        appendCustomMessage("assistant", `
                            <div class="text-xs text-slate-200">
                                <div class="font-bold text-emerald-400 flex items-center mb-1">
                                    <i class="fa-solid fa-folder-open mr-1.5"></i> Opened ${data.report.report_type} Report (${data.report.report_id})
                                </div>
                                <p class="text-[11px] text-slate-300">Active study loaded for <strong>${data.patient.patient_name}</strong> (${data.patient.patient_id}). You can ask clinical follow-up questions (e.g. <em>"Summarize her report"</em> or <em>"What did the radiologist note about tooth findings?"</em>).</p>
                            </div>
                        `, "bg-slate-700/60 border border-emerald-500/40");
                    }
                }
            } catch (err) {
                console.error("Failed to open report:", err);
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

                // Display Verified Clinical Action Indicator
                let agentBadgeHtml = "";
                if (data.agent_trace) {
                    const actionMap = {
                        "search_patient_reports": "Patient Records Search",
                        "structure_clinical_dictation": "Clinical Dictation Processing",
                        "summarize_active_report": "Report Consultation & Summary",
                        "general_query": "Clinical Database Query",
                        "query_clinic_database": "Clinical Database Query",
                        "check_appointment_slots": "Scan Appointment Scheduling",
                        "confirm_scan_booking": "Appointment Booking Confirmed"
                    };
                    const actionName = actionMap[data.agent_trace.tool_called] || "Clinical Task Executed";
                    agentBadgeHtml = `
                        <div class="mb-2 pb-1.5 border-b border-slate-700/80 flex items-center justify-between text-[11px] text-slate-300 font-sans">
                            <span class="flex items-center text-emerald-400"><i class="fa-solid fa-circle-check mr-1.5 text-xs"></i> <strong class="text-slate-200">${actionName}</strong></span>
                            <span class="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded border border-slate-700">Verified</span>
                        </div>
                    `;
                }

                // CASE 1: Clarification Required (Ambiguity Safety Check)
                if (data.status === "clarification_required" && data.candidates) {
                    let html = `${agentBadgeHtml}<p class="font-semibold text-amber-300 text-xs mb-2"><i class="fa-solid fa-triangle-exclamation mr-1"></i> ${data.message}</p>`;
                    html += `<div class="space-y-1.5 mt-2">`;
                    data.candidates.forEach(c => {
                        html += `
                            <div class="flex items-center justify-between p-2.5 bg-slate-800 hover:bg-slate-700/80 border border-slate-700 rounded-xl transition">
                                <div>
                                    <div class="flex items-center space-x-2">
                                        <span class="font-bold text-white text-xs">${c.patient_name}</span>
                                        <span class="text-[10px] bg-emerald-500/10 text-emerald-300 font-mono px-1.5 py-0.5 rounded border border-emerald-500/20">${c.patient_id}</span>
                                    </div>
                                    <div class="text-slate-400 text-[11px] mt-0.5">DOB: ${c.date_of_birth} • Sex: ${c.sex}</div>
                                </div>
                                <button onclick="selectPatientById('${c.patient_id}')" 
                                        class="text-[11px] bg-sky-600 hover:bg-sky-500 text-white font-semibold px-2.5 py-1 rounded-lg transition shadow-sm">
                                    Select Patient
                                </button>
                            </div>
                        `;
                    });
                    html += `</div>`;
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-amber-500/40");
                    return;
                }

                // CASE 2: Scan Booking Confirmation Required (Synthetic Appointment Slots)
                if (data.status === "confirmation_required" && data.available_slots) {
                    let html = `${agentBadgeHtml}<p class="font-semibold text-sky-300 text-xs mb-2"><i class="fa-solid fa-calendar-check mr-1.5"></i> Available Appointment Slots (${data.modality || 'Scan'}):</p>`;
                    html += `<div class="space-y-2 mt-2">`;
                    data.available_slots.forEach(s => {
                        html += `
                            <div class="flex items-center justify-between p-2.5 bg-slate-800 hover:bg-slate-700/80 border border-slate-700 rounded-xl transition">
                                <div>
                                    <div class="flex items-center space-x-2">
                                        <span class="font-bold text-white text-xs">${s.datetime_str}</span>
                                        <span class="text-[10px] bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded border border-sky-500/30">${s.modality}</span>
                                    </div>
                                    <span class="text-slate-400 text-[11px]">${s.room} • [${s.slot_id}]</span>
                                </div>
                                <button onclick="sendChat('Confirm slot ${s.slot_id}')" 
                                        class="text-[11px] bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-3 py-1.5 rounded-lg transition shadow-sm flex items-center">
                                    <i class="fa-solid fa-check mr-1"></i> Confirm & Book
                                </button>
                            </div>
                        `;
                    });
                    html += `</div>`;
                    html += `<p class="text-[11px] text-amber-300/90 mt-2.5 flex items-center"><i class="fa-solid fa-circle-exclamation mr-1.5"></i> Explicit clinician/patient confirmation required before booking.</p>`;
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-sky-500/40");
                    return;
                }

                // CASE 3: Scan Booking Confirmed
                if (data.status === "success" && data.booking_reference) {
                    let html = `${agentBadgeHtml}`;
                    html += `
                        <div class="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-slate-200">
                            <p class="font-bold text-emerald-400 text-xs flex items-center mb-1">
                                <i class="fa-solid fa-calendar-check mr-1.5 text-sm"></i> Appointment Successfully Confirmed
                            </p>
                            <p class="text-xs mb-2">${data.message}</p>
                            <div class="text-[11px] bg-slate-800 p-2.5 rounded-lg border border-slate-700 font-mono text-emerald-300 space-y-0.5">
                                <div><strong>Booking Reference:</strong> ${data.booking_reference}</div>
                                <div><strong>Modality:</strong> ${data.slot ? data.slot.modality : 'Scan'}</div>
                                <div><strong>Scheduled Time:</strong> ${data.slot ? data.slot.datetime_str : 'Confirmed'}</div>
                                <div><strong>Location:</strong> ${data.slot ? data.slot.room : 'Suite'}</div>
                            </div>
                        </div>
                    `;
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-emerald-500/40");
                    return;
                }

                // CASE 4: Report Successfully Retrieved
                if (data.status === "success" && data.report) {
                    renderReport(data.report, data.patient);
                    updateContextBanner(data.patient, data.report);
                    if (data.patient && data.patient.patient_id) {
                        selectPatientById(data.patient.patient_id, data.report.report_id, false);
                    }
                    let html = `${agentBadgeHtml}<p class="text-xs text-emerald-300 font-medium"><i class="fa-solid fa-circle-check mr-1 text-emerald-400"></i> ${data.message}</p>`;
                    appendCustomMessage("assistant", html, "bg-slate-700/60 border border-emerald-500/40");
                    return;
                }

                // CASE 5: Conversational Answer (Evidence-Grounded Q&A)
                if (data.answer) {
                    let html = `${agentBadgeHtml}<p class="whitespace-pre-line text-xs text-slate-200">${data.answer}</p>`;
                    if (data.citations && data.citations.length > 0) {
                        html += `<div class="mt-3 pt-2.5 border-t border-slate-700 text-[11px] text-slate-400">`;
                        html += `<span class="font-bold text-slate-300 flex items-center mb-1"><i class="fa-solid fa-quote-left mr-1.5 text-sky-400"></i> Source Evidence Citations:</span>`;
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
                appendMessage("assistant", "Unable to connect to clinical workstation service.");
            }
        }

        function appendMessage(sender, text) {
            const container = document.getElementById("chatMessages");
            const div = document.createElement("div");
            if (sender === "user") {
                div.className = "bg-sky-600 text-white rounded-2xl p-3 max-w-[85%] ml-auto text-xs shadow-md";
                div.innerHTML = `<p class="font-bold text-[10px] text-sky-200 mb-0.5">Clinician</p>${text}`;
            } else {
                div.className = "bg-slate-700/80 border border-slate-600 text-slate-100 rounded-2xl p-3 max-w-[85%] text-xs shadow-md";
                div.innerHTML = `<p class="font-bold text-[10px] text-sky-400 mb-0.5">Clinical Assistant</p>${text}`;
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
            btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-2"></i> Structuring Clinical Report...`;

            try {
                const resp = await fetch(`${API_BASE}/dictation/structure`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text })
                });
                if (!resp.ok) {
                    const errPayload = await resp.json().catch(() => ({}));
                    throw new Error(errPayload.detail || `Server returned status ${resp.status}`);
                }
                const data = await resp.json();

                document.getElementById("dictationOutputContainer").classList.remove("hidden");
                document.getElementById("jsonOutput").innerText = JSON.stringify(data, null, 2);
                document.getElementById("narrativeOutput").innerText = data.narrative_draft || (data.report_metadata && data.report_metadata.extraction_status === "NON_CLINICAL_INPUT_REJECTED" ? "ATTENTION: Non-clinical or instructional text detected. Please dictate or enter actual dental/maxillofacial radiological findings." : "No narrative generated.");

                // Render Findings Badges
                let badgesHtml = "";
                if (data.report_metadata && data.report_metadata.extraction_status === "NON_CLINICAL_INPUT_REJECTED") {
                    badgesHtml += `
                        <div class="w-full p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-300 text-xs flex items-center">
                            <i class="fa-solid fa-triangle-exclamation mr-2 text-sm"></i>
                            <span><strong>Non-Clinical Text Detected:</strong> Input contains software development notes or prompt instructions. No clinical dental findings were extracted.</span>
                        </div>
                    `;
                }
                if (data.anatomical_sites && data.anatomical_sites.length > 0) {
                    data.anatomical_sites.forEach(s => {
                        badgesHtml += `
                            <span class="inline-flex items-center text-xs bg-sky-500/20 border border-sky-500/40 text-sky-300 px-2.5 py-1 rounded-lg">
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
                console.error("Clinical structuring error:", err);
                alert("Clinical Dictation Notice: " + (err.message || "Unable to structure dictation. Please check server logs."));
            } finally {
                btn.innerHTML = `<i class="fa-solid fa-file-circle-check mr-2 text-sm"></i> Generate Verified Clinical Report & PDF`;
            }
        }

        async function resetSession() {
            await fetch(`${API_BASE}/sessions/${currentSessionId}/reset`, { method: "POST" });
            updateContextBanner(null, null);
            document.getElementById("chatMessages").innerHTML = `
                <div class="bg-slate-700/50 border border-slate-600 rounded-2xl p-3 text-slate-300 text-xs">
                    <p class="font-bold text-sky-400 mb-1"><i class="fa-solid fa-rotate-left mr-1"></i> Context Cleared</p>
                    Session memory reset. You can search or request any new patient study.
                </div>
            `;
            document.getElementById("reportModalityBadge").innerText = "NO STUDY LOADED";
            document.getElementById("reportTitle").innerText = "Diagnostic Radiology Report";
            document.getElementById("reportPatientMeta").innerText = "Patient: None selected • Study ID: -";
            document.getElementById("reportContent").innerHTML = `
                <div class="p-8 bg-slate-900/60 border border-dashed border-slate-700 rounded-2xl text-center text-slate-400">
                    <i class="fa-solid fa-file-waveform text-4xl mb-3 text-slate-600"></i>
                    <p class="font-medium text-slate-300">No diagnostic imaging study currently active.</p>
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

        // Initialize Patient Directory on Load
        window.addEventListener('DOMContentLoaded', loadPatientDirectory);
    </script>
</body>
</html>
"""
