"""
ViScan Dental AI Assistant - Streamlit Cloud Web Application
-----------------------------------------------------------
Production clinical dashboard implementing:
- Feature A: Safe Patient Retrieval & Ambiguity Stops (Rule 1)
- Feature B: Clinical Dictation to Structured Report (Rule 2, 3, 4)
- Feature C: Context-Dependent Follow-Up with Citations (Rule 5, 6)
- Bonus: Report-Aware Chatbot, Claim Verifier Sandbox & MCP Inspector
"""

import os
import sys
import json
import streamlit as st

# Ensure repository root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.orchestrator_agent import run_multi_agent_pipeline
from backend.app.services.retrieval import get_all_patients, get_report_by_id, resolve_report_retrieval
from backend.app.services.mcp_tools import DENTAL_ASSISTANT_TOOLS, execute_tool, handle_mcp_jsonrpc_request
from backend.app.services.claim_verification import verify_report_claim, VerificationStatus
from backend.app.services.conversation_service import get_or_create_session
from backend.app.api.dictation import compose_narrative, generate_dictation_pdf

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="ViScan Dental AI Assistant",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header { font-size: 24px; font-weight: 700; color: #38bdf8; margin-bottom: 2px; }
    .sub-header { font-size: 13px; color: #94a3b8; margin-bottom: 15px; }
    .amber-banner { background-color: rgba(245, 158, 11, 0.15); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 6px; margin: 10px 0; }
    .emerald-banner { background-color: rgba(16, 185, 129, 0.15); border-left: 4px solid #10b981; padding: 12px; border-radius: 6px; margin: 10px 0; }
    .card-box { background-color: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session_1"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_patient" not in st.session_state:
    st.session_state.active_patient = None
if "active_report" not in st.session_state:
    st.session_state.active_report = None

session = get_or_create_session(st.session_state.session_id)
all_patients = get_all_patients()

# =====================================================================
# SIDEBAR - CLINIC DIRECTORY & SESSION CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("### 🦷 ViScan Dental AI")
    st.markdown("**Model:** Meta Llama-3.1 • **Protocol:** MCP JSON-RPC")
    st.divider()

    st.markdown("#### 👥 Clinic Patient Directory")
    patient_options = {f"{p['patient_name']} ({p['patient_id']} • DOB: {p['date_of_birth']})": p for p in all_patients}
    selected_label = st.selectbox("Select Patient to Query:", list(patient_options.keys()))
    selected_patient = patient_options[selected_label]

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📂 Open OPG", use_container_width=True):
            query = f"Open {selected_patient['patient_name']}'s latest OPG report"
            res = run_multi_agent_pipeline(query, session_id=st.session_state.session_id)
            st.session_state.chat_history.append({"role": "user", "text": query})
            st.session_state.chat_history.append({"role": "assistant", "data": res})
            if res.get("report"):
                st.session_state.active_patient = res.get("patient")
                st.session_state.active_report = res.get("report")
            st.rerun()

    with col_btn2:
        if st.button("🧊 Open CBCT", use_container_width=True):
            query = f"Open {selected_patient['patient_name']}'s latest CBCT report"
            res = run_multi_agent_pipeline(query, session_id=st.session_state.session_id)
            st.session_state.chat_history.append({"role": "user", "text": query})
            st.session_state.chat_history.append({"role": "assistant", "data": res})
            if res.get("report"):
                st.session_state.active_patient = res.get("patient")
                st.session_state.active_report = res.get("report")
            st.rerun()

    if st.button("⚠️ Test Ambiguity Stop", use_container_width=True, help="Tests duplicate name 'John Smith' (Rule 1)"):
        query = "Open John Smith's latest report"
        res = run_multi_agent_pipeline(query, session_id=st.session_state.session_id)
        st.session_state.chat_history.append({"role": "user", "text": query})
        st.session_state.chat_history.append({"role": "assistant", "data": res})
        st.rerun()

    st.divider()
    st.markdown("#### 🩺 Active Context")
    if st.session_state.active_patient and st.session_state.active_report:
        st.success(f"**Patient:** {st.session_state.active_patient['patient_name']} ({st.session_state.active_patient['patient_id']})\n\n**Report:** {st.session_state.active_report['report_id']} ({st.session_state.active_report['report_type']})")
    else:
        st.info("No active scan selected.")

    if st.button("🔄 Reset Session Context", use_container_width=True):
        session.clear()
        st.session_state.active_patient = None
        st.session_state.active_report = None
        st.session_state.chat_history = []
        st.rerun()

# =====================================================================
# MAIN TABS INTERFACE
# =====================================================================
tab_chat, tab_dictation, tab_safety, tab_mcp = st.tabs([
    "💬 1. Conversational Voice & Report Assistant (Features A & C)",
    "🎙️ 2. Clinical Dictation Studio (Feature B)",
    "🛡️ 3. Safety Guardrails & Claim Verifier Sandbox (Rule 2 & 3)",
    "🤖 4. Multi-Agent & MCP Inspector"
])

# ---------------------------------------------------------------------
# TAB 1: CONVERSATIONAL ASSISTANT & REPORT VIEWER
# ---------------------------------------------------------------------
with tab_chat:
    col_chat, col_viewer = st.columns([6, 6])

    with col_chat:
        st.markdown("<div class='main-header'>Voice & Conversational AI Assistant</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Routes clinical requests to specialized agents with safety stops.</div>", unsafe_allow_html=True)

        # Chat message history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["text"])
            else:
                with st.chat_message("assistant"):
                    data = msg["data"]
                    if data.get("agent_trace"):
                        trace = data["agent_trace"]
                        st.caption(f"🤖 **Agent:** `{trace.get('active_agent')}` | **Tool:** `{trace.get('tool_called')}`")

                    # Status: Ambiguity Safety Stop
                    if data.get("status") == "clarification_required" and data.get("candidates"):
                        st.warning(f"⚠️ {data.get('message')}")
                        st.markdown("**Multiple Matching Records Detected. Please Select Candidate:**")
                        for cand in data.get("candidates", []):
                            col_c1, col_c2 = st.columns([3, 1])
                            with col_c1:
                                st.write(f"• **{cand['patient_name']}** (ID: `{cand['patient_id']}` | DOB: `{cand['date_of_birth']}` | {cand['sex']})")
                            with col_c2:
                                if st.button(f"Select {cand['patient_id']}", key=f"btn_{cand['patient_id']}"):
                                    sub_query = f"Open {cand['patient_id']}'s latest report"
                                    sub_res = run_multi_agent_pipeline(sub_query, session_id=st.session_state.session_id)
                                    st.session_state.chat_history.append({"role": "user", "text": sub_query})
                                    st.session_state.chat_history.append({"role": "assistant", "data": sub_res})
                                    if sub_res.get("report"):
                                        st.session_state.active_patient = sub_res.get("patient")
                                        st.session_state.active_report = sub_res.get("report")
                                    st.rerun()

                    # Status: Report Retrieved
                    elif data.get("status") == "success" and data.get("report"):
                        st.success(data.get("message"))

                    # Status: Conversational Answer (Feature C / General)
                    elif data.get("answer"):
                        st.write(data["answer"])
                        if data.get("citations"):
                            with st.expander("📌 Source Evidence Citations"):
                                for cit in data["citations"]:
                                    st.code(cit)
                    else:
                        st.write(data.get("message", "Request completed."))

        # Chat Input Bar
        user_prompt = st.chat_input("Speak or type (e.g. 'how many reports have', 'Open Vikram Malhotra CBCT', 'Summarize her report')...")
        if user_prompt:
            res = run_multi_agent_pipeline(user_prompt, session_id=st.session_state.session_id)
            st.session_state.chat_history.append({"role": "user", "text": user_prompt})
            st.session_state.chat_history.append({"role": "assistant", "data": res})
            if res.get("report"):
                st.session_state.active_patient = res.get("patient")
                st.session_state.active_report = res.get("report")
            st.rerun()

    # Right Column: Diagnostic Report Viewer
    with col_viewer:
        st.markdown("<div class='main-header'>Radiology Report Viewer</div>", unsafe_allow_html=True)
        if st.session_state.active_report and st.session_state.active_patient:
            rep = st.session_state.active_report
            pat = st.session_state.active_patient

            st.markdown(f"### {rep.get('report_type', 'Scan')} Report: `{rep.get('report_id')}`")
            st.caption(f"**Patient:** {pat.get('patient_name')} ({pat.get('patient_id')}) • **Date:** {rep.get('report_date')} • **Modality:** {rep.get('modality')}")
            
            st.markdown(f"**Clinical Indication:** {rep.get('clinical_indication', 'Routine evaluation')}")
            st.divider()

            st.markdown("#### 🔬 Detailed Findings")
            for f in rep.get("findings", []):
                st.markdown(f"- **{f.get('region')}**: {f.get('observation')} `[Certainty: {f.get('certainty', 'Confirmed')}]`")

            st.markdown("#### 💡 Radiologist Impression")
            for imp in rep.get("impression", []):
                st.info(f"• {imp}")

            st.markdown(f"**Recommendation:** {rep.get('recommendation', 'Clinical correlation advised.')}")

            # Check if PDF file exists on disk to provide download
            pdf_name = f"{rep['report_id']}_{pat['patient_name'].replace(' ', '_')}_{rep['report_type'].replace(' ', '_')}.pdf"
            pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_reports", pdf_name)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pf:
                    st.download_button("📄 Download Official Clinical PDF", data=pf.read(), file_name=pdf_name, mime="application/pdf")
        else:
            st.info("No active radiological report selected. Use the patient directory in the sidebar or ask the assistant on the left.")

# ---------------------------------------------------------------------
# TAB 2: CLINICAL DICTATION STUDIO
# ---------------------------------------------------------------------
with tab_dictation:
    st.markdown("<div class='main-header'>Feature B: Clinical Speech & Audio Dictation Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Normalizes dental tooth numbering (FDI, Palmer, Universal), classifies certainty, and generates signed reports.</div>", unsafe_allow_html=True)

    # Preset templates
    col_t1, col_t2, col_t3 = st.columns(3)
    template_text = ""
    with col_t1:
        if st.button("📋 Template: Mandibular Molar Impaction"):
            template_text = "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal. Mild pericoronal radiolucency. Suggestive of pericoronitis."
    with col_t2:
        if st.button("📋 Template: Maxillary Premolar (UR5, UR4)"):
            template_text = "UR5 shows complete endodontic obturation with well-defined apical radiolucency 4 mm. UR4 displays occlusal coronal radiolucency into enamel-dentin junction. Possible secondary caries."
    with col_t3:
        if st.button("📋 Template: Universal #29 & TMJ"):
            template_text = "Tooth #29 shows normal periapical lamina dura. Right temporomandibular joint reveals mild cortical flattening of the condylar head, indeterminate for early degenerative joint disease."

    dict_input = st.text_area("Clinician Voice Dictation Audio Transcript:", value=template_text or "Lower right third molar is horizontally impacted with roots approximating the inferior alveolar nerve canal. Possible apical rarefaction.", height=110)

    col_dp1, col_dp2 = st.columns([4, 2])
    with col_dp1:
        dict_patient = st.selectbox("Associate Draft with Patient:", [f"{p['patient_name']} ({p['patient_id']})" for p in all_patients], index=0)
    with col_dp2:
        st.write("")
        st.write("")
        synthesize_clicked = st.button("✨ Synthesize Structured Draft & Generate PDF", type="primary", use_container_width=True)

    if synthesize_clicked and dict_input:
        pid = dict_patient.split("(")[1].replace(")", "")
        pname = dict_patient.split("(")[0].strip()

        with st.spinner("Executing Clinical Scribe Agent with Llama-3.1..."):
            tool_res = execute_tool("structure_clinical_dictation", {
                "dictation_text": dict_input,
                "patient_id": pid
            })

            narrative = compose_narrative(pname, tool_res.get("findings", []), tool_res.get("impressions", []), tool_res.get("recommendations", []))

            # Mandatory Review Banner
            st.markdown("""
            <div class='amber-banner'>
                <strong>⚠️ MANDATORY CLINICIAN REVIEW REQUIRED (Rule 4)</strong><br/>
                <em>AI synthesized draft report from spoken audio transcript. Must be reviewed, verified, and signed by attending dentist.</em>
            </div>
            """, unsafe_allow_html=True)

            # Tooth normalization badges
            st.markdown("#### 🦷 Normalized Anatomical Sites")
            badge_cols = st.columns(len(tool_res.get("anatomical_sites", [])) or 1)
            for i, site in enumerate(tool_res.get("anatomical_sites", [])):
                with badge_cols[i % len(badge_cols)]:
                    st.metric(label=site.get("quadrant", "Dental Arch"), value=f"FDI: {site.get('fdi_number', 'N/A')}", help=site.get("anatomical_name"))

            # Side-by-side JSON and narrative
            col_json, col_narr = st.columns(2)
            with col_json:
                st.markdown("**Validated Structured JSON (FHIR Schema)**")
                st.json(tool_res)
            with col_narr:
                st.markdown("**Radiology Narrative Draft**")
                st.text_area("Formatted Draft:", value=narrative, height=350)

# ---------------------------------------------------------------------
# TAB 3: SAFETY GUARDRAILS & CLAIM VERIFICATION SANDBOX
# ---------------------------------------------------------------------
with tab_safety:
    st.markdown("<div class='main-header'>Clinical Claim Verification Sandbox (Revised Rule 2 & 3)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Multi-dimensional claim verification across tooth numbers, negation, certainty, laterality, and measurements.</div>", unsafe_allow_html=True)

    sample_source = "Coronal radiolucency noted on the occlusal surface of tooth 16 extending into mid-dentin without periapical rarefaction. Lesion in left ramus measuring 4 mm."
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        source_text = st.text_area("Source Report Text Evidence:", value=sample_source, height=120)
    with col_s2:
        claim_input = st.text_input("AI-Generated Claim to Test:", value="Caries observed in tooth 46 measuring 8 mm.")

    if st.button("🛡️ Run 8-Dimension Verification Engine", type="primary"):
        mock_rep = {
            "report_id": "RPT-TEST",
            "patient_id": "PAT-TEST",
            "findings": [{"region": "Tooth 16", "observation": source_text}],
            "impression": ["Dental evaluation"]
        }
        verif_res = verify_report_claim(claim_input, mock_rep, patient_id="PAT-TEST")

        if verif_res.passed:
            st.success(f"✅ **CLAIM VERIFIED: {verif_res.status.value.upper()}**")
            if verif_res.evidence_citations:
                st.markdown("**Supporting Citations:**")
                for c in verif_res.evidence_citations:
                    st.code(c)
        else:
            st.error(f"❌ **SAFETY VIOLATION: {verif_res.status.value.upper()}**")
            st.markdown("**Violations Detected:**")
            for v in verif_res.violations:
                st.warning(f"• {v}")

# ---------------------------------------------------------------------
# TAB 4: MULTI-AGENT & MCP INSPECTOR
# ---------------------------------------------------------------------
with tab_mcp:
    st.markdown("<div class='main-header'>Model Context Protocol (MCP) & Multi-Agent Trace</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Inspect live JSON-RPC 2.0 tool definitions and agent traces.</div>", unsafe_allow_html=True)

    st.markdown("#### 📦 Registered MCP Clinical Tools")
    st.json(DENTAL_ASSISTANT_TOOLS)

    st.markdown("#### ⚡ Live JSON-RPC 2.0 Tool Invocation Tester")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        rpc_method = st.selectbox("JSON-RPC Method:", ["tools/list", "tools/call"])
        selected_tool_name = st.selectbox("Tool Name:", ["search_patient_reports", "structure_clinical_dictation", "summarize_active_report", "verify_clinical_claim"])
    with col_m2:
        default_args = '{"patient_name": "Priya Sharma", "modality": "OPG"}'
        tool_args_str = st.text_area("Tool Arguments (JSON):", value=default_args, height=90)

    if st.button("Execute MCP JSON-RPC Request"):
        try:
            args_obj = json.loads(tool_args_str)
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": rpc_method,
                "params": {"name": selected_tool_name, "arguments": args_obj} if rpc_method == "tools/call" else {}
            }
            rpc_resp = handle_mcp_jsonrpc_request(rpc_payload)
            st.markdown("**JSON-RPC 2.0 Protocol Response:**")
            st.json(rpc_resp)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
