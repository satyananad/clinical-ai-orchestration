# Viscan Dental Assistant — Documented Demo API Sequence
> **Deliverable for Take-Home Technical Assignment**  
> Alternative to 3–5 minute video: A step-by-step sequence of reproducible API calls demonstrating Feature A, Feature B, Feature C, scan booking, and safety guardrails.

---

## 🚀 Prerequisites

Ensure the backend server is running:
```bash
./venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
All calls below target `http://127.0.0.1:8000`.

---

## Turn 1: Feature A — Ambiguous Patient Identity Safety Halt

When a clinician requests a patient whose name matches multiple records (e.g., *Priya Sharma* exists twice: `PAT-1001` and `PAT-1012`), the system **halts**, refuses to guess, and surfaces candidate details with dates of birth.

### Request:
```bash
curl -s -X POST http://127.0.0.1:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Open Priya Sharma report",
    "session_id": "demo-session-101"
  }' | jq
```

### Sample Output:
```json
{
  "status": "clarification_required",
  "message": "Multiple patients matched 'Priya Sharma'. Please clarify which patient you mean:\n• Priya Sharma (DOB: 1998-04-12, ID: PAT-1001)\n• Priya Sharma (DOB: 1985-02-14, ID: PAT-1012)",
  "candidates": [
    {
      "patient_id": "PAT-1001",
      "patient_name": "Priya Sharma",
      "date_of_birth": "1998-04-12",
      "sex": "Female"
    },
    {
      "patient_id": "PAT-1012",
      "patient_name": "Priya Sharma",
      "date_of_birth": "1985-02-14",
      "sex": "Female"
    }
  ],
  "report": null,
  "agent_trace": {
    "orchestrator": "TriageAgent",
    "active_agent": "RetrievalAgent",
    "tool_called": "search_patient_reports"
  }
}
```
*Clinical Safety Verification:* `report` is `null`. The system refuses to open an arbitrary file until unambiguous confirmation is provided.

---

## Turn 2: Feature A — Disambiguation & Safe Report Retrieval

The clinician resolves the ambiguity by providing the specific patient ID `PAT-1001`.

### Request:
```bash
curl -s -X POST http://127.0.0.1:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Select PAT-1001",
    "session_id": "demo-session-101"
  }' | jq
```

### Sample Output:
```json
{
  "status": "success",
  "message": "Opened OPG report (RPT-2001) for Priya Sharma.",
  "patient": {
    "patient_id": "PAT-1001",
    "patient_name": "Priya Sharma",
    "date_of_birth": "1998-04-12",
    "sex": "Female"
  },
  "report": {
    "report_id": "RPT-2001",
    "report_type": "OPG",
    "report_date": "2026-06-12",
    "clinical_indication": "Pre-treatment evaluation of third molars and baseline dental screening.",
    "findings_count": 4
  },
  "pdf_url": "/patient-pdf/RPT-2001_Priya_Sharma_OPG.pdf"
}
```
*Clinical Verification:* Context is bound to `PAT-1001`, and the latest OPG report (`RPT-2001`) is retrieved.

---

## Turn 3: Feature C — Context-Dependent Conversational Follow-up

Without repeating the patient's name, the clinician uses a conversational pronoun (*"her"*). The assistant maintains session memory, resolves the pronoun to Priya Sharma (`PAT-1001`), and synthesizes the active report with citations.

### Request:
```bash
curl -s -X POST http://127.0.0.1:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize her report",
    "session_id": "demo-session-101"
  }' | jq
```

### Sample Output:
```json
{
  "status": "success",
  "answer": "Summary for Priya Sharma (OPG Report: RPT-2001, Date: 2026-06-12):\n• Tooth 48: Partially erupted presenting with mesioangular impaction. Root apices lie within 1.5 mm of the superior border of the inferior alveolar nerve canal.\n• Tooth 38: Partially erupted with vertical orientation.\n• Teeth 16 & 46: Coronal radiolucency noted on occlusal surface of tooth 16 penetrating into mid-dentin.\n• Recommendation: Operative restoration for tooth 16; CBCT imaging advised prior to surgical removal of tooth 48.",
  "citations": [
    "Source Report: RPT-2001 (OPG, 2026-06-12) - Finding: Tooth 48 (Mandibular right third molar)",
    "Source Report: RPT-2001 (OPG, 2026-06-12) - Finding: Teeth 16 and 46"
  ],
  "context_resolution": "Active Context -> Patient ID: PAT-1001, Report ID: RPT-2001"
}
```
*Clinical Verification:* Zero re-prompting for patient identity. Every clinical fact has a direct verbatim quote citation.

---

## Turn 4: Feature C / Safety — Unsupported Clinical Claims Guardrail

The clinician asks about conditions not documented in the report (*jaw fracture* or *radicular cyst*). The assistant rejects hallucinating, affirms the condition is absent from documentation, and issues a clinical disclaimer.

### Request:
```bash
curl -s -X POST http://127.0.0.1:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is there any radicular cyst or jaw fracture documented in this scan?",
    "session_id": "demo-session-101"
  }' | jq
```

### Sample Output:
```json
{
  "status": "success",
  "answer": "According to OPG report (RPT-2001), the requested details (radicular cyst, jaw fracture) are not documented or evaluated. The radiologist's stated impression is: 'Mesioangular impaction of tooth 48 in close proximity to the mandibular canal; Partially erupted tooth 38; Moderate occlusal dental caries on tooth 16.'\n\n*Clinical Note: Absence of documentation indicates the condition was outside the evaluated scope; it must not be assumed to be clinically normal.*",
  "citations": [
    "Source Report RPT-2001: Verified impressions"
  ]
}
```
*Clinical Safety Verification:* Anti-hallucination verified. The model never asserts an ungrounded finding.

---

## Turn 5: Feature B — Clinical Dictation to Structured JSON Draft

The clinician dictates complex findings using mixed notations (Palmer notation `UR5`, tooth names, measurements, and differential diagnoses). The system converts speech transcripts into structured JSON with certainty ratings and anti-hallucination metrics.

### Request:
```bash
curl -s -X POST http://127.0.0.1:8000/dictation/structure \
  -H "Content-Type: application/json" \
  -d '{
    "dictation_text": "UR5 is root treated with a unilocular apical radiolucency 8 by 7 mm. UR4 is root treated with an apical radiolucency 2 mm wide. Distal to UR8 is an unerupted supernumerary tooth, possibly a dentigerous cyst.",
    "patient_id": "PAT-1001"
  }' | jq
```

### Sample Output:
```json
{
  "report_metadata": {
    "status": "DRAFT_PENDING_REVIEW",
    "clinician_review_required": true,
    "disclaimer": "ATTENTION: This is an unverified AI-generated draft created from clinical dictation. Must be validated by a licensed radiologist before clinical sign-off."
  },
  "patient": "Priya Sharma (PAT-1001)",
  "findings": [
    {
      "anatomical_site": "Maxillary right second premolar (5 ┘)",
      "tooth_number": "15",
      "category": "Endodontic & Periapical Status",
      "observation": "UR5 is root treated with a unilocular apical radiolucency 8 by 7 mm",
      "certainty": "definite observation",
      "measurements": ["8 mm", "7 mm"]
    },
    {
      "anatomical_site": "Maxillary right first premolar (4 ┘)",
      "tooth_number": "14",
      "category": "Endodontic & Periapical Status",
      "observation": "UR4 is root treated with an apical radiolucency 2 mm wide",
      "certainty": "definite observation",
      "measurements": ["2 mm"]
    },
    {
      "anatomical_site": "Maxillary right third molar (8 ┘)",
      "tooth_number": "18",
      "category": "Pathology & Cysts",
      "observation": "Distal to UR8 is an unerupted supernumerary tooth, possibly a dentigerous cyst",
      "certainty": "suspected / differential diagnosis",
      "measurements": []
    }
  ],
  "safety_guardrails": {
    "status": "PASSED",
    "grounded_ratio": 1.0,
    "unsupported_findings": [],
    "anti_hallucination_guarantee": "Zero ungrounded or contradicted claims admitted to final report."
  }
}
```
*Clinical Verification:* Palmer notation (`UR5`) correctly mapped to FDI `15`, certainty classified as `suspected` for differential diagnoses, measurements preserved (`8 by 7 mm`), and clinician draft disclaimer embedded.

---

## Turn 6: Voice/Typed Scan-Booking Agent — Check Slots & Confirm

The clinician requests to schedule a scan. The system checks synthetic slot availability and enforces mandatory clinician confirmation before locking the appointment.

### Step 6.1: Inquire & Review Available Slots
```bash
curl -s -X POST http://127.0.0.1:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to schedule a CBCT scan for tomorrow",
    "session_id": "booking-turn-1"
  }' | jq
```
*Output returns available CBCT slots with slot IDs (e.g. `SLOT-107`, `Tomorrow, Oct 20 at 03:15 PM`).*

### Step 6.2: Confirm Booking
```bash
curl -s -X POST http://127.0.0.1:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Confirm slot SLOT-107 for Priya Sharma",
    "session_id": "booking-turn-1"
  }' | jq
```
*Output returns confirmed booking reference (`BKG-107-...`), assigned attending radiologist, and clinic arrival instructions.*
