# Viscan AI Dental Assistant — UI Demo Evidence
> **Visual Walkthrough & Proof of Requirements**  
> This document details the 6 core visual checkpoints in the Web Workstation UI (`http://localhost:8000`) representing verifiable demo evidence for the technical assessment.

---

## 📸 Overview of Visual Evidence Checkpoints

| Checkpoint | Clinical Workflow | What the UI Displays | Assessment Requirement |
|---|---|---|---|
| **Screenshot 1** | Ambiguity Safety Stop | 2 Candidate cards with DOBs (`PAT-1001` vs `PAT-1012`) | Feature A (Zero guessing) |
| **Screenshot 2** | Safe Report Retrieval | Active OPG report, findings viewer, & diagnostic PDF | Feature A (Retrieval by date) |
| **Screenshot 3** | Conversational Follow-Up | Multi-turn summary citing verbatim report evidence | Feature C (Pronoun resolution) |
| **Screenshot 4** | Anti-Hallucination Guardrail | Explicit refusal to claim unmentioned jaw fracture | Safety Guardrail (Grounded QA) |
| **Screenshot 5** | Clinical Dictation Studio | Structured FDI JSON findings & certainty ratings | Feature B (Clinical Scribe) |
| **Screenshot 6** | Scan-Booking Agent | Confirmed appointment card with token & doctor | Scan-Booking Agent (Two-phase) |

---

## 1. Checkpoint 1: Ambiguity Safety Stop (Feature A)
* **How to trigger in UI:** Type `"Open Priya Sharma"` into the chat box and press Enter.
* **What to screenshot:**
  * The chat window displaying the safety warning banner:  
    `"Multiple patients matched 'Priya Sharma'. Please clarify which patient you mean:"`
  * The two patient candidate selection cards:
    * `Priya Sharma` • DOB: 1998-04-12 • ID: `PAT-1001` • Female
    * `Priya Sharma` • DOB: 1985-02-14 • ID: `PAT-1012` • Female
  * The agent trace badge: `[Agent: RetrievalAgent | MCP Tool: search_patient_reports]`
* **Image path:** `docs/demo_evidence/01_ambiguity_safety_halt.png`

---

## 2. Checkpoint 2: Safe Report Retrieval & PDF Viewer (Feature A)
* **How to trigger in UI:** Click the button **"Select PAT-1001"** (or type `"Select PAT-1001"`).
* **What to screenshot:**
  * The left panel showing `PAT-1001` highlighted in the Patient Directory.
  * The right panel showing the active report card:
    * **OPG (Digital Panoramic Radiograph)** — Exam Date: `2026-06-12`
    * Structured findings table (Tooth 48 mesioangular impaction, Tooth 16 caries, Alveolar crest).
  * The button **"Open High-Res Diagnostic PDF"** linked to `RPT-2001_Priya_Sharma_OPG.pdf`.
* **Image path:** `docs/demo_evidence/02_safe_report_retrieval.png`

---

## 3. Checkpoint 3: Conversational Follow-Up with Citations (Feature C)
* **How to trigger in UI:** Type `"Summarize her report"` in the chat box and press Enter.
* **What to screenshot:**
  * The assistant's structured response resolving `"her"` to Priya Sharma.
  * The blue **Source Evidence Citations** box at the bottom of the chat bubble quoting verbatim findings from `RPT-2001`.
  * The agent trace badge: `[Agent: ConsultationAgent | MCP Tool: summarize_active_report]`
* **Image path:** `docs/demo_evidence/03_conversational_followup_citations.png`

---

## 4. Checkpoint 4: Anti-Hallucination & Unsupported Claims
* **How to trigger in UI:** Type `"Is there any jaw fracture or radicular cyst?"` and press Enter.
* **What to screenshot:**
  * The assistant's response:  
    `"According to OPG report (RPT-2001), the requested details are not documented or evaluated. The radiologist's stated impression is: ..."`
  * The highlighted *Clinical Note* explaining that absence of documentation indicates the condition was outside the evaluated scope.
* **Image path:** `docs/demo_evidence/04_anti_hallucination_guardrail.png`

---

## 5. Checkpoint 5: Clinical Dictation Studio (Feature B)
* **How to trigger in UI:**
  1. Click the top tab **"Clinical Dictation"**.
  2. Click the quick sample button **"Impacted 3rd Molar (LR8)"**.
  3. Click **"Generate Verified Clinical Report & PDF"**.
* **What to screenshot:**
  * The **Extracted Findings** table showing:
    * Site: `Mandibular right third molar (8 ┐)` • FDI: `48` • Category: `Impaction`
    * Certainty: `definite observation`
  * The yellow safety banner: `DRAFT - MANDATORY CLINICIAN REVIEW REQUIRED PRIOR TO SIGN-OFF`.
  * The green guardrail status: `PASSED (1.0 grounded claims)`.
  * The generated PDF download button: `PAT-1001_Priya_Sharma_Dictation_Report.pdf`.
* **Image path:** `docs/demo_evidence/05_clinical_dictation_studio.png`

---

## 6. Checkpoint 6: Voice/Typed Scan-Booking Agent
* **How to trigger in UI:**
  * Type: `"Confirm slot SLOT-107 for Vikram Malhotra"` and press Enter.
* **What to screenshot:**
  * The green confirmed appointment card:
    * **Appointment Successfully Confirmed**
    * Token / Queue #: `TK-107`
    * Booking Reference: `BKG-107-XXXX`
    * Patient: `Vikram Malhotra` (ID: `PAT-1009`, Age: `43 years`)
    * Attending Doctor: `Dr. R. Mehta, MDS (Attending Radiologist)`
    * Scheduled Time: `Tomorrow, Oct 20 at 03:15 PM`
  * The 4 hospital arrival instructions.
* **Image path:** `docs/demo_evidence/06_scan_booking_confirmation.png`
