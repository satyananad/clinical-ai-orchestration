from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CandidatePatient(BaseModel):
    patient_id: str
    patient_name: str
    date_of_birth: str
    sex: str

class ReportFinding(BaseModel):
    region: str
    observation: str
    certainty: Optional[str] = "descriptive"

class ReportDetail(BaseModel):
    report_id: str
    patient_id: Optional[str] = None
    report_date: str
    report_type: str
    modality: Optional[str] = None
    clinical_indication: Optional[str] = None
    comparison: Optional[str] = None
    technique: Optional[str] = None
    findings: List[ReportFinding] = []
    impression: List[str] = []
    recommendation: Optional[str] = None
    report_text: Optional[str] = None
    pdf_filename: Optional[str] = None

class PatientResponse(BaseModel):
    patient_id: str
    patient_name: str
    date_of_birth: str
    sex: str
    reports: List[ReportDetail] = []

class DictationRequest(BaseModel):
    text: str = Field(..., description="Spoken or typed dictation transcript")
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None

class AnatomicalSite(BaseModel):
    site_type: str
    fdi_number: Optional[int] = None
    palmer_notation: Optional[str] = None
    universal_number: Optional[Any] = None
    anatomical_name: str
    quadrant: str

class ClinicalFindingItem(BaseModel):
    anatomical_site: str
    category: str
    observation: str
    certainty: str
    measurements: Optional[List[str]] = []

class SafetyGuardrails(BaseModel):
    status: str
    grounded_ratio: str
    unsupported_findings: List[str] = []
    anti_hallucination_guarantee: str

class DictationReportResponse(BaseModel):
    report_metadata: Dict[str, Any]
    patient_context: Dict[str, Any]
    raw_dictation: str
    anatomical_sites: List[AnatomicalSite]
    findings: List[ClinicalFindingItem]
    impressions: List[str]
    recommendations: List[str]
    narrative_draft: str
    safety_guardrails: SafetyGuardrails
    dictation_json_path: Optional[str] = None
    dictation_pdf_path: Optional[str] = None

class AssistantChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"

class AssistantChatResponse(BaseModel):
    status: str
    message: str
    candidates: Optional[List[CandidatePatient]] = None
    report: Optional[ReportDetail] = None
    patient: Optional[CandidatePatient] = None
    answer: Optional[str] = None
    citations: Optional[List[str]] = None
    context_resolution: Optional[str] = None
    agent_trace: Optional[Dict[str, Any]] = None
