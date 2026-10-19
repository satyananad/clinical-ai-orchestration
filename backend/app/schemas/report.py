from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

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
    model_config = ConfigDict(extra="allow")
    site_type: str
    fdi_number: Optional[int] = None
    palmer_notation: Optional[str] = None
    universal_number: Optional[Any] = None
    anatomical_name: str
    quadrant: Optional[str] = None

from typing import List, Optional, Dict, Any, Union

class ClinicalFindingItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    anatomical_site: str
    tooth_number: Optional[str] = None
    category: str
    observation: str
    polarity: Optional[str] = "present"
    certainty: str
    measurements: Optional[List[str]] = []
    source_quote: Optional[str] = None
    related_sites: Optional[List[str]] = []
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None

class SafetyGuardrails(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    grounded_ratio: Union[float, str]
    unsupported_findings: List[str] = []
    anti_hallucination_guarantee: str

class DictationReportResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
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

class ConfirmDictationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    dictation_json_path: Optional[str] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    findings: Optional[List[ClinicalFindingItem]] = None
    impressions: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    report_metadata: Optional[Dict[str, Any]] = None
    patient_context: Optional[Dict[str, Any]] = None
    raw_dictation: Optional[str] = None


class AssistantChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"
    active_patient_id: Optional[str] = None
    active_report_id: Optional[str] = None

class AssistantChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    message: str
    candidates: Optional[List[CandidatePatient]] = None
    report: Optional[ReportDetail] = None
    patient: Optional[CandidatePatient] = None
    answer: Optional[str] = None
    citations: Optional[List[str]] = None
    context_resolution: Optional[str] = None
    available_reports: Optional[List[Dict[str, Any]]] = None
    agent_trace: Optional[Dict[str, Any]] = None


# Dictation Pydantic models for structured LLM parsing
class PatientDetailsModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: Optional[int] = None
    gender: Optional[str] = None
    demographics_summary: Optional[str] = None
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    study_type: Optional[str] = None

class AnatomicalSiteModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    site_type: str = "Tooth"
    anatomical_name: str
    quadrant: Optional[str] = None
    fdi_number: Optional[int] = None
    palmer_notation: Optional[str] = None
    universal_number: Optional[Any] = None

class FindingModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    anatomical_site: Optional[str] = "Dentition"
    tooth_number: Optional[str] = None
    category: str = "General Observation"
    observation: str
    polarity: str = "present"
    certainty: str = "unspecified"
    measurements: List[str] = Field(default_factory=list)
    source_quote: Optional[str] = None
    related_sites: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None

class DictationDraftReportModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    patient_details: Optional[PatientDetailsModel] = None
    anatomical_sites: List[AnatomicalSiteModel] = Field(default_factory=list)
    findings: List[FindingModel] = Field(default_factory=list)
    impressions: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    extraction_status: str = Field(default="COMPLETE")

class RetrievalIntentModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    patient_name: str
    report_type: str = "Any"

