export interface Patient {
  patient_id: string;
  patient_name: string;
  date_of_birth: string;
  sex: string;
  reports?: ReportDetail[];
}

export interface ReportFinding {
  region: string;
  observation: string;
  certainty?: string;
}

export interface ReportDetail {
  report_id: string;
  patient_id?: string;
  patient_name?: string;
  report_date: string;
  report_type: string;
  modality?: string;
  clinical_indication?: string;
  findings: ReportFinding[];
  impression: string[];
  recommendation?: string;
  report_text?: string;
  pdf_filename?: string;
}

export interface AssistantResponse {
  status: 'success' | 'clarification_required' | 'not_found' | 'missing_report';
  message: string;
  candidates?: Patient[];
  report?: ReportDetail;
  patient?: Patient;
  answer?: string;
  citations?: string[];
  context_resolution?: string;
}

export interface DictationResponse {
  report_metadata: any;
  patient_context: any;
  raw_dictation: string;
  anatomical_sites: Array<{
    site_type: string;
    fdi_number?: number;
    palmer_notation?: string;
    universal_number?: number;
    anatomical_name: string;
    quadrant: string;
  }>;
  findings: Array<{
    anatomical_site: string;
    category: string;
    observation: string;
    certainty: string;
  }>;
  impressions: string[];
  recommendations: string[];
  narrative_draft: string;
  safety_guardrails: {
    status: string;
    grounded_ratio: string;
    unsupported_findings: string[];
    anti_hallucination_guarantee: string;
  };
  dictation_json_path?: string;
  dictation_pdf_path?: string;
}

const API_BASE = 'http://localhost:8000';

export async function searchPatients(query?: string): Promise<Patient[]> {
  const url = query ? `${API_BASE}/patients?query=${encodeURIComponent(query)}` : `${API_BASE}/patients`;
  const res = await fetch(url);
  return res.json();
}

export async function getPatientReports(patientId: string, modality?: string): Promise<ReportDetail[]> {
  const url = modality 
    ? `${API_BASE}/patients/${patientId}/reports?modality=${encodeURIComponent(modality)}`
    : `${API_BASE}/patients/${patientId}/reports`;
  const res = await fetch(url);
  return res.json();
}

export async function getReportById(reportId: string): Promise<ReportDetail> {
  const res = await fetch(`${API_BASE}/reports/${reportId}`);
  return res.json();
}

export async function sendAssistantChat(query: string, sessionId: string = 'default_session'): Promise<AssistantResponse> {
  const res = await fetch(`${API_BASE}/assistant/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId })
  });
  return res.json();
}

export async function structureDictation(text: string, patientId?: string, patientName?: string): Promise<DictationResponse> {
  const res = await fetch(`${API_BASE}/dictation/structure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, patient_id: patientId, patient_name: patientName })
  });
  return res.json();
}

export async function resetSession(sessionId: string = 'default_session'): Promise<any> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/reset`, { method: 'POST' });
  return res.json();
}
