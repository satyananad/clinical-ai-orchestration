from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from ..schemas.report import PatientResponse, ReportDetail
from ..services.retrieval import search_patients, get_patient, get_patient_reports

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.get("", response_model=List[PatientResponse])
def get_patients_endpoint(query: Optional[str] = Query(None, description="Search by patient name or ID")):
    """Search patients and return matching candidates or all patients."""
    if query:
        return search_patients(query)
    from ..services.retrieval import get_all_patients
    return get_all_patients()

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient_by_id_endpoint(patient_id: str):
    """Fetch patient details and all historical reports by patient ID."""
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return patient

@router.get("/{patient_id}/reports", response_model=List[ReportDetail])
def get_patient_reports_endpoint(patient_id: str, modality: Optional[str] = Query(None, description="Filter by modality e.g. OPG, CBCT")):
    """Retrieve all reports for a specific patient."""
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return get_patient_reports(patient_id, modality=modality)
