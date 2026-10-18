from fastapi import APIRouter, HTTPException
from ..schemas.report import ReportDetail
from ..services.retrieval import get_report_by_id

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/{report_id}", response_model=ReportDetail)
def get_single_report_endpoint(report_id: str):
    """Fetch one report by its exact report ID."""
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return report
