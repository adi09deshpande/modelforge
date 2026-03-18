from fastapi import APIRouter, HTTPException
from services.services_jobs import get_job_status, list_jobs

router = APIRouter(prefix="/jobs", tags=["Jobs"])


# =====================================================
# GET JOB STATUS (FOR UI PROGRESS BAR)
# =====================================================
@router.get("/{job_id}")
def job_status(job_id: str):
    status = get_job_status(job_id)

    if status["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return status


# =====================================================
# LIST ALL JOBS (ADMIN / DEBUG ONLY)
# =====================================================
@router.get("/")
def all_jobs():
    return list_jobs()
