from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

from sqlmodel import select

from db.db import get_session
from db.db_models import DatasetPreparationConfig, ModelArtifact
from services.services_jobs import enqueue_training_job, get_job_status

router = APIRouter()


# =====================================================
# SCHEMA
# =====================================================
class TrainRequest(BaseModel):
    project_id: int
    dataset_id: int
    algorithm: str
    random_state: int = 42

    # Parameter tuning
    use_default_params: bool = True
    hyperparameters: Optional[Dict] = None


# =====================================================
# START TRAINING (BACKGROUND)
# =====================================================
@router.post("/train")
def train_model(req: TrainRequest):
    db = get_session()
    try:
        # -------------------------------------------------
        # LOAD DATA PREPARATION CONFIG (SINGLE SOURCE)
        # -------------------------------------------------
        config = db.exec(
            select(DatasetPreparationConfig)
            .where(DatasetPreparationConfig.dataset_id == req.dataset_id)
        ).first()

        if not config:
            raise HTTPException(
                status_code=400,
                detail="Dataset preparation not completed",
            )

        if config.problem_type not in ("Classification", "Regression"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported problem type: {config.problem_type}",
            )

        if not config.features:
            raise HTTPException(
                status_code=400,
                detail="No features selected during data preparation",
            )

        # -------------------------------------------------
        # PREP CONFIG PASSED TO WORKER (AS ONE OBJECT)
        # -------------------------------------------------
        prep_config = {
            "target": config.target,
            "features": config.features,
            "test_size": config.test_size,
            "stratify": config.stratify,
            "encoding": config.encoding,
            "scaling": config.scaling,
            "random_state": req.random_state,
        }

        # -------------------------------------------------
        # PARAMETER HANDLING
        # -------------------------------------------------
        user_params = None
        if not req.use_default_params:
            user_params = req.hyperparameters or {}

        # -------------------------------------------------
        # ENQUEUE BACKGROUND JOB
        # -------------------------------------------------
        job_id = enqueue_training_job(
            project_id=req.project_id,
            dataset_id=req.dataset_id,
            algorithm=req.algorithm,
            problem_type=config.problem_type,
            prep_config=prep_config,
            user_params=user_params,
        )

        return {
            "status": "queued",
            "job_id": job_id,
        }

    finally:
        db.close()


# =====================================================
# JOB STATUS (FOR UI PROGRESS BAR)
# =====================================================
@router.get("/status/{job_id}")
def training_status(job_id: str):
    status = get_job_status(job_id)

    if status["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail="Training job not found",
        )

    return status


# =====================================================
# GET LATEST MODEL FOR PROJECT (UNCHANGED)
# =====================================================
@router.get("/latest/{project_id}")
def get_latest_model(project_id: int):
    db = get_session()
    try:
        artifact = db.exec(
            select(ModelArtifact)
            .where(ModelArtifact.project_id == project_id)
            .order_by(ModelArtifact.created_at.desc())
        ).first()

        if not artifact:
            raise HTTPException(
                status_code=404,
                detail="No trained model found for this project",
            )

        return {
            "id": artifact.id,
            "algorithm": artifact.algorithm,
            "metrics": artifact.metrics,
            "created_at": artifact.created_at,
        }

    finally:
        db.close()
