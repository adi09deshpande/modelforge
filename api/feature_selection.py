from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from io import BytesIO

from db.db import get_session
from db.db_models import DatasetPreparationConfig
from services.services_dataset import get_current_file_path
from services.services_feature_selection import run_feature_selection

router = APIRouter()


# =====================================================
# SCHEMA
# =====================================================
class FeatureSelectionRequest(BaseModel):
    dataset_id: int
    method: str                         # importance | correlation | rfe
    top_n: int = 10
    importance_threshold: float = 0.01
    correlation_threshold: float = 0.9
    n_rfe_features: int = 5


# =====================================================
# RUN FEATURE SELECTION
# =====================================================
@router.post("/run")
def run_selection(req: FeatureSelectionRequest):
    db = get_session()
    try:
        # Load prep config to get target + features + problem_type
        config = db.exec(
            __import__("sqlmodel", fromlist=["select"]).select(DatasetPreparationConfig)
            .where(DatasetPreparationConfig.dataset_id == req.dataset_id)
        ).first()

        if not config:
            raise HTTPException(400, "Dataset preparation config not found. Complete Data Preparation first.")

        # Load dataset
        file_path = get_current_file_path(db, req.dataset_id)
        if not file_path:
            raise HTTPException(404, "Dataset file not found")

        df = pd.read_csv(file_path)

        if config.target not in df.columns:
            raise HTTPException(400, f"Target column '{config.target}' not found in dataset")

        # Use only the configured features
        available_features = [f for f in config.features if f in df.columns]
        X = df[available_features]
        y = df[config.target]

        result = run_feature_selection(
            X=X,
            y=y,
            problem_type=config.problem_type,
            method=req.method,
            top_n=req.top_n,
            importance_threshold=req.importance_threshold,
            correlation_threshold=req.correlation_threshold,
            n_rfe_features=req.n_rfe_features,
        )

        return {
            "dataset_id":    req.dataset_id,
            "problem_type":  config.problem_type,
            "target":        config.target,
            "total_features": len(available_features),
            **result,
        }

    finally:
        db.close()
