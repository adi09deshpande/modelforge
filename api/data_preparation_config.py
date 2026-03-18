from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from db.db import get_session
from db.db_models import DatasetPreparationConfig

router = APIRouter()


# =====================================================
# SCHEMA
# =====================================================

class PreparationConfigRequest(BaseModel):
    problem_type: str                 # Classification | Regression
    target: str
    features: List[str]
    test_size: float
    stratify: bool
    encoding: Optional[str] = None    # label | onehot
    scaling: Optional[str] = None     # standard | minmax


# =====================================================
# SAVE / UPDATE CONFIG
# =====================================================

@router.post("/{dataset_id}")
def save_preparation_config(dataset_id: int, req: PreparationConfigRequest):
    if not 0 < req.test_size < 1:
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0 and 1",
        )

    db = get_session()
    try:
        config = (
            db.query(DatasetPreparationConfig)
            .filter(DatasetPreparationConfig.dataset_id == dataset_id)
            .first()
        )

        if config:
            # UPDATE
            config.problem_type = req.problem_type
            config.target = req.target
            config.features = req.features
            config.test_size = req.test_size
            config.stratify = req.stratify
            config.encoding = req.encoding
            config.scaling = req.scaling
        else:
            # CREATE
            config = DatasetPreparationConfig(
                dataset_id=dataset_id,
                problem_type=req.problem_type,
                target=req.target,
                features=req.features,
                test_size=req.test_size,
                stratify=req.stratify,
                encoding=req.encoding,
                scaling=req.scaling,
            )
            db.add(config)

        db.commit()
        db.refresh(config)

        return {"message": "Preparation config saved successfully"}

    finally:
        db.close()


# =====================================================
# LOAD CONFIG
# =====================================================

@router.get("/{dataset_id}")
def get_preparation_config(dataset_id: int):
    db = get_session()
    try:
        config = (
            db.query(DatasetPreparationConfig)
            .filter(DatasetPreparationConfig.dataset_id == dataset_id)
            .first()
        )

        if not config:
            raise HTTPException(404, "Preparation config not found")

        return {
            "problem_type": config.problem_type,
            "target": config.target,
            "features": config.features,
            "test_size": config.test_size,
            "stratify": config.stratify,
            "encoding": config.encoding,
            "scaling": config.scaling,
        }

    finally:
        db.close()
