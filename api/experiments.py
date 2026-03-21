from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import select

from db.db import get_session
from db.db_models import Experiment, DatasetPreparationConfig
from services.services_experiments import (
    list_experiments,
    get_experiment,
    delete_experiment,
    update_experiment_notes,
    extract_comparison_metrics,
)

router = APIRouter()


# =====================================================
# LIST ALL EXPERIMENTS FOR A PROJECT
# =====================================================
@router.get("/project/{project_id}")
def get_experiments(project_id: int):
    db = get_session()
    try:
        experiments = list_experiments(db, project_id)
        return [
            {
                "id":            e.id,
                "name":          e.name,
                "algorithm":     e.algorithm,
                "problem_type":  e.problem_type,
                "tuning_method": e.tuning_method,
                "cv_folds":      e.cv_folds,
                "metrics":       e.metrics,
                "params":        e.params,
                "training_time": e.training_time_seconds,
                "dataset_version": e.dataset_version,
                "tags":          e.tags,
                "notes":         e.notes,
                "model_id":      e.model_id,
                "created_at":    str(e.created_at),
            }
            for e in experiments
        ]
    finally:
        db.close()


# =====================================================
# GET SINGLE EXPERIMENT
# =====================================================
@router.get("/{experiment_id}")
def get_single_experiment(experiment_id: int):
    db = get_session()
    try:
        exp = get_experiment(db, experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        return {
            "id":            exp.id,
            "name":          exp.name,
            "algorithm":     exp.algorithm,
            "problem_type":  exp.problem_type,
            "tuning_method": exp.tuning_method,
            "cv_folds":      exp.cv_folds,
            "metrics":       exp.metrics,
            "params":        exp.params,
            "training_time": exp.training_time_seconds,
            "tags":          exp.tags,
            "notes":         exp.notes,
            "model_id":      exp.model_id,
            "created_at":    str(exp.created_at),
        }
    finally:
        db.close()


# =====================================================
# COMPARE EXPERIMENTS (multiple IDs)
# =====================================================
@router.get("/compare/{project_id}")
def compare_experiments(project_id: int, ids: str):
    """
    ids: comma-separated experiment IDs e.g. ?ids=1,2,3
    """
    db = get_session()
    try:
        id_list = [int(i) for i in ids.split(",") if i.strip()]
        experiments = db.exec(
            select(Experiment)
            .where(Experiment.id.in_(id_list))
        ).all()

        if not experiments:
            raise HTTPException(404, "No experiments found")

        problem_type = experiments[0].problem_type
        rows = extract_comparison_metrics(experiments, problem_type)
        return {"problem_type": problem_type, "comparison": rows}
    finally:
        db.close()


# =====================================================
# UPDATE NOTES / TAGS
# =====================================================
class UpdateExperimentRequest(BaseModel):
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


@router.patch("/{experiment_id}")
def update_experiment(experiment_id: int, req: UpdateExperimentRequest):
    db = get_session()
    try:
        exp = update_experiment_notes(
            db, experiment_id,
            notes=req.notes,
            tags=req.tags,
        )
        if not exp:
            raise HTTPException(404, "Experiment not found")
        return {"message": "Updated successfully"}
    finally:
        db.close()


# =====================================================
# DELETE EXPERIMENT
# =====================================================
@router.delete("/{experiment_id}")
def remove_experiment(experiment_id: int):
    db = get_session()
    try:
        ok = delete_experiment(db, experiment_id)
        if not ok:
            raise HTTPException(404, "Experiment not found")
        return {"message": "Deleted successfully"}
    finally:
        db.close()
