"""
Experiment tracking service for ModelForge.
Logs every training run with full metadata.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlmodel import Session, select
from sqlalchemy import text

from db.db_models import Experiment


# =====================================================
# AUTO-NAME GENERATOR
# =====================================================
_ALGO_SHORT = {
    "Logistic Regression":         "LR",
    "Linear Regression":           "LinR",
    "Random Forest Classifier":    "RFC",
    "Random Forest Regressor":     "RFR",
    "Decision Tree Classifier":    "DTC",
    "Decision Tree Regressor":     "DTR",
}


def _generate_name(db: Session, project_id: int, algorithm: str) -> str:
    short = _ALGO_SHORT.get(algorithm, algorithm[:3].upper())
    try:
        count = db.exec(
            select(Experiment)
            .where(Experiment.project_id == project_id)
            .where(Experiment.algorithm == algorithm)
        ).all()
        return f"{short}-{len(count) + 1:03d}"
    except Exception:
        # Fallback to timestamp-based name if query fails
        ts = datetime.utcnow().strftime("%H%M%S")
        return f"{short}-{ts}"


# =====================================================
# LOG EXPERIMENT
# =====================================================
def log_experiment(
    db: Session,
    *,
    project_id: int,
    dataset_id: int,
    model_id: Optional[int],
    algorithm: str,
    problem_type: str,
    tuning_method: str,
    cv_folds: int,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    training_time_seconds: float,
    dataset_version: Optional[int] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Experiment:

    name = _generate_name(db, project_id, algorithm)

    # ── Sanitize metrics — remove non-serializable values ────────────────
    safe_metrics = _sanitize_dict(metrics)
    safe_params  = _sanitize_dict(params)

    experiment = Experiment(
        project_id=project_id,
        dataset_id=dataset_id,
        model_id=model_id,
        name=name,
        algorithm=algorithm,
        problem_type=problem_type,
        tuning_method=tuning_method,
        cv_folds=cv_folds,
        params=safe_params,
        metrics=safe_metrics,
        training_time_seconds=round(training_time_seconds, 2),
        dataset_version=dataset_version,
        tags=tags or [],
        notes=notes,
        created_at=datetime.utcnow(),
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def _sanitize_dict(d: Any) -> Any:
    """Recursively convert non-JSON-serializable values to strings."""
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(i) for i in d]
    elif isinstance(d, (int, float, str, bool, type(None))):
        return d
    else:
        return str(d)


# =====================================================
# LIST EXPERIMENTS FOR PROJECT
# =====================================================
def list_experiments(
    db: Session,
    project_id: int,
) -> List[Experiment]:
    return db.exec(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
    ).all()


# =====================================================
# GET SINGLE EXPERIMENT
# =====================================================
def get_experiment(db: Session, experiment_id: int) -> Optional[Experiment]:
    return db.exec(
        select(Experiment).where(Experiment.id == experiment_id)
    ).first()


# =====================================================
# DELETE EXPERIMENT
# =====================================================
def delete_experiment(db: Session, experiment_id: int) -> bool:
    exp = db.exec(
        select(Experiment).where(Experiment.id == experiment_id)
    ).first()
    if not exp:
        return False
    db.delete(exp)
    db.commit()
    return True


# =====================================================
# UPDATE TAGS / NOTES
# =====================================================
def update_experiment_notes(
    db: Session,
    experiment_id: int,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Experiment]:
    exp = db.exec(
        select(Experiment).where(Experiment.id == experiment_id)
    ).first()
    if not exp:
        return None
    if notes is not None:
        exp.notes = notes
    if tags is not None:
        exp.tags = tags
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


# =====================================================
# COMPARISON HELPER
# =====================================================
def extract_comparison_metrics(
    experiments: List[Experiment],
    problem_type: str,
) -> List[Dict]:
    rows = []
    for exp in experiments:
        m = exp.metrics or {}
        row = {
            "id":            exp.id,
            "name":          exp.name,
            "algorithm":     exp.algorithm,
            "tuning_method": exp.tuning_method,
            "cv_folds":      exp.cv_folds,
            "training_time": exp.training_time_seconds,
            "created_at":    exp.created_at.strftime("%Y-%m-%d %H:%M") if exp.created_at else "",
        }

        if problem_type == "Classification":
            row["accuracy"]    = round(m.get("accuracy", 0), 4)
            row["f1_weighted"] = round(m.get("weighted avg", {}).get("f1-score", 0), 4)
            row["precision"]   = round(m.get("weighted avg", {}).get("precision", 0), 4)
            row["recall"]      = round(m.get("weighted avg", {}).get("recall", 0), 4)
            row["roc_auc"]     = round(m.get("roc_auc", 0), 4)
        else:
            row["r2"]   = round(m.get("r2",   0), 4)
            row["rmse"] = round(m.get("rmse", 0), 4)
            row["mae"]  = round(m.get("mae",  0), 4)

        cv = m.get("cross_validation", {})
        if cv:
            first_metric = next(iter(cv.values()), {})
            row["cv_mean"] = first_metric.get("mean", "-")
            row["cv_std"]  = first_metric.get("std",  "-")
        else:
            row["cv_mean"] = "-"
            row["cv_std"]  = "-"

        rows.append(row)
    return rows
