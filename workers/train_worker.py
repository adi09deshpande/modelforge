import traceback
import hashlib
import time
from pathlib import Path

import pandas as pd

from db.db import get_session
from db.db_models import Experiment
from services.services_jobs import (
    update_job_progress,
    mark_job_completed,
    mark_job_failed,
)
from services.services_dataset import get_current_file_path, list_versions
from services.services_training import train_and_evaluate
from services.services_models import add_model_artifact_file
from services.services_explainability import generate_and_store_explainability
from services.services_experiments import log_experiment, _sanitize_dict


# =====================================================
# MAIN WORKER FUNCTION
# =====================================================

def run_training_job(
    *,
    job_id: str,
    project_id: int,
    dataset_id: int,
    algorithm: str,
    problem_type: str,
    prep_config: dict,
    user_params: dict,
):
    try:
        start_time = time.time()
        update_job_progress(job_id, progress=5, message="Initializing training job")

        # ── Load dataset ──────────────────────────────────────────────────
        update_job_progress(job_id, progress=10, message="Loading dataset")

        with get_session() as db:
            file_path = get_current_file_path(db, dataset_id)

        if not file_path:
            raise RuntimeError("Dataset file not found")

        df = pd.read_csv(file_path)
        X  = df[prep_config["features"]]
        y  = df[prep_config["target"]]

        # ── Resolve tuning config ─────────────────────────────────────────
        tuning_method = prep_config.get("tuning_method", "manual")
        cv_folds      = int(prep_config.get("cv_folds", 0))

        tune_messages = {
            "grid":     "Running Grid Search (this may take a while)...",
            "random":   "Running Randomized Search...",
            "bayesian": "Running Bayesian Optimization...",
        }
        update_job_progress(
            job_id, progress=20,
            message=tune_messages.get(tuning_method, "Training model..."),
        )

        # ── Train ─────────────────────────────────────────────────────────
        model_bytes, params, metrics = train_and_evaluate(
            X=X, y=y,
            problem_type=problem_type,
            model_name=algorithm,
            test_size=prep_config["test_size"],
            random_state=prep_config.get("random_state", 42),
            encoding=prep_config.get("encoding"),
            scaling=prep_config.get("scaling"),
            stratify=prep_config.get("stratify", False),
            user_params=user_params,
            tuning_method=tuning_method,
            cv_folds=cv_folds,
        )

        training_time = round(time.time() - start_time, 2)

        # ── Save model file ───────────────────────────────────────────────
        update_job_progress(job_id, progress=65, message="Saving model artifact")

        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"{algorithm.replace(' ', '_')}_{job_id}.pkl"
        model_path     = models_dir / model_filename
        model_path.write_bytes(model_bytes)
        checksum = hashlib.sha256(model_bytes).hexdigest()

        # ── Save metadata ─────────────────────────────────────────────────
        update_job_progress(job_id, progress=75, message="Persisting model metadata")

        with get_session() as db:
            artifact = add_model_artifact_file(
                db=db,
                project_id=project_id,
                algorithm=algorithm,
                params=params,
                metrics=metrics,
                file_path=str(model_path),
                checksum=checksum,
            )

        # ── Log experiment ────────────────────────────────────────────────
        update_job_progress(job_id, progress=80, message="Logging experiment")

        try:
            with get_session() as db:
                # Get current dataset version
                ds_versions    = list_versions(db, dataset_id)
                current_version = ds_versions[0].version_number if ds_versions else None

                # Sanitize metrics and params before saving
                safe_metrics = _sanitize_dict(metrics)
                safe_params  = _sanitize_dict(params)

                exp = log_experiment(
                    db=db,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    model_id=artifact.id,
                    algorithm=algorithm,
                    problem_type=problem_type,
                    tuning_method=tuning_method,
                    cv_folds=cv_folds,
                    params=safe_params,
                    metrics=safe_metrics,
                    training_time_seconds=training_time,
                    dataset_version=current_version,
                )
                print(f"[INFO] Experiment logged: {exp.name} (id={exp.id})")

        except Exception as e:
            # Print full traceback so we can see the REAL error
            print(f"[WARN] Experiment logging failed:")
            traceback.print_exc()

        # ── Explainability ────────────────────────────────────────────────
        update_job_progress(
            job_id, progress=85,
            message="Generating SHAP + LIME explainability",
        )

        try:
            with get_session() as db:
                generate_and_store_explainability(
                    db=db,
                    model_artifact=artifact,
                    X=X,
                    problem_type=problem_type,
                )
        except Exception as e:
            print(f"[WARN] Explainability failed: {e}")

        # ── Complete ──────────────────────────────────────────────────────
        mark_job_completed(
            job_id,
            model_id=artifact.id,
            metrics=metrics,
            model_file=model_filename,
        )

    except Exception as e:
        traceback.print_exc()
        mark_job_failed(job_id, error=str(e))
