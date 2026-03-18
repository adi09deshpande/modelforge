import traceback
import hashlib
from pathlib import Path

import pandas as pd

from db.db import get_session
from services.services_jobs import (
    update_job_progress,
    mark_job_completed,
    mark_job_failed,
)
from services.services_dataset import get_current_file_path
from services.services_training import train_and_evaluate
from services.services_models import add_model_artifact_file
from services.services_explainability import generate_and_store_explainability


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
    """
    Background training worker.

    Responsibilities:
    - Execute training in background
    - Update job progress
    - Persist model + metrics
    - Trigger explainability
    """

    try:
        # =================================================
        # INIT
        # =================================================
        update_job_progress(
            job_id,
            progress=5,
            message="Initializing training job",
        )

        # =================================================
        # LOAD DATASET
        # =================================================
        update_job_progress(
            job_id,
            progress=10,
            message="Loading dataset",
        )

        with get_session() as db:
            file_path = get_current_file_path(db, dataset_id)

        if not file_path:
            raise RuntimeError("Dataset file not found")

        df = pd.read_csv(file_path)

        target = prep_config["target"]
        features = prep_config["features"]

        X = df[features]
        y = df[target]

        # =================================================
        # TRAIN MODEL
        # =================================================
        update_job_progress(
            job_id,
            progress=30,
            message="Training model",
        )

        model_bytes, params, metrics = train_and_evaluate(
            X=X,
            y=y,
            problem_type=problem_type,
            model_name=algorithm,
            test_size=prep_config["test_size"],
            random_state=prep_config.get("random_state", 42),
            encoding=prep_config.get("encoding"),
            scaling=prep_config.get("scaling"),
            stratify=prep_config.get("stratify", False),
            user_params=user_params,
        )

        # =================================================
        # SAVE MODEL FILE
        # =================================================
        update_job_progress(
            job_id,
            progress=65,
            message="Saving model artifact",
        )

        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"{algorithm.replace(' ', '_')}_{job_id}.pkl"
        model_path = models_dir / model_filename
        model_path.write_bytes(model_bytes)

        checksum = hashlib.sha256(model_bytes).hexdigest()

        # =================================================
        # SAVE METADATA
        # =================================================
        update_job_progress(
            job_id,
            progress=75,
            message="Persisting model metadata",
        )

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

        # =================================================
        # EXPLAINABILITY (NON-BLOCKING)
        # =================================================
        update_job_progress(
            job_id,
            progress=90,
            message="Generating explainability",
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
            # Explainability must NOT fail training
            print(f"[WARN] Explainability failed: {e}")

        # =================================================
        # COMPLETE
        # =================================================
        mark_job_completed(
            job_id,
            model_id=artifact.id,
            metrics=metrics,
            model_file=model_filename,
        )

    except Exception as e:
        traceback.print_exc()
        mark_job_failed(
            job_id,
            error=str(e),
        )
