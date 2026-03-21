from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from sqlmodel import select

from db.db import get_session
from db.db_models import ModelArtifact, DatasetPreparationConfig, Experiment

router = APIRouter()


# =====================================================
# SCHEMAS
# =====================================================
class SinglePredictRequest(BaseModel):
    model_id: int
    features: Dict[str, Any]


class BatchPredictRequest(BaseModel):
    model_id: int
    rows: List[Dict[str, Any]]


# =====================================================
# HELPER — extract feature names from a fitted pipeline
# =====================================================
def _get_feature_names_from_pipeline(pipeline) -> List[str]:
    """
    Extract the feature names the pipeline was trained with.
    Works for sklearn ColumnTransformer pipelines.
    """
    try:
        preprocessor = pipeline.named_steps.get("preprocess")
        if preprocessor and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)
        if preprocessor and hasattr(preprocessor, "transformers_"):
            features = []
            for _, _, cols in preprocessor.transformers_:
                if isinstance(cols, list):
                    features.extend(cols)
            if features:
                return features
    except Exception:
        pass
    return []


# =====================================================
# SINGLE ROW PREDICTION
# =====================================================
@router.post("/single")
def predict_single(req: SinglePredictRequest):
    db = get_session()
    try:
        artifact = db.exec(
            select(ModelArtifact).where(ModelArtifact.id == req.model_id)
        ).first()

        if not artifact:
            raise HTTPException(404, "Model not found")

        if not Path(artifact.file_path).exists():
            raise HTTPException(404, "Model file missing on disk")

        pipeline = joblib.load(artifact.file_path)

        # ── Get the exact features the model was trained with ─────────────
        trained_features = _get_feature_names_from_pipeline(pipeline)

        if not trained_features:
            # Fallback: use whatever the user sent
            trained_features = list(req.features.keys())

        # ── Build DataFrame with EXACTLY the trained features ─────────────
        # Fill missing features with 0, ignore extra features
        row_data = {}
        for feat in trained_features:
            row_data[feat] = req.features.get(feat, 0)

        df = pd.DataFrame([row_data])[trained_features]

        # ── Predict ───────────────────────────────────────────────────────
        prediction = pipeline.predict(df)
        result = {"prediction": prediction[0]}

        # Probabilities for classification
        model_step = pipeline.named_steps.get("model", pipeline)
        if hasattr(model_step, "predict_proba"):
            try:
                proba   = pipeline.predict_proba(df)[0]
                classes = model_step.classes_
                result["probabilities"] = {
                    str(cls): round(float(p), 4)
                    for cls, p in zip(classes, proba)
                }
                result["confidence"] = round(float(max(proba)), 4)
            except Exception:
                pass

        # Convert numpy types
        if isinstance(result["prediction"], (np.integer, np.floating)):
            result["prediction"] = result["prediction"].item()
        else:
            result["prediction"] = str(result["prediction"])

        return result

    finally:
        db.close()


# =====================================================
# BATCH PREDICTION (CSV UPLOAD)
# =====================================================
@router.post("/batch")
def predict_batch(
    model_id: int,
    file: UploadFile = File(...),
):
    db = get_session()
    try:
        artifact = db.exec(
            select(ModelArtifact).where(ModelArtifact.id == model_id)
        ).first()

        if not artifact:
            raise HTTPException(404, "Model not found")

        if not Path(artifact.file_path).exists():
            raise HTTPException(404, "Model file missing on disk")

        pipeline = joblib.load(artifact.file_path)

        # Get trained features
        trained_features = _get_feature_names_from_pipeline(pipeline)

        content = file.file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))

        if df.empty:
            raise HTTPException(400, "Uploaded file is empty")

        # If we know trained features, ensure correct columns
        if trained_features:
            missing = [f for f in trained_features if f not in df.columns]
            if missing:
                raise HTTPException(
                    400,
                    f"CSV is missing columns that the model needs: {missing}. "
                    f"Required columns: {trained_features}"
                )
            df = df[trained_features]

        predictions = pipeline.predict(df)
        result_df   = df.copy()
        result_df["prediction"] = predictions

        # Probabilities
        model_step = pipeline.named_steps.get("model", pipeline)
        if hasattr(model_step, "predict_proba"):
            try:
                proba   = pipeline.predict_proba(df)
                classes = model_step.classes_
                for i, cls in enumerate(classes):
                    result_df[f"prob_{cls}"] = proba[:, i].round(4)
                result_df["confidence"] = proba.max(axis=1).round(4)
            except Exception:
                pass

        return {
            "total_rows":  len(result_df),
            "predictions": result_df.to_dict(orient="records"),
            "columns":     list(result_df.columns),
        }

    finally:
        db.close()


# =====================================================
# GET MODEL INFO — returns the ACTUAL features the model needs
# =====================================================
@router.get("/model-info/{model_id}")
def get_model_info(model_id: int):
    db = get_session()
    try:
        artifact = db.exec(
            select(ModelArtifact).where(ModelArtifact.id == model_id)
        ).first()

        if not artifact:
            raise HTTPException(404, "Model not found")

        # Load pipeline to get actual trained features
        trained_features = []
        if Path(artifact.file_path).exists():
            try:
                pipeline = joblib.load(artifact.file_path)
                trained_features = _get_feature_names_from_pipeline(pipeline)
            except Exception:
                pass

        # Fallback: get from experiment record
        if not trained_features:
            exp = db.exec(
                select(Experiment)
                .where(Experiment.model_id == model_id)
            ).first()
            if exp and exp.params:
                trained_features = exp.params.get("features", [])

        return {
            "model_id":        artifact.id,
            "algorithm":       artifact.algorithm,
            "params":          artifact.params,
            "metrics":         artifact.metrics,
            "trained_features": trained_features,
            "created_at":      str(artifact.created_at),
        }

    finally:
        db.close()
