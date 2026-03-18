# services/services_explainability.py
from typing import Dict
import numpy as np
import pandas as pd
import joblib

import shap
import lime.lime_tabular

from sqlmodel import Session

from db.db_models import ModelArtifact, ModelExplainability


# =====================================================
# HELPERS
# =====================================================
def _is_tree_model(model) -> bool:
    return any(
        k in type(model).__name__
        for k in ["Tree", "Forest", "XGB", "LGBM", "CatBoost"]
    )


def _get_pipeline_parts(pipeline):
    """
    Extract preprocessor + estimator from sklearn Pipeline
    """
    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    return preprocessor, model


def _get_feature_names(preprocessor, X: pd.DataFrame) -> list[str]:
    """
    Safely extract feature names after ColumnTransformer
    """
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "passthrough":
            feature_names.extend(cols)
        else:
            try:
                names = transformer.get_feature_names_out(cols)
                feature_names.extend(names.tolist())
            except Exception:
                feature_names.extend(cols)

    return feature_names


# =====================================================
# GENERATE + STORE EXPLAINABILITY
# =====================================================
def generate_and_store_explainability(
    db: Session,
    model_artifact: ModelArtifact,
    X: pd.DataFrame,
    problem_type: str,
    sample_index: int = 0,
) -> None:
    """
    Generates SHAP (global + local) and LIME (local)
    for PIPELINE-BASED models and persists them.
    """

    # -------------------------------------------------
    # Load pipeline
    # -------------------------------------------------
    pipeline = joblib.load(model_artifact.file_path)
    preprocessor, model = _get_pipeline_parts(pipeline)

    # -------------------------------------------------
    # Transform data using SAME preprocessing
    # -------------------------------------------------
    X_transformed = preprocessor.transform(X)
    feature_names = _get_feature_names(preprocessor, X)

    X_np = np.asarray(X_transformed)

    # Clamp index
    idx = min(sample_index, X_np.shape[0] - 1)

    # =================================================
    # SHAP
    # =================================================
    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_np)
        expected_value = explainer.expected_value
    else:
        explainer = shap.Explainer(model, X_np)
        exp = explainer(X_np)
        shap_values = exp.values
        expected_value = exp.base_values

    # -----------------------------
    # Classification handling
    # -----------------------------
    if isinstance(shap_values, list):
        # Global: mean across classes
        shap_matrix = np.mean(np.abs(shap_values), axis=0)

        # Local: predicted class
        pred = model.predict(X_np[[idx]])[0]
        class_idx = list(model.classes_).index(pred)

        local_shap = shap_values[class_idx][idx]
        base_value = expected_value[class_idx]
    else:
        shap_matrix = np.abs(shap_values)
        local_shap = shap_values[idx]
        base_value = expected_value[idx]

    # Global importance
    global_importance = dict(
        zip(
            feature_names,
            shap_matrix.mean(axis=0).tolist(),
        )
    )

    # Local SHAP
    local_shap_dict = dict(
        zip(
            feature_names,
            local_shap.tolist(),
        )
    )

    # =================================================
    # LIME (LOCAL)
    # =================================================
    mode = "classification" if problem_type == "Classification" else "regression"

    class_names = (
        [str(c) for c in model.classes_]
        if mode == "classification" and hasattr(model, "classes_")
        else None
    )

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_np,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
    )

    predict_fn = (
        model.predict_proba
        if mode == "classification" and hasattr(model, "predict_proba")
        else model.predict
    )

    lime_exp = lime_explainer.explain_instance(
        X_np[idx],
        predict_fn,
    )

    lime_local = dict(lime_exp.as_list())

    # =================================================
    # PERSIST (UPSERT)
    # =================================================
    payload = {
        "shap": local_shap_dict,
        "lime": lime_local,
        "base_value": float(base_value),
    }

    existing = (
        db.query(ModelExplainability)
        .filter(ModelExplainability.model_id == model_artifact.id)
        .first()
    )

    if existing:
        existing.method = "shap+lime"
        existing.global_importance = global_importance
        existing.local_explanation = payload
    else:
        explain = ModelExplainability(
            model_id=model_artifact.id,
            method="shap+lime",
            global_importance=global_importance,
            local_explanation=payload,
        )
        db.add(explain)

    db.commit()
