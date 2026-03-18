# api/explainability.py
from fastapi import APIRouter, HTTPException
from sqlmodel import select

from db.db import get_session
from db.db_models import ModelExplainability, ModelArtifact

router = APIRouter()


# =====================================================
# GET EXPLAINABILITY FOR A MODEL (READ-ONLY)
# =====================================================
@router.get("/{model_id}")
def get_model_explainability(model_id: int):
    """
    Returns persisted explainability for a trained model.

    IMPORTANT:
    - NO SHAP / LIME computation happens here
    - Explainability is generated during training
    - This endpoint is READ-ONLY
    """

    db = get_session()
    try:
        # -------------------------------------------------
        # Validate model exists
        # -------------------------------------------------
        model = db.exec(
            select(ModelArtifact)
            .where(ModelArtifact.id == model_id)
        ).first()

        if not model:
            raise HTTPException(
                status_code=404,
                detail="Model not found",
            )

        # -------------------------------------------------
        # Fetch persisted explainability
        # -------------------------------------------------
        explanation = db.exec(
            select(ModelExplainability)
            .where(ModelExplainability.model_id == model_id)
        ).first()

        if not explanation:
            raise HTTPException(
                status_code=404,
                detail="Explainability not generated for this model",
            )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------
        return {
            "model_id": model_id,
            "algorithm": model.algorithm,
            "method": explanation.method,              # shap | lime | shap+lime
            "global_importance": explanation.global_importance,
            "local_explanation": explanation.local_explanation,
            "created_at": explanation.created_at,
        }

    finally:
        db.close()
