from typing import Dict, Any, Optional, List
from sqlmodel import Session, select

from db.db_models import ModelArtifact


# =====================================================
# ADD MODEL ARTIFACT (PHASE-2: FILE-BASED)
# =====================================================
def add_model_artifact_file(
    db: Session,
    project_id: int,
    algorithm: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    file_path: str,
    checksum: str,
) -> ModelArtifact:
    """
    Store model artifact metadata in DB.
    The actual model binary lives on disk (file_path).
    """

    artifact = ModelArtifact(
        project_id=project_id,
        algorithm=algorithm,
        params=params,       # JSON column
        metrics=metrics,     # JSON column
        file_path=file_path,
        checksum=checksum,
    )

    db.add(artifact)
    db.commit()
    db.refresh(artifact)

    return artifact


# =====================================================
# LIST MODELS FOR A PROJECT
# =====================================================
def list_models(
    db: Session,
    project_id: int,
) -> List[ModelArtifact]:
    """
    List all trained models for a project (newest first).
    """

    return db.exec(
        select(ModelArtifact)
        .where(ModelArtifact.project_id == project_id)
        .order_by(ModelArtifact.created_at.desc())
    ).all()


# =====================================================
# GET LATEST MODEL FOR A PROJECT
# =====================================================
def get_latest_model_for_project(
    db: Session,
    project_id: int,
) -> Optional[ModelArtifact]:
    """
    Fetch the most recent trained model for explainability/inference.
    """

    return db.exec(
        select(ModelArtifact)
        .where(ModelArtifact.project_id == project_id)
        .order_by(ModelArtifact.created_at.desc())
    ).first()
