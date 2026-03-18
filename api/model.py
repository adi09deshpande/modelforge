from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from sqlmodel import select
from db.db import get_session
from db.db_models import ModelArtifact

router = APIRouter()


@router.get("/{model_id}/download")
def download_model(model_id: int):
    db = get_session()
    try:
        artifact = db.exec(
            select(ModelArtifact).where(ModelArtifact.id == model_id)
        ).first()

        if not artifact:
            raise HTTPException(404, "Model not found")

        path = Path(artifact.file_path)

        if not path.exists():
            raise HTTPException(404, "Model file missing on disk")

        return FileResponse(
            path=path,
            filename=path.name,
            media_type="application/octet-stream",
        )

    finally:
        db.close()
