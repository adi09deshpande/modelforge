from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlmodel import select
from pathlib import Path
import os
import pandas as pd

from db.db import get_session
from db.db_models import Dataset
from services.services_dataset import (
    create_dataset,
    get_current_file_path,
)

router = APIRouter()

# =====================================================
# LIST DATASETS FOR A PROJECT
# =====================================================
@router.get("/project/{project_id}")
def list_datasets(project_id: int):
    db = get_session()
    try:
        datasets = db.exec(
            select(Dataset).where(Dataset.project_id == project_id)
        ).all()

        return [
            {"id": d.id, "name": d.name, "created_at": d.created_at}
            for d in datasets
        ]
    finally:
        db.close()


# =====================================================
# UPLOAD NEW DATASET (STREAMING)
# =====================================================
@router.post("/upload")
def upload_dataset(
    project_id: int = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    db = get_session()
    try:
        dataset = create_dataset(
            db=db,
            project_id=project_id,
            name=name,
            uploaded_file=file.file,   # streaming
        )

        return {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "message": "Dataset uploaded successfully",
        }
    finally:
        db.close()


# =====================================================
# DATASET PREVIEW (SAMPLE ONLY – PHASE 2 SAFE)
# =====================================================
@router.get("/{dataset_id}/preview")
def preview_dataset(
    dataset_id: int,
    n: int = Query(50, ge=1, le=500),
):
    db = get_session()
    try:
        file_path = get_current_file_path(db, dataset_id)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = pd.read_csv(file_path, nrows=n)

        return {
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "rows_returned": len(df),
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    finally:
        db.close()


# =====================================================
# DATASET STATS (🔥 STABLE API CONTRACT 🔥)
# =====================================================
@router.get("/{dataset_id}/stats")
def dataset_stats(dataset_id: int):
    """
    Phase-2 SAFE dataset statistics.
    This response is BACKWARD + FORWARD compatible
    with ALL UI pages.
    """

    db = get_session()
    try:
        file_path = get_current_file_path(db, dataset_id)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        # ---- Sample only (cheap) ----
        sample_df = pd.read_csv(file_path, nrows=200)

        # ---- Row estimate (streaming, safe for huge files) ----
        with open(file_path, "rb") as f:
            rows_estimated = max(sum(1 for _ in f) - 1, 0)

        missing = sample_df.isnull().sum().to_dict()
        columns = list(sample_df.columns)

        return {
            # ===== Upload page =====
            "rows": rows_estimated,
            "columns": columns,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),

            # ===== EDA page =====
            "rows_estimated": rows_estimated,
            "columns_count": len(columns),
            "dtypes": {c: str(t) for c, t in sample_df.dtypes.items()},
            "missing_sample": missing,

            # ===== Backward safety =====
            "missing_values": missing,
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    finally:
        db.close()


# =====================================================
# DOWNLOAD CURRENT DATASET (STREAMING ONLY)
# =====================================================
@router.get("/{dataset_id}/current")
def download_current_dataset(dataset_id: int):
    db = get_session()
    try:
        file_path = get_current_file_path(db, dataset_id)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        path = Path(file_path)

        def file_iterator():
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    yield chunk

        return StreamingResponse(
            file_iterator(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={path.name}"
            },
        )

    finally:
        db.close()
