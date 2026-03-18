from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path

from db.db import get_session
from db.db_models import Dataset
from services.services_dataset import (
    get_current_file_path,
    add_version,
)
from services.services_eda import (
    convert_dtype,
    handle_missing,
    drop_duplicates,
    drop_columns,
)

router = APIRouter()

# =====================================================
# REQUEST SCHEMAS
# =====================================================

class DtypeRequest(BaseModel):
    dataset_id: int
    column: str
    dtype: str   # int | float | str | category


class MissingValueRequest(BaseModel):
    dataset_id: int
    strategy: str   # drop | mean | median | mode | custom
    value: str | None = None


class DropColumnsRequest(BaseModel):
    dataset_id: int
    columns: list[str]


class DatasetOnlyRequest(BaseModel):
    dataset_id: int


# =====================================================
# HELPERS
# =====================================================

def load_df_and_project(
    db, dataset_id: int
) -> tuple[pd.DataFrame, Path, int]:
    """
    Load FULL dataset + project context.
    Phase-2 safe.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = get_current_file_path(db, dataset_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Dataset file not found")

    df = pd.read_csv(file_path)
    return df, Path(file_path), dataset.project_id


def save_df(
    db,
    project_id: int,
    dataset_id: int,
    df: pd.DataFrame,
    note: str,
):
    """
    Save processed dataset as NEW VERSION.
    """
    tmp_path = Path(f"data/datasets/tmp_{dataset_id}_{note}.csv")
    df.to_csv(tmp_path, index=False)

    with tmp_path.open("rb") as f:
        add_version(
            db=db,
            project_id=project_id,
            dataset_id=dataset_id,
            uploaded_file=f,
            note=note,
        )

    tmp_path.unlink(missing_ok=True)


# =====================================================
# ENDPOINTS
# =====================================================

@router.post("/convert-dtype")
def api_convert_dtype(req: DtypeRequest):
    db = get_session()
    try:
        df, _, project_id = load_df_and_project(db, req.dataset_id)

        if req.column not in df.columns:
            raise HTTPException(status_code=400, detail="Column not found")

        df = convert_dtype(df, req.column, req.dtype)
        save_df(
            db,
            project_id,
            req.dataset_id,
            df,
            f"astype_{req.column}_{req.dtype}",
        )

        return {"message": "Column datatype converted"}
    finally:
        db.close()


@router.post("/missing-values")
def api_missing_values(req: MissingValueRequest):
    db = get_session()
    try:
        df, _, project_id = load_df_and_project(db, req.dataset_id)

        df = handle_missing(df, req.strategy, req.value)
        save_df(
            db,
            project_id,
            req.dataset_id,
            df,
            f"na_{req.strategy}",
        )

        return {"message": "Missing values handled"}
    finally:
        db.close()


@router.post("/drop-duplicates")
def api_drop_duplicates(req: DatasetOnlyRequest):
    db = get_session()
    try:
        df, _, project_id = load_df_and_project(db, req.dataset_id)

        df = drop_duplicates(df)
        save_df(
            db,
            project_id,
            req.dataset_id,
            df,
            "drop_duplicates",
        )

        return {"message": "Duplicates removed"}
    finally:
        db.close()


@router.post("/drop-columns")
def api_drop_columns(req: DropColumnsRequest):
    db = get_session()
    try:
        df, _, project_id = load_df_and_project(db, req.dataset_id)

        for c in req.columns:
            if c not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column not found: {c}",
                )

        df = drop_columns(df, req.columns)
        save_df(
            db,
            project_id,
            req.dataset_id,
            df,
            "drop_columns",
        )

        return {"message": "Columns dropped"}
    finally:
        db.close()
