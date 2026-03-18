from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path

from db.db import get_session
from db.db_models import Dataset  # ✅ REQUIRED
from services.services_dataset import (
    get_current_file_path,
    add_version,
)
from services.services_feature_engineering import (
    create_numeric_feature,
    transform_numeric,
    extract_date_features,
    create_age_feature,
)

router = APIRouter()

# =====================================================
# SCHEMAS
# =====================================================

class NumericFeatureRequest(BaseModel):
    dataset_id: int
    col1: str
    col2: str | None
    operation: str
    new_name: str


class NumericTransformRequest(BaseModel):
    dataset_id: int
    column: str
    transform: str
    new_name: str
    power: int | None = None
    bins: int | None = None


class DateFeatureRequest(BaseModel):
    dataset_id: int
    column: str
    features: list[str]
    prefix: str
    keep_original: bool = True


class AgeRequest(BaseModel):
    dataset_id: int
    dob_column: str
    new_name: str
    keep_original: bool = False


# =====================================================
# HELPERS
# =====================================================

def load_df(db, dataset_id: int) -> pd.DataFrame:
    file_path = get_current_file_path(db, dataset_id)
    if not file_path:
        raise HTTPException(404, "Dataset not found")
    return pd.read_csv(file_path)


def get_project_id(db, dataset_id: int) -> int:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset.project_id


def save_df(db, dataset_id: int, df: pd.DataFrame, note: str):
    project_id = get_project_id(db, dataset_id)  # ✅ FIX

    out = Path(f"data/datasets/{dataset_id}_{note}.csv")
    df.to_csv(out, index=False)

    with open(out, "rb") as f:
        add_version(
            db=db,
            project_id=project_id,     # ✅ REQUIRED
            dataset_id=dataset_id,
            uploaded_file=f,
            note=note,
        )


# =====================================================
# ENDPOINTS
# =====================================================

@router.post("/numeric-feature")
def api_numeric_feature(req: NumericFeatureRequest):
    db = get_session()
    try:
        df = load_df(db, req.dataset_id)

        if req.col1 not in df.columns:
            raise HTTPException(400, f"Column not found: {req.col1}")
        if req.col2 and req.col2 not in df.columns:
            raise HTTPException(400, f"Column not found: {req.col2}")

        df = create_numeric_feature(
            df, req.col1, req.col2, req.operation, req.new_name
        )

        save_df(db, req.dataset_id, df, f"numfeat_{req.operation}")
        return {"message": "Numeric feature created"}

    finally:
        db.close()


@router.post("/numeric-transform")
def api_numeric_transform(req: NumericTransformRequest):
    db = get_session()
    try:
        df = load_df(db, req.dataset_id)

        if req.column not in df.columns:
            raise HTTPException(400, f"Column not found: {req.column}")

        df = transform_numeric(
            df,
            req.column,
            req.transform,
            req.new_name,
            req.power,
            req.bins,
        )

        save_df(db, req.dataset_id, df, f"transform_{req.transform}")
        return {"message": "Numeric transformation applied"}

    finally:
        db.close()


@router.post("/date-features")
def api_date_features(req: DateFeatureRequest):
    db = get_session()
    try:
        df = load_df(db, req.dataset_id)

        if req.column not in df.columns:
            raise HTTPException(400, f"Column not found: {req.column}")

        df = extract_date_features(
            df,
            req.column,
            req.features,
            req.prefix,
            req.keep_original,
        )

        save_df(db, req.dataset_id, df, f"datefeat_{req.column}")
        return {"message": "Date features extracted"}

    finally:
        db.close()


@router.post("/age-from-dob")
def api_age_from_dob(req: AgeRequest):
    db = get_session()
    try:
        df = load_df(db, req.dataset_id)

        if req.dob_column not in df.columns:
            raise HTTPException(400, f"Column not found: {req.dob_column}")

        df = create_age_feature(
            df,
            req.dob_column,
            req.new_name,
            req.keep_original,
        )

        save_df(db, req.dataset_id, df, f"age_{req.dob_column}")
        return {"message": "Age feature created"}

    finally:
        db.close()
