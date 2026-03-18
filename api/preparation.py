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
from services.services_Data_Preparation import (
    split_dataset,
    encode_categorical,
    scale_numeric,
)

router = APIRouter()

# =====================================================
# HELPERS
# =====================================================

def load_df(db, dataset_id: int) -> tuple[pd.DataFrame, int]:
    file_path = get_current_file_path(db, dataset_id)
    if not file_path:
        raise HTTPException(404, "Dataset not found")

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    return pd.read_csv(file_path), dataset.project_id


def save_df(db, dataset_id: int, project_id: int, df: pd.DataFrame, note: str):
    out = Path(f"data/datasets/{dataset_id}_{note}.csv")
    df.to_csv(out, index=False)

    with open(out, "rb") as f:
        add_version(
            db=db,
            project_id=project_id,
            dataset_id=dataset_id,
            uploaded_file=f,
            note=note,
        )

# =====================================================
# SCHEMAS
# =====================================================

class SplitRequest(BaseModel):
    dataset_id: int
    target: str
    features: list[str]
    test_size: float = 0.2
    stratify: bool = False


class EncodingRequest(BaseModel):
    dataset_id: int
    columns: list[str]
    method: str   # label | onehot


class ScalingRequest(BaseModel):
    dataset_id: int
    columns: list[str]
    method: str   # standard | minmax

# =====================================================
# ENDPOINTS
# =====================================================

@router.post("/split")
def split_data(req: SplitRequest):
    db = get_session()
    try:
        df, project_id = load_df(db, req.dataset_id)

        if req.target not in df.columns:
            raise HTTPException(400, "Target column not found")

        for c in req.features:
            if c not in df.columns:
                raise HTTPException(400, f"Feature not found: {c}")

        X_train, X_test, y_train, y_test = split_dataset(
            df,
            req.features,
            req.target,
            test_size=req.test_size,
            stratify=req.stratify,
        )

        return {
            "X_train_rows": int(X_train.shape[0]),
            "X_test_rows": int(X_test.shape[0]),
            "num_features": int(X_train.shape[1]),
            "message": "Train-test split successful",
        }

    finally:
        db.close()


@router.post("/encode")
def encode_data(req: EncodingRequest):
    db = get_session()
    try:
        df, project_id = load_df(db, req.dataset_id)

        for c in req.columns:
            if c not in df.columns:
                raise HTTPException(400, f"Column not found: {c}")

        df = encode_categorical(df, req.columns, req.method)

        save_df(db, req.dataset_id, project_id, df, f"encode_{req.method}")
        return {"message": "Encoding applied"}

    finally:
        db.close()


@router.post("/scale")
def scale_data(req: ScalingRequest):
    db = get_session()
    try:
        df, project_id = load_df(db, req.dataset_id)

        for c in req.columns:
            if c not in df.columns:
                raise HTTPException(400, f"Column not found: {c}")

        df = scale_numeric(df, req.columns, req.method)

        save_df(db, req.dataset_id, project_id, df, f"scale_{req.method}")
        return {"message": "Scaling applied"}

    finally:
        db.close()
