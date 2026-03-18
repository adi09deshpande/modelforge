from typing import Optional
from sqlmodel import Session, select # type: ignore
from db.db_models import Dataset

def get_dataset_for_project(db: Session, project_id: int) -> Optional[Dataset]:
    stmt = select(Dataset).where(Dataset.project_id == project_id).limit(1)
    return db.exec(stmt).first()

def replace_project_dataset(
    db: Session,
    project_id: int,
    data_bytes: bytes,
    original_name: str,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> Dataset:
    existing = get_dataset_for_project(db, project_id)
    if existing:
        existing.original_name = original_name
        existing.n_rows = n_rows
        existing.n_cols = n_cols
        existing.data = data_bytes
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing
    ds = Dataset(
        project_id=project_id,
        original_name=original_name,
        n_rows=n_rows,
        n_cols=n_cols,
        data=data_bytes,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds
