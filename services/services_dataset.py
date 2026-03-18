from sqlmodel import Session, select
from db.db_models import Dataset, DatasetVersion
from pathlib import Path
import hashlib
import shutil
import tempfile
from typing import Optional

# ======================================================
# CONFIG
# ======================================================
MAX_VERSIONS = 10
DATASET_ROOT = Path("data/datasets")
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

# ======================================================
# HELPERS
# ======================================================
def _hash_file(path: Path) -> str:
    """
    Compute SHA256 hash without loading full file into memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_write(uploaded_file, dest: Path):
    """
    Atomic streaming write.
    IMPORTANT: resets stream pointer before reading.
    """

    # 🔴 CRITICAL FIX — reset stream
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = Path(tmp.name)

    shutil.move(tmp_path, dest)


# ======================================================
# DATASET LISTING
# ======================================================
def list_datasets_for_project(db: Session, project_id: int):
    return db.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).all()


# ======================================================
# DATASET CREATION
# ======================================================
def create_dataset(
    db: Session,
    project_id: int,
    name: str,
    uploaded_file,
):
    """
    Create dataset metadata and store initial version on disk.
    uploaded_file must be UploadFile.file
    """

    dataset = Dataset(project_id=project_id, name=name)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    add_version(
        db=db,
        project_id=project_id,
        dataset_id=dataset.id,
        uploaded_file=uploaded_file,
        note="Initial upload",
    )

    return dataset


# ======================================================
# VERSIONING
# ======================================================
def add_version(
    db: Session,
    project_id: int,
    dataset_id: int,
    uploaded_file,
    note: Optional[str] = None,
):
    versions = db.exec(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
    ).all()

    next_version = versions[0].version_number + 1 if versions else 1

    # --------------------------------------------------
    # Directory layout
    # --------------------------------------------------
    version_dir = DATASET_ROOT / str(project_id) / str(dataset_id)
    version_dir.mkdir(parents=True, exist_ok=True)

    file_path = version_dir / f"v{next_version}.csv"

    # --------------------------------------------------
    # Write file (atomic + streaming)
    # --------------------------------------------------
    _safe_write(uploaded_file, file_path)

    checksum = _hash_file(file_path)

    # --------------------------------------------------
    # Deduplication
    # --------------------------------------------------
    if versions and versions[0].checksum == checksum:
        file_path.unlink(missing_ok=True)
        return versions[0]

    # --------------------------------------------------
    # Mark previous versions inactive
    # --------------------------------------------------
    for v in versions:
        v.is_current = False

    new_v = DatasetVersion(
        dataset_id=dataset_id,
        version_number=next_version,
        file_path=str(file_path),
        checksum=checksum,
        note=note,
        is_current=True,
    )

    db.add(new_v)
    db.commit()

    # --------------------------------------------------
    # Retain last N versions
    # --------------------------------------------------
    if len(versions) + 1 > MAX_VERSIONS:
        for old in versions[MAX_VERSIONS - 1:]:
            Path(old.file_path).unlink(missing_ok=True)
            db.delete(old)
        db.commit()

    return new_v


# ======================================================
# CURRENT DATA ACCESS
# ======================================================
def get_current_file_path(db: Session, dataset_id: int) -> Optional[str]:
    """
    Return file path of the current dataset version.
    Returns None if not found (API decides response).
    """

    v = db.exec(
        select(DatasetVersion)
        .where(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.is_current == True,
        )
        .order_by(DatasetVersion.version_number.desc())
    ).first()

    if not v:
        return None

    return v.file_path


# ======================================================
# VERSION LISTING
# ======================================================
def list_versions(db: Session, dataset_id: int):
    return db.exec(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
    ).all()


# ======================================================
# ROLLBACK
# ======================================================
def rollback_version(db: Session, dataset_id: int, version_number: int):
    versions = db.exec(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
    ).all()

    found = False
    for v in versions:
        if v.version_number == version_number:
            v.is_current = True
            found = True
        else:
            v.is_current = False

    if not found:
        return False

    db.commit()
    return True
