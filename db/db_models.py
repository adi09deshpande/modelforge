from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint, Column, JSON


# -----------------------------
# USER
# -----------------------------
class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    name: str
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    projects: List["Project"] = Relationship(back_populates="owner")


# -----------------------------
# PROJECT
# -----------------------------
class Project(SQLModel, table=True):
    __tablename__ = "project"

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="users.id", index=True)
    title: str
    slug: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    owner: Optional[User] = Relationship(back_populates="projects")
    datasets: List["Dataset"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


# -----------------------------
# DATASET
# -----------------------------
class Dataset(SQLModel, table=True):
    __tablename__ = "dataset"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    project: Optional[Project] = Relationship(back_populates="datasets")
    versions: List["DatasetVersion"] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


# -----------------------------
# DATASET VERSION
# -----------------------------
class DatasetVersion(SQLModel, table=True):
    __tablename__ = "datasetversion"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)

    version_number: int
    file_path: str
    checksum: str
    note: Optional[str] = None

    is_current: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    dataset: Optional["Dataset"] = Relationship(back_populates="versions")


# -----------------------------
# MODEL ARTIFACT
# -----------------------------
class ModelArtifact(SQLModel, table=True):
    __tablename__ = "modelartifact"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)

    algorithm: str

    params: Dict[str, Any] = Field(
        sa_column=Column(JSON),
        default_factory=dict,
    )
    metrics: Dict[str, Any] = Field(
        sa_column=Column(JSON),
        default_factory=dict,
    )

    file_path: str
    checksum: str

    created_at: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------
# DATASET PREPARATION CONFIG
# -----------------------------
class DatasetPreparationConfig(SQLModel, table=True):
    __tablename__ = "dataset_preparation_config"

    id: Optional[int] = Field(default=None, primary_key=True)

    dataset_id: int = Field(
        foreign_key="dataset.id",
        index=True,
        unique=True,
    )

    problem_type: str
    target: str

    features: List[str] = Field(
        sa_column=Column(JSON),
    )

    test_size: float
    stratify: bool = False

    encoding: Optional[str] = None
    scaling: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------
# MODEL EXPLAINABILITY
# -----------------------------
class ModelExplainability(SQLModel, table=True):
    __tablename__ = "modelexplainability"

    id: Optional[int] = Field(default=None, primary_key=True)

    model_id: int = Field(
        foreign_key="modelartifact.id",
        index=True,
        unique=True,
    )

    method: str

    global_importance: Dict[str, float] = Field(
        sa_column=Column(JSON),
    )

    local_explanation: Dict[str, float] = Field(
        sa_column=Column(JSON),
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------
# EXPERIMENT
# -----------------------------
class Experiment(SQLModel, table=True):
    __tablename__ = "experiment"

    id: Optional[int] = Field(default=None, primary_key=True)

    project_id: int = Field(foreign_key="project.id", index=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    model_id: Optional[int] = Field(
        foreign_key="modelartifact.id",
        index=True,
        default=None,
    )

    name: str                    # Auto-generated e.g. "RFC-001"
    algorithm: str
    problem_type: str
    tuning_method: str = "manual"
    cv_folds: int = 0

    params: Dict[str, Any] = Field(
        sa_column=Column(JSON),
        default_factory=dict,
    )
    metrics: Dict[str, Any] = Field(
        sa_column=Column(JSON),
        default_factory=dict,
    )

    training_time_seconds: Optional[float] = None
    dataset_version: Optional[int] = None

    tags: Optional[List[str]] = Field(
        sa_column=Column(JSON),
        default=None,
    )
    notes: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
