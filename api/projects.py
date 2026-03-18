from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

from sqlmodel import Session
from db.db import get_session
from services.services_projects import (
    create_project,
    list_projects,
)

router = APIRouter()


# -------------------------
# SCHEMAS
# -------------------------
class ProjectCreate(BaseModel):
    title: str
    user_id: int


class ProjectRead(BaseModel):
    id: int
    title: str
    slug: str


# -------------------------
# ROUTES
# -------------------------
@router.post("/", response_model=ProjectRead)
def create_project_api(
    payload: ProjectCreate,
    db: Session = Depends(get_session),
):
    if not payload.title.strip():
        raise HTTPException(status_code=400, detail="Project title required")

    project = create_project(
        db=db,
        user_id=payload.user_id,
        title=payload.title.strip(),
    )
    return project


@router.get("/", response_model=List[ProjectRead])
def list_projects_api(
    user_id: int,
    db: Session = Depends(get_session),
):
    return list_projects(db, user_id)
