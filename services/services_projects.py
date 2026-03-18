# services/services_projects.py

from typing import List, Optional
from sqlmodel import Session, select
from db.db_models import Project
from slugify import slugify


def create_project(db: Session, user_id: int, title: str) -> Project:
    """
    Create a new project for the given user_id.
    Slug is generated uniquely per project.
    """

    slug = slugify(title, allow_unicode=True)
    base = slug
    i = 1

    # Ensure slug uniqueness
    while db.exec(select(Project).where(Project.slug == slug)).first():
        slug = f"{base}-{i}"
        i += 1

    p = Project(
        owner_id=user_id,
        title=title,
        slug=slug,
    )

    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def list_projects(db: Session, user_id: int) -> List[Project]:
    """
    List all projects owned by the given user_id.
    """
    return db.exec(
        select(Project)
        .where(Project.owner_id == user_id)
        .order_by(Project.created_at.desc())
    ).all()


def get_project_by_slug(db: Session, slug: str) -> Optional[Project]:
    """
    Fetch a project by its slug.
    """
    return db.exec(
        select(Project).where(Project.slug == slug)
    ).first()
