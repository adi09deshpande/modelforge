# services_users.py
from typing import Dict, Optional
from sqlmodel import Session, select  # type: ignore
from passlib.hash import bcrypt  # type: ignore
from db.db_models import User


def list_users_for_auth(db: Session) -> Dict[str, Dict]:
    users = db.exec(select(User)).all()
    return {
        "usernames": {
            u.username: {
                "name": u.name or u.username,
                "email": u.email,
                "password": u.password_hash,
            }
            for u in users
        }
    }


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.exec(
        select(User).where(User.username == username)
    ).first()


def create_user(
    db: Session,
    username: str,
    name: str,
    email: str,
    password: str,
) -> User:
    existing = db.exec(
        select(User).where(
            (User.username == username) | (User.email == email)
        )
    ).first()

    if existing:
        raise ValueError("Username or email already exists")

    pw_hash = bcrypt.hash(password)
    u = User(
        username=username,
        name=name,
        email=email,
        password_hash=pw_hash,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u
