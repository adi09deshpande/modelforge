import streamlit as st  # type: ignore
from sqlmodel import Session, SQLModel  # type: ignore
from sqlalchemy.engine import Engine  # type: ignore

def get_engine() -> Engine:
    # Uses .streamlit/secrets.toml [connections.pg]
    conn = st.connection("pg", type="sql")  # keep name 'pg' consistent with secrets.toml
    return conn.engine

def get_session() -> Session:
    engine = get_engine()
    return Session(engine)

def init_db(models_module) -> None:
    """
    Create all tables defined in models_module one time.
    Import your models first: import storage.db_models as m; init_db(m)
    """
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
