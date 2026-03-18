# ModelForge

End-to-End Machine Learning Workflow and Explainability Platform.

## Features

- Dataset Upload & Management
- EDA & Preprocessing
- Feature Engineering
- Model Training
- Model Explainability (SHAP, LIME)
- PostgreSQL Database Backend
- Project & Experiment Tracking
- Streamlit UI

## Tech Stack

- Python
- Streamlit
- PostgreSQL
- SQLModel
- SHAP
- LIME
- Docker
- Alembic

## Architecture

UI → API → Services → Database

## Running

Backend:

pip install -r requirements-backend.txt

UI:

pip install -r requirements-ui.txt
streamlit run ui/Home.py
