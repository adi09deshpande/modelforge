from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# ROUTERS — existing
# -------------------------------------------------
from api.projects import router as project_router
from api.dataset import router as dataset_router
from api.preprocessing import router as preprocessing_router
from api.feature_engineering import router as feature_router
from api.preparation import router as preparation_router
from api.train import router as train_router
from api.explainability import router as explainability_router
from api.model import router as model_router
from api.jobs import router as jobs_router

# Persistent preparation config
from api.data_preparation_config import router as prep_config_router

# -------------------------------------------------
# ROUTERS — new features
# -------------------------------------------------
from api.experiments import router as experiments_router
from api.predict import router as predict_router
from api.feature_selection import router as feature_selection_router
from api.chat import router as chat_router


# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI(
    title="ModelForge API",
    version="0.4.0",
    description="Backend API for ModelForge ML lifecycle platform",
)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok"}

# -------------------------------------------------
# ROUTER REGISTRATION — existing
# -------------------------------------------------

# Core resources
app.include_router(project_router, prefix="/projects", tags=["Projects"])
app.include_router(dataset_router, prefix="/dataset",  tags=["Dataset"])

# Data processing
app.include_router(preprocessing_router, prefix="/preprocessing", tags=["Preprocessing"])
app.include_router(feature_router,       prefix="/feature",       tags=["Feature Engineering"])

# Stateless preparation actions (split / encode / scale)
app.include_router(preparation_router, prefix="/prep", tags=["Data Preparation"])

# Persistent preparation configuration (SINGLE SOURCE OF TRUTH)
app.include_router(prep_config_router, prefix="/dataset-preparation", tags=["Dataset Preparation Config"])

# Training (enqueue jobs)
app.include_router(train_router, prefix="/train", tags=["Training"])

# Background jobs (progress, status)
app.include_router(jobs_router, prefix="/jobs", tags=["Jobs"])

# Explainability
app.include_router(explainability_router, prefix="/explain", tags=["Explainability"])

# Model artifacts (download)
app.include_router(model_router, prefix="/model", tags=["Model"])

# -------------------------------------------------
# ROUTER REGISTRATION — new features
# -------------------------------------------------

# Experiment tracking + model comparison
app.include_router(experiments_router, prefix="/experiments", tags=["Experiments"])

# Prediction / inference (single row + batch CSV)
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])

# Feature selection (importance / correlation / RFE)
app.include_router(feature_selection_router, prefix="/feature-selection", tags=["Feature Selection"])

# AI Assistant (Ollama-powered chat)
app.include_router(chat_router, prefix="/chat", tags=["AI Assistant"])
