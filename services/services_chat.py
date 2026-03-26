"""
Chat service for ModelForge.
Powered by Groq API — fast, free, high quality.
Model: llama3-70b-8192 (best quality on Groq)
"""

import os
import numpy as np
from typing import List, Dict, Optional, Iterator
from sqlmodel import Session, select
import streamlit as st

from db.db_models import (
    Experiment,
    ModelArtifact,
    ModelExplainability,
    DatasetPreparationConfig,
    DatasetVersion,
    Dataset,
    Project,
)

# =====================================================
# GROQ CONFIG
# =====================================================
def _get_api_key(override: str = "") -> str:
    """Get Groq API key from secrets.toml or environment."""
    if override:
        return override
    # Try streamlit secrets first
    try:
        return st.secrets["groq"]["api_key"]
    except Exception:
        pass
    # Fallback to environment variable
    return os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = ""  # loaded dynamically via _get_api_key()
DEFAULT_MODEL  = "llama-3.3-70b-versatile"   # best quality on Groq, free

# Alternative models (faster but smaller):
# "llama3-8b-8192"         — faster, good quality
# "mixtral-8x7b-32768"     — large context window
# "gemma2-9b-it"           — Google Gemma via Groq


# =====================================================
# SYSTEM PROMPT
# =====================================================
SYSTEM_PROMPT = """You are ModelForge Assistant — an expert AI data scientist and ML engineer
embedded inside the ModelForge machine learning platform.

You have two modes:

## MODE 1: Project-Specific Questions
When the user asks about THEIR data, models, or experiments, use the
project context provided to give specific, data-driven answers.
Always quote actual numbers from the context. Never make up metrics.

## MODE 2: General ML/DS Knowledge
When the user asks general questions about machine learning, data science,
statistics, algorithms, or concepts, answer from your expert knowledge.

## What you can help with:
- Data quality issues: missing values, outliers, distributions, dtypes
- Feature engineering: creating features, interactions, date features
- Feature selection: importance, correlation, RFE — when to use each
- Model selection: comparing algorithms, pros/cons of each
- Hyperparameters: what they do and how to tune them
- Evaluation metrics: accuracy, F1, ROC AUC, R², RMSE, MAE — interpretation
- Cross-validation: k-fold, stratified, interpreting mean ± std
- Overfitting/underfitting: how to detect and fix
- SHAP/LIME: how to interpret feature importance
- Encoding/scaling: when to use label vs onehot, standard vs minmax
- Workflow guidance: what to do next based on current state

## Rules:
- Be concise but thorough
- Always be actionable — end with a suggestion when possible
- For project-specific questions, only use data from the provided context
- For general questions, draw from your expert ML/DS knowledge
- Never hallucinate metrics, model names, or results not in the context
- Format responses with markdown for readability
"""


# =====================================================
# PAGE CONTEXT
# =====================================================
PAGE_CONTEXT = {
    "Upload_Data":         "User is uploading or reviewing a dataset.",
    "EDA_Preprocessing":   "User is exploring data, handling missing values, dropping duplicates, converting dtypes.",
    "Feature_Engineering": "User is creating new features: numeric combinations, date features, age calculations.",
    "Feature_Selection":   "User is selecting important features using importance scores, correlation, or RFE.",
    "Data_Preparation":    "User is configuring train/test split, encoding, and scaling before training.",
    "Train_Model":         "User is selecting an algorithm, configuring hyperparameters, and training models.",
    "Model_Explainability":"User is viewing SHAP global importance and LIME local explanations.",
    "Experiments":         "User is comparing multiple training runs and model performances.",
    "Predict":             "User is running inference on new data using a trained model.",
    "Projects":            "User is creating or selecting a project.",
}


# =====================================================
# BUILD PROJECT CONTEXT
# =====================================================
def build_project_context(
    db: Session,
    project_id: int,
    dataset_id: Optional[int] = None,
    current_page: Optional[str] = None,
) -> str:
    lines = []

    if current_page and current_page in PAGE_CONTEXT:
        lines.append(f"CURRENT PAGE: {PAGE_CONTEXT[current_page]}\n")

    project = db.exec(select(Project).where(Project.id == project_id)).first()
    if project:
        lines.append(f"PROJECT: {project.title}")

    datasets = db.exec(select(Dataset).where(Dataset.project_id == project_id)).all()
    lines.append(f"DATASETS: {len(datasets)} total")
    for d in datasets:
        marker = " ← ACTIVE" if dataset_id and d.id == dataset_id else ""
        lines.append(f"  - {d.name} (ID:{d.id}){marker}")

    for d in datasets:
        config = db.exec(
            select(DatasetPreparationConfig)
            .where(DatasetPreparationConfig.dataset_id == d.id)
        ).first()
        if config:
            active = " (ACTIVE)" if dataset_id and d.id == dataset_id else ""
            lines.append(f"\nDATA PREP — {d.name}{active}:")
            lines.append(f"  Problem Type : {config.problem_type}")
            lines.append(f"  Target       : {config.target}")
            lines.append(f"  Features ({len(config.features)}): {', '.join(config.features)}")
            lines.append(f"  Test Size    : {int(config.test_size*100)}%")
            lines.append(f"  Stratify     : {'Yes' if config.stratify else 'No'}")
            lines.append(f"  Encoding     : {config.encoding or 'None'}")
            lines.append(f"  Scaling      : {config.scaling or 'None'}")

    experiments = db.exec(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
    ).all()

    if experiments:
        prob_type = experiments[0].problem_type
        lines.append(f"\nEXPERIMENTS ({len(experiments)} total | {prob_type}):")

        for exp in experiments:
            m = exp.metrics or {}
            lines.append(f"\n  [{exp.name}] {exp.algorithm}")
            lines.append(f"    Tuning     : {exp.tuning_method} | CV: {exp.cv_folds} folds | Time: {exp.training_time_seconds}s")
            lines.append(f"    Date       : {str(exp.created_at)[:16]} | Dataset v{exp.dataset_version or '?'}")

            if prob_type == "Classification":
                wt = m.get("weighted avg", {})
                lines.append(f"    Accuracy   : {round(m.get('accuracy', 0), 4)}")
                lines.append(f"    F1         : {round(wt.get('f1-score', 0), 4)}")
                lines.append(f"    Precision  : {round(wt.get('precision', 0), 4)}")
                lines.append(f"    Recall     : {round(wt.get('recall', 0), 4)}")
                if "roc_auc" in m:
                    lines.append(f"    ROC AUC    : {round(m['roc_auc'], 4)}")
            else:
                lines.append(f"    R²         : {round(m.get('r2', 0), 4)}")
                lines.append(f"    RMSE       : {round(m.get('rmse', 0), 4)}")
                lines.append(f"    MAE        : {round(m.get('mae', 0), 4)}")

            if "best_params" in m:
                bp = ", ".join(f"{k}={v}" for k, v in list(m["best_params"].items())[:6])
                lines.append(f"    Best Params: {bp}")

            if "cross_validation" in m:
                for metric, res in m["cross_validation"].items():
                    lines.append(f"    CV {metric}: mean={res['mean']} std={res['std']} folds={res['scores']}")

        # Best model
        if prob_type == "Classification":
            best = max(experiments, key=lambda e: (e.metrics or {}).get("accuracy", 0))
            score = round((best.metrics or {}).get("accuracy", 0), 4)
            lines.append(f"\nBEST MODEL: {best.name} ({best.algorithm}) — accuracy={score}")

            lines.append("\nRANKING by accuracy:")
            ranked = sorted(experiments, key=lambda e: (e.metrics or {}).get("accuracy", 0), reverse=True)
        else:
            best = max(experiments, key=lambda e: (e.metrics or {}).get("r2", -999))
            score = round((best.metrics or {}).get("r2", 0), 4)
            lines.append(f"\nBEST MODEL: {best.name} ({best.algorithm}) — R²={score}")

            lines.append("\nRANKING by R²:")
            ranked = sorted(experiments, key=lambda e: (e.metrics or {}).get("r2", -999), reverse=True)

        for i, exp in enumerate(ranked, 1):
            m = exp.metrics or {}
            if prob_type == "Classification":
                val = round(m.get("accuracy", 0), 4)
                lines.append(f"  #{i} {exp.name} ({exp.algorithm}) — accuracy={val}")
            else:
                val = round(m.get("r2", 0), 4)
                lines.append(f"  #{i} {exp.name} ({exp.algorithm}) — R²={val}")
    else:
        lines.append("\nEXPERIMENTS: None yet.")

    # SHAP
    latest_model = db.exec(
        select(ModelArtifact)
        .where(ModelArtifact.project_id == project_id)
        .order_by(ModelArtifact.created_at.desc())
    ).first()

    if latest_model:
        exp_row = db.exec(
            select(ModelExplainability)
            .where(ModelExplainability.model_id == latest_model.id)
        ).first()
        if exp_row and exp_row.global_importance:
            flat = {}
            for k, v in exp_row.global_importance.items():
                flat[k] = float(np.mean(np.abs(v))) if isinstance(v, list) else float(v)
            top = sorted(flat.items(), key=lambda x: x[1], reverse=True)
            lines.append(f"\nFEATURE IMPORTANCE (SHAP — {latest_model.algorithm}):")
            for feat, score in top:
                lines.append(f"  {feat}: {round(score, 6)}")

    # Workflow status
    has_dataset  = len(datasets) > 0
    has_config   = any(db.exec(select(DatasetPreparationConfig).where(
        DatasetPreparationConfig.dataset_id == d.id)).first() for d in datasets)
    has_model    = latest_model is not None
    has_exps     = len(experiments) > 0

    lines.append("\nWORKFLOW STATUS:")
    lines.append(f"  Dataset uploaded     : {'✅' if has_dataset else '❌'}")
    lines.append(f"  Data prep configured : {'✅' if has_config else '❌'}")
    lines.append(f"  Model trained        : {'✅' if has_model else '❌'}")
    lines.append(f"  Experiments logged   : {'✅' if has_exps else '❌'}")

    if not has_dataset:
        lines.append("  SUGGESTED NEXT STEP: Upload a dataset.")
    elif not has_config:
        lines.append("  SUGGESTED NEXT STEP: Configure data preparation.")
    elif not has_model:
        lines.append("  SUGGESTED NEXT STEP: Train a model.")
    else:
        lines.append("  SUGGESTED NEXT STEP: Try feature selection or compare more models.")

    return "\n".join(lines)


# =====================================================
# QUERY GROQ (STREAMING)
# =====================================================
def query_groq_stream(
    question: str,
    context: str,
    history: List[Dict],
    model: str = DEFAULT_MODEL,
    api_key: str = "",
) -> Iterator[str]:
    """
    Streams response from Groq token by token.
    Yields string chunks as they arrive.
    """
    try:
        from groq import Groq
    except ImportError:
        yield "❌ Groq SDK not installed. Run: `pip install groq`"
        return

    key = _get_api_key(api_key)
    if not key:
        yield "❌ Groq API key not set. Add GROQ_API_KEY to your environment or .streamlit/secrets.toml"
        return

    client = Groq(api_key=key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"=== PROJECT DATA ===\n{context}\n=== END PROJECT DATA ==="},
    ]
    for msg in history[-10:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        err = str(e)
        if "401" in err or "auth" in err.lower():
            yield "❌ Invalid Groq API key. Check your key at console.groq.com"
        elif "429" in err:
            yield "⚠️ Groq rate limit hit. Wait a moment and try again."
        elif "model" in err.lower():
            yield f"❌ Model not found. Try changing DEFAULT_MODEL in services_chat.py\nError: {err}"
        else:
            yield f"❌ Groq error: {err}"


# =====================================================
# QUERY GROQ (NON-STREAMING — for sidebar widget)
# =====================================================
def query_groq(
    question: str,
    context: str,
    history: List[Dict],
    model: str = DEFAULT_MODEL,
    api_key: str = "",
) -> str:
    """Non-streaming version for sidebar widget."""
    result = ""
    for chunk in query_groq_stream(question, context, history, model, api_key):
        result += chunk
    return result


# =====================================================
# CHECK GROQ STATUS
# =====================================================
def check_groq_status(api_key: str = "") -> Dict:
    key = api_key or GROQ_API_KEY
    if not key:
        return {"running": False, "error": "No API key", "models": []}
    try:
        from groq import Groq
        client = Groq(api_key=key)
        models = client.models.list()
        model_ids = [m.id for m in models.data if "llama" in m.id or "mixtral" in m.id or "gemma" in m.id]
        return {"running": True, "models": model_ids[:10]}
    except ImportError:
        return {"running": False, "error": "groq not installed", "models": []}
    except Exception as e:
        return {"running": False, "error": str(e), "models": []}


# =====================================================
# DYNAMIC SUGGESTIONS
# =====================================================
def get_suggestions(
    db: Session,
    project_id: int,
    current_page: Optional[str] = None,
) -> List[str]:
    experiments = db.exec(
        select(Experiment).where(Experiment.project_id == project_id)
    ).all()
    datasets = db.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).all()

    prob_type = experiments[0].problem_type if experiments else None

    page_suggestions = {
        "EDA_Preprocessing": [
            "How should I handle missing values in my dataset?",
            "When should I drop columns vs fill missing values?",
            "How do I detect outliers in my data?",
            "When should I convert a column to category dtype?",
            "What does a skewed distribution mean for my model?",
        ],
        "Feature_Engineering": [
            "What new features could improve my model?",
            "When should I create ratio features?",
            "How do date features help ML models?",
            "How do I know if a new feature is actually useful?",
            "What is feature interaction and when is it helpful?",
        ],
        "Feature_Selection": [
            "How many features should I keep?",
            "What is the difference between feature importance and RFE?",
            "What does a low SHAP importance score mean?",
            "When should I use correlation-based feature selection?",
            "Can removing features hurt my model?",
        ],
        "Data_Preparation": [
            "Should I use label encoding or one-hot encoding?",
            "When should I use standardization vs normalization?",
            "Does scaling matter for tree-based models?",
            "What test size should I use?",
            "When should I enable stratified split?",
        ],
        "Train_Model": [
            "Which algorithm should I start with for my problem?",
            "What is the difference between Grid Search and Randomized Search?",
            "How many cross-validation folds should I use?",
            "What does overfitting look like in my metrics?",
            "When should I use Random Forest vs Logistic Regression?",
        ],
        "Model_Explainability": [
            "What are my most important features?",
            "Explain the SHAP values for my model",
            "What is the difference between global and local SHAP?",
            "Why does feature X have high importance?",
            "How do I use SHAP values to improve my model?",
        ],
        "Experiments": [
            "Which of my models is best and why?",
            "Compare all my experiments",
            "What do my cross-validation results tell me?",
            "Should I try more hyperparameter tuning?",
            "Why is one model better than another here?",
        ],
        "Predict": [
            "How confident is my model in its predictions?",
            "What does the prediction probability mean?",
            "How do I interpret a low confidence score?",
            "Can I trust predictions on data very different from training data?",
        ],
    }

    if current_page and current_page in page_suggestions:
        base = page_suggestions[current_page]
    elif prob_type == "Classification":
        base = [
            "Which model has the best accuracy?",
            "Compare all my experiments",
            "What are my most important features?",
            "How can I improve my F1 score?",
            "What does my ROC AUC score mean?",
        ]
    elif prob_type == "Regression":
        base = [
            "Which model has the best R²?",
            "Compare all my experiments",
            "What are my most important features?",
            "How can I reduce my RMSE?",
            "What is a good R² score for my problem?",
        ]
    elif datasets:
        base = [
            "What should I do after uploading my dataset?",
            "How do I handle missing values?",
            "What encoding should I use for categorical columns?",
            "What is the ML workflow in ModelForge?",
        ]
    else:
        base = [
            "How do I get started with ModelForge?",
            "What is the difference between classification and regression?",
            "What is cross-validation and why is it important?",
            "How do I choose the right ML algorithm?",
        ]

    general = [
        "What is the bias-variance tradeoff?",
        "Explain precision vs recall in simple terms",
        "How do I know if my model is overfitting?",
    ]

    return base[:5] + general[:2]
