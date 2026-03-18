# pages/6_Train_Model.py
import bootstrap  # noqa: F401
import streamlit as st  # type: ignore
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="ModelForge • Train Model", layout="wide")

API_BASE = "http://127.0.0.1:8000"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Navigate")
    st.page_link("Home.py", label="Home")
    st.page_link("pages/1_Projects.py", label="Projects")
    st.page_link("pages/2_Upload_Data.py", label="Upload & Explore")
    st.page_link("pages/3_EDA_Preprocessing.py", label="EDA & Preprocessing")
    st.page_link("pages/4_Feature_Engineering.py", label="Feature Engineering")
    st.page_link("pages/5_Data_Preparation.py", label="Data Preparation")
    st.page_link("pages/6_Train_Model.py", label="Train Model")
    st.page_link("pages/7_Model_Explainability.py", label="Model Explainability")

# ---------------- AUTH ----------------
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

st.header("🚀 Train Model")

# ---------------- CONTEXT ----------------
if not st.session_state.get("project_id") or not st.session_state.get("dataset_id"):
    st.info("Select a project and dataset first.")
    st.stop()

project_id = st.session_state["project_id"]
dataset_id = st.session_state["dataset_id"]

# =====================================================
# LOAD PREPARATION CONFIG
# =====================================================
prep = requests.get(
    f"{API_BASE}/dataset-preparation/{dataset_id}",
    timeout=30,
).json()

problem_type = prep["problem_type"]

# =====================================================
# DATA PREPARATION SUMMARY
# =====================================================
st.subheader("📋 Data Preparation Summary")

summary_df = pd.DataFrame(
    [
        ("Problem Type", prep["problem_type"]),
        ("Target Column", prep["target"]),
        ("Test Size", f"{int(prep['test_size'] * 100)}%"),
        ("Stratify", "Yes" if prep["stratify"] else "No"),
        ("Encoding", prep.get("encoding") or "None"),
        ("Scaling", prep.get("scaling") or "None"),
        ("Number of Features", len(prep["features"])),
    ],
    columns=["Setting", "Value"],
)
st.table(summary_df)

# =====================================================
# MODEL SELECTION
# =====================================================
st.divider()
st.subheader("📌 Model Selection")

models = (
    ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"]
    if problem_type == "Classification"
    else ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]
)
algorithm = st.selectbox("Select Model", models)

# =====================================================
# HYPERPARAMETERS
# =====================================================
st.divider()
st.subheader("⚙️ Hyperparameter Tuning")

use_defaults = st.checkbox("Use default model parameters", value=True)
hyperparameters = {}

if not use_defaults:
    st.caption("Unset parameters will automatically use sklearn defaults.")

    if algorithm == "Random Forest Classifier":
        hyperparameters["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100)
        md = st.number_input("max_depth (0 = None)", 0, 50, 0)
        hyperparameters["max_depth"] = None if md == 0 else md

    elif algorithm == "Decision Tree Classifier":
        md = st.number_input("max_depth (0 = None)", 0, 50, 0)
        hyperparameters["max_depth"] = None if md == 0 else md

    elif algorithm == "Logistic Regression":
        hyperparameters["C"] = st.number_input("C", 0.01, 10.0, 1.0)
        hyperparameters["max_iter"] = st.number_input("max_iter", 100, 5000, 1000)

# =====================================================
# TRAIN (BACKGROUND JOB)
# =====================================================
st.divider()

if st.button("🚀 Train Model", type="primary"):
    resp = requests.post(
        f"{API_BASE}/train/train",
        json={
            "project_id": project_id,
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "use_default_params": use_defaults,
            "hyperparameters": hyperparameters,
        },
    )
    resp.raise_for_status()
    st.session_state["job_id"] = resp.json()["job_id"]
    st.session_state["poll"] = True

# =====================================================
# TRAINING PROGRESS (SAFE POLLING)
# =====================================================
if st.session_state.get("poll") and "job_id" in st.session_state:
    st.subheader("⏳ Training Progress")

    job = requests.get(
        f"{API_BASE}/jobs/{st.session_state['job_id']}"
    ).json()

    # ✅ FIX 1: SAFE ACCESS
    status = job.get("status", "running")
    progress = job.get("progress", 0)
    message = job.get("message", "Training...")

    st.progress(progress)
    st.info(message)

    if status == "completed":
        st.session_state["poll"] = False

        model_info = requests.get(
            f"{API_BASE}/train/latest/{project_id}"
        ).json()

        st.session_state["metrics"] = model_info["metrics"]
        st.session_state["model_id"] = model_info["id"]

        st.success("✅ Model trained successfully!")
        st.rerun()


    elif status == "failed":
        st.session_state["poll"] = False
        st.error(job.get("message", "❌ Training failed"))

    else:
        time.sleep(2)
        st.rerun()


# =====================================================
# EVALUATION METRICS (UNCHANGED)
# =====================================================
if "metrics" in st.session_state:
    st.divider()
    st.subheader("📊 Evaluation Metrics")
    metrics = st.session_state["metrics"]

    if problem_type == "Classification":

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", round(metrics["accuracy"], 4))
        c2.metric("Precision (weighted)", round(metrics["weighted avg"]["precision"], 4))
        c3.metric("Recall (weighted)", round(metrics["weighted avg"]["recall"], 4))
        c4.metric("F1-score (weighted)", round(metrics["weighted avg"]["f1-score"], 4))

        st.markdown("### 📄 Classification Report")
        rows = {
            k: v for k, v in metrics.items()
            if isinstance(v, dict)
            and all(m in v for m in ["precision", "recall", "f1-score"])
        }
        st.dataframe(pd.DataFrame(rows).transpose().round(4))

        if "confusion_matrix" in metrics:
            st.markdown("### 🔲 Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            st.dataframe(cm)

        if "roc_curve" in metrics and "roc_auc" in metrics:
            st.markdown("### 📈 ROC Curve")
            fpr = metrics["roc_curve"]["fpr"]
            tpr = metrics["roc_curve"]["tpr"]
            auc = metrics["roc_auc"]

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.legend()
            st.pyplot(fig)

    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", round(metrics["r2"], 4))
        c2.metric("RMSE", round(metrics["rmse"], 4))
        c3.metric("MAE", round(metrics["mae"], 4))

# =====================================================
# DOWNLOAD MODEL
# =====================================================
if "model_id" in st.session_state:
    st.divider()
    st.subheader("⬇️ Download Trained Model")

    download_resp = requests.get(
        f"{API_BASE}/model/{st.session_state['model_id']}/download",
        timeout=30,
    )
    download_resp.raise_for_status()

    st.download_button(
        "Download Model (.pkl)",
        download_resp.content,
        "trained_model.pkl",
        "application/octet-stream",
    )
