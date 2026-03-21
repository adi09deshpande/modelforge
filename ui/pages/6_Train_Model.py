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

    st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.35);text-transform:uppercase;margin:8px 0 4px 4px;">📁 Core</div>', unsafe_allow_html=True)
    st.page_link("Home.py",                        label="🏠 Home")
    st.page_link("pages/1_Projects.py",            label="📁 Projects")

    st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.35);text-transform:uppercase;margin:8px 0 4px 4px;">📊 Data</div>', unsafe_allow_html=True)
    st.page_link("pages/2_Upload_Data.py",         label="📂 Upload & Explore")
    st.page_link("pages/3_EDA_Preprocessing.py",   label="📊 EDA & Preprocessing")
    st.page_link("pages/4_Feature_Engineering.py", label="🛠️ Feature Engineering")
    st.page_link("pages/10_Feature_Selection.py",  label="🎯 Feature Selection")
    st.page_link("pages/5_Data_Preparation.py",    label="📦 Data Preparation")

    st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.35);text-transform:uppercase;margin:8px 0 4px 4px;">🤖 Modelling</div>', unsafe_allow_html=True)
    st.page_link("pages/6_Train_Model.py",         label="🚀 Train Model")
    st.page_link("pages/8_Experiments.py",         label="🧪 Experiments")
    st.page_link("pages/7_Model_Explainability.py",label="🧠 Explainability")
    st.page_link("pages/9_Predict.py",             label="🎯 Predict")

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

# ── Clear stale metrics when problem type OR dataset changes ──────────────
current_context = f"{problem_type}_{dataset_id}"
if st.session_state.get("last_train_context") != current_context:
    st.session_state.pop("metrics",               None)
    st.session_state.pop("model_id",              None)
    st.session_state.pop("poll",                  None)
    st.session_state.pop("job_id",                None)
    st.session_state.pop("metrics_problem_type",  None)
    st.session_state["last_train_context"] = current_context

# =====================================================
# DATA PREPARATION SUMMARY
# =====================================================
st.subheader("📋 Data Preparation Summary")

summary_df = pd.DataFrame(
    [
        ("Problem Type",       str(prep["problem_type"])),
        ("Target Column",      str(prep["target"])),
        ("Test Size",          f"{int(prep['test_size'] * 100)}%"),
        ("Stratify",           "Yes" if prep["stratify"] else "No"),
        ("Encoding",           str(prep.get("encoding") or "None")),
        ("Scaling",            str(prep.get("scaling") or "None")),
        ("Number of Features", str(len(prep["features"]))),
    ],
    columns=["Setting", "Value"],
)
summary_df["Value"] = summary_df["Value"].astype(str)
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
# HYPERPARAMETER MODE
# =====================================================
st.divider()
st.subheader("⚙️ Hyperparameter Configuration")

param_mode = st.radio(
    "Parameter Mode",
    [
        "🎯 Use Default Parameters",
        "✏️ Manual Tuning",
        "🔍 Grid Search (exhaustive)",
        "🎲 Randomized Search (fast)",
        "🧠 Bayesian Optimization (smart)",
    ],
    index=0,
)

MODE_MAP = {
    "🎯 Use Default Parameters":        "manual",
    "✏️ Manual Tuning":                 "manual",
    "🔍 Grid Search (exhaustive)":      "grid",
    "🎲 Randomized Search (fast)":      "random",
    "🧠 Bayesian Optimization (smart)": "bayesian",
}
tuning_method   = MODE_MAP[param_mode]
use_defaults    = (param_mode == "🎯 Use Default Parameters")
hyperparameters = {}

if param_mode == "🔍 Grid Search (exhaustive)":
    st.info("**Grid Search** tests every combination in a predefined parameter grid.\n\n✅ Finds the best result within the grid\n⚠️ Can be slow for complex models like Random Forest")
elif param_mode == "🎲 Randomized Search (fast)":
    st.info("**Randomized Search** samples 20 random combinations from the search space.\n\n✅ Much faster than Grid Search\n✅ Often finds near-optimal parameters\n✅ Best default choice for most use cases")
elif param_mode == "🧠 Bayesian Optimization (smart)":
    st.info("**Bayesian Optimization** uses past results to intelligently choose the next parameters.\n\n✅ Most sample-efficient — fewest trials needed\n✅ Best for expensive-to-train models\nℹ️ Falls back to Randomized Search if `scikit-optimize` is not installed")
elif param_mode == "✏️ Manual Tuning":
    st.caption("Set parameters below. Unset values use sklearn defaults.")

    if algorithm == "Random Forest Classifier":
        hyperparameters["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100)
        md = st.number_input("max_depth (0 = None)", 0, 50, 0)
        hyperparameters["max_depth"] = None if md == 0 else md
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", 2, 20, 2)
    elif algorithm == "Decision Tree Classifier":
        md = st.number_input("max_depth (0 = None)", 0, 50, 0)
        hyperparameters["max_depth"] = None if md == 0 else md
        hyperparameters["criterion"] = st.selectbox("criterion", ["gini", "entropy"])
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", 2, 20, 2)
    elif algorithm == "Logistic Regression":
        hyperparameters["C"]        = st.number_input("C (regularization)", 0.01, 100.0, 1.0)
        hyperparameters["max_iter"] = st.number_input("max_iter", 100, 5000, 1000)
        hyperparameters["solver"]   = st.selectbox("solver", ["lbfgs", "liblinear", "saga"])
    elif algorithm == "Random Forest Regressor":
        hyperparameters["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100)
        md = st.number_input("max_depth (0 = None)", 0, 50, 0)
        hyperparameters["max_depth"] = None if md == 0 else md
        hyperparameters["min_samples_split"] = st.number_input("min_samples_split", 2, 20, 2)
    elif algorithm == "Decision Tree Regressor":
        md = st.number_input("max_depth (0 = None)", 0, 50, 0)
        hyperparameters["max_depth"] = None if md == 0 else md
        hyperparameters["criterion"] = st.selectbox("criterion", ["squared_error", "absolute_error"])
    elif algorithm == "Linear Regression":
        st.info("Linear Regression has no hyperparameters to tune.")

# =====================================================
# CROSS-VALIDATION
# =====================================================
st.divider()
st.subheader("🔁 Cross-Validation")

enable_cv = st.checkbox("Enable Cross-Validation", value=False)
cv_folds = 0
if enable_cv:
    cv_folds = st.selectbox("Number of folds (k)", [3, 5, 10], index=1)
    st.info(
        f"**{cv_folds}-Fold Cross-Validation:** Splits dataset into {cv_folds} equal parts, "
        f"trains on {cv_folds-1}, tests on 1 — repeated {cv_folds} times. "
        "Reports mean ± std for robust generalisation estimate."
    )

# =====================================================
# TRAIN BUTTON
# =====================================================
st.divider()

if param_mode == "🔍 Grid Search (exhaustive)" and "Random Forest" in algorithm:
    st.warning("⚠️ Grid Search on Random Forest can take several minutes. Consider **Randomized Search** for faster results.")

if st.button("🚀 Train Model", type="primary"):
    with st.spinner("Submitting training job..."):
        resp = requests.post(
            f"{API_BASE}/train/train",
            json={
                "project_id":         project_id,
                "dataset_id":         dataset_id,
                "algorithm":          algorithm,
                "use_default_params": use_defaults,
                "hyperparameters":    hyperparameters,
                "tuning_method":      tuning_method,
                "cv_folds":           cv_folds,
            },
        )
        resp.raise_for_status()

    # Clear ALL previous results before new training
    st.session_state["job_id"]               = resp.json()["job_id"]
    st.session_state["poll"]                 = True
    st.session_state["metrics_problem_type"] = problem_type
    st.session_state.pop("metrics",  None)
    st.session_state.pop("model_id", None)

# =====================================================
# TRAINING PROGRESS
# =====================================================
if st.session_state.get("poll") and "job_id" in st.session_state:
    st.subheader("⏳ Training Progress")

    try:
        job_resp = requests.get(f"{API_BASE}/jobs/{st.session_state['job_id']}", timeout=10)
        job = job_resp.json() if job_resp.status_code == 200 else {"status": "not_found"}
    except Exception:
        job = {"status": "running", "progress": 50, "message": "Training in progress..."}

    status   = job.get("status",   "running")
    progress = job.get("progress", 0)
    message  = job.get("message",  "Training...")

    if status == "completed":
        st.session_state["poll"] = False
        try:
            model_info = requests.get(f"{API_BASE}/train/latest/{project_id}", timeout=30).json()
            st.session_state["metrics"]              = model_info["metrics"]
            st.session_state["model_id"]             = model_info["id"]
            st.session_state["metrics_problem_type"] = problem_type
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"Training finished but could not load results: {e}")
        st.rerun()

    elif status == "not_found":
        st.info("⏳ Waiting for worker to pick up the job...")
        try:
            model_resp = requests.get(f"{API_BASE}/train/latest/{project_id}", timeout=10)
            if model_resp.status_code == 200:
                model_info = model_resp.json()
                if model_info.get("id"):
                    st.session_state["poll"]             = False
                    st.session_state["metrics"]          = model_info["metrics"]
                    st.session_state["model_id"]         = model_info["id"]
                    st.session_state["metrics_problem_type"] = problem_type
                    st.success("✅ Model trained successfully!")
                    st.rerun()
        except Exception:
            pass
        time.sleep(2)
        st.rerun()

    elif status == "failed":
        st.session_state["poll"] = False
        st.error(f"❌ Training failed: {job.get('message', 'Unknown error')}")
    else:
        st.progress(progress)
        st.info(message)
        time.sleep(2)
        st.rerun()

# =====================================================
# EVALUATION METRICS
# =====================================================
if "metrics" in st.session_state:
    st.divider()
    st.subheader("📊 Evaluation Metrics")
    metrics = st.session_state["metrics"]

    # Always use the problem type the model was ACTUALLY trained with
    trained_pt = st.session_state.get("metrics_problem_type", problem_type)

    if "best_params" in metrics:
        st.subheader("🏆 Best Parameters Found (Auto-Tuning)")
        bp_df = pd.DataFrame(list(metrics["best_params"].items()), columns=["Parameter", "Best Value"])
        bp_df["Best Value"] = bp_df["Best Value"].astype(str)
        st.table(bp_df)

    if trained_pt == "Classification":
        if "accuracy" not in metrics:
            st.warning("⚠️ Stale metrics detected. Click **🚀 Train Model** to retrain.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",             round(metrics.get("accuracy", 0), 4))
            c2.metric("Precision (weighted)", round(metrics.get("weighted avg", {}).get("precision", 0), 4))
            c3.metric("Recall (weighted)",    round(metrics.get("weighted avg", {}).get("recall", 0), 4))
            c4.metric("F1-score (weighted)",  round(metrics.get("weighted avg", {}).get("f1-score", 0), 4))

            st.markdown("### 📄 Classification Report")
            rows = {k: v for k, v in metrics.items() if isinstance(v, dict) and all(m in v for m in ["precision", "recall", "f1-score"])}
            if rows:
                st.dataframe(pd.DataFrame(rows).transpose().round(4).astype(str))

            if "confusion_matrix" in metrics:
                st.markdown("### 🔲 Confusion Matrix")
                st.dataframe(pd.DataFrame(np.array(metrics["confusion_matrix"])).astype(str))

            if "roc_curve" in metrics and "roc_auc" in metrics:
                st.markdown("### 📈 ROC Curve")
                fig, ax = plt.subplots()
                ax.plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], label=f"AUC = {metrics['roc_auc']:.4f}")
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.legend()
                st.pyplot(fig)
    else:
        if "r2" not in metrics:
            st.warning("⚠️ Stale metrics detected. Click **🚀 Train Model** to retrain.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", round(metrics.get("r2",   0), 4))
            c2.metric("RMSE",     round(metrics.get("rmse", 0), 4))
            c3.metric("MAE",      round(metrics.get("mae",  0), 4))

    if "cross_validation" in metrics:
        st.divider()
        st.subheader("🔁 Cross-Validation Results")
        cv_data = metrics["cross_validation"]
        cv_rows = []
        for metric_name, result in cv_data.items():
            cv_rows.append({
                "Metric":          metric_name.upper(),
                "Mean":            result["mean"],
                "Std Dev (±)":     result["std"],
                "Per-Fold Scores": "  |  ".join([str(s) for s in result["scores"]]),
            })
        st.dataframe(pd.DataFrame(cv_rows).astype(str), use_container_width=True)
        mean_df = pd.DataFrame({row["Metric"]: [float(row["Mean"])] for row in cv_rows}).T.rename(columns={0: "Mean Score"})
        st.bar_chart(mean_df)

# =====================================================
# DOWNLOAD MODEL
# =====================================================
if "model_id" in st.session_state:
    st.divider()
    st.subheader("⬇️ Download Trained Model")
    download_resp = requests.get(f"{API_BASE}/model/{st.session_state['model_id']}/download", timeout=30)
    download_resp.raise_for_status()
    st.download_button("Download Model (.pkl)", download_resp.content, "trained_model.pkl", "application/octet-stream")
