# pages/8_Experiments.py
import bootstrap  # noqa: F401
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="ModelForge • Experiments", layout="wide")
API_BASE = "http://127.0.0.1:8000"

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
    
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

st.header("🧪 Experiment Tracking")

if not st.session_state.get("project_id"):
    st.info("Select a project first.")
    st.stop()

project_id   = st.session_state["project_id"]
dataset_id   = st.session_state.get("dataset_id")

# ── Load prep config to know problem type ─────────────────────────────────
problem_type = "Classification"
if dataset_id:
    try:
        prep = requests.get(
            f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=10
        ).json()
        problem_type = prep.get("problem_type", "Classification")
    except Exception:
        pass

# ── Load experiments ──────────────────────────────────────────────────────
try:
    resp = requests.get(
        f"{API_BASE}/experiments/project/{project_id}", timeout=30
    )
    experiments = resp.json() if resp.status_code == 200 else []
except Exception:
    experiments = []

if not experiments:
    st.info("No experiments yet. Train a model to start tracking experiments.")
    st.stop()

# =====================================================
# EXPERIMENT TABLE
# =====================================================
st.subheader(f"📋 All Experiments ({len(experiments)} total)")

# Build display dataframe
rows = []
for e in experiments:
    m = e.get("metrics", {})
    row = {
        "ID":        e["id"],
        "Name":      e["name"],
        "Algorithm": e["algorithm"],
        "Tuning":    e["tuning_method"],
        "CV Folds":  e["cv_folds"],
        "Train Time (s)": e.get("training_time") or "-",
        "Date":      e["created_at"][:16],
    }
    if problem_type == "Classification":
        row["Accuracy"]   = round(m.get("accuracy", 0), 4)
        row["F1 (weighted)"] = round(m.get("weighted avg", {}).get("f1-score", 0), 4)
        row["AUC"]        = round(m.get("roc_auc", 0), 4)
    else:
        row["R²"]   = round(m.get("r2",   0), 4)
        row["RMSE"] = round(m.get("rmse", 0), 4)
        row["MAE"]  = round(m.get("mae",  0), 4)
    rows.append(row)

exp_df = pd.DataFrame(rows).astype(str)
st.dataframe(exp_df, use_container_width=True)

# =====================================================
# MODEL COMPARISON
# =====================================================
st.divider()
st.subheader("📊 Model Comparison")

exp_names  = [f"{e['name']} — {e['algorithm']}" for e in experiments]
exp_ids    = [e["id"] for e in experiments]

selected_labels = st.multiselect(
    "Select experiments to compare (2 or more)",
    options=exp_names,
    default=exp_names[:min(3, len(exp_names))],
)

if len(selected_labels) >= 2:
    selected_ids = [
        exp_ids[exp_names.index(label)]
        for label in selected_labels
    ]
    ids_param = ",".join(str(i) for i in selected_ids)

    try:
        cmp_resp = requests.get(
            f"{API_BASE}/experiments/compare/{project_id}",
            params={"ids": ids_param},
            timeout=30,
        )
        if cmp_resp.status_code == 200:
            cmp_data   = cmp_resp.json()
            comparison = cmp_data["comparison"]

            cmp_df = pd.DataFrame(comparison).astype(str)
            st.dataframe(cmp_df, use_container_width=True)

            # ── Visual comparison bar chart ───────────────────────────────
            st.markdown("### 📈 Visual Comparison")

            if problem_type == "Classification":
                metrics_to_plot = ["accuracy", "f1_weighted", "roc_auc"]
            else:
                metrics_to_plot = ["r2", "rmse", "mae"]

            available_metrics = [
                m for m in metrics_to_plot
                if m in comparison[0]
            ]

            for metric in available_metrics:
                fig, ax = plt.subplots(figsize=(8, 3))
                names  = [c["name"] for c in comparison]
                values = [float(c.get(metric, 0)) for c in comparison]
                colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
                bars   = ax.barh(names, values, color=colors)
                ax.bar_label(bars, fmt="%.4f", padding=3)
                ax.set_xlabel(metric.upper())
                ax.set_title(f"{metric.upper()} Comparison")
                ax.set_xlim(0, max(values) * 1.2 if values else 1)
                st.pyplot(fig)
                plt.close()

        else:
            st.error("Could not load comparison data.")
    except Exception as e:
        st.error(f"Comparison failed: {e}")

elif len(selected_labels) == 1:
    st.info("Select at least 2 experiments to compare.")

# =====================================================
# EXPERIMENT DETAIL
# =====================================================
st.divider()
st.subheader("🔍 Experiment Details")

selected_detail = st.selectbox(
    "Select experiment to inspect",
    options=exp_names,
)

if selected_detail:
    detail_id  = exp_ids[exp_names.index(selected_detail)]
    detail_exp = next((e for e in experiments if e["id"] == detail_id), None)

    if detail_exp:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Configuration**")
            cfg_df = pd.DataFrame(
                [
                    ("Algorithm",    detail_exp["algorithm"]),
                    ("Problem Type", detail_exp.get("problem_type", "-")),
                    ("Tuning",       detail_exp["tuning_method"]),
                    ("CV Folds",     str(detail_exp["cv_folds"])),
                    ("Train Time",   f"{detail_exp.get('training_time', '-')}s"),
                    ("Date",         detail_exp["created_at"][:16]),
                ],
                columns=["Setting", "Value"],
            )
            cfg_df["Value"] = cfg_df["Value"].astype(str)
            st.table(cfg_df)

        with col2:
            st.markdown("**Hyperparameters**")
            params = detail_exp.get("params", {})
            if params:
                param_df = pd.DataFrame(
                    list(params.items()),
                    columns=["Parameter", "Value"],
                ).astype(str)
                st.dataframe(param_df, use_container_width=True)
            else:
                st.info("No hyperparameters recorded.")

        # Best params from auto-tuning
        best = detail_exp.get("metrics", {}).get("best_params")
        if best:
            st.markdown("**🏆 Best Auto-Tuned Parameters**")
            bp_df = pd.DataFrame(
                list(best.items()), columns=["Parameter", "Best Value"]
            ).astype(str)
            st.table(bp_df)

        # CV results
        cv = detail_exp.get("metrics", {}).get("cross_validation")
        if cv:
            st.markdown("**🔁 Cross-Validation Results**")
            cv_rows = []
            for metric_name, result in cv.items():
                cv_rows.append({
                    "Metric":  metric_name.upper(),
                    "Mean":    result["mean"],
                    "Std Dev": result["std"],
                    "Scores":  " | ".join(str(s) for s in result["scores"]),
                })
            st.dataframe(pd.DataFrame(cv_rows).astype(str), use_container_width=True)

        # Notes
        st.markdown("**📝 Notes**")
        current_notes = detail_exp.get("notes") or ""
        new_notes = st.text_area("Add/edit notes", value=current_notes, key=f"notes_{detail_id}")

        if st.button("💾 Save Notes", key=f"save_{detail_id}"):
            requests.patch(
                f"{API_BASE}/experiments/{detail_id}",
                json={"notes": new_notes},
            )
            st.success("Notes saved!")
            st.rerun()

        # Delete
        if st.button("🗑️ Delete Experiment", key=f"del_{detail_id}", type="secondary"):
            requests.delete(f"{API_BASE}/experiments/{detail_id}")
            st.success("Experiment deleted.")
            st.rerun()
