import bootstrap  # noqa: F401
import streamlit as st  # type: ignore
import requests
import pandas as pd

st.set_page_config(
    page_title="ModelForge • Model Explainability",
    layout="wide",
)

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

st.header("🔍 Model Explainability")

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
# LOAD LATEST MODEL (SAFE)
# =====================================================
try:
    model_info = requests.get(
        f"{API_BASE}/train/latest/{project_id}",
        timeout=30,
    ).json()
except Exception:
    st.info("No trained model found for this project.")
    st.stop()

model_id = model_info.get("id")
if not model_id:
    st.info("No trained model available yet.")
    st.stop()

# =====================================================
# LOAD EXPLAINABILITY (SAFE)
# =====================================================
try:
    explanation = requests.get(
        f"{API_BASE}/explain/{model_id}",
        timeout=60,
    ).json()
except Exception:
    st.info("Explainability data not available for this model.")
    st.stop()

if not explanation:
    st.info("Explainability not generated for this model.")
    st.stop()

# =====================================================
# MODEL SUMMARY
# =====================================================
st.subheader("📌 Model Summary")

summary_df = pd.DataFrame(
    [
        ("Algorithm", model_info.get("algorithm")),
        ("Problem Type", problem_type),
        ("Target Column", prep["target"]),
        ("Encoding", prep.get("encoding") or "None"),
        ("Scaling", prep.get("scaling") or "None"),
        ("Test Size", f"{int(prep['test_size'] * 100)}%"),
        ("No. of Features", len(prep["features"])),
    ],
    columns=["Setting", "Value"],
)

st.table(summary_df)

# =====================================================
# GLOBAL SHAP
# =====================================================
st.divider()
st.subheader("🌍 Global Feature Importance (SHAP)")

global_imp = explanation.get("global_importance")

if not global_imp:
    st.info("Global SHAP importance not available for this model.")
else:
    global_df = (
        pd.DataFrame.from_dict(global_imp, orient="index", columns=["Importance"])
        .sort_values("Importance", ascending=False)
    )

    top_k = st.slider(
        "Show Top K Features",
        min_value=5,
        max_value=min(30, len(global_df)),
        value=10,
    )

    st.bar_chart(global_df.head(top_k))

    with st.expander("📋 View Full SHAP Importance Table"):
        st.dataframe(global_df, use_container_width=True)

# =====================================================
# LOCAL SHAP
# =====================================================
st.divider()
st.subheader("🔬 Local Explanation (SHAP)")

local = explanation.get("local_explanation")

if not local or "shap" not in local:
    st.info("Local SHAP explanation not available.")
else:
    shap_df = (
        pd.DataFrame.from_dict(local["shap"], orient="index", columns=["Contribution"])
        .sort_values("Contribution", ascending=False)
    )

    st.write("**Feature contribution for a representative sample**")
    st.bar_chart(shap_df)
    st.caption(f"Base value: {local.get('base_value')}")

# =====================================================
# LIME
# =====================================================
st.divider()
st.subheader("🍋 Local Explanation (LIME)")

if not local or "lime" not in local:
    st.info("LIME explanation not available.")
else:
    lime_df = (
        pd.DataFrame.from_dict(local["lime"], orient="index", columns=["Contribution"])
        .sort_values("Contribution", ascending=False)
    )

    st.write("**LIME feature contributions (local interpretability)**")
    st.bar_chart(lime_df)

    with st.expander("📋 View LIME Table"):
        st.dataframe(lime_df, use_container_width=True)
