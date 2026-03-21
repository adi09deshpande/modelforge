# pages/7_Model_Explainability.py
import bootstrap  # noqa: F401
import streamlit as st  # type: ignore
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ModelForge • Model Explainability",
    layout="wide",
)

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

st.header("🔍 Model Explainability")

if not st.session_state.get("project_id") or not st.session_state.get("dataset_id"):
    st.info("Select a project and dataset first.")
    st.stop()

project_id = st.session_state["project_id"]
dataset_id = st.session_state["dataset_id"]

# ── Load prep config ──────────────────────────────────────────────────────
prep = requests.get(
    f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30
).json()
problem_type = prep["problem_type"]

# ── Load latest model ─────────────────────────────────────────────────────
try:
    model_info = requests.get(
        f"{API_BASE}/train/latest/{project_id}", timeout=30
    ).json()
except Exception:
    st.info("No trained model found for this project.")
    st.stop()

model_id = model_info.get("id")
if not model_id:
    st.info("No trained model available yet.")
    st.stop()

# ── Load explainability ───────────────────────────────────────────────────
try:
    exp_resp    = requests.get(f"{API_BASE}/explain/{model_id}", timeout=60)
    explanation = exp_resp.json() if exp_resp.status_code == 200 else None
except Exception:
    explanation = None

# ── Model summary ─────────────────────────────────────────────────────────
st.subheader("📌 Model Summary")

summary_df = pd.DataFrame(
    [
        ("Algorithm",       str(model_info.get("algorithm"))),
        ("Problem Type",    str(problem_type)),
        ("Target Column",   str(prep["target"])),
        ("Encoding",        str(prep.get("encoding") or "None")),
        ("Scaling",         str(prep.get("scaling") or "None")),
        ("Test Size",       f"{int(prep['test_size'] * 100)}%"),
        ("No. of Features", str(len(prep["features"]))),
    ],
    columns=["Setting", "Value"],
)
summary_df["Value"] = summary_df["Value"].astype(str)
st.table(summary_df)

# ── Handle missing explainability ─────────────────────────────────────────
if not explanation:
    st.warning(
        "⚠️ Explainability data not available for this model.\n\n"
        "This usually happens when:\n"
        "- The model was trained before explainability was enabled\n"
        "- Explainability generation failed during training\n\n"
        "**Fix:** Retrain the model — explainability is generated automatically."
    )
    st.page_link("pages/6_Train_Model.py", label="🚀 Go to Train Model", icon="➡️")
    st.stop()


# =====================================================
# HELPER — flatten SHAP values safely
# =====================================================
def flatten_importance(raw: dict) -> dict:
    out = {}
    for feat, val in raw.items():
        if isinstance(val, (list, tuple)):
            out[feat] = float(np.mean(np.abs(val)))
        else:
            try:
                out[feat] = float(val)
            except (TypeError, ValueError):
                out[feat] = 0.0
    return out


# =====================================================
# GLOBAL SHAP
# =====================================================
st.divider()
st.subheader("🌍 Global Feature Importance (SHAP)")

global_imp = explanation.get("global_importance")

if not global_imp:
    st.info("Global SHAP importance not available for this model. Retrain to generate it.")
else:
    flat_imp  = flatten_importance(global_imp)
    global_df = (
        pd.DataFrame.from_dict(flat_imp, orient="index", columns=["Importance"])
        .sort_values("Importance", ascending=False)
    )

    top_k = st.slider(
        "Show Top K Features",
        min_value=1,
        max_value=min(30, len(global_df)),
        value=min(10, len(global_df)),
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
    st.info("Local SHAP explanation not available. Retrain to generate it.")
else:
    flat_local = flatten_importance(local["shap"])
    shap_df    = (
        pd.DataFrame.from_dict(flat_local, orient="index", columns=["Contribution"])
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
    st.info("LIME explanation not available. Retrain to generate it.")
else:
    flat_lime = flatten_importance(local["lime"])
    lime_df   = (
        pd.DataFrame.from_dict(flat_lime, orient="index", columns=["Contribution"])
        .sort_values("Contribution", ascending=False)
    )
    st.write("**LIME feature contributions (local interpretability)**")
    st.bar_chart(lime_df)

    with st.expander("📋 View LIME Table"):
        st.dataframe(lime_df, use_container_width=True)
