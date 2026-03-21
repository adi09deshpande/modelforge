# pages/10_Feature_Selection.py
import bootstrap  # noqa: F401
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ModelForge • Feature Selection", layout="wide")
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

st.header("🎯 Feature Selection")

if not st.session_state.get("dataset_id"):
    st.info("Select a dataset first.")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ── Load prep config ──────────────────────────────────────────────────────
try:
    prep = requests.get(
        f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30
    ).json()
except Exception:
    st.error("Could not load data preparation config. Complete Data Preparation first.")
    st.stop()

problem_type = prep.get("problem_type", "Classification")
target       = prep.get("target", "")
features     = prep.get("features", [])

st.info(
    f"**Problem Type:** {problem_type}  |  "
    f"**Target:** `{target}`  |  "
    f"**Current Features:** {len(features)}"
)

# =====================================================
# METHOD SELECTION
# =====================================================
st.divider()
st.subheader("🔧 Select Feature Selection Method")

method = st.radio(
    "Method",
    [
        "🌲 Feature Importance (Tree-Based)",
        "📉 Correlation Threshold",
        "🔁 Recursive Feature Elimination (RFE)",
    ],
    index=0,
)

METHOD_MAP = {
    "🌲 Feature Importance (Tree-Based)": "importance",
    "📉 Correlation Threshold":           "correlation",
    "🔁 Recursive Feature Elimination (RFE)": "rfe",
}
selected_method = METHOD_MAP[method]

# ── Method descriptions ───────────────────────────────────────────────────
if selected_method == "importance":
    st.info(
        "**Feature Importance** trains a Random Forest on your data and ranks features "
        "by how much they contribute to predictions.\n\n"
        "✅ Works for both Classification and Regression\n"
        "✅ Fast and interpretable\n"
        "✅ Good starting point for feature selection"
    )
elif selected_method == "correlation":
    st.info(
        "**Correlation Threshold** removes features that are highly correlated with each other "
        "(multicollinearity), keeping only one from each correlated group.\n\n"
        "✅ Reduces redundancy in features\n"
        "✅ Improves model stability\n"
        "✅ Works well before linear models"
    )
elif selected_method == "rfe":
    st.info(
        "**RFE** recursively removes the least important features and retrains until "
        "the desired number of features remains.\n\n"
        "✅ Model-guided selection\n"
        "✅ Very precise\n"
        "⚠️ Slower than other methods for large feature sets"
    )

# ── Method parameters ─────────────────────────────────────────────────────
st.subheader("⚙️ Parameters")

top_n = 10
importance_threshold = 0.01
correlation_threshold = 0.9
n_rfe_features = 5

if selected_method == "importance":
    top_n = st.slider(
        "Max features to keep (Top N)",
        min_value=1,
        max_value=len(features),
        value=min(10, len(features)),
    )
    importance_threshold = st.slider(
        "Minimum importance threshold",
        min_value=0.0,
        max_value=0.1,
        value=0.01,
        step=0.005,
        format="%.3f",
    )

elif selected_method == "correlation":
    correlation_threshold = st.slider(
        "Correlation threshold (remove above this)",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05,
        format="%.2f",
    )
    st.caption(
        "Features with correlation **above** this threshold with another feature will be removed. "
        "Lower = more aggressive removal."
    )

elif selected_method == "rfe":
    n_rfe_features = st.slider(
        "Number of features to select",
        min_value=1,
        max_value=len(features),
        value=min(5, len(features)),
    )

# =====================================================
# RUN FEATURE SELECTION
# =====================================================
st.divider()

if st.button("🚀 Run Feature Selection", type="primary"):
    with st.spinner("Running feature selection..."):
        resp = requests.post(
            f"{API_BASE}/feature-selection/run",
            json={
                "dataset_id":             dataset_id,
                "method":                 selected_method,
                "top_n":                  top_n,
                "importance_threshold":   importance_threshold,
                "correlation_threshold":  correlation_threshold,
                "n_rfe_features":         n_rfe_features,
            },
            timeout=120,
        )

    if resp.status_code == 200:
        result = resp.json()
        st.session_state["fs_result"] = result
    else:
        st.error(f"Feature selection failed: {resp.text}")

# =====================================================
# DISPLAY RESULTS
# =====================================================
if "fs_result" in st.session_state:
    result = st.session_state["fs_result"]

    selected_features = result.get("selected_features", [])
    selected_list     = result.get("selected", [])
    removed_list      = result.get("removed", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Features",    result.get("total_features", len(features)))
    col2.metric("✅ Selected",        len(selected_features))
    col3.metric("❌ Removed",         len(removed_list))

    st.divider()

    # ── Importance bar chart ──────────────────────────────────────────────
    if selected_method == "importance" and selected_list:
        st.subheader("📊 Feature Importance Scores")

        imp_df = pd.DataFrame(selected_list).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.4)))
        ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
        ax.set_xlabel("Importance Score")
        ax.set_title("Selected Features by Importance")
        st.pyplot(fig)
        plt.close()

    # ── Correlation target chart ──────────────────────────────────────────
    if selected_method == "correlation":
        target_corr = result.get("target_correlations", {})
        if target_corr:
            st.subheader(f"📈 Feature Correlation with Target (`{target}`)")
            corr_df = pd.DataFrame(
                sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True),
                columns=["Feature", "Correlation with Target"],
            )
            st.dataframe(corr_df.astype(str), use_container_width=True)

    # ── Selected features table ───────────────────────────────────────────
    st.subheader("✅ Selected Features")
    if selected_list:
        sel_df = pd.DataFrame(selected_list).astype(str)
        st.dataframe(sel_df, use_container_width=True)
    else:
        st.info("No features selected.")

    # ── Removed features ─────────────────────────────────────────────────
    with st.expander(f"❌ Removed Features ({len(removed_list)})"):
        if removed_list:
            rem_df = pd.DataFrame(removed_list).astype(str)
            st.dataframe(rem_df, use_container_width=True)
        else:
            st.info("No features removed.")

    # ── Apply to Data Preparation ─────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Apply to Data Preparation")

    st.write(
        f"The following **{len(selected_features)} features** will be used for training:"
    )
    st.code(", ".join(selected_features))

    if st.button("✅ Apply Selected Features to Data Preparation", type="primary"):
        # Load current prep config
        current_prep = requests.get(
            f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=10
        ).json()

        # Update features list
        updated_payload = {
            "problem_type": current_prep["problem_type"],
            "target":       current_prep["target"],
            "features":     selected_features,
            "test_size":    current_prep["test_size"],
            "stratify":     current_prep["stratify"],
            "encoding":     current_prep.get("encoding"),
            "scaling":      current_prep.get("scaling"),
        }

        resp = requests.post(
            f"{API_BASE}/dataset-preparation/{dataset_id}",
            json=updated_payload,
            timeout=10,
        )

        if resp.status_code == 200:
            st.success(
                f"✅ Data preparation updated with {len(selected_features)} selected features! "
                "Go to Train Model to retrain."
            )
        else:
            st.error(f"Failed to update: {resp.text}")
