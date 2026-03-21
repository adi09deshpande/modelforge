# pages/9_Predict.py
import bootstrap  # noqa: F401
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ModelForge • Predict", layout="wide")
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

st.header("🎯 Prediction & Inference")

if not st.session_state.get("project_id") or not st.session_state.get("dataset_id"):
    st.info("Select a project and dataset first.")
    st.stop()

project_id = st.session_state["project_id"]
dataset_id = st.session_state["dataset_id"]

# ── Load prep config for target + problem type ────────────────────────────
try:
    prep = requests.get(
        f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30
    ).json()
except Exception:
    st.error("Could not load data preparation config.")
    st.stop()

target       = prep.get("target", "")
problem_type = prep.get("problem_type", "Classification")

# ── Load experiments to pick a model ─────────────────────────────────────
try:
    experiments = requests.get(
        f"{API_BASE}/experiments/project/{project_id}", timeout=30
    ).json()
except Exception:
    experiments = []

if not experiments:
    st.info("No trained models found. Train a model first.")
    st.stop()

# ── Model selector ────────────────────────────────────────────────────────
st.subheader("🤖 Select Model")

exp_labels = [
    f"{e['name']} — {e['algorithm']} (trained {e['created_at'][:10]})"
    for e in experiments
]
selected_label = st.selectbox("Choose a trained model", exp_labels)
selected_exp   = experiments[exp_labels.index(selected_label)]
model_id       = selected_exp.get("model_id")

if not model_id:
    st.error("Selected experiment has no associated model file.")
    st.stop()

# ── Get the ACTUAL features the model was trained with ────────────────────
try:
    model_info_resp = requests.get(
        f"{API_BASE}/predict/model-info/{model_id}", timeout=30
    )
    model_info = model_info_resp.json() if model_info_resp.status_code == 200 else {}
    trained_features = model_info.get("trained_features", [])
except Exception:
    trained_features = []

# Fallback to prep config features if we couldn't get from model
if not trained_features:
    trained_features = prep.get("features", [])

st.info(
    f"**Model:** {selected_exp['algorithm']}  |  "
    f"**Tuning:** {selected_exp['tuning_method']}  |  "
    f"**Target:** `{target}`  |  "
    f"**Features:** {len(trained_features)}"
)

# =====================================================
# PREDICTION MODE
# =====================================================
st.divider()
pred_mode = st.radio(
    "Prediction Mode",
    ["✏️ Single Row (Form)", "📁 Batch (CSV Upload)"],
    horizontal=True,
)

# =====================================================
# MODE 1: SINGLE ROW FORM
# =====================================================
if pred_mode == "✏️ Single Row (Form)":
    st.subheader("✏️ Enter Feature Values")

    # Load sample row for default values
    sample_values = {}
    try:
        preview = requests.get(
            f"{API_BASE}/dataset/{dataset_id}/preview",
            params={"n": 1}, timeout=10
        ).json()
        if preview.get("rows"):
            sample_values = preview["rows"][0]
    except Exception:
        pass

    # Load dtypes
    dtypes = {}
    try:
        stats  = requests.get(f"{API_BASE}/dataset/{dataset_id}/stats", timeout=10).json()
        dtypes = stats.get("dtypes", {})
    except Exception:
        pass

    # Build form using TRAINED features
    col1, col2 = st.columns(2)
    input_values = {}

    for i, feature in enumerate(trained_features):
        col = col1 if i % 2 == 0 else col2
        dtype   = dtypes.get(feature, "object")
        default = sample_values.get(feature, "")

        with col:
            if dtype.startswith(("int", "float")):
                try:
                    default_num = float(default) if default != "" else 0.0
                except (ValueError, TypeError):
                    default_num = 0.0
                input_values[feature] = st.number_input(
                    feature, value=default_num, key=f"input_{feature}"
                )
            else:
                input_values[feature] = st.text_input(
                    feature,
                    value=str(default) if default else "",
                    key=f"input_{feature}",
                )

    st.divider()
    if st.button("🚀 Predict", type="primary"):
        with st.spinner("Running prediction..."):
            resp = requests.post(
                f"{API_BASE}/predict/single",
                json={"model_id": model_id, "features": input_values},
                timeout=30,
            )

        if resp.status_code == 200:
            result = resp.json()
            st.success("✅ Prediction Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"Predicted {target}", value=str(result["prediction"]))
            with col2:
                if "confidence" in result:
                    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

            if "probabilities" in result:
                st.subheader("📊 Class Probabilities")
                prob_df = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Class", "Probability"],
                ).astype(str)
                st.dataframe(prob_df, use_container_width=True)
                prob_data = pd.DataFrame({"Probability": result["probabilities"]})
                st.bar_chart(prob_data)
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Prediction failed: {detail}")


# =====================================================
# MODE 2: BATCH CSV UPLOAD
# =====================================================
else:
    st.subheader("📁 Upload CSV for Batch Predictions")

    st.info(
        f"Upload a CSV with these **{len(trained_features)} columns** "
        f"(the model was trained with these exact features):\n\n"
        f"`{', '.join(trained_features)}`\n\n"
        f"The `{target}` column is NOT needed — it will be predicted."
    )

    # Template download
    template_df  = pd.DataFrame(columns=trained_features)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        csv_template,
        "prediction_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        preview_df = pd.read_csv(uploaded)
        st.write(f"**Preview** ({len(preview_df)} rows)")
        st.dataframe(preview_df.head(10), use_container_width=True)

        missing_cols = [f for f in trained_features if f not in preview_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner(f"Running predictions on {len(preview_df)} rows..."):
                    uploaded.seek(0)
                    resp = requests.post(
                        f"{API_BASE}/predict/batch",
                        params={"model_id": model_id},
                        files={"file": ("data.csv", uploaded, "text/csv")},
                        timeout=120,
                    )

                if resp.status_code == 200:
                    result    = resp.json()
                    result_df = pd.DataFrame(result["predictions"])

                    st.success(f"✅ Predictions complete for {result['total_rows']} rows!")

                    pred_col = "prediction"
                    if pred_col in result_df.columns:
                        st.subheader("📊 Prediction Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Prediction Distribution**")
                            st.dataframe(
                                result_df[pred_col].value_counts().reset_index(),
                                use_container_width=True,
                            )
                        with col2:
                            if problem_type == "Classification":
                                st.bar_chart(result_df[pred_col].value_counts())

                    st.subheader("📋 Full Results")
                    st.dataframe(result_df, use_container_width=True)

                    csv_out = result_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        csv_out,
                        "predictions.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"Batch prediction failed: {resp.json().get('detail', resp.text)}")
