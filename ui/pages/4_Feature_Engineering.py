# pages/4_Feature_Engineering.py
import bootstrap  # noqa: F401
import requests
import streamlit as st  # type: ignore
import pandas as pd

st.set_page_config(page_title="ModelForge • Feature Engineering", layout="wide")

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

st.header("🛠️ Feature Engineering")

# ---------------- CONTEXT ----------------
if not st.session_state.get("dataset_id"):
    st.info("Select a dataset first.")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# =====================================================
# LOAD STATS (API)
# =====================================================
try:
    resp = requests.get(
        f"{API_BASE}/dataset/{dataset_id}/stats",
        timeout=30,
    )
    resp.raise_for_status()
    stats = resp.json()
except Exception as e:
    st.error(f"Failed to load dataset stats: {e}")
    st.stop()

# ---- SAFE METRICS ----
rows = stats.get("rows") or stats.get("rows_estimated", 0)
columns_count = (
    stats.get("columns_count")
    or len(stats.get("column_names", []))
    or len(stats.get("dtypes", {}))
)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", rows)
c2.metric("Columns", columns_count)
c3.metric("Size (MB)", stats.get("file_size_mb", 0))

# =====================================================
# LOAD PREVIEW (SAMPLE ONLY)
# =====================================================
try:
    preview_resp = requests.get(
        f"{API_BASE}/dataset/{dataset_id}/preview",
        params={"n": 200},
        timeout=30,
    )
    preview_resp.raise_for_status()
    preview = preview_resp.json()
    df = pd.DataFrame(preview.get("rows", []))
except Exception as e:
    st.error(f"Failed to load preview: {e}")
    st.stop()

if df.empty:
    st.warning("Dataset preview is empty. Feature engineering cannot be performed.")
    st.stop()

st.subheader("Dataset Preview (Sample)")
st.dataframe(df, use_container_width=True)

# =====================================================
# COLUMN GROUPS
# =====================================================
dtypes = stats.get("dtypes", {})

numeric_cols = [
    c for c, t in dtypes.items()
    if t.startswith(("int", "float"))
]

all_cols = list(dtypes.keys())

# =====================================================
# NUMERIC FEATURE
# =====================================================
st.subheader("Numeric Feature Creation")

if numeric_cols:
    col1 = st.selectbox("Column 1", numeric_cols, key="nf_col1")
    col2 = st.selectbox(
        "Column 2 (optional)",
        ["None"] + numeric_cols,
        key="nf_col2",
    )
    op = st.selectbox(
        "Operation",
        ["sum", "diff", "product", "ratio"],
        key="nf_op",
    )
    new_name = st.text_input(
        "New feature name",
        key="nf_name",
    )

    if st.button("Create Numeric Feature", key="nf_btn"):
        if not new_name:
            st.warning("Enter a new feature name")
        else:
            try:
                requests.post(
                    f"{API_BASE}/feature/numeric-feature",
                    json={
                        "dataset_id": dataset_id,
                        "col1": col1,
                        "col2": None if col2 == "None" else col2,
                        "operation": op,
                        "new_name": new_name,
                    },
                ).raise_for_status()
                st.success("Numeric feature created")
                st.rerun()
            except Exception as e:
                st.error(e)
else:
    st.info("No numeric columns available.")

# =====================================================
# NUMERIC TRANSFORM
# =====================================================
st.subheader("Numeric Transformation")

if numeric_cols:
    tcol = st.selectbox(
        "Numeric column",
        numeric_cols,
        key="nt_col",
    )
    transform = st.selectbox(
        "Transform",
        ["log", "square", "sqrt", "power", "bin"],
        key="nt_transform",
    )
    new_col = st.text_input(
        "New column name",
        key="nt_name",
    )
    power = (
        st.number_input("Power", value=2, key="nt_power")
        if transform == "power"
        else None
    )
    bins = (
        st.number_input("Bins", value=5, min_value=2, key="nt_bins")
        if transform == "bin"
        else None
    )

    if st.button("Apply Transformation", key="nt_btn"):
        if not new_col:
            st.warning("Enter a new column name")
        else:
            try:
                requests.post(
                    f"{API_BASE}/feature/numeric-transform",
                    json={
                        "dataset_id": dataset_id,
                        "column": tcol,
                        "transform": transform,
                        "new_name": new_col,
                        "power": power,
                        "bins": bins,
                    },
                ).raise_for_status()
                st.success("Transformation applied")
                st.rerun()
            except Exception as e:
                st.error(e)

# =====================================================
# DATE FEATURES
# =====================================================
st.subheader("Date Feature Extraction")

dcol = st.selectbox(
    "Date column",
    all_cols,
    key="df_col",
)
features = st.multiselect(
    "Extract",
    ["year", "month", "day", "weekday", "hour", "minute", "second", "quarter"],
    key="df_features",
)
prefix = st.text_input(
    "Prefix for new columns",
    key="df_prefix",
)
keep_orig = st.checkbox(
    "Keep original column",
    value=True,
    key="df_keep",
)

if st.button("Extract Date Features", key="df_btn"):
    if not features or not prefix:
        st.warning("Select features and prefix")
    else:
        try:
            requests.post(
                f"{API_BASE}/feature/date-features",
                json={
                    "dataset_id": dataset_id,
                    "column": dcol,
                    "features": features,
                    "prefix": prefix,
                    "keep_original": keep_orig,
                },
            ).raise_for_status()
            st.success("Date features extracted")
            st.rerun()
        except Exception as e:
            st.error(e)

# =====================================================
# AGE FROM DOB
# =====================================================
st.subheader("Age from DOB")

dob_col = st.selectbox(
    "DOB column",
    all_cols,
    key="age_col",
)
age_name = st.text_input(
    "Age column name",
    key="age_name",
)
keep_dob = st.checkbox(
    "Keep DOB",
    value=False,
    key="age_keep",
)

if st.button("Create Age Feature", key="age_btn"):
    if not age_name:
        st.warning("Enter age column name")
    else:
        try:
            requests.post(
                f"{API_BASE}/feature/age-from-dob",
                json={
                    "dataset_id": dataset_id,
                    "dob_column": dob_col,
                    "new_name": age_name,
                    "keep_original": keep_dob,
                },
            ).raise_for_status()
            st.success("Age feature created")
            st.rerun()
        except Exception as e:
            st.error(e)
