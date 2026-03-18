import bootstrap  # noqa: F401
import requests
import streamlit as st  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="ModelForge • EDA & Preprocessing", layout="wide")

API_BASE = "http://127.0.0.1:8000"

# =====================================================
# SIDEBAR
# =====================================================
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

    st.divider()
    st.subheader("🕒 Dataset Versioning")

    if st.session_state.get("dataset_id"):
        try:
            versions = requests.get(
                f"{API_BASE}/dataset/{st.session_state['dataset_id']}/versions",
                timeout=10,
            ).json()

            version_numbers = [v["version_number"] for v in versions]
            if version_numbers:
                selected = st.selectbox("Rollback to version", version_numbers)
                if st.button("Rollback Dataset"):
                    requests.post(
                        f"{API_BASE}/dataset/{st.session_state['dataset_id']}/rollback",
                        json={"version_number": selected},
                    )
                    st.success("Dataset rolled back")
                    st.rerun()
        except Exception:
            st.caption("Version history unavailable")

# =====================================================
# AUTH
# =====================================================
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

st.header("📊 EDA & Preprocessing")

# =====================================================
# CONTEXT
# =====================================================
if not st.session_state.get("project_id"):
    st.info("Select a project first.")
    st.stop()

if not st.session_state.get("dataset_id"):
    st.info("Select a dataset first.")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# =====================================================
# DATASET STATS
# =====================================================
stats = requests.get(
    f"{API_BASE}/dataset/{dataset_id}/stats",
    timeout=60,
).json()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", stats["rows_estimated"])
c2.metric("Columns", stats["columns_count"])
c3.metric("Size (MB)", stats["file_size_mb"])

# =====================================================
# LOAD FULL DATASET (ONCE)
# =====================================================
with st.spinner("Loading full dataset… this may take time for large files"):
    csv_bytes = requests.get(
        f"{API_BASE}/dataset/{dataset_id}/current",
        timeout=180,
    ).content
    df = pd.read_csv(BytesIO(csv_bytes))

# =====================================================
# PREVIEW
# =====================================================
st.subheader("Dataset Preview")
st.dataframe(df.head(200), use_container_width=True)
st.caption(f"Dataset shape: {df.shape}")

# =====================================================
# DATASET INFO (FULL)
# =====================================================
st.subheader("Dataset Info (Full Dataset)")

info_df = pd.DataFrame(
    {
        "Dtype": df.dtypes.astype(str),
        "Missing values": df.isnull().sum(),
    }
)

st.dataframe(info_df, use_container_width=True)

# =====================================================
# DESCRIBE (FULL DATASET)
# =====================================================
st.subheader("📐 Statistical Summary (df.describe)")
st.caption("Computed on the full dataset")

st.dataframe(df.describe(include="all"), use_container_width=True)

# =====================================================
# TYPE CONVERSION
# =====================================================
st.subheader("Convert Column Datatypes")

col = st.selectbox("Column", df.columns)
dtype = st.selectbox("Target dtype", ["int", "float", "str", "category"])

if st.button("Convert"):
    try:
        requests.post(
            f"{API_BASE}/preprocessing/convert-dtype",
            json={
                "dataset_id": dataset_id,
                "column": col,
                "dtype": dtype,
            },
        ).raise_for_status()
        st.toast("Column datatype converted")
        st.rerun()
    except Exception as e:
        st.error(e)

# =====================================================
# MISSING VALUES
# =====================================================
st.subheader("Missing Values Handling (Full Dataset)")

strategy = st.selectbox(
    "Strategy",
    ["Do nothing", "Drop rows", "Fill mean", "Fill median", "Fill mode", "Fill custom"],
)

custom = st.text_input("Custom value") if strategy == "Fill custom" else None

strategy_map = {
    "Drop rows": "drop",
    "Fill mean": "mean",
    "Fill median": "median",
    "Fill mode": "mode",
    "Fill custom": "custom",
}

if st.button("Apply Missing Value Strategy") and strategy != "Do nothing":
    requests.post(
        f"{API_BASE}/preprocessing/missing-values",
        json={
            "dataset_id": dataset_id,
            "strategy": strategy_map[strategy],
            "value": custom,
        },
    )
    st.toast("Missing values handled")
    st.rerun()

# =====================================================
# DUPLICATES
# =====================================================
st.subheader("Duplicate Rows (Full Dataset)")

if st.button("Drop Duplicates"):
    requests.post(
        f"{API_BASE}/preprocessing/drop-duplicates",
        json={"dataset_id": dataset_id},
    )
    st.toast("Duplicates removed")
    st.rerun()

# =====================================================
# DROP COLUMNS
# =====================================================
st.subheader("Drop Columns")

cols = st.multiselect("Columns to drop", df.columns)

if st.button("Drop Selected Columns") and cols:
    requests.post(
        f"{API_BASE}/preprocessing/drop-columns",
        json={
            "dataset_id": dataset_id,
            "columns": cols,
        },
    )
    st.toast("Columns dropped")
    st.rerun()

# =====================================================
# VISUALISATIONS (FULL DATASET)
# =====================================================
st.subheader("📈 Visualisations (Full Dataset)")

st.warning(
    "Visualisations are computed on the FULL dataset. "
    "This may take time for large datasets. "
    "Background processing will be added later."
)

col_to_plot = st.selectbox("Select column", df.columns)

fig, ax = plt.subplots()

if pd.api.types.is_numeric_dtype(df[col_to_plot]):
    sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax)
else:
    df[col_to_plot].value_counts().plot(kind="bar", ax=ax)

st.pyplot(fig)
