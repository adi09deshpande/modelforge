import bootstrap  # noqa: F401
import requests
import streamlit as st  # type: ignore
import pandas as pd

st.set_page_config(page_title="ModelForge • Upload & Explore", layout="wide")

API_BASE = "http://127.0.0.1:8000"

# -------------------------
# SIDEBAR
# -------------------------
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
    
# -------------------------
# AUTH
# -------------------------
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# -------------------------
# PROJECT REQUIRED
# -------------------------
if not st.session_state.get("project_id"):
    st.info("Select a project first.")
    st.stop()

project_id = st.session_state["project_id"]

st.header("📂 Upload & Explore Data")
st.caption(f"Active project ID: {project_id}")

# =====================================================
# FETCH DATASETS
# =====================================================
def fetch_datasets():
    r = requests.get(f"{API_BASE}/dataset/project/{project_id}", timeout=30)
    r.raise_for_status()
    return r.json()

try:
    datasets = fetch_datasets()
except Exception as e:
    st.error(f"Failed to load datasets: {e}")
    datasets = []

# =====================================================
# EXISTING DATASETS
# =====================================================
st.subheader("📊 Existing datasets")

if datasets:
    default_idx = 0
    if "dataset_id" in st.session_state:
        for i, d in enumerate(datasets):
            if d["id"] == st.session_state["dataset_id"]:
                default_idx = i
                break

    ds = st.selectbox(
        "Select dataset",
        datasets,
        index=default_idx,
        format_func=lambda d: d["name"],
    )

    dataset_id = ds["id"]
    st.session_state["dataset_id"] = dataset_id

    # -----------------------------
    # STATS (PHASE-2 SAFE)
    # -----------------------------
    st.subheader("📈 Dataset Stats")

    try:
        stats_resp = requests.get(
            f"{API_BASE}/dataset/{dataset_id}/stats",
            timeout=30,
        )
        stats_resp.raise_for_status()
        stats = stats_resp.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows (estimated)", stats["rows_estimated"])
        c2.metric("Columns", stats["columns_count"])
        c3.metric("Size (MB)", stats["file_size_mb"])

    except Exception as e:
        st.warning(f"Could not load stats: {e}")

    st.divider()

    # -----------------------------
    # PREVIEW (SAFE)
    # -----------------------------
    st.subheader("Preview (first 50 rows)")

    try:
        preview_resp = requests.get(
            f"{API_BASE}/dataset/{dataset_id}/preview",
            params={"n": 50},
            timeout=30,
        )
        preview_resp.raise_for_status()

        preview = preview_resp.json()
        df = pd.DataFrame(preview.get("rows", []))

        if df.empty:
            st.warning("Dataset has no rows.")
        else:
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load preview: {e}")

else:
    st.info("No datasets yet.")

st.divider()

# =====================================================
# UPLOAD
# =====================================================
st.subheader("➕ Upload new dataset")

with st.form("upload_form", clear_on_submit=True):
    name = st.text_input("Dataset name (optional)")
    file = st.file_uploader("Upload CSV", type=["csv"])
    submit = st.form_submit_button("Upload")

if submit:
    if not file:
        st.warning("Please select a CSV file.")
    else:
        try:
            with st.spinner("Uploading dataset…"):
                r = requests.post(
                    f"{API_BASE}/dataset/upload",
                    data={
                        "project_id": project_id,
                        "name": name.strip() or file.name,
                    },
                    files={"file": (file.name, file, "text/csv")},
                    timeout=300,
                )
                r.raise_for_status()

            payload = r.json()
            st.session_state["dataset_id"] = payload["dataset_id"]

            st.success(f"Dataset '{payload['name']}' uploaded successfully.")
            st.rerun()

        except Exception as e:
            st.error(f"Upload error: {e}")
