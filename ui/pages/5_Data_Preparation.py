# pages/5_Data_Preparation.py
import bootstrap  # noqa: F401
import requests
import streamlit as st  # type: ignore
import pandas as pd
import json
import hashlib

st.set_page_config(page_title="ModelForge • Data Preparation", layout="wide")

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

st.header("🛠️ Data Preparation")

# ---------------- CONTEXT ----------------
if not st.session_state.get("dataset_id"):
    st.info("Select a dataset first.")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# =====================================================
# HELPERS
# =====================================================
def hash_payload(payload: dict) -> str:
    return hashlib.md5(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def autosave(payload: dict):
    try:
        requests.post(
            f"{API_BASE}/dataset-preparation/{dataset_id}",
            json=payload,
            timeout=10,
        )
    except Exception as e:
        st.warning(f"Auto-save failed: {e}")

# =====================================================
# LOAD DATASET STATS
# =====================================================
stats = requests.get(
    f"{API_BASE}/dataset/{dataset_id}/stats",
    timeout=30,
).json()

dtypes = stats.get("dtypes", {})
all_cols = list(dtypes.keys())
num_cols = [c for c, t in dtypes.items() if t.startswith(("int", "float"))]
cat_cols = [c for c, t in dtypes.items() if t in ("object", "category")]

# =====================================================
# LOAD SAVED PREPARATION CONFIG
# =====================================================
prep_config = None
try:
    r = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=10)
    if r.status_code == 200:
        prep_config = r.json()
except Exception:
    pass

# =====================================================
# DATA PREVIEW
# =====================================================
st.subheader("📄 Dataset Preview")

preview = requests.get(
    f"{API_BASE}/dataset/{dataset_id}/preview",
    params={"n": 200},
).json()

st.dataframe(
    pd.DataFrame(preview.get("rows", [])),
    use_container_width=True,
)

# =====================================================
# PROBLEM TYPE
# =====================================================
st.subheader("🧩 Problem Type")

problem_type = st.selectbox(
    "Problem Type",
    ["Classification", "Regression"],
    index=(
        ["Classification", "Regression"].index(prep_config["problem_type"])
        if prep_config
        else 0
    ),
)

# =====================================================
# TARGET & FEATURES (AUTO-DERIVED)
# =====================================================
st.subheader("🎯 Target & Features")

target = st.selectbox(
    "Target column",
    all_cols,
    index=(
        all_cols.index(prep_config["target"])
        if prep_config and prep_config["target"] in all_cols
        else 0
    ),
)

# ✅ AUTO-DERIVE FEATURES
features = [c for c in all_cols if c != target]

st.caption(
    f"Automatically selected {len(features)} feature columns "
    f"(all columns except target)."
)


# =====================================================
# TRAIN–TEST SPLIT CONFIG
# =====================================================
st.subheader("📦 Train–Test Split")

test_size = st.slider(
    "Test size (%)",
    10,
    50,
    int(prep_config["test_size"] * 100) if prep_config else 20,
)

stratify = False
if problem_type == "Classification":
    stratify = st.checkbox(
        "Stratify (preserve class balance)",
        value=prep_config["stratify"] if prep_config else True,
    )


# =====================================================
# ENCODING & SCALING GUIDANCE (EDUCATIONAL)
# =====================================================
st.divider()
st.subheader("ℹ️ Encoding & Scaling Guidance")

col1, col2 = st.columns(2)

with col1:
    st.info(
        """
        **🔢 Encoding**
        
        - **Label Encoding**  
          ✔ Tree-based models  
          ❌ Linear / distance-based models
        
        - **One-Hot Encoding**  
          ✔ Logistic / Linear models  
          ✔ Distance-based models  
          ❌ High-cardinality columns
        """
    )

with col2:
    st.info(
        """
        **📏 Scaling**
        
        - **Standardization (Z-score)**  
          ✔ Logistic / Linear models  
          ✔ Gradient-based models
        
        - **Normalization (Min-Max)**  
          ✔ Distance-based models (KNN, SVM)
        
        - **No scaling**  
          ✔ Tree-based models
        """
    )
    
# =====================================================
# ENCODING & SCALING
# =====================================================
st.subheader("🔧 Encoding & Scaling")

encoding = st.selectbox(
    "Encoding",
    ["None", "Label Encoding", "One-Hot Encoding"],
    index={
        None: 0,
        "label": 1,
        "onehot": 2,
    }.get(prep_config.get("encoding") if prep_config else None, 0),
)

scaling = st.selectbox(
    "Scaling",
    ["None", "Standardization", "Normalization"],
    index={
        None: 0,
        "standard": 1,
        "minmax": 2,
    }.get(prep_config.get("scaling") if prep_config else None, 0),
)

# =====================================================
# AUTO-SAVE CONFIG (SINGLE SOURCE OF TRUTH)
# =====================================================
if target and features:
    payload = {
        "problem_type": problem_type,
        "target": target,
        "features": features,
        "test_size": test_size / 100,
        "stratify": stratify,
        "encoding": {
            "None": None,
            "Label Encoding": "label",
            "One-Hot Encoding": "onehot",
        }[encoding],
        "scaling": {
            "None": None,
            "Standardization": "standard",
            "Normalization": "minmax",
        }[scaling],
    }

    payload_hash = hash_payload(payload)

    if st.session_state.get("last_prep_hash") != payload_hash:
        autosave(payload)
        st.session_state["last_prep_hash"] = payload_hash
        st.toast("Preparation auto-saved", icon="💾")
