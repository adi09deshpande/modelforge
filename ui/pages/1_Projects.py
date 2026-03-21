import bootstrap  # noqa
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="ModelForge • Projects", layout="wide")

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
# AUTH GATE
# -------------------------
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

st.header("📁 Projects")

mode = st.radio(
    "Choose action",
    ["Select existing project", "Create new project"],
    index=0,
)

st.divider()

# =========================
# SELECT PROJECT
# =========================
if mode == "Select existing project":
    resp = requests.get(
        f"{API_BASE}/projects",
        params={"user_id": st.session_state["user_id"]},
    )

    if resp.status_code != 200:
        st.error("Failed to load projects")
        st.stop()

    projects = resp.json()
    options = ["<None>"] + [f"{p['title']} ({p['slug']})" for p in projects]

    sel = st.selectbox("Select a project", options)

    if sel != "<None>":
        idx = options.index(sel) - 1
        p = projects[idx]

        st.session_state["project_id"] = p["id"]
        st.session_state["project_slug"] = p["slug"]

        st.success(f"Loaded project: {p['title']}")

# =========================
# CREATE PROJECT
# =========================
else:
    st.subheader("Create new project")

    name = st.text_input("Project name")
    create = st.button("Create project", type="primary")

    if create:
        if not name.strip():
            st.warning("Enter a project name")
        else:
            resp = requests.post(
                f"{API_BASE}/projects",
                json={
                    "title": name,
                    "user_id": st.session_state["user_id"],
                },
            )

            if resp.status_code != 200:
                st.error(resp.text)
            else:
                p = resp.json()
                st.session_state["project_id"] = p["id"]
                st.session_state["project_slug"] = p["slug"]

                st.success(f"Project created: {p['title']}")
                st.info("Next: Upload a dataset")

st.divider()

# -------------------------
# FOOTER
# -------------------------
if st.session_state.get("project_id"):
    st.success("Project ready 🚀")
else:
    st.info("No active project")
