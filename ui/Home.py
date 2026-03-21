# Home.py
import bootstrap  # noqa: F401
import streamlit as st # type: ignore
import streamlit_authenticator as stauth # type: ignore
from streamlit_cookies_controller import CookieController # type: ignore

# DB helpers
from db.db import get_session
from services.services_users import get_user_by_username, list_users_for_auth

st.set_page_config(page_title="ModelForge • Home", layout="wide")

# -------------------------------------------------
# Logout helpers
# -------------------------------------------------
cookie = CookieController()

def purge_auth_cookie():
    try:
        cookie.delete("modelforge_auth")
    except Exception:
        pass

COOKIE_NAME = "modelforge_auth"
COOKIE_KEY  = "b4jF2zQ-4y8p9T0cNfE6rWmX3kV9qL2sY7hD1uJ0aP5cR8eG6iM4nB2vQ1tZ3x5"

def hard_logout_and_redirect():
    purge_auth_cookie()
    try:
        db    = get_session()
        creds = {"credentials": list_users_for_auth(db)}
        _auth = stauth.Authenticate(
            credentials=creds["credentials"],
            cookie_name=COOKIE_NAME,
            key=COOKIE_KEY,
            cookie_expiry_days=30,
            preauthorized={"emails": []},
        )
        _auth.logout("Logout", "main", key="logout_force")
    except Exception:
        pass

    for k in [
        "user_id", "display_name", "project_id", "project_path",
        "undo_stack", "redo_stack", "data", "authentication_status", "project_slug",
    ]:
        st.session_state.pop(k, None)

    try:
        st.switch_page("pages/0_Master.py")
    except Exception:
        st.rerun()

# -------------------------------------------------
# Guard: require login
# -------------------------------------------------
if not st.session_state.get("user_id"):
    purge_auth_cookie()
    hard_logout_and_redirect()

# -------------------------------------------------
# Authenticator setup for sidebar logout
# -------------------------------------------------
db            = get_session()
creds         = {"credentials": list_users_for_auth(db)}
authenticator = stauth.Authenticate(
    credentials=creds["credentials"],
    cookie_name=COOKIE_NAME,
    key=COOKIE_KEY,
    cookie_expiry_days=30,
    preauthorized={"emails": []},
)

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

.hero {
  background: radial-gradient(85% 120% at 10% 10%, #153041 0%, #0e1620 60%),
              linear-gradient(135deg,#0a0d14,#16232d,#1a3848);
  color: #fff; padding: 28px 26px; border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 26px rgba(0,0,0,.35), inset 0 0 80px rgba(255,255,255,.02);
  text-align: center;
}
.hero h1 { margin: 0 0 6px 0; font-size: 42px; letter-spacing: .4px; }
.hero h2 { margin: 6px 0 0 0; font-weight: 500; color: #cfe7ff; font-size: 20px; }

.section-title {
  margin: 16px 0 8px 2px; color: #dbe8f7; font-weight: 700;
  letter-spacing: .3px; font-size: 18px;
}

.card {
  background: #1d212c; border-radius: 12px; padding: 14px; height: 170px;
  border: 1px solid rgba(255,255,255,0.06);
  transition: border-color .16s ease, transform .16s ease, box-shadow .16s ease;
}
.card:hover {
  border-color: rgba(120,180,255,0.35);
  transform: translateY(-3px);
  box-shadow: 0 12px 26px rgba(0,0,0,.28);
}
.card-title  { color: #f3f6ff; font-weight: 700; }
.card-desc   { color: #b9c7d8; margin-top: 6px; font-size: 14px; }
.card-footer { margin-top: 14px; }
.open-wrap   { display: flex; justify-content: flex-start; }

.nav-section-label {
  font-size: 11px; font-weight: 700; letter-spacing: 1px;
  color: rgba(255,255,255,0.35); text-transform: uppercase;
  margin: 12px 0 4px 4px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown('<div><h2>Account</h2></div>', unsafe_allow_html=True)

    if st.button("Logout", type="secondary", key="logout_click"):
        try:
            authenticator.logout("Logout", "sidebar", key="logout_sidebar_hidden")
        except Exception:
            pass
        hard_logout_and_redirect()

    display_name = st.session_state.get("display_name") or st.session_state.get("user_id", "User")
    st.markdown(
        f'<div style="margin-top:.25rem;"><strong>👋 Welcome, {display_name}</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.10);margin:0.75rem 0 0.5rem 0;"></div>',
        unsafe_allow_html=True,
    )

    # ── Navigation ────────────────────────────────────────────────────────
    st.markdown('<div style="text-align:center;"><h3 style="margin:.2rem 0;">Navigate</h3></div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">📁 Core</div>', unsafe_allow_html=True)
    st.page_link("Home.py",                        label="🏠 Home")
    st.page_link("pages/1_Projects.py",            label="📁 Projects")

    st.markdown('<div class="nav-section-label">📊 Data</div>', unsafe_allow_html=True)
    st.page_link("pages/2_Upload_Data.py",         label="📂 Upload & Explore")
    st.page_link("pages/3_EDA_Preprocessing.py",   label="📊 EDA & Preprocessing")
    st.page_link("pages/4_Feature_Engineering.py", label="🛠️ Feature Engineering")
    st.page_link("pages/10_Feature_Selection.py",  label="🎯 Feature Selection")
    st.page_link("pages/5_Data_Preparation.py",    label="📦 Data Preparation")

    st.markdown('<div class="nav-section-label">🤖 Modelling</div>', unsafe_allow_html=True)
    st.page_link("pages/6_Train_Model.py",         label="🚀 Train Model")
    st.page_link("pages/8_Experiments.py",         label="🧪 Experiments")
    st.page_link("pages/7_Model_Explainability.py",label="🧠 Explainability")
    st.page_link("pages/9_Predict.py",             label="🎯 Predict")

# -------------------------------------------------
# Hero
# -------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>⚡ Model Forge</h1>
  <h2>Build, Compare & Explain ML Models in Minutes 🚀</h2>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Workflow grid — Data Pipeline
# -------------------------------------------------
st.markdown('<div class="section-title">📊 Data Pipeline</div>', unsafe_allow_html=True)

data_steps = [
    ("📁 Projects",             "pages/1_Projects.py",            "Create or select a project workspace"),
    ("📂 Upload & Explore",     "pages/2_Upload_Data.py",         "Upload your dataset to the active project"),
    ("📊 EDA & Preprocessing",  "pages/3_EDA_Preprocessing.py",   "Explore, clean and fix your data"),
    ("🛠️ Feature Engineering",  "pages/4_Feature_Engineering.py", "Create and transform features"),
    ("🎯 Feature Selection",    "pages/10_Feature_Selection.py",  "Select the most important features"),
    ("📦 Data Preparation",     "pages/5_Data_Preparation.py",    "Split, encode and scale your data"),
]

cols = st.columns(3, gap="large")
for idx, (title, page, desc) in enumerate(data_steps):
    with cols[idx % 3]:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="card-footer open-wrap">
        """, unsafe_allow_html=True)
        st.page_link(page, label="Open", icon="➡️", use_container_width=False)
        st.markdown("</div></div>", unsafe_allow_html=True)

# -------------------------------------------------
# Workflow grid — Modelling
# -------------------------------------------------
st.markdown('<div class="section-title">🤖 Modelling</div>', unsafe_allow_html=True)

model_steps = [
    ("🚀 Train Model",     "pages/6_Train_Model.py",          "Train models with auto-tuning & CV"),
    ("🧪 Experiments",     "pages/8_Experiments.py",          "Compare and track all training runs"),
    ("🧠 Explainability",  "pages/7_Model_Explainability.py", "SHAP & LIME feature explanations"),
    ("🎯 Predict",         "pages/9_Predict.py",              "Run predictions on new data"),
]

cols2 = st.columns(3, gap="large")
for idx, (title, page, desc) in enumerate(model_steps):
    with cols2[idx % 3]:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="card-footer open-wrap">
        """, unsafe_allow_html=True)
        st.page_link(page, label="Open", icon="➡️", use_container_width=False)
        st.markdown("</div></div>", unsafe_allow_html=True)
