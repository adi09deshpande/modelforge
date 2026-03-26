# Home.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "components"))

import streamlit as st
import streamlit_authenticator as stauth
from streamlit_cookies_controller import CookieController

from db.db import get_session
from services.services_users import list_users_for_auth

st.set_page_config(page_title="ModelForge", layout="wide", initial_sidebar_state="expanded")

from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

# ── Sidebar toggle fix ────────────────────────────────────────────────────
st.html("""
<style>
[data-testid="collapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    height: 100vh !important;
    width: 24px !important;
    background: rgba(12,14,22,0.6) !important;
    border-right: 1px solid rgba(79,142,247,.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    top: 0 !important;
    left: 0 !important;
    z-index: 999999 !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(79,142,247,0.1) !important;
    border-right-color: rgba(79,142,247,.7) !important;
    width: 32px !important;
}
[data-testid="collapsedControl"] svg {
    color: #4f8ef7 !important;
    stroke: #4f8ef7 !important;
    width: 16px !important;
    height: 16px !important;
}
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] button {
    color: #6070a0 !important;
    opacity: 0.8 !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    color: #a0a8d0 !important;
    opacity: 1 !important;
}
</style>
""")

# ── Auth setup ────────────────────────────────────────────────────────────
cookie      = CookieController()
COOKIE_NAME = "modelforge_auth"
COOKIE_KEY  = "b4jF2zQ-4y8p9T0cNfE6rWmX3kV9qL2sY7hD1uJ0aP5cR8eG6iM4nB2vQ1tZ3x5"

def purge_auth_cookie():
    try: cookie.delete(COOKIE_NAME)
    except Exception: pass

def hard_logout_and_redirect():
    purge_auth_cookie()
    try:
        db    = get_session()
        creds = {"credentials": list_users_for_auth(db)}
        _auth = stauth.Authenticate(
            credentials=creds["credentials"], cookie_name=COOKIE_NAME,
            key=COOKIE_KEY, cookie_expiry_days=30, preauthorized={"emails": []})
        _auth.logout("Logout", "main", key="logout_force")
    except Exception: pass
    for k in ["user_id","display_name","project_id","project_path","undo_stack",
              "redo_stack","data","authentication_status","project_slug"]:
        st.session_state.pop(k, None)
    try: st.switch_page("pages/0_Master.py")
    except Exception: st.rerun()

if not st.session_state.get("user_id"):
    purge_auth_cookie()
    hard_logout_and_redirect()

db            = get_session()
creds         = {"credentials": list_users_for_auth(db)}
authenticator = stauth.Authenticate(
    credentials=creds["credentials"], cookie_name=COOKIE_NAME,
    key=COOKIE_KEY, cookie_expiry_days=30, preauthorized={"emails": []})

# ── Sidebar ───────────────────────────────────────────────────────────────
from sidebar import render_sidebar
render_sidebar(
    current_page="Home",
    authenticator=authenticator,
    logout_callback=hard_logout_and_redirect,
)

# ── Page styles ───────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

.mf-card {
    background: rgba(12,14,22,.9);
    border: 1px solid rgba(255,255,255,.05);
    border-radius: 12px 12px 0 0;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: all 0.22s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    animation: mf-card-in 0.5s ease both;
    backdrop-filter: blur(8px);
    min-height: 130px;
}
.mf-card::before {
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    opacity:0;transition:opacity .22s;
}
.mf-card::after {
    content:'';position:absolute;inset:0;
    background:radial-gradient(circle at 50% 0%,rgba(79,142,247,.04),transparent 60%);
    opacity:0;transition:opacity .22s;
}
.mf-card:hover {
    border-color: rgba(255,255,255,.1);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,.7), 0 0 0 1px rgba(255,255,255,.04);
}
.mf-card:hover::before { opacity:1; }
.mf-card:hover::after  { opacity:1; }
.mf-card.blue::before   { background:linear-gradient(90deg,transparent,rgba(79,142,247,.5),transparent); }
.mf-card.purple::before { background:linear-gradient(90deg,transparent,rgba(139,92,246,.5),transparent); }
.mf-card.pink::before   { background:linear-gradient(90deg,transparent,rgba(236,72,153,.5),transparent); }

.c-num {
    font-family:'Syne',sans-serif;font-size:11px;font-weight:800;
    letter-spacing:1px;margin-bottom:8px;
}
.mf-card.blue   .c-num { color:rgba(79,142,247,.6); }
.mf-card.purple .c-num { color:rgba(139,92,246,.6); }
.mf-card.pink   .c-num { color:rgba(236,72,153,.6); }

.c-title {
    font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
    color:#e0e4ff;margin-bottom:4px;letter-spacing:-.2px;
}
.c-desc { font-size:13px;color:#8a90b4;line-height:1.55; }
.c-arrow {
    position:absolute;bottom:12px;right:14px;
    font-size:15px;color:#555870;
    transition:all .2s ease;font-family:monospace;
}
.mf-card:hover .c-arrow { color:#8899cc;transform:translateX(3px); }

/* Open link styling scoped to main content */
.block-container div[data-testid="stPageLink"] {
    margin: -6px 0 14px 0 !important;
    padding: 0 !important;
}
.block-container div[data-testid="stPageLink"] a[data-testid="stPageLink"],
.block-container div[data-testid="stPageLink"] a {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    padding: 8px 0 !important;
    border-radius: 0 0 12px 12px !important;
    border: 1px solid rgba(255,255,255,.05) !important;
    border-top: none !important;
    background: rgba(12,14,22,.9) !important;
    color: #525870 !important;
    font-size: 11px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: .4px !important;
    text-decoration: none !important;
    transition: all .2s ease !important;
    transform: none !important;
}
.block-container div[data-testid="stPageLink"] a:hover {
    background: rgba(20,22,35,.9) !important;
    color: #9098c0 !important;
    transform: none !important;
    border-color: rgba(255,255,255,.1) !important;
}
.block-container div[data-testid="stPageLink"] p {
    font-size: 11px !important;
    color: inherit !important;
    margin: 0 !important;
    line-height: 1 !important;
}

/* Stagger delays */
.mf-d1  { animation-delay:.04s; } .mf-d2  { animation-delay:.08s; }
.mf-d3  { animation-delay:.12s; } .mf-d4  { animation-delay:.16s; }
.mf-d5  { animation-delay:.20s; } .mf-d6  { animation-delay:.24s; }
.mf-d7  { animation-delay:.28s; } .mf-d8  { animation-delay:.32s; }
.mf-d9  { animation-delay:.36s; } .mf-d10 { animation-delay:.40s; }
.mf-d11 { animation-delay:.44s; }

@keyframes mf-card-in {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
}
</style>
""")

# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.html("""
<div style="padding:48px 0 36px;border-bottom:1px solid rgba(255,255,255,.04);
            margin-bottom:28px;position:relative;overflow:hidden;">

    <div style="position:absolute;top:-40px;right:80px;width:280px;height:280px;border-radius:50%;
                background:radial-gradient(circle,rgba(139,92,246,.06),transparent 70%);pointer-events:none;"></div>
    <div style="position:absolute;top:20px;right:20px;width:160px;height:160px;border-radius:50%;
                background:radial-gradient(circle,rgba(79,142,247,.04),transparent 70%);pointer-events:none;"></div>
    <div style="position:absolute;right:40px;top:20px;width:140px;height:140px;border-radius:50%;
                border:1px solid rgba(255,255,255,.04);opacity:.5;pointer-events:none;"></div>
    <div style="position:absolute;right:60px;top:40px;width:100px;height:100px;border-radius:50%;
                border:1px solid rgba(255,255,255,.03);opacity:.5;pointer-events:none;"></div>

    <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(255,255,255,.04);backdrop-filter:blur(12px);
                border:1px solid rgba(255,255,255,.08);border-radius:20px;
                padding:5px 14px;margin-bottom:20px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#10b981;"></div>
        <span style="font-size:11px;color:#7080a0;font-weight:500;letter-spacing:.7px;">
            MODELFORGE PLATFORM &#183; v0.4
        </span>
    </div>

    <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:800;
                letter-spacing:-2px;line-height:1.1;margin-bottom:14px;
                background:linear-gradient(135deg,#ffffff 0%,#9099cc 35%,#ffffff 70%,#9099cc 100%);
                background-size:250% auto;-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;background-clip:text;">
        Build, Compare &amp;<br>Explain ML Models
    </div>

    <p style="font-size:16px;color:#7880a8;max-width:500px;line-height:1.75;
              margin:0 0 26px;font-weight:300;">
        End-to-end machine learning from raw data to production-ready models.
        Full explainability, experiment tracking, and an AI assistant that
        actually knows your project.
    </p>

    <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <div style="background:rgba(6,14,29,.85);border:1px solid rgba(29,58,110,.5);
                    border-radius:9px;padding:7px 14px;font-family:'DM Sans',sans-serif;
                    font-size:13px;font-weight:500;color:#7eb3f7;
                    box-shadow:0 0 20px rgba(79,142,247,.08);">
            SHAP + LIME Explainability
        </div>
        <div style="background:rgba(13,8,25,.85);border:1px solid rgba(76,29,149,.5);
                    border-radius:9px;padding:7px 14px;font-family:'DM Sans',sans-serif;
                    font-size:13px;font-weight:500;color:#a78bfa;
                    box-shadow:0 0 20px rgba(139,92,246,.08);">
            Groq AI Assistant
        </div>
        <div style="background:rgba(5,18,9,.85);border:1px solid rgba(20,83,45,.5);
                    border-radius:9px;padding:7px 14px;font-family:'DM Sans',sans-serif;
                    font-size:13px;font-weight:500;color:#10b981;
                    box-shadow:0 0 20px rgba(16,185,129,.08);">
            Experiment Tracking
        </div>
        <div style="background:rgba(18,5,5,.85);border:1px solid rgba(127,29,29,.5);
                    border-radius:9px;padding:7px 14px;font-family:'DM Sans',sans-serif;
                    font-size:13px;font-weight:500;color:#f87171;
                    box-shadow:0 0 20px rgba(248,113,113,.06);">
            Auto Hyperparameter Tuning
        </div>
    </div>
</div>
""")

# ══════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════
st.html("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
    <div style="height:1px;width:20px;background:rgba(255,255,255,.08);"></div>
    <span style="font-size:11px;font-weight:700;color:#6070a0;
                 letter-spacing:1.5px;text-transform:uppercase;">Data Pipeline</span>
    <div style="height:1px;flex:1;background:rgba(255,255,255,.04);"></div>
</div>
""")

data_steps = [
    ("01","Projects",           "pages/1_Projects.py",            "Create or select a project workspace", "blue",  "mf-d1"),
    ("02","Upload & Explore",   "pages/2_Upload_Data.py",         "Upload and preview your CSV dataset",  "blue",  "mf-d2"),
    ("03","EDA & Preprocessing","pages/3_EDA_Preprocessing.py",   "Explore, clean and fix data issues",   "blue",  "mf-d3"),
    ("04","Feature Engineering","pages/4_Feature_Engineering.py", "Create and transform features",        "blue",  "mf-d4"),
    ("05","Feature Selection",  "pages/10_Feature_Selection.py",  "Pick the most impactful features",     "blue",  "mf-d5"),
    ("06","Data Preparation",   "pages/5_Data_Preparation.py",    "Split, encode and scale for training", "blue",  "mf-d6"),
]

cols = st.columns(3, gap="small")
for i, (num, title, page, desc, color, delay) in enumerate(data_steps):
    with cols[i % 3]:
        st.html(f"""
        <div class="mf-card {color} {delay}" style="margin-bottom:4px;">
            <div class="c-num">{num}</div>
            <div class="c-title">{title}</div>
            <div class="c-desc">{desc}</div>
            <div class="c-arrow">&#8594;</div>
        </div>
        """)
        st.page_link(page, label="Open", use_container_width=True)

# ══════════════════════════════════════════════════════════
# MODELLING
# ══════════════════════════════════════════════════════════
st.html("""
<div style="display:flex;align-items:center;gap:8px;margin:26px 0 14px;">
    <div style="height:1px;width:20px;background:rgba(255,255,255,.08);"></div>
    <span style="font-size:11px;font-weight:700;color:#6070a0;
                 letter-spacing:1.5px;text-transform:uppercase;">Modelling</span>
    <div style="height:1px;flex:1;background:rgba(255,255,255,.04);"></div>
</div>
""")

model_steps = [
    ("07","Train Model",    "pages/6_Train_Model.py",          "Auto-tuning, grid search, CV scoring",   "purple","mf-d7"),
    ("08","Experiments",    "pages/8_Experiments.py",          "Compare all training runs side by side",  "purple","mf-d8"),
    ("09","Explainability", "pages/7_Model_Explainability.py", "SHAP global and LIME local explanations", "purple","mf-d9"),
    ("10","Predict",        "pages/9_Predict.py",              "Single row and batch CSV inference",      "purple","mf-d10"),
    ("11","AI Assistant",   "pages/11_Chat.py",                "Groq-powered chat that knows your data",  "pink",  "mf-d11"),
]

cols2 = st.columns(3, gap="small")
for i, (num, title, page, desc, color, delay) in enumerate(model_steps):
    with cols2[i % 3]:
        st.html(f"""
        <div class="mf-card {color} {delay}" style="margin-bottom:4px;">
            <div class="c-num">{num}</div>
            <div class="c-title">{title}</div>
            <div class="c-desc">{desc}</div>
            <div class="c-arrow">&#8594;</div>
        </div>
        """)
        st.page_link(page, label="Open", use_container_width=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.html("""
<div style="margin-top:48px;padding:20px 0;
            border-top:1px solid rgba(255,255,255,.04);
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:10px;">
    <div style="display:flex;align-items:center;gap:9px;">
        <div style="width:18px;height:18px;border-radius:5px;
                    background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
                    display:flex;align-items:center;justify-content:center;
                    font-family:'Syne',sans-serif;font-size:9px;font-weight:800;color:#fff;">M</div>
        <span style="font-family:'Syne',sans-serif;font-size:12px;color:#445070;
                     font-weight:700;letter-spacing:-.2px;">ModelForge</span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-size:10px;color:#445070;">PostgreSQL</span>
        <span style="font-size:10px;color:#333650;">+</span>
        <span style="font-size:10px;color:#445070;">Redis</span>
        <span style="font-size:10px;color:#333650;">+</span>
        <span style="font-size:10px;color:#445070;">FastAPI</span>
        <span style="font-size:10px;color:#333650;">+</span>
        <span style="font-size:10px;color:#445070;">Streamlit</span>
        <span style="font-size:10px;color:#333650;">+</span>
        <span style="font-size:10px;color:#445070;">Groq</span>
    </div>
</div>
""")
