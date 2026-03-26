# pages/6_Train_Model.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="ModelForge • Train Model", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# ── Page styles ───────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

.mf-sec { display:flex;align-items:center;gap:10px;margin-bottom:16px; }
.mf-sec-line { height:1px;background:rgba(255,255,255,.1); }
.mf-sec-label {
    font-size:11px;font-weight:700;color:#7a82a6;
    letter-spacing:1.4px;text-transform:uppercase;white-space:nowrap;
}

.mf-info-card {
    background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.4);
    border-radius:10px;padding:14px 18px;font-size:13px;
    color:#8890aa;line-height:1.65;margin-top:8px;
}

.mf-metric {
    background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.07);
    border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;
    backdrop-filter:blur(8px);transition:all .22s ease;text-align:center;
}
.mf-metric:hover {
    border-color:rgba(255,255,255,.13);transform:translateY(-2px);
    box-shadow:0 10px 30px rgba(0,0,0,.5);
}
.mf-metric-label {
    font-size:11px;font-weight:700;color:#7a82a6;
    letter-spacing:.8px;text-transform:uppercase;margin-bottom:10px;
}
.mf-metric-value {
    font-family:'Syne',sans-serif;
    font-size:clamp(20px,2.5vw,28px);
    font-weight:800;letter-spacing:-.5px;line-height:1.1;
    white-space:nowrap;overflow:visible;
}

.mf-prep-row {
    display:flex;align-items:center;padding:11px 16px;
    border-bottom:1px solid rgba(255,255,255,.04);transition:background .15s;
}
.mf-prep-row:last-child { border-bottom:none; }
.mf-prep-row:hover { background:rgba(255,255,255,.02); }
.mf-prep-key { font-size:12px;color:#7a82a6;font-weight:500;width:180px;flex-shrink:0; }
.mf-prep-val { font-size:13px;color:#d0d4f0;font-weight:500; }

.mf-warn {
    background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.5);
    border-radius:10px;padding:13px 18px;font-size:13px;color:#f59e0b;margin-bottom:12px;
}

/* Mode selector — pure st.button styled as cards */
.mode-btn-selected,
.mode-btn-unsel {
    display: block !important;
    width: 100% !important;
}
/* Selected state */
.mode-btn-selected > button {
    background: linear-gradient(135deg,rgba(13,25,41,.95),rgba(13,13,40,.95)) !important;
    border: 1px solid rgba(79,142,247,.6) !important;
    color: #ffffff !important;
    text-align: center !important;
    padding: 14px 8px !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    width: 100% !important;
    height: auto !important;
    min-height: 90px !important;
    white-space: normal !important;
    line-height: 1.5 !important;
    box-shadow: 0 0 18px rgba(79,142,247,.18), inset 0 0 0 1px rgba(79,142,247,.2) !important;
    margin-bottom: 0 !important;
    justify-content: center !important;
}
/* Unselected state */
.mode-btn-unsel > button {
    background: rgba(12,14,22,.85) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    color: #8890aa !important;
    text-align: center !important;
    padding: 14px 8px !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    width: 100% !important;
    height: auto !important;
    min-height: 90px !important;
    white-space: normal !important;
    line-height: 1.5 !important;
    margin-bottom: 0 !important;
    justify-content: center !important;
}
.mode-btn-unsel > button:hover {
    background: rgba(13,25,41,.6) !important;
    border-color: rgba(79,142,247,.3) !important;
    color: #c8ccee !important;
}

/* Launch button — dark gradient */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d3a6e, #4c1d95) !important;
    color: #e0e8ff !important;
    border: 1px solid rgba(79,142,247,.35) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: .3px !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: all .25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #234a8a, #5b22b0) !important;
    border-color: rgba(79,142,247,.55) !important;
    box-shadow: 0 0 28px rgba(79,142,247,.2), 0 6px 20px rgba(0,0,0,.5) !important;
    transform: translateY(-2px) !important;
    color: #fff !important;
}

.stProgress > div > div { height:6px !important;border-radius:6px !important; }
</style>
""")

# ── Sidebar ───────────────────────────────────────────────────────────────
render_sidebar(current_page="Train Model") 
# ── Auth ──────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

if not st.session_state.get("project_id") or not st.session_state.get("dataset_id"):
    st.html("""
    <div style="background:rgba(6,14,29,.8);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:20px 24px;margin-top:20px;">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No project or dataset selected</div>
        <div style="font-size:13px;color:#8890aa;">Select a project and dataset first.</div>
    </div>
    """)
    st.stop()

project_id = st.session_state["project_id"]
dataset_id = st.session_state["dataset_id"]

# ── Load prep config ──────────────────────────────────────────────────────
prep = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30).json()
problem_type = prep["problem_type"]

current_context = f"{problem_type}_{dataset_id}"
if st.session_state.get("last_train_context") != current_context:
    for k in ["metrics","model_id","poll","job_id","metrics_problem_type","train_mode"]:
        st.session_state.pop(k, None)
    st.session_state["last_train_context"] = current_context

# Init mode
if "train_mode" not in st.session_state:
    st.session_state["train_mode"] = "Use Default Parameters"

# ── Page header ───────────────────────────────────────────────────────────
pt_color = "#7eb3f7" if problem_type == "Classification" else "#a78bfa"
pt_bg    = "rgba(6,14,29,.8)"   if problem_type == "Classification" else "rgba(13,8,25,.8)"
pt_bdr   = "rgba(29,58,110,.5)" if problem_type == "Classification" else "rgba(76,29,149,.5)"

st.html(f"""
<div style="padding:40px 0 28px;border-bottom:1px solid rgba(255,255,255,.07);
            margin-bottom:28px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-20px;right:0;width:220px;height:220px;border-radius:50%;
                background:radial-gradient(circle,rgba(139,92,246,.05),transparent 70%);pointer-events:none;"></div>
    <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.05);
                border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:5px 14px;margin-bottom:18px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#8b5cf6;"></div>
        <span style="font-size:11px;color:#8890aa;font-weight:600;letter-spacing:.7px;">
            MODELLING &#183; TRAIN
        </span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.15;margin-bottom:14px;color:#fff;">
        Train Model
    </div>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <p style="font-size:15px;color:#8890aa;margin:0;font-weight:400;line-height:1.7;">
            Configure your algorithm, tune hyperparameters, and launch a training job.
        </p>
        <span style="background:{pt_bg};border:1px solid {pt_bdr};border-radius:20px;
                     padding:4px 13px;font-size:12px;font-weight:600;color:{pt_color};">
            {problem_type}
        </span>
    </div>
</div>
""")

# ══════════════════════════════════════════════════════════
# PREP SUMMARY — full-width horizontal strip
# ══════════════════════════════════════════════════════════
prep_rows = [
    ("Problem Type",       str(prep["problem_type"])),
    ("Target Column",      str(prep["target"])),
    ("Test Size",          f"{int(prep['test_size'] * 100)}%"),
    ("Stratify",           "Yes" if prep["stratify"] else "No"),
    ("Encoding",           str(prep.get("encoding") or "None")),
    ("Scaling",            str(prep.get("scaling") or "None")),
    ("Number of Features", str(len(prep["features"]))),
]

prep_items_html = "".join([
    f'<div style="display:flex;flex-direction:column;gap:5px;padding:14px 20px;'
    f'border-right:1px solid rgba(255,255,255,.05);flex-shrink:0;">'
    f'<span style="font-size:10px;font-weight:700;color:#7a82a6;letter-spacing:.8px;text-transform:uppercase;">{k}</span>'
    f'<span style="font-size:13px;font-weight:500;color:#d0d4f0;white-space:nowrap;">{v}</span>'
    f'</div>'
    for k, v in prep_rows
])

features = prep.get("features", [])
pills = "".join([
    f'<span style="display:inline-flex;background:rgba(255,255,255,.04);'
    f'border:1px solid rgba(255,255,255,.09);border-radius:20px;'
    f'padding:3px 10px;font-size:11px;color:#8890aa;margin:2px 2px 2px 0;white-space:nowrap;">{f}</span>'
    for f in features
])
features_section = f"""
<div style="display:flex;flex-direction:column;gap:5px;padding:14px 20px;flex:1;min-width:0;">
    <span style="font-size:10px;font-weight:700;color:#7a82a6;letter-spacing:.8px;text-transform:uppercase;">Selected Features</span>
    <div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:2px;">{pills}</div>
</div>
""" if features else ""

st.html(f"""
<div style="background:rgba(12,14,22,.85);border:1px solid rgba(255,255,255,.07);
            border-radius:12px;overflow:hidden;margin-bottom:28px;">
    <div style="height:2px;background:linear-gradient(90deg,transparent,rgba(79,142,247,.4),transparent);"></div>
    <div style="display:flex;align-items:stretch;flex-wrap:wrap;">
        {prep_items_html}
        {features_section}
    </div>
</div>
""")

# ══════════════════════════════════════════════════════════
# LAYOUT — config only (no right column needed)
# ══════════════════════════════════════════════════════════
# ── Model selection ───────────────────────────────────────────────────────
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Model Selection</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")
_sel_col, _ = st.columns([1, 1], gap="small")
with _sel_col:
    models = (
        ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"]
        if problem_type == "Classification"
        else ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]
    )
    algorithm = st.selectbox("Select Model", models, label_visibility="collapsed")

algo_desc = {
    "Logistic Regression":       ("Linear boundary, fast and interpretable", "#4f8ef7"),
    "Random Forest Classifier":  ("Ensemble of trees, high accuracy, robust", "#8b5cf6"),
    "Decision Tree Classifier":  ("Rule-based splits, highly interpretable", "#10b981"),
    "Linear Regression":         ("Linear relationship, fast baseline model", "#4f8ef7"),
    "Random Forest Regressor":   ("Ensemble for regression, very powerful", "#8b5cf6"),
    "Decision Tree Regressor":   ("Rule-based regression, interpretable", "#10b981"),
}
desc, acolor = algo_desc.get(algorithm, ("", "#7a82a6"))
st.html(f"""
<div style="background:{acolor}11;border:1px solid {acolor}33;border-radius:9px;
            padding:10px 14px;font-size:12px;color:{acolor};margin:6px 0 22px;display:inline-block;">
    {desc}
</div>
""")

# ── Hyperparameter mode cards — full width ────────────────────────────────
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Hyperparameter Configuration</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")

MODES = [
    ("Use Default Parameters",        "⚡", "Best starting point — sklearn defaults",              "#4f8ef7"),
    ("Manual Tuning",                 "🎛️", "Set each parameter yourself",                         "#8b5cf6"),
    ("Grid Search (exhaustive)",      "🔍", "Try every combination — thorough but slow",           "#f59e0b"),
    ("Randomized Search (fast)",      "🎲", "Sample 20 random combos — fast and effective",        "#10b981"),
    ("Bayesian Optimization (smart)", "🧠", "Learns from past trials — fewest evaluations",        "#ec4899"),
]

current_mode = st.session_state["train_mode"]
clicked_mode = None

mode_cols = st.columns(5, gap="small")
for col, (mode_name, icon, mode_desc, mcolor) in zip(mode_cols, MODES):
    is_sel = (current_mode == mode_name)
    with col:
        st.html(f'<div class="{"mode-btn-selected" if is_sel else "mode-btn-unsel"}" data-color="{mcolor}"'
                f' style="--mc:{mcolor};">')
        clicked = st.button(
            f"{'**' if is_sel else ''}{icon} {mode_name}{'**' if is_sel else ''}\n\n{mode_desc}",
            key=f"modecard_{mode_name}",
            use_container_width=True,
        )
        st.html('</div>')
        if clicked and current_mode != mode_name:
            clicked_mode = mode_name

if clicked_mode:
    st.session_state["train_mode"] = clicked_mode
    st.rerun()

# Refresh current mode after possible click
param_mode = st.session_state["train_mode"]

# Show selected mode indicator
sel_mode_data = {m[0]: (m[3], m[2]) for m in MODES}
sel_color, sel_desc = sel_mode_data[param_mode]
st.html(f"""
<div style="display:flex;align-items:center;gap:10px;margin:10px 0 4px;
            background:{sel_color}0d;border:1px solid {sel_color}33;
            border-radius:9px;padding:10px 16px;">
    <div style="width:6px;height:6px;border-radius:50%;background:{sel_color};
                box-shadow:0 0 8px {sel_color};flex-shrink:0;"></div>
    <span style="font-size:12px;font-weight:600;color:{sel_color};">Selected: {param_mode}</span>
    <span style="font-size:12px;color:#555870;margin-left:4px;">— {sel_desc}</span>
</div>
""")

MODE_MAP = {
    "Use Default Parameters":       "manual",
    "Manual Tuning":                "manual",
    "Grid Search (exhaustive)":     "grid",
    "Randomized Search (fast)":     "random",
    "Bayesian Optimization (smart)":"bayesian",
}
tuning_method = MODE_MAP[param_mode]
use_defaults  = (param_mode == "Use Default Parameters")
hyperparameters = {}

# Mode info / manual params
mode_infos = {
    "Grid Search (exhaustive)": (
        "Grid Search", "Tests every combination in a predefined parameter grid.",
        "Finds the best result — can be slow for Random Forest.", "#f59e0b",
    ),
    "Randomized Search (fast)": (
        "Randomized Search", "Samples 20 random combinations from the search space.",
        "Much faster than Grid Search, finds near-optimal parameters.", "#10b981",
    ),
    "Bayesian Optimization (smart)": (
        "Bayesian Optimization", "Uses past results to intelligently select next parameters.",
        "Most sample-efficient. Falls back to Randomized if scikit-optimize is missing.", "#ec4899",
    ),
}

if param_mode in mode_infos:
    name, l1, l2, c = mode_infos[param_mode]
    st.html(f"""
    <div class="mf-info-card" style="border-color:{c}33;margin-top:8px;">
        <span style="color:{c};font-weight:600;">{name}</span> &mdash; {l1}
        <br><span style="color:#555870;font-size:12px;">{l2}</span>
    </div>
    """)

elif param_mode == "Manual Tuning":
    st.html("""
    <div style="height:1px;background:rgba(255,255,255,.07);margin:14px 0 12px;"></div>
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Manual Parameters</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """)
    _mp1, _mp2, _mp3 = st.columns(3, gap="medium")
    if algorithm == "Random Forest Classifier":
        with _mp1: hyperparameters["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100)
        with _mp2:
            md = st.number_input("max_depth (0 = None)", 0, 50, 0)
            hyperparameters["max_depth"] = None if md == 0 else md
        with _mp3: hyperparameters["min_samples_split"] = st.number_input("min_samples_split", 2, 20, 2)
    elif algorithm == "Decision Tree Classifier":
        with _mp1:
            md = st.number_input("max_depth (0 = None)", 0, 50, 0)
            hyperparameters["max_depth"] = None if md == 0 else md
        with _mp2: hyperparameters["criterion"] = st.selectbox("criterion", ["gini", "entropy"])
        with _mp3: hyperparameters["min_samples_split"] = st.number_input("min_samples_split", 2, 20, 2)
    elif algorithm == "Logistic Regression":
        with _mp1: hyperparameters["C"] = st.number_input("C (regularization)", 0.01, 100.0, 1.0)
        with _mp2: hyperparameters["max_iter"] = st.number_input("max_iter", 100, 5000, 1000)
        with _mp3: hyperparameters["solver"] = st.selectbox("solver", ["lbfgs", "liblinear", "saga"])
    elif algorithm == "Random Forest Regressor":
        with _mp1: hyperparameters["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100)
        with _mp2:
            md = st.number_input("max_depth (0 = None)", 0, 50, 0)
            hyperparameters["max_depth"] = None if md == 0 else md
        with _mp3: hyperparameters["min_samples_split"] = st.number_input("min_samples_split", 2, 20, 2)
    elif algorithm == "Decision Tree Regressor":
        with _mp1:
            md = st.number_input("max_depth (0 = None)", 0, 50, 0)
            hyperparameters["max_depth"] = None if md == 0 else md
        with _mp2: hyperparameters["criterion"] = st.selectbox("criterion", ["squared_error", "absolute_error"])
    elif algorithm == "Linear Regression":
        st.html('<div class="mf-info-card">Linear Regression has no hyperparameters to tune manually.</div>')

# ── Cross-validation ──────────────────────────────────────────────────────
st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:22px 0 18px;"></div>')
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Cross-Validation</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")
_cv1, _cv2 = st.columns([1, 3], gap="medium")
with _cv1:
    enable_cv = st.checkbox("Enable Cross-Validation", value=False)
cv_folds = 0
if enable_cv:
    with _cv2:
        cv_folds = st.selectbox("Number of folds (k)", [3, 5, 10], index=1, label_visibility="collapsed")
    st.html(f"""
    <div class="mf-info-card" style="border-color:rgba(16,185,129,.3);margin-top:8px;">
        <span style="color:#10b981;font-weight:600;">{cv_folds}-Fold Cross-Validation</span>
        &mdash; Splits into {cv_folds} parts, trains on {cv_folds-1}, tests on 1,
        repeated {cv_folds} times. Reports mean &plusmn; std.
    </div>
    """)

# ── Launch button ─────────────────────────────────────────────────────────
st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:22px 0 18px;"></div>')

if param_mode == "Grid Search (exhaustive)" and "Random Forest" in algorithm:
    st.html('<div class="mf-warn">Grid Search on Random Forest can take several minutes. Consider <strong>Randomized Search</strong> for faster results.</div>')

_btn_col, _ = st.columns([1, 2])
with _btn_col:
    train_clicked = st.button("Launch Training Job", type="primary", use_container_width=True)

    if train_clicked:
        with st.spinner("Submitting training job..."):
            resp = requests.post(
                f"{API_BASE}/train/train",
                json={
                    "project_id":         project_id,
                    "dataset_id":         dataset_id,
                    "algorithm":          algorithm,
                    "use_default_params": use_defaults,
                    "hyperparameters":    hyperparameters,
                    "tuning_method":      tuning_method,
                    "cv_folds":           cv_folds,
                },
            )
            resp.raise_for_status()
        st.session_state["job_id"]               = resp.json()["job_id"]
        st.session_state["poll"]                 = True
        st.session_state["metrics_problem_type"] = problem_type
        st.session_state.pop("metrics",  None)
        st.session_state.pop("model_id", None)


# ══════════════════════════════════════════════════════════
# TRAINING PROGRESS — full width
# ══════════════════════════════════════════════════════════
if st.session_state.get("poll") and "job_id" in st.session_state:
    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:28px 0 20px;"></div>')
    st.html("""
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Training Progress</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """)
    try:
        job_resp = requests.get(f"{API_BASE}/jobs/{st.session_state['job_id']}", timeout=10)
        job = job_resp.json() if job_resp.status_code == 200 else {"status": "not_found"}
    except Exception:
        job = {"status": "running", "progress": 50, "message": "Training in progress..."}

    status   = job.get("status",   "running")
    progress = job.get("progress", 0)
    message  = job.get("message",  "Training...")

    if status == "completed":
        st.session_state["poll"] = False
        try:
            model_info = requests.get(f"{API_BASE}/train/latest/{project_id}", timeout=30).json()
            st.session_state["metrics"]              = model_info["metrics"]
            st.session_state["model_id"]             = model_info["id"]
            st.session_state["metrics_problem_type"] = problem_type
            st.html("""
            <div style="background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.5);
                        border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:10px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#10b981;
                            box-shadow:0 0 8px #10b981;flex-shrink:0;"></div>
                <span style="font-size:13px;font-weight:600;color:#10b981;">Model trained successfully!</span>
            </div>
            """)
        except Exception as e:
            st.error(f"Training finished but could not load results: {e}")
        st.rerun()

    elif status == "not_found":
        st.html('<div class="mf-info-card">Waiting for worker to pick up the job...</div>')
        try:
            model_resp = requests.get(f"{API_BASE}/train/latest/{project_id}", timeout=10)
            if model_resp.status_code == 200:
                model_info = model_resp.json()
                if model_info.get("id"):
                    st.session_state["poll"]                 = False
                    st.session_state["metrics"]              = model_info["metrics"]
                    st.session_state["model_id"]             = model_info["id"]
                    st.session_state["metrics_problem_type"] = problem_type
                    st.rerun()
        except Exception:
            pass
        time.sleep(2)
        st.rerun()

    elif status == "failed":
        st.session_state["poll"] = False
        st.html(f"""
        <div style="background:rgba(18,5,5,.7);border:1px solid rgba(127,29,29,.4);
                    border-radius:10px;padding:14px 16px;font-size:13px;color:#f87171;">
            Training failed: {job.get('message', 'Unknown error')}
        </div>
        """)
    else:
        st.progress(progress)
        st.html(f'<div class="mf-info-card" style="margin-top:8px;">{message}</div>')
        time.sleep(2)
        st.rerun()


# ══════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════
if "metrics" in st.session_state:
    metrics    = st.session_state["metrics"]
    trained_pt = st.session_state.get("metrics_problem_type", problem_type)

    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:28px 0 20px;"></div>')
    st.html("""
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Evaluation Metrics</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """)

    if "best_params" in metrics:
        st.html("""
        <div class="mf-sec" style="margin-top:4px;">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Best Parameters Found</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """)
        param_pills = "".join([
            f'<div style="background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.08);'
            f'border-radius:9px;padding:10px 14px;margin:4px;">'
            f'<div style="font-size:10px;color:#7a82a6;font-weight:600;letter-spacing:.5px;margin-bottom:4px;">{k.upper()}</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:#c8ccee;">{v}</div>'
            f'</div>'
            for k, v in metrics["best_params"].items()
        ])
        st.html(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:18px;">{param_pills}</div>')

    if trained_pt == "Classification":
        if "accuracy" not in metrics:
            st.html('<div class="mf-warn">Stale metrics detected. Retrain the model to see updated results.</div>')
        else:
            c1, c2, c3, c4 = st.columns(4, gap="small")
            for col, (label, val, color) in zip([c1,c2,c3,c4], [
                ("Accuracy",             round(metrics.get("accuracy", 0), 4),                                "#4f8ef7"),
                ("Precision (weighted)", round(metrics.get("weighted avg",{}).get("precision",0), 4),         "#8b5cf6"),
                ("Recall (weighted)",    round(metrics.get("weighted avg",{}).get("recall",0), 4),            "#10b981"),
                ("F1-score (weighted)",  round(metrics.get("weighted avg",{}).get("f1-score",0), 4),          "#ec4899"),
            ]):
                with col:
                    st.html(f"""
                    <div class="mf-metric">
                        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                                    background:linear-gradient(90deg,transparent,{color}55,transparent);
                                    border-radius:12px 12px 0 0;"></div>
                        <div class="mf-metric-label">{label}</div>
                        <div class="mf-metric-value" style="color:{color};">{val}</div>
                    </div>
                    """)

            st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
            st.html("""
            <div class="mf-sec">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Classification Report</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            rows = {k: v for k, v in metrics.items()
                    if isinstance(v, dict) and all(m in v for m in ["precision","recall","f1-score"])}
            if rows:
                st.dataframe(pd.DataFrame(rows).transpose().round(4).astype(str), use_container_width=True)

            if "confusion_matrix" in metrics:
                st.html("""
                <div class="mf-sec" style="margin-top:18px;">
                    <div class="mf-sec-line" style="width:16px;"></div>
                    <span class="mf-sec-label">Confusion Matrix</span>
                    <div class="mf-sec-line" style="flex:1;"></div>
                </div>
                """)
                st.dataframe(pd.DataFrame(np.array(metrics["confusion_matrix"])).astype(str), use_container_width=True)

            if "roc_curve" in metrics and "roc_auc" in metrics:
                st.html("""
                <div class="mf-sec" style="margin-top:18px;">
                    <div class="mf-sec-line" style="width:16px;"></div>
                    <span class="mf-sec-label">ROC Curve</span>
                    <div class="mf-sec-line" style="flex:1;"></div>
                </div>
                """)
                fig, ax = plt.subplots(facecolor="#0c0e16")
                ax.set_facecolor("#0c0e16")
                ax.plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"],
                        color="#4f8ef7", linewidth=2, label=f"AUC = {metrics['roc_auc']:.4f}")
                ax.plot([0,1],[0,1],"--",color="#2a2d3e",linewidth=1)
                ax.legend(facecolor="#111318", labelcolor="#8890aa")
                ax.tick_params(colors="#525870")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1f2235")
                st.pyplot(fig)
    else:
        if "r2" not in metrics:
            st.html('<div class="mf-warn">Stale metrics detected. Retrain the model to see updated results.</div>')
        else:
            c1, c2, c3 = st.columns(3, gap="small")
            for col, (label, val, color) in zip([c1,c2,c3],[
                ("R² Score", round(metrics.get("r2",   0), 4), "#4f8ef7"),
                ("RMSE",     round(metrics.get("rmse", 0), 4), "#8b5cf6"),
                ("MAE",      round(metrics.get("mae",  0), 4), "#10b981"),
            ]):
                with col:
                    st.html(f"""
                    <div class="mf-metric">
                        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                                    background:linear-gradient(90deg,transparent,{color}55,transparent);
                                    border-radius:12px 12px 0 0;"></div>
                        <div class="mf-metric-label">{label}</div>
                        <div class="mf-metric-value" style="color:{color};">{val}</div>
                    </div>
                    """)

    if "cross_validation" in metrics:
        st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:22px 0 18px;"></div>')
        st.html("""
        <div class="mf-sec">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Cross-Validation Results</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """)
        cv_rows = []
        for metric_name, result in metrics["cross_validation"].items():
            cv_rows.append({
                "Metric":          metric_name.upper(),
                "Mean":            result["mean"],
                "Std Dev (±)":     result["std"],
                "Per-Fold Scores": "  |  ".join([str(s) for s in result["scores"]]),
            })
        st.dataframe(pd.DataFrame(cv_rows).astype(str), use_container_width=True)
        mean_df = pd.DataFrame({r["Metric"]: [float(r["Mean"])] for r in cv_rows}).T.rename(columns={0:"Mean Score"})
        st.bar_chart(mean_df)


# ══════════════════════════════════════════════════════════
# DOWNLOAD MODEL
# ══════════════════════════════════════════════════════════
if "model_id" in st.session_state:
    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:28px 0 20px;"></div>')
    st.html("""
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Download Trained Model</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """)
    try:
        dl = requests.get(f"{API_BASE}/model/{st.session_state['model_id']}/download", timeout=30)
        dl.raise_for_status()
        st.download_button("Download Model (.pkl)", dl.content, "trained_model.pkl", "application/octet-stream")
    except Exception as e:
        st.html(f"""
        <div style="background:rgba(18,5,5,.7);border:1px solid rgba(127,29,29,.4);
                    border-radius:10px;padding:14px 16px;font-size:13px;color:#f87171;">
            Could not fetch model: {e}
        </div>
        """)
