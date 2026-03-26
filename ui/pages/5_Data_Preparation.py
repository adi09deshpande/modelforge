# pages/5_Data_Preparation.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import requests
import streamlit as st
import pandas as pd
import json
import hashlib

st.set_page_config(page_title="ModelForge • Data Preparation", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)
API_BASE = "http://127.0.0.1:8000"

# ── Inject CSS via st.html ────────────────────────────────────────────────
st.html("""
<style>
.section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: #7a82a6 !important;
    text-transform: uppercase !important;
    margin: 20px 0 10px 0 !important;
    display: flex; align-items: center; gap: 6px;
}
.page-header { margin-bottom: 24px; }
.page-header h1 { font-size: 28px; font-weight: 800; color: #e2e8f0; margin: 0 0 8px 0; }
.page-header p  { font-size: 15px; color: #8890aa; font-weight: 400; margin: 0; line-height: 1.6; }

.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent);
    margin: 24px 0;
}
.guidance-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0 20px 0; }
.guidance-card {
    border-radius: 14px; border: 1px solid rgba(99,102,241,0.18);
    background: rgba(12,15,35,0.75); padding: 20px 22px;
    position: relative; overflow: hidden;
}
.guidance-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.guidance-card.indigo::before { background: linear-gradient(90deg, #6366f1, #38bdf8); }
.guidance-card.green::before  { background: linear-gradient(90deg, #34d399, #6ee7b7); }
.guidance-card .gc-title {
    font-size: 14px; font-weight: 700; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.guidance-card.indigo .gc-title { color: #a5b4fc; }
.guidance-card.green  .gc-title { color: #6ee7b7; }
.guidance-card .gc-item {
    margin-bottom: 14px; padding-bottom: 14px;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}
.guidance-card .gc-item:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
.guidance-card .gc-item-name { font-size: 12px; font-weight: 700; color: #c8cfe8; margin-bottom: 6px; }
.guidance-card .gc-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.guidance-card .gc-tag { font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 12px; letter-spacing: 0.3px; }
.gc-tag.good { background: rgba(52,211,153,0.12); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.2); }
.gc-tag.bad  { background: rgba(248,113,113,0.1);  color: #fca5a5; border: 1px solid rgba(248,113,113,0.2); }

.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0 20px 0; }
.summary-card {
    border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);
    background: rgba(15,18,40,0.8); padding: 16px 18px; position: relative; overflow: visible;
}
.summary-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 12px 12px 0 0; }
.summary-card.blue::before   { background: linear-gradient(90deg, #38bdf8, #818cf8); }
.summary-card.purple::before { background: linear-gradient(90deg, #a78bfa, #6366f1); }
.summary-card.green::before  { background: linear-gradient(90deg, #34d399, #6ee7b7); }
.summary-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.summary-card .sc-label { font-size: 10px; font-weight: 700; letter-spacing: 1.2px; color: #7a82a6; text-transform: uppercase; margin-bottom: 6px; }
.summary-card .sc-value { font-size: clamp(13px, 1.5vw, 15px); font-weight: 700; line-height: 1.3; word-break: break-word; }
.summary-card.blue   .sc-value { color: #38bdf8; }
.summary-card.purple .sc-value { color: #a78bfa; }
.summary-card.green  .sc-value { color: #34d399; }
.summary-card.amber  .sc-value { color: #fcd34d; }

.summary-box { background: rgba(12,15,35,0.7); border: 1px solid rgba(99,102,241,0.18); border-radius: 14px; overflow: hidden; }
.summary-table { width: 100%; border-collapse: collapse; }
.summary-table tr { border-bottom: 1px solid rgba(99,102,241,0.08); }
.summary-table tr:last-child { border-bottom: none; }
.summary-table td { padding: 12px 16px; font-size: 13px; vertical-align: middle; }
.summary-table td:first-child { color: #7a82a6; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; width: 38%; }
.sv-mono   { font-family: monospace; color: #38bdf8; background: rgba(56,189,248,0.08); padding: 2px 8px; border-radius: 4px; }
.sv-purple { font-family: monospace; color: #a78bfa; background: rgba(167,139,250,0.08); padding: 2px 8px; border-radius: 4px; }
.sv-green  { color: #34d399; background: rgba(52,211,153,0.08); padding: 2px 8px; border-radius: 4px; }
.sv-amber  { color: #fcd34d; background: rgba(245,158,11,0.08); padding: 2px 8px; border-radius: 4px; }
.sv-plain  { color: #c8cfe8; }

.info-note {
    background: linear-gradient(135deg, rgba(56,189,248,0.07) 0%, rgba(99,102,241,0.07) 100%);
    border: 1px solid rgba(99,102,241,0.2); border-left: 3px solid #6366f1;
    border-radius: 10px; padding: 12px 16px; font-size: 13px; color: #8890aa; line-height: 1.6; margin: 10px 0;
}
.info-note.amber { border-left-color: #f59e0b; }

.feature-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.feature-tag {
    font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px;
    background: rgba(99,102,241,0.1); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.2);
}
</style>
""")

# ── Sidebar ───────────────────────────────────────────────────────────────
render_sidebar(current_page="Preparation") 
# ── Auth ──────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Page header ───────────────────────────────────────────────────────────
st.html("""
<div class="page-header">
    <h1>🛠️ Data Preparation</h1>
    <p>Configure your target, features, train-test split, encoding and scaling before model training.</p>
</div>
""")

if not st.session_state.get("dataset_id"):
    st.info("Select a dataset first.")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ── Helpers ───────────────────────────────────────────────────────────────
def hash_payload(payload: dict) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def autosave(payload: dict):
    try:
        requests.post(f"{API_BASE}/dataset-preparation/{dataset_id}", json=payload, timeout=10)
    except Exception as e:
        st.warning(f"Auto-save failed: {e}")

# ── Load dataset stats ────────────────────────────────────────────────────
stats    = requests.get(f"{API_BASE}/dataset/{dataset_id}/stats", timeout=30).json()
dtypes   = stats.get("dtypes", {})
all_cols = list(dtypes.keys())

# ── Load saved prep config ────────────────────────────────────────────────
prep_config = None
try:
    r = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=10)
    if r.status_code == 200:
        prep_config = r.json()
except Exception:
    pass

# =====================================================
# DATASET PREVIEW
# =====================================================
st.html('<div class="section-label">📄 Dataset Preview</div>')
preview = requests.get(f"{API_BASE}/dataset/{dataset_id}/preview", params={"n": 200}).json()
st.dataframe(pd.DataFrame(preview.get("rows", [])), use_container_width=True)
st.html('<div class="fancy-divider"></div>')

# =====================================================
# PROBLEM TYPE
# =====================================================
st.html('<div class="section-label">🧩 Problem Type</div>')

problem_type = st.selectbox(
    "Problem Type",
    ["Classification", "Regression"],
    index=(
        ["Classification", "Regression"].index(prep_config["problem_type"])
        if prep_config else 0
    ),
    label_visibility="collapsed",
)

if problem_type == "Classification":
    st.html('<div class="info-note">Predicts a <strong style="color:#a5b4fc">discrete category</strong> — e.g. spam/not spam, disease/no disease. Uses metrics like accuracy, F1, and AUC.</div>')
else:
    st.html('<div class="info-note amber">Predicts a <strong style="color:#fcd34d">continuous numeric value</strong> — e.g. price, temperature, score. Uses metrics like RMSE, MAE, and R².</div>')

st.html('<div class="fancy-divider"></div>')

# =====================================================
# TARGET & FEATURES
# =====================================================
st.html('<div class="section-label">🎯 Target &amp; Features</div>')

target = st.selectbox(
    "Target column",
    all_cols,
    index=(
        all_cols.index(prep_config["target"])
        if prep_config and prep_config["target"] in all_cols else 0
    ),
)

saved_features = None
if prep_config and prep_config.get("features"):
    saved_features = [f for f in prep_config["features"] if f in all_cols and f != target]

available_features = [c for c in all_cols if c != target]
default_features   = saved_features if saved_features else available_features

features = st.multiselect(
    "Feature columns",
    options=available_features,
    default=default_features,
    help="These are the columns used for training. Feature Selection page updates this automatically.",
)

if saved_features and saved_features != available_features:
    excluded = len(available_features) - len(saved_features)
    st.html(f'<div class="info-note"><strong style="color:#34d399">{len(saved_features)} features</strong> were set automatically by Feature Selection. You can modify them here if needed. <span style="color:#7a82a6">({excluded} feature{"s" if excluded != 1 else ""} excluded)</span></div>')
else:
    st.html(f'<div class="info-note amber">Using all <strong style="color:#fcd34d">{len(available_features)} available features</strong>. Run Feature Selection to automatically pick the most impactful ones.</div>')

if not features:
    st.warning("⚠️ Please select at least one feature column.")
    st.stop()

st.html('<div class="fancy-divider"></div>')

# =====================================================
# TRAIN–TEST SPLIT
# =====================================================
st.html('<div class="section-label">📦 Train–Test Split</div>')

test_size = st.slider(
    "Test size (%)", 10, 50,
    int(prep_config["test_size"] * 100) if prep_config else 20,
)

stratify = False
if problem_type == "Classification":
    stratify = st.checkbox(
        "Stratify split (preserve class balance in train & test sets)",
        value=prep_config["stratify"] if prep_config else True,
    )

st.html('<div class="fancy-divider"></div>')

# =====================================================
# ENCODING & SCALING GUIDANCE
# =====================================================
st.html('<div class="section-label">ℹ️ Encoding &amp; Scaling Guidance</div>')

st.html("""
<div class="guidance-grid">
    <div class="guidance-card indigo">
        <div class="gc-title">🔢 Encoding</div>
        <div class="gc-item">
            <div class="gc-item-name">Label Encoding</div>
            <div class="gc-tags">
                <span class="gc-tag good">✔ Tree-based models</span>
                <span class="gc-tag bad">✘ Linear models</span>
                <span class="gc-tag bad">✘ Distance-based models</span>
            </div>
        </div>
        <div class="gc-item">
            <div class="gc-item-name">One-Hot Encoding</div>
            <div class="gc-tags">
                <span class="gc-tag good">✔ Logistic / Linear models</span>
                <span class="gc-tag good">✔ Distance-based models</span>
                <span class="gc-tag bad">✘ High-cardinality columns</span>
            </div>
        </div>
        <div class="gc-item">
            <div class="gc-item-name">No Encoding</div>
            <div class="gc-tags">
                <span class="gc-tag good">✔ Numeric-only datasets</span>
            </div>
        </div>
    </div>
    <div class="guidance-card green">
        <div class="gc-title">📏 Scaling</div>
        <div class="gc-item">
            <div class="gc-item-name">Standardization (Z-score)</div>
            <div class="gc-tags">
                <span class="gc-tag good">✔ Logistic / Linear models</span>
                <span class="gc-tag good">✔ Gradient-based models</span>
            </div>
        </div>
        <div class="gc-item">
            <div class="gc-item-name">Normalization (Min-Max)</div>
            <div class="gc-tags">
                <span class="gc-tag good">✔ KNN &amp; SVM</span>
                <span class="gc-tag good">✔ Distance-based models</span>
            </div>
        </div>
        <div class="gc-item">
            <div class="gc-item-name">No Scaling</div>
            <div class="gc-tags">
                <span class="gc-tag good">✔ Tree-based models</span>
                <span class="gc-tag good">✔ Ensemble methods</span>
            </div>
        </div>
    </div>
</div>
""")

# =====================================================
# ENCODING & SCALING SELECTS
# =====================================================
st.html('<div class="section-label">🔧 Encoding &amp; Scaling</div>')

col_enc, col_scl = st.columns(2)

with col_enc:
    encoding = st.selectbox(
        "Encoding method",
        ["None", "Label Encoding", "One-Hot Encoding"],
        index={None: 0, "label": 1, "onehot": 2}.get(
            prep_config.get("encoding") if prep_config else None, 0
        ),
    )

with col_scl:
    scaling = st.selectbox(
        "Scaling method",
        ["None", "Standardization", "Normalization"],
        index={None: 0, "standard": 1, "minmax": 2}.get(
            prep_config.get("scaling") if prep_config else None, 0
        ),
    )

st.html('<div class="fancy-divider"></div>')

# =====================================================
# AUTO-SAVE
# =====================================================
ENCODING_MAP = {"None": None, "Label Encoding": "label", "One-Hot Encoding": "onehot"}
SCALING_MAP  = {"None": None, "Standardization": "standard", "Normalization": "minmax"}

if target and features:
    payload = {
        "problem_type": problem_type,
        "target":       target,
        "features":     features,
        "test_size":    test_size / 100,
        "stratify":     stratify,
        "encoding":     ENCODING_MAP[encoding],
        "scaling":      SCALING_MAP[scaling],
    }
    payload_hash = hash_payload(payload)
    if st.session_state.get("last_prep_hash") != payload_hash:
        autosave(payload)
        st.session_state["last_prep_hash"] = payload_hash
        st.toast("Configuration auto-saved", icon="💾")

# =====================================================
# SUMMARY
# =====================================================
st.html('<div class="section-label">✅ Current Configuration Summary</div>')

train_pct    = 100 - test_size
enc_display  = encoding if encoding != "None" else "No Encoding"
scl_display  = scaling  if scaling  != "None" else "No Scaling"
strat_display = "Yes" if stratify else "No / N/A"

st.html(f"""
<div class="summary-grid">
    <div class="summary-card blue">
        <div class="sc-label">Problem Type</div>
        <div class="sc-value">{problem_type}</div>
    </div>
    <div class="summary-card purple">
        <div class="sc-label">Target Column</div>
        <div class="sc-value">{target}</div>
    </div>
    <div class="summary-card green">
        <div class="sc-label">Features Selected</div>
        <div class="sc-value">{len(features)} columns</div>
    </div>
    <div class="summary-card amber">
        <div class="sc-label">Train / Test Split</div>
        <div class="sc-value">{train_pct}% / {test_size}%</div>
    </div>
</div>
""")

st.html(f"""
<div class="summary-box">
    <table class="summary-table">
        <tr><td>Problem Type</td><td><span class="sv-purple">{problem_type}</span></td></tr>
        <tr><td>Target Column</td><td><span class="sv-mono">{target}</span></td></tr>
        <tr><td>Features</td><td><span class="sv-green">{len(features)} columns selected</span></td></tr>
        <tr><td>Test Size</td><td><span class="sv-amber">{test_size}%</span></td></tr>
        <tr><td>Stratify Split</td><td><span class="sv-plain">{strat_display}</span></td></tr>
        <tr><td>Encoding</td><td><span class="sv-plain">{enc_display}</span></td></tr>
        <tr><td>Scaling</td><td><span class="sv-plain">{scl_display}</span></td></tr>
    </table>
</div>
""")

# ── Feature list expander ─────────────────────────────────────────────────
with st.expander("📋 View selected feature columns"):
    tags_html = "".join([f'<span class="feature-tag">{f}</span>' for f in features])
    st.html(f'<div class="feature-list">{tags_html}</div>')
