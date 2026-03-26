import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ModelForge • Feature Engineering", layout="wide", initial_sidebar_state="expanded")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# ── Auth ───────────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar(current_page="Features") 
# ── Page styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarCollapseButton"] { display:none !important; }

.mf-section-label {
    display:flex; align-items:center; gap:8px; margin:32px 0 16px;
}
.mf-section-label .line { height:1px; background:rgba(255,255,255,.06); }
.mf-section-label .line-short { width:20px; }
.mf-section-label .line-long  { flex:1; }
.mf-section-label span {
    font-size:11px; font-weight:700; color:#7a82a6;
    letter-spacing:1.5px; text-transform:uppercase;
}

/* Stat cards */
.mf-stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:28px; }
.mf-stat-card {
    background:rgba(12,14,22,.9); border:1px solid rgba(255,255,255,.06);
    border-radius:14px; padding:20px 22px; position:relative; overflow:visible;
    animation:mf-card-in .4s ease both;
}
.mf-stat-card::before {
    content:''; position:absolute; top:0;left:0;right:0;height:1px; border-radius:14px 14px 0 0;
}
.mf-stat-card.blue::before   { background:linear-gradient(90deg,transparent,rgba(79,142,247,.5),transparent); }
.mf-stat-card.purple::before { background:linear-gradient(90deg,transparent,rgba(139,92,246,.5),transparent); }
.mf-stat-card.green::before  { background:linear-gradient(90deg,transparent,rgba(16,185,129,.5),transparent); }
.mf-stat-label { font-size:10px; font-weight:700; color:#7a82a6; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:10px; }
.mf-stat-value { font-family:'Syne',sans-serif; font-size:clamp(20px,3vw,30px); font-weight:800; letter-spacing:-1px; line-height:1; }
.mf-stat-card.blue   .mf-stat-value { color:#7eb3f7; }
.mf-stat-card.purple .mf-stat-value { color:#a78bfa; }
.mf-stat-card.green  .mf-stat-value { color:#10b981; }

/* Action card titles/descs rendered via st.html */
.mf-action-title {
    font-family:'Syne',sans-serif; font-size:16px; font-weight:700;
    color:#e0e4ff; margin-bottom:4px; letter-spacing:-.2px;
}
.mf-action-desc { font-size:12px; color:#7a82a6; margin-bottom:18px; line-height:1.5; }

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(12,14,22,.9) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 14px !important;
    height: 100% !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255,255,255,.12) !important;
}
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex !important; flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important; display: flex !important; flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 !important; display: flex !important; flex-direction: column !important;
}

/* Primary buttons */
.block-container .stButton > button[kind="primary"] {
    background: rgba(79,142,247,.15) !important;
    border: 1px solid rgba(79,142,247,.4) !important;
    border-radius: 10px !important; color: #7eb3f7 !important;
    font-weight: 600 !important; font-size: 13px !important;
    letter-spacing: .3px !important; transition: all .2s ease !important;
}
.block-container .stButton > button[kind="primary"]:hover {
    background: rgba(79,142,247,.25) !important;
    border-color: rgba(79,142,247,.7) !important; color: #b8d4ff !important;
    transform: translateY(-1px) !important;
}
/* Secondary buttons */
.block-container .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important; color: #6070a0 !important; font-size: 13px !important;
}
.block-container .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,.06) !important;
    border-color: rgba(255,255,255,.15) !important; color: #a0a8d0 !important;
}

/* Labels */
.stSelectbox label, .stMultiselect label, .stTextInput label,
.stNumberInput label, .stCheckbox label {
    font-size:12px !important; color:#7a82a6 !important;
    font-weight:500 !important; letter-spacing:.3px !important;
}
.block-container div[data-testid="stPageLink"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Context guards ─────────────────────────────────────────────────────────────
if not st.session_state.get("dataset_id"):
    st.html("""
    <div style="text-align:center;padding:80px 0;">
        <div style="font-size:36px;margin-bottom:14px;">⊞</div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#e0e4ff;margin-bottom:8px;">No dataset selected</div>
        <div style="font-size:13px;color:#525870;">Upload a dataset first via the Upload page.</div>
    </div>
    """)
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ── Load stats ─────────────────────────────────────────────────────────────────
try:
    resp = requests.get(f"{API_BASE}/dataset/{dataset_id}/stats", timeout=30)
    resp.raise_for_status()
    stats = resp.json()
except Exception as e:
    st.error(f"Failed to load dataset stats: {e}")
    st.stop()

rows         = stats.get("rows") or stats.get("rows_estimated", 0)
columns_count = (stats.get("columns_count") or len(stats.get("column_names", [])) or len(stats.get("dtypes", {})))

# ── Page header ────────────────────────────────────────────────────────────────
st.html("""
<div style="padding:36px 0 28px;border-bottom:1px solid rgba(255,255,255,.04);margin-bottom:28px;position:relative;">
    <div style="position:absolute;top:-20px;right:0;width:300px;height:200px;border-radius:50%;
                background:radial-gradient(circle,rgba(79,142,247,.05),transparent 70%);pointer-events:none;"></div>
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.04);
                border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:4px 12px;margin-bottom:16px;">
        <span style="font-size:13px;">⊞</span>
        <span style="font-size:10px;color:#7a82a6;font-weight:600;letter-spacing:.8px;text-transform:uppercase;">Feature Engineering</span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.1;margin-bottom:10px;
                background:linear-gradient(135deg,#ffffff 0%,#9099cc 60%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        Build &amp; Transform Features
    </div>
    <p style="font-size:14px;color:#8890aa;margin:0;font-weight:400;line-height:1.6;max-width:600px;">
        Create new features from existing columns, apply numeric transforms, extract date components, and engineer age features.
    </p>
</div>
""")

# ── Stat cards ─────────────────────────────────────────────────────────────────
st.html(f"""
<div class="mf-stat-grid">
    <div class="mf-stat-card blue" style="animation-delay:.04s">
        <div class="mf-stat-label">Rows</div>
        <div class="mf-stat-value">{rows:,}</div>
    </div>
    <div class="mf-stat-card purple" style="animation-delay:.08s">
        <div class="mf-stat-label">Columns</div>
        <div class="mf-stat-value">{columns_count}</div>
    </div>
    <div class="mf-stat-card green" style="animation-delay:.12s">
        <div class="mf-stat-label">Size (MB)</div>
        <div class="mf-stat-value">{stats.get('file_size_mb', 0)}</div>
    </div>
</div>
""")

# ── Load preview ───────────────────────────────────────────────────────────────
try:
    preview_resp = requests.get(
        f"{API_BASE}/dataset/{dataset_id}/preview", params={"n": 200}, timeout=30,
    )
    preview_resp.raise_for_status()
    preview = preview_resp.json()
    df = pd.DataFrame(preview.get("rows", []))
except Exception as e:
    st.error(f"Failed to load preview: {e}")
    st.stop()

if df.empty:
    st.warning("Dataset preview is empty.")
    st.stop()

# ── Dataset preview ────────────────────────────────────────────────────────────
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Dataset Preview</span>
    <div class="line line-long"></div>
</div>
""")
st.dataframe(df, use_container_width=True)
st.markdown(
    f'<div style="font-size:12px;color:#7a82a6;margin-top:6px;">'
    f'Sample: <span style="color:#a0a8d0;font-weight:600;">{len(df)} rows shown</span></div>',
    unsafe_allow_html=True
)

# ── Column groups ──────────────────────────────────────────────────────────────
dtypes      = stats.get("dtypes", {})
numeric_cols = [c for c, t in dtypes.items() if t.startswith(("int", "float"))]
all_cols     = list(dtypes.keys())

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Numeric Feature Creation + Numeric Transformation
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Numeric Features</span>
    <div class="line line-long"></div>
</div>
""")

col_l, col_r = st.columns(2, gap="medium")

with col_l:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Numeric Feature Creation</div>'
                '<div class="mf-action-desc">Combine two numeric columns using sum, difference, product, or ratio to create a new feature.</div>')
        if numeric_cols:
            col1  = st.selectbox("Column 1", numeric_cols, key="nf_col1")
            col2  = st.selectbox("Column 2 (optional)", ["None"] + numeric_cols, key="nf_col2")
            op    = st.selectbox("Operation", ["sum", "diff", "product", "ratio"], key="nf_op")
            nname = st.text_input("New feature name", placeholder="e.g. bmi_age_ratio", key="nf_name")
            if st.button("Create feature →", key="nf_btn", use_container_width=True, type="primary"):
                if not nname:
                    st.toast("Enter a new feature name", icon="⚠️")
                else:
                    try:
                        requests.post(f"{API_BASE}/feature/numeric-feature", json={
                            "dataset_id": dataset_id, "col1": col1,
                            "col2": None if col2 == "None" else col2,
                            "operation": op, "new_name": nname,
                        }).raise_for_status()
                        st.toast("✓ Numeric feature created", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.toast(f"Error: {e}", icon="❌")
        else:
            st.html('<div style="color:#525870;font-size:13px;padding:8px 0;">No numeric columns available.</div>')

with col_r:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Numeric Transformation</div>'
                '<div class="mf-action-desc">Apply log, square, sqrt, power, or binning transforms to a numeric column.</div>')
        if numeric_cols:
            tcol      = st.selectbox("Numeric column", numeric_cols, key="nt_col")
            transform = st.selectbox("Transform", ["log", "square", "sqrt", "power", "bin"], key="nt_transform")
            new_col   = st.text_input("New column name", placeholder="e.g. log_age", key="nt_name")
            power = st.number_input("Power", value=2, key="nt_power") if transform == "power" else None
            bins  = st.number_input("Bins", value=5, min_value=2, key="nt_bins") if transform == "bin" else None
            if st.button("Apply transform →", key="nt_btn", use_container_width=True, type="primary"):
                if not new_col:
                    st.toast("Enter a new column name", icon="⚠️")
                else:
                    try:
                        requests.post(f"{API_BASE}/feature/numeric-transform", json={
                            "dataset_id": dataset_id, "column": tcol,
                            "transform": transform, "new_name": new_col,
                            "power": power, "bins": bins,
                        }).raise_for_status()
                        st.toast("✓ Transformation applied", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.toast(f"Error: {e}", icon="❌")
        else:
            st.html('<div style="color:#525870;font-size:13px;padding:8px 0;">No numeric columns available.</div>')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Date Feature Extraction + Age from DOB
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Date &amp; Time Features</span>
    <div class="line line-long"></div>
</div>
""")

col_l2, col_r2 = st.columns(2, gap="medium")

with col_l2:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Date Feature Extraction</div>'
                '<div class="mf-action-desc">Extract year, month, day, weekday, hour and more from a date column.</div>')
        dcol       = st.selectbox("Date column", all_cols, key="df_col")
        features   = st.multiselect("Extract", ["year","month","day","weekday","hour","minute","second","quarter"], key="df_features")
        prefix     = st.text_input("Prefix for new columns", placeholder="e.g. signup_", key="df_prefix")
        keep_orig  = st.checkbox("Keep original column", value=True, key="df_keep")
        if st.button("Extract date features →", key="df_btn", use_container_width=True, type="primary"):
            if not features or not prefix:
                st.toast("Select features and enter a prefix", icon="⚠️")
            else:
                try:
                    requests.post(f"{API_BASE}/feature/date-features", json={
                        "dataset_id": dataset_id, "column": dcol,
                        "features": features, "prefix": prefix,
                        "keep_original": keep_orig,
                    }).raise_for_status()
                    st.toast("✓ Date features extracted", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.toast(f"Error: {e}", icon="❌")

with col_r2:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Age from Date of Birth</div>'
                '<div class="mf-action-desc">Compute age in years from a date-of-birth column and add it as a new feature.</div>')
        dob_col  = st.selectbox("DOB column", all_cols, key="age_col")
        age_name = st.text_input("Age column name", placeholder="e.g. patient_age", key="age_name")
        keep_dob = st.checkbox("Keep DOB column", value=False, key="age_keep")
        if st.button("Create age feature →", key="age_btn", use_container_width=True, type="primary"):
            if not age_name:
                st.toast("Enter an age column name", icon="⚠️")
            else:
                try:
                    requests.post(f"{API_BASE}/feature/age-from-dob", json={
                        "dataset_id": dataset_id, "dob_column": dob_col,
                        "new_name": age_name, "keep_original": keep_dob,
                    }).raise_for_status()
                    st.toast("✓ Age feature created", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.toast(f"Error: {e}", icon="❌")
