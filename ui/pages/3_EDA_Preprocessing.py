import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import requests
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="ModelForge • EDA & Preprocessing", layout="wide", initial_sidebar_state="expanded")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# ── Auth ───────────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar(current_page="EDA") 
# ── Page-level styles ──────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarCollapseButton"] { display:none !important; }

/* Section label */
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
    background:rgba(12,14,22,.9);
    border:1px solid rgba(255,255,255,.06);
    border-radius:14px; padding:20px 22px;
    position:relative; overflow:visible;
    animation:mf-card-in .4s ease both;
}
.mf-stat-card::before {
    content:''; position:absolute; top:0;left:0;right:0;height:1px;
    border-radius:14px 14px 0 0;
}
.mf-stat-card.blue::before  { background:linear-gradient(90deg,transparent,rgba(79,142,247,.5),transparent); }
.mf-stat-card.purple::before{ background:linear-gradient(90deg,transparent,rgba(139,92,246,.5),transparent); }
.mf-stat-card.green::before { background:linear-gradient(90deg,transparent,rgba(16,185,129,.5),transparent); }
.mf-stat-label { font-size:10px; font-weight:700; color:#7a82a6; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:10px; }
.mf-stat-value {
    font-family:'Syne',sans-serif;
    font-size:clamp(20px,3vw,30px);
    font-weight:800; letter-spacing:-1px; line-height:1;
}
.mf-stat-card.blue   .mf-stat-value { color:#7eb3f7; }
.mf-stat-card.purple .mf-stat-value { color:#a78bfa; }
.mf-stat-card.green  .mf-stat-value { color:#10b981; }

/* Action cards (preprocessing) */
.mf-action-card {
    background:rgba(12,14,22,.9);
    border:1px solid rgba(255,255,255,.06);
    border-radius:14px; padding:24px 24px 20px;
    margin-bottom:16px;
}
.mf-action-title {
    font-family:'Syne',sans-serif;
    font-size:15px; font-weight:700; color:#e0e4ff;
    margin-bottom:4px; letter-spacing:-.2px;
}
.mf-action-desc {
    font-size:12px; color:#525870; margin-bottom:18px; line-height:1.5;
}

/* Warning banner */
.mf-warn {
    background:rgba(18,12,0,.8);
    border:1px solid rgba(120,53,15,.4);
    border-radius:10px; padding:12px 18px;
    color:#f59e0b; font-size:13px; margin-bottom:16px;
    display:flex; align-items:flex-start; gap:10px;
}

/* Override mf_theme p color inside action cards */
.mf-action-card .stMarkdown p { color:#8890aa !important; font-size:13px !important; }

/* Page link hide in main */
.block-container div[data-testid="stPageLink"] { display:none !important; }

/* Style st.container(border=True) to match dark card theme */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(12,14,22,.9) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 14px !important;
    height: 100% !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255,255,255,.12) !important;
}
/* Equal height columns - stretch cards to same height per row */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Theme-matched primary buttons */
.block-container .stButton > button[kind="primary"] {
    background: rgba(79,142,247,.15) !important;
    border: 1px solid rgba(79,142,247,.4) !important;
    border-radius: 10px !important;
    color: #7eb3f7 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: .3px !important;
    transition: all .2s ease !important;
    box-shadow: 0 0 16px rgba(79,142,247,.08) !important;
}
.block-container .stButton > button[kind="primary"]:hover {
    background: rgba(79,142,247,.25) !important;
    border-color: rgba(79,142,247,.7) !important;
    color: #b8d4ff !important;
    box-shadow: 0 0 24px rgba(79,142,247,.2) !important;
    transform: translateY(-1px) !important;
}
/* Secondary buttons */
.block-container .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important;
    color: #6070a0 !important;
    font-size: 13px !important;
    transition: all .2s !important;
}
.block-container .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,.06) !important;
    border-color: rgba(255,255,255,.15) !important;
    color: #a0a8d0 !important;
}

/* Selectbox & inputs */
.stSelectbox label, .stMultiselect label, .stTextInput label {
    font-size:12px !important; color:#7a82a6 !important;
    font-weight:500 !important; letter-spacing:.3px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Context guards ─────────────────────────────────────────────────────────────
if not st.session_state.get("project_id"):
    st.html("""
    <div style="text-align:center;padding:80px 0;">
        <div style="font-size:36px;margin-bottom:14px;">◫</div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#e0e4ff;margin-bottom:8px;">No project selected</div>
        <div style="font-size:13px;color:#525870;">Go to Projects and select a workspace first.</div>
    </div>
    """)
    st.stop()

if not st.session_state.get("dataset_id"):
    st.html("""
    <div style="text-align:center;padding:80px 0;">
        <div style="font-size:36px;margin-bottom:14px;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#e0e4ff;margin-bottom:8px;">No dataset selected</div>
        <div style="font-size:13px;color:#525870;">Upload a dataset first via the Upload page.</div>
    </div>
    """)
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ── Page header ────────────────────────────────────────────────────────────────
st.html("""
<div style="padding:36px 0 28px;border-bottom:1px solid rgba(255,255,255,.04);margin-bottom:28px;position:relative;">
    <div style="position:absolute;top:-20px;right:0;width:300px;height:200px;border-radius:50%;
                background:radial-gradient(circle,rgba(139,92,246,.05),transparent 70%);pointer-events:none;"></div>
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.04);
                border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:4px 12px;margin-bottom:16px;">
        <span style="font-size:14px;">◈</span>
        <span style="font-size:10px;color:#7a82a6;font-weight:600;letter-spacing:.8px;text-transform:uppercase;">EDA &amp; Preprocessing</span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.1;margin-bottom:10px;
                background:linear-gradient(135deg,#ffffff 0%,#9099cc 60%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        Explore &amp; Clean Your Data
    </div>
    <p style="font-size:14px;color:#8890aa;margin:0;font-weight:400;line-height:1.6;max-width:600px;">
        Inspect statistics, fix data quality issues, visualise distributions, and prepare your dataset for modelling.
    </p>
</div>
""")

# ── Dataset stats ──────────────────────────────────────────────────────────────
stats = requests.get(f"{API_BASE}/dataset/{dataset_id}/stats", timeout=60).json()

st.html(f"""
<div class="mf-stat-grid">
    <div class="mf-stat-card blue" style="animation-delay:.04s">
        <div class="mf-stat-label">Rows (estimated)</div>
        <div class="mf-stat-value">{stats['rows_estimated']:,}</div>
    </div>
    <div class="mf-stat-card purple" style="animation-delay:.08s">
        <div class="mf-stat-label">Columns</div>
        <div class="mf-stat-value">{stats['columns_count']}</div>
    </div>
    <div class="mf-stat-card green" style="animation-delay:.12s">
        <div class="mf-stat-label">Size (MB)</div>
        <div class="mf-stat-value">{stats['file_size_mb']}</div>
    </div>
</div>
""")

# ── Load dataset ───────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    csv_bytes = requests.get(f"{API_BASE}/dataset/{dataset_id}/current", timeout=180).content
    df = pd.read_csv(BytesIO(csv_bytes))

# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Dataset Preview</span>
    <div class="line line-long"></div>
</div>
""")

st.dataframe(df.head(200), use_container_width=True)
st.markdown(
    f'<div style="font-size:12px;color:#7a82a6;margin-top:6px;margin-bottom:4px;">'
    f'Shape: <span style="color:#a0a8d0;font-weight:600;">{df.shape[0]:,} rows × {df.shape[1]} columns</span></div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET INFO
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Column Info</span>
    <div class="line line-long"></div>
</div>
""")

info_df = pd.DataFrame({
    "Dtype": df.dtypes.astype(str),
    "Missing": df.isnull().sum(),
    "Missing %": (df.isnull().sum() / len(df) * 100).round(2).astype(str) + "%",
    "Unique": df.nunique(),
})
st.dataframe(info_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Statistical Summary</span>
    <div class="line line-long"></div>
</div>
""")
st.markdown('<div style="font-size:12px;color:#7a82a6;margin-bottom:10px;">Computed on the full dataset via df.describe()</div>', unsafe_allow_html=True)
st.dataframe(df.describe(include="all"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Preprocessing Actions</span>
    <div class="line line-long"></div>
</div>
""")

col_left, col_right = st.columns(2, gap="medium")

# ── Type conversion ────────────────────────────────────────────────────────────
with col_left:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Convert Column Dtype</div><div class="mf-action-desc">Change a column\'s data type to int, float, string, or category.</div>')
        col = st.selectbox("Column", df.columns, key="dtype_col")
        dtype = st.selectbox("Target dtype", ["int", "float", "str", "category"], key="dtype_target")
        if st.button("Convert dtype →", key="btn_dtype", use_container_width=True, type="primary"):
            try:
                requests.post(f"{API_BASE}/preprocessing/convert-dtype", json={
                    "dataset_id": dataset_id, "column": col, "dtype": dtype,
                }).raise_for_status()
                st.toast("✓ Column datatype converted", icon="✅")
                st.rerun()
            except Exception as e:
                st.toast(f"Error: {e}", icon="❌")

# ── Missing values (paired with convert — both have 2 selects) ─────────────────
with col_right:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Handle Missing Values</div><div class="mf-action-desc">Choose a strategy to fill or remove null values across the full dataset.</div>')
        strategy = st.selectbox("Strategy", [
            "Do nothing", "Drop rows", "Fill mean", "Fill median", "Fill mode", "Fill custom"
        ], key="missing_strategy")
        custom = None
        if strategy == "Fill custom":
            custom = st.text_input("Custom fill value", key="missing_custom")
        strategy_map = {
            "Drop rows": "drop", "Fill mean": "mean", "Fill median": "median",
            "Fill mode": "mode", "Fill custom": "custom",
        }
        if st.button("Apply strategy →", key="btn_missing", use_container_width=True, type="primary"):
            if strategy != "Do nothing":
                requests.post(f"{API_BASE}/preprocessing/missing-values", json={
                    "dataset_id": dataset_id,
                    "strategy": strategy_map[strategy],
                    "value": custom,
                })
                st.toast("✓ Missing values handled", icon="✅")
                st.rerun()

col_left2, col_right2 = st.columns(2, gap="medium")

# ── Drop columns (paired with drop duplicates — both have 1 widget + button) ───
with col_left2:
    with st.container(border=True):
        st.html('<div class="mf-action-title">Drop Columns</div><div class="mf-action-desc">Remove one or more columns from the dataset permanently.</div>')
        cols_to_drop = st.multiselect("Columns to drop", df.columns, key="drop_cols")
        if st.button("Drop selected →", key="btn_drop_cols", use_container_width=True, type="primary"):
            if cols_to_drop:
                requests.post(f"{API_BASE}/preprocessing/drop-columns", json={
                    "dataset_id": dataset_id, "columns": cols_to_drop,
                })
                st.toast("✓ Columns dropped", icon="✅")
                st.rerun()
            else:
                st.toast("Select at least one column", icon="⚠️")

# ── Drop duplicates ────────────────────────────────────────────────────────────
with col_right2:
    dup_count = int(df.duplicated().sum())
    with st.container(border=True):
        st.html(f"""
        <div class="mf-action-title">Drop Duplicate Rows</div>
        <div class="mf-action-desc">Remove all exact duplicate rows from the dataset.</div>
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                    border-radius:8px;padding:6px 14px;margin-bottom:16px;">
            <span style="font-size:12px;color:#7a82a6;">Detected duplicates:</span>
            <span style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                         color:{'#f87171' if dup_count > 0 else '#10b981'};">{dup_count:,}</span>
        </div>
        """)
        if st.button("Drop duplicates →", key="btn_dups", use_container_width=True,
                     type="primary", disabled=(dup_count == 0)):
            requests.post(f"{API_BASE}/preprocessing/drop-duplicates", json={"dataset_id": dataset_id})
            st.toast("✓ Duplicates removed", icon="✅")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="mf-section-label">
    <div class="line line-short"></div>
    <span>Visualisations</span>
    <div class="line line-long"></div>
</div>
""")

st.markdown("""
<div style="background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.35);
            border-radius:10px;padding:11px 16px;color:#f59e0b;font-size:12px;
            margin-bottom:20px;display:flex;align-items:center;gap:10px;">
    <span style="font-size:16px;">⚡</span>
    <span>Visualisations run on the full dataset and may take time for large files.</span>
</div>
""", unsafe_allow_html=True)

viz_col, _ = st.columns([1, 1])
with viz_col:
    col_to_plot = st.selectbox("Select column to visualise", df.columns, key="viz_col")

# Dark-themed matplotlib
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0c0e16")
ax.set_facecolor("#0c0e16")

if pd.api.types.is_numeric_dtype(df[col_to_plot]):
    sns.histplot(
        df[col_to_plot].dropna(), kde=True, ax=ax,
        color="#4f8ef7", alpha=0.6,
        line_kws={"color": "#8b5cf6", "linewidth": 2},
    )
    ax.set_xlabel(col_to_plot, color="#7a82a6", fontsize=11)
    ax.set_ylabel("Count", color="#7a82a6", fontsize=11)
else:
    vc = df[col_to_plot].value_counts().head(20)
    ax.bar(vc.index.astype(str), vc.values, color="#4f8ef7", alpha=0.75, edgecolor="#8b5cf6", linewidth=.8)
    ax.set_xlabel(col_to_plot, color="#7a82a6", fontsize=11)
    ax.set_ylabel("Count", color="#7a82a6", fontsize=11)
    plt.xticks(rotation=35, ha="right")

ax.tick_params(colors="#525870")
ax.spines["bottom"].set_color("#1a1d2e")
ax.spines["left"].set_color("#1a1d2e")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#111320", linewidth=.8)
ax.set_axisbelow(True)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)
