# pages/10_Feature_Selection.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ModelForge • Feature Selection", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)
API_BASE = "http://127.0.0.1:8000"

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Section labels ────────────────────────────────────── */
.section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: #7a82a6 !important;
    text-transform: uppercase !important;
    margin: 20px 0 10px 0 !important;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Page header ───────────────────────────────────────── */
.page-header { margin-bottom: 24px; }
.page-header h1 { font-size: 28px; font-weight: 800; color: #e2e8f0; margin: 0 0 8px 0; }
.page-header p  { font-size: 15px; color: #8890aa; font-weight: 400; margin: 0; line-height: 1.5; }

/* ── Info banner ───────────────────────────────────────── */
.info-banner {
    background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
}
.info-banner .label { font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: #7a82a6; text-transform: uppercase; margin-bottom: 10px; }
.info-banner .row   { display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }
.info-banner .pill  { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #c8cfe8; }
.info-banner .pill span.key { color: #8890aa; font-weight: 500; }
.info-banner .pill span.val        { color: #38bdf8; font-weight: 700; font-family: monospace; background: rgba(56,189,248,0.1);   padding: 2px 8px; border-radius: 4px; }
.info-banner .pill span.val-purple { color: #a78bfa; font-weight: 700; font-family: monospace; background: rgba(167,139,250,0.1); padding: 2px 8px; border-radius: 4px; }
.info-banner .pill span.val-green  { color: #34d399; font-weight: 700;                         background: rgba(52,211,153,0.1);  padding: 2px 8px; border-radius: 4px; }

/* ── Method tile grid ──────────────────────────────────── */
.method-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0 0 0; }
.method-tile {
    border-radius: 14px; border: 1.5px solid rgba(99,102,241,0.2);
    background: rgba(15,18,40,0.8); padding: 20px 18px 20px 18px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s, transform 0.2s;
}
.method-tile::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: rgba(99,102,241,0.15); transition: background 0.2s;
}
.method-tile.active {
    border-color: #6366f1;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(56,189,248,0.06) 100%);
    box-shadow: 0 0 0 1px rgba(99,102,241,0.25), 0 8px 28px rgba(99,102,241,0.18);
}
.method-tile.active::before { background: linear-gradient(90deg, #6366f1, #38bdf8); }
.method-tile .mt-icon  { font-size: 30px; margin-bottom: 10px; display: block; }
.method-tile .mt-name  { font-size: 13px; font-weight: 700; color: #c8cfe8; margin-bottom: 4px; }
.method-tile.active .mt-name { color: #e2e8f0; }
.method-tile .mt-sub   { font-size: 11px; color: #4a5280; line-height: 1.4; }
.method-tile.active .mt-sub  { color: #7a82a6; }
.method-tile .mt-badge {
    display: inline-flex; align-items: center; gap: 4px;
    margin-top: 12px; font-size: 10px; font-weight: 700; letter-spacing: 0.4px;
    padding: 3px 10px; border-radius: 20px;
    background: rgba(99,102,241,0.0); color: transparent; border: 1px solid transparent;
    transition: all 0.2s;
}
.method-tile.active .mt-badge { background: rgba(99,102,241,0.15); color: #a5b4fc; border-color: rgba(99,102,241,0.3); }

/* ── Tile select buttons ───────────────────────────────── */
div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {
    background: rgba(15,18,40,0.5) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    color: #4a5280 !important;
    font-size: 12px !important; font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 8px 0 !important;
    margin-top: 8px !important;
    letter-spacing: 0.5px !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.08) !important;
    color: #7a82a6 !important;
    border-color: rgba(99,102,241,0.35) !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(99,102,241,0.22), rgba(56,189,248,0.12)) !important;
    border: 1px solid rgba(99,102,241,0.45) !important;
    color: #a5b4fc !important;
    font-size: 12px !important; font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 8px 0 !important;
    margin-top: 8px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.15) !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, rgba(99,102,241,0.32), rgba(56,189,248,0.2)) !important;
    transform: none !important;
}

/* ── Description panel ─────────────────────────────────── */
.desc-panel {
    border-radius: 14px; padding: 20px 22px; margin: 16px 0 20px 0;
    display: flex; gap: 18px; align-items: flex-start;
    border: 1px solid rgba(99,102,241,0.18);
    background: rgba(12,15,35,0.7);
    border-left: 3px solid #6366f1;
}
.desc-panel.green { border-left-color: #34d399; }
.desc-panel.amber { border-left-color: #f59e0b; }
.desc-panel .dp-icon  { font-size: 34px; line-height: 1; flex-shrink: 0; margin-top: 2px; }
.desc-panel .dp-body  { flex: 1; }
.desc-panel .dp-title { font-size: 15px; font-weight: 700; color: #a5b4fc; margin-bottom: 7px; }
.desc-panel.green .dp-title { color: #6ee7b7; }
.desc-panel.amber .dp-title { color: #fcd34d; }
.desc-panel .dp-desc  { font-size: 13px; color: #8890aa; line-height: 1.65; margin-bottom: 14px; }
.desc-panel .dp-tags  { display: flex; flex-wrap: wrap; gap: 8px; }
.desc-panel .dp-tag   { font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px; letter-spacing: 0.3px; background: rgba(99,102,241,0.12); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }
.desc-panel.green .dp-tag { background: rgba(52,211,153,0.1);  color: #6ee7b7; border-color: rgba(52,211,153,0.2); }
.desc-panel.amber .dp-tag { background: rgba(245,158,11,0.1);  color: #fcd34d; border-color: rgba(245,158,11,0.2); }
.desc-panel.amber .dp-tag.warn { background: rgba(251,146,60,0.1); color: #fb923c; border-color: rgba(251,146,60,0.2); }

/* ── Metric cards ──────────────────────────────────────── */
.metric-row { display: flex; gap: 16px; margin: 16px 0; }
.metric-card { flex: 1; background: rgba(20,24,44,0.8); border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; padding: 18px 20px; text-align: center; position: relative; overflow: visible; }
.metric-card .m-label { font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: #7a82a6; text-transform: uppercase; margin-bottom: 8px; }
.metric-card .m-value { font-size: clamp(20px, 3vw, 30px); font-weight: 800; color: #c8cfe8; line-height: 1; }
.metric-card.blue  .m-value { color: #38bdf8; }
.metric-card.green .m-value { color: #34d399; }
.metric-card.red   .m-value { color: #f87171; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 12px 12px 0 0; }
.metric-card.blue::before  { background: linear-gradient(90deg, #38bdf8, #818cf8); }
.metric-card.green::before { background: linear-gradient(90deg, #34d399, #6ee7b7); }
.metric-card.red::before   { background: linear-gradient(90deg, #f87171, #fb923c); }

/* ── Apply box ─────────────────────────────────────────── */
.apply-box { background: linear-gradient(135deg, rgba(52,211,153,0.06) 0%, rgba(99,102,241,0.06) 100%); border: 1px solid rgba(52,211,153,0.2); border-radius: 12px; padding: 18px 22px; margin: 8px 0 16px 0; }
.apply-box .apply-title { font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: #7a82a6; text-transform: uppercase; margin-bottom: 8px; }
.apply-box .apply-desc  { font-size: 13px; color: #8890aa; }

/* ── Divider ───────────────────────────────────────────── */
.fancy-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent); margin: 24px 0; }

/* ── Primary action buttons (Run / Apply) ──────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 55%, #38bdf8 100%) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; font-size: 14px !important;
    border-radius: 10px !important; padding: 10px 28px !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 55%, #0ea5e9 100%) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.55) !important;
    transform: translateY(-1px) !important;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
render_sidebar(current_page="Selection") 
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Page header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🎯 Feature Selection</h1>
    <p>Identify the most impactful features and remove noise from your dataset before training.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("dataset_id"):
    st.info("Select a dataset first.")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ── Load prep config ──────────────────────────────────────────────────────
try:
    prep = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30).json()
except Exception:
    st.error("Could not load data preparation config. Complete Data Preparation first.")
    st.stop()

problem_type = prep.get("problem_type", "Classification")
target       = prep.get("target", "")
features     = prep.get("features", [])

# ── Info banner ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="info-banner">
    <div class="label">⚙️ Active Configuration</div>
    <div class="row">
        <div class="pill"><span class="key">Problem Type</span><span class="val-purple">{problem_type}</span></div>
        <div class="pill"><span class="key">Target</span><span class="val">{target}</span></div>
        <div class="pill"><span class="key">Current Features</span><span class="val-green">{len(features)}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# =====================================================
# METHOD SELECTION
# =====================================================
st.markdown('<div class="section-label">🔧 Select Feature Selection Method</div>', unsafe_allow_html=True)

if "fs_method" not in st.session_state:
    st.session_state["fs_method"] = "importance"

METHODS = [
    ("importance",  "🌲", "Feature Importance",           "Tree-Based Ranking",          "Best Starting Point"),
    ("correlation", "📉", "Correlation Threshold",        "Remove Multicollinearity",    "Linear Model Friendly"),
    ("rfe",         "🔁", "Recursive Feature Elimination","Iterative Model-Guided",      "Maximum Precision"),
]

# Render tile visuals + select buttons in 3 columns
cols = st.columns(3)
for col, (mkey, icon, name, sub, badge) in zip(cols, METHODS):
    with col:
        active = st.session_state["fs_method"] == mkey
        st.markdown(f"""
        <div class="method-tile {'active' if active else ''}">
            <span class="mt-icon">{icon}</span>
            <div class="mt-name">{name}</div>
            <div class="mt-sub">{sub}</div>
            <span class="mt-badge">{'✦ ' + badge if active else ''}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(
            "✓ Selected" if active else "Select",
            key=f"btn_{mkey}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state["fs_method"] = mkey
            st.rerun()

selected_method = st.session_state["fs_method"]

# ── Description panel ─────────────────────────────────────────────────────
if selected_method == "importance":
    st.markdown("""
    <div class="desc-panel">
        <div class="dp-icon">🌲</div>
        <div class="dp-body">
            <div class="dp-title">Feature Importance (Tree-Based)</div>
            <div class="dp-desc">
                Trains a Random Forest on your data and ranks every feature by how much it
                contributes to accurate predictions. Features scoring below your minimum threshold
                are automatically dropped, leaving only the most impactful signals.
            </div>
            <div class="dp-tags">
                <span class="dp-tag">✅ Classification &amp; Regression</span>
                <span class="dp-tag">✅ Fast &amp; Interpretable</span>
                <span class="dp-tag">✅ Handles Non-linearity</span>
                <span class="dp-tag">✅ Great Starting Point</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif selected_method == "correlation":
    st.markdown("""
    <div class="desc-panel green">
        <div class="dp-icon">📉</div>
        <div class="dp-body">
            <div class="dp-title">Correlation Threshold</div>
            <div class="dp-desc">
                Scans all feature pairs and removes those that are highly correlated with each
                other — keeping only one from each redundant group. This eliminates multicollinearity
                and reduces noise, especially beneficial before training linear or logistic models.
            </div>
            <div class="dp-tags">
                <span class="dp-tag">✅ Reduces Feature Redundancy</span>
                <span class="dp-tag">✅ Improves Model Stability</span>
                <span class="dp-tag">✅ Best Before Linear Models</span>
                <span class="dp-tag">✅ No Target Column Needed</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif selected_method == "rfe":
    st.markdown("""
    <div class="desc-panel amber">
        <div class="dp-icon">🔁</div>
        <div class="dp-body">
            <div class="dp-title">Recursive Feature Elimination (RFE)</div>
            <div class="dp-desc">
                Iteratively trains a model, removes the least important feature, then retrains —
                repeating until your target feature count is reached. Gives you the most
                model-aligned subset, at the cost of extra computation time on large datasets.
            </div>
            <div class="dp-tags">
                <span class="dp-tag">✅ Model-Guided Selection</span>
                <span class="dp-tag">✅ Eliminates Weakest Features First</span>
                <span class="dp-tag">✅ Very Precise Results</span>
                <span class="dp-tag warn">⚠️ Slower on Large Feature Sets</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Parameters ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">⚙️ Parameters</div>', unsafe_allow_html=True)

top_n = 10
importance_threshold = 0.01
correlation_threshold = 0.9
n_rfe_features = 5

if selected_method == "importance":
    top_n = st.slider(
        "Max features to keep (Top N)",
        min_value=1, max_value=len(features), value=min(10, len(features)),
    )
    importance_threshold = st.slider(
        "Minimum importance threshold",
        min_value=0.0, max_value=0.1, value=0.01, step=0.005, format="%.3f",
    )

elif selected_method == "correlation":
    correlation_threshold = st.slider(
        "Correlation threshold (remove above this)",
        min_value=0.5, max_value=1.0, value=0.9, step=0.05, format="%.2f",
    )
    st.caption(
        "Features with correlation **above** this threshold with another feature will be removed. "
        "Lower = more aggressive removal."
    )

elif selected_method == "rfe":
    n_rfe_features = st.slider(
        "Number of features to select",
        min_value=1, max_value=len(features), value=min(5, len(features)),
    )

# =====================================================
# RUN FEATURE SELECTION
# =====================================================
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

if st.button("🚀 Run Feature Selection", type="primary"):
    with st.spinner("Running feature selection..."):
        resp = requests.post(
            f"{API_BASE}/feature-selection/run",
            json={
                "dataset_id":            dataset_id,
                "method":                selected_method,
                "top_n":                 top_n,
                "importance_threshold":  importance_threshold,
                "correlation_threshold": correlation_threshold,
                "n_rfe_features":        n_rfe_features,
            },
            timeout=120,
        )
    if resp.status_code == 200:
        st.session_state["fs_result"] = resp.json()
        st.success("✅ Feature selection complete!")
    else:
        st.error(f"Feature selection failed: {resp.text}")

# =====================================================
# DISPLAY RESULTS
# =====================================================
if "fs_result" in st.session_state:
    result            = st.session_state["fs_result"]
    selected_features = result.get("selected_features", [])
    selected_list     = result.get("selected", [])
    removed_list      = result.get("removed", [])

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Results Summary</div>', unsafe_allow_html=True)

    total         = result.get("total_features", len(features))
    kept          = len(selected_features)
    removed_count = len(removed_list)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="m-label">Total Features</div>
            <div class="m-value">{total}</div>
        </div>
        <div class="metric-card green">
            <div class="m-label">✅ Selected</div>
            <div class="m-value">{kept}</div>
        </div>
        <div class="metric-card red">
            <div class="m-label">❌ Removed</div>
            <div class="m-value">{removed_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Importance bar chart ──────────────────────────────────────────────
    if selected_method == "importance" and selected_list:
        st.markdown('<div class="section-label">📊 Feature Importance Scores</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame(selected_list).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(imp_df) * 0.45)))
        fig.patch.set_facecolor("#0d0f1e")
        ax.set_facecolor("#0d0f1e")
        colors = plt.cm.cool([i / max(len(imp_df) - 1, 1) for i in range(len(imp_df))])
        ax.barh(imp_df["feature"], imp_df["importance"], color=colors, height=0.6)
        ax.set_xlabel("Importance Score", color="#7a82a6", fontsize=11)
        ax.set_title("Selected Features by Importance", color="#c8cfe8", fontsize=13, pad=14, fontweight="bold")
        ax.tick_params(colors="#8890aa", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2240")
            spine.set_alpha(0.5)
        ax.grid(axis="x", color="#1e2240", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Correlation table ─────────────────────────────────────────────────
    if selected_method == "correlation":
        target_corr = result.get("target_correlations", {})
        if target_corr:
            st.markdown(f'<div class="section-label">📈 Feature Correlation with Target ({target})</div>', unsafe_allow_html=True)
            corr_df = pd.DataFrame(
                sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True),
                columns=["Feature", "Correlation with Target"],
            )
            st.dataframe(corr_df.astype(str), use_container_width=True)

    # ── Selected features table ───────────────────────────────────────────
    st.markdown('<div class="section-label">✅ Selected Features</div>', unsafe_allow_html=True)
    if selected_list:
        st.dataframe(pd.DataFrame(selected_list).astype(str), use_container_width=True)
    else:
        st.info("No features selected.")

    with st.expander(f"❌ Removed Features ({len(removed_list)})"):
        if removed_list:
            st.dataframe(pd.DataFrame(removed_list).astype(str), use_container_width=True)
        else:
            st.info("No features removed.")

    # ── Apply ─────────────────────────────────────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="apply-box">
        <div class="apply-title">⚡ Apply to Data Preparation</div>
        <div class="apply-desc">The following <strong style="color:#34d399">{kept} features</strong> will be saved and used for model training.</div>
    </div>
    """, unsafe_allow_html=True)

    st.code(", ".join(selected_features))

    if st.button("✅ Apply Selected Features to Data Preparation", type="primary"):
        current_prep = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=10).json()
        resp = requests.post(
            f"{API_BASE}/dataset-preparation/{dataset_id}",
            json={
                "problem_type": current_prep["problem_type"],
                "target":       current_prep["target"],
                "features":     selected_features,
                "test_size":    current_prep["test_size"],
                "stratify":     current_prep["stratify"],
                "encoding":     current_prep.get("encoding"),
                "scaling":      current_prep.get("scaling"),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            st.success(
                f"✅ Data preparation updated with {len(selected_features)} selected features! "
                "Go to Train Model to retrain."
            )
        else:
            st.error(f"Failed to update: {resp.text}")
