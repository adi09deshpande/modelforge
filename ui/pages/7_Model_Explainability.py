# pages/7_Model_Explainability.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "figure.facecolor":  "#0c0e16",
    "axes.facecolor":    "#0c0e16",
    "axes.edgecolor":    "#1f2235",
    "axes.labelcolor":   "#7a82a6",
    "xtick.color":       "#525870",
    "ytick.color":       "#525870",
    "text.color":        "#c8ccee",
    "grid.color":        "#1a1d2e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "sans-serif",
})

st.set_page_config(page_title="ModelForge • Explainability", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# ── Styles ────────────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

.mf-sec { display:flex;align-items:center;gap:10px;margin-bottom:16px; }
.mf-sec-line { height:1px;background:rgba(255,255,255,.1); }
.mf-sec-label {
    font-size:11px;font-weight:700;color:#7a82a6;
    letter-spacing:1.4px;text-transform:uppercase;white-space:nowrap;
}

.mf-card {
    background:rgba(12,14,22,.9);
    border:1px solid rgba(255,255,255,.07);
    border-radius:12px;
    padding:20px 22px;
    position:relative;
    overflow:hidden;
    backdrop-filter:blur(8px);
    transition:all .22s ease;
}
.mf-card:hover {
    border-color:rgba(255,255,255,.12);
    box-shadow:0 8px 30px rgba(0,0,0,.5);
}

.mf-prep-row {
    display:flex;align-items:center;padding:11px 16px;
    border-bottom:1px solid rgba(255,255,255,.04);transition:background .15s;
}
.mf-prep-row:last-child { border-bottom:none; }
.mf-prep-row:hover { background:rgba(255,255,255,.02); }
.mf-prep-key { font-size:12px;color:#7a82a6;font-weight:500;width:180px;flex-shrink:0; }
.mf-prep-val { font-size:13px;color:#d0d4f0;font-weight:500; }

.mf-feat-bar-wrap {
    background:rgba(12,14,22,.85);
    border:1px solid rgba(255,255,255,.06);
    border-radius:12px;
    padding:18px 20px;
    position:relative;overflow:hidden;
}
.mf-feat-bar-wrap::before {
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
}
.mf-feat-bar-wrap.blue::before  { background:linear-gradient(90deg,transparent,rgba(79,142,247,.5),transparent); }
.mf-feat-bar-wrap.purple::before{ background:linear-gradient(90deg,transparent,rgba(139,92,246,.5),transparent); }
.mf-feat-bar-wrap.green::before { background:linear-gradient(90deg,transparent,rgba(16,185,129,.5),transparent); }

.mf-feat-row {
    display:flex;align-items:center;gap:12px;
    padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);
}
.mf-feat-row:last-child { border-bottom:none; }
.mf-feat-name {
    font-size:12px;color:#c8ccee;font-weight:500;
    width:140px;flex-shrink:0;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;
}
.mf-feat-track {
    flex:1;height:6px;background:rgba(255,255,255,.06);
    border-radius:6px;overflow:hidden;
}
.mf-feat-fill { height:100%;border-radius:6px;transition:width .5s ease; }
.mf-feat-val {
    font-family:'Syne',sans-serif;font-size:12px;
    font-weight:700;color:#c8ccee;width:60px;text-align:right;flex-shrink:0;
}

.mf-info-card {
    background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.4);
    border-radius:10px;padding:14px 18px;font-size:13px;color:#8890aa;line-height:1.65;
}

.mf-warn {
    background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.5);
    border-radius:10px;padding:13px 18px;font-size:13px;color:#f59e0b;
}
</style>
""")

# ── Sidebar ───────────────────────────────────────────────────────────────
render_sidebar(current_page="Explainability") 
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

# ── Load data ─────────────────────────────────────────────────────────────
prep = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30).json()
problem_type = prep["problem_type"]

try:
    model_info = requests.get(f"{API_BASE}/train/latest/{project_id}", timeout=30).json()
except Exception:
    st.html("""
    <div style="background:rgba(6,14,29,.8);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:20px 24px;margin-top:20px;">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No trained model found</div>
        <div style="font-size:13px;color:#8890aa;">Train a model first to view explainability.</div>
    </div>
    """)
    st.stop()

model_id = model_info.get("id")
if not model_id:
    st.html("""
    <div style="background:rgba(6,14,29,.8);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:20px 24px;margin-top:20px;">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No trained model yet</div>
        <div style="font-size:13px;color:#8890aa;">Train a model to generate explanations.</div>
    </div>
    """)
    st.stop()

try:
    exp_resp    = requests.get(f"{API_BASE}/explain/{model_id}", timeout=60)
    explanation = exp_resp.json() if exp_resp.status_code == 200 else None
except Exception:
    explanation = None

# ── Helper ────────────────────────────────────────────────────────────────
def flatten_importance(raw: dict) -> dict:
    out = {}
    for feat, val in raw.items():
        if isinstance(val, (list, tuple)):
            out[feat] = float(np.mean(np.abs(val)))
        else:
            try:    out[feat] = float(val)
            except: out[feat] = 0.0
    return out

def feat_bar_html(data: dict, color: str, color_class: str, max_rows: int = 15) -> str:
    """Renders a custom horizontal bar chart as HTML — no matplotlib needed."""
    if not data:
        return ""
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:max_rows]
    max_val      = max(v for _, v in sorted_items) or 1
    rows = ""
    for feat, val in sorted_items:
        pct  = (val / max_val) * 100
        rows += f"""
        <div class="mf-feat-row">
            <span class="mf-feat-name" title="{feat}">{feat}</span>
            <div class="mf-feat-track">
                <div class="mf-feat-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <span class="mf-feat-val">{val:.4f}</span>
        </div>"""
    return f'<div class="mf-feat-bar-wrap {color_class}">{rows}</div>'

# ══════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════
algo = model_info.get("algorithm","Unknown")
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
            MODELLING &#183; EXPLAINABILITY
        </span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.15;margin-bottom:14px;color:#fff;">
        Model Explainability
    </div>
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <p style="font-size:15px;color:#8890aa;margin:0;font-weight:400;line-height:1.7;">
            Understand why your model makes predictions using SHAP and LIME.
        </p>
        <span style="background:{pt_bg};border:1px solid {pt_bdr};border-radius:20px;
                     padding:4px 13px;font-size:12px;font-weight:600;color:{pt_color};">{problem_type}</span>
        <span style="background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.08);border-radius:20px;
                     padding:4px 13px;font-size:12px;font-weight:600;color:#c8ccee;">{algo}</span>
    </div>
</div>
""")

# ══════════════════════════════════════════════════════════
# TWO-COLUMN: Summary left, charts right — but stacked for explainability
# ══════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.6], gap="large")

# ── LEFT: Model summary ────────────────────────────────────────────────────
with col_left:
    st.html("""
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Model Summary</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """)

    summary_rows = [
        ("Algorithm",       str(model_info.get("algorithm", "N/A"))),
        ("Problem Type",    str(problem_type)),
        ("Target Column",   str(prep["target"])),
        ("Encoding",        str(prep.get("encoding") or "None")),
        ("Scaling",         str(prep.get("scaling") or "None")),
        ("Test Size",       f"{int(prep['test_size'] * 100)}%"),
        ("No. of Features", str(len(prep["features"]))),
    ]
    rows_html = "".join([
        f'<div class="mf-prep-row">'
        f'<span class="mf-prep-key">{k}</span>'
        f'<span class="mf-prep-val">{v}</span>'
        f'</div>'
        for k, v in summary_rows
    ])
    st.html(f"""
    <div style="background:rgba(12,14,22,.85);border:1px solid rgba(255,255,255,.07);
                border-radius:12px;overflow:hidden;">
        <div style="height:2px;background:linear-gradient(90deg,transparent,rgba(139,92,246,.4),transparent);"></div>
        {rows_html}
    </div>
    """)

    # Features
    features = prep.get("features", [])
    if features:
        st.html("""
        <div class="mf-sec" style="margin-top:20px;">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Features Used</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """)
        pills = "".join([
            f'<span style="display:inline-flex;background:rgba(12,14,22,.9);'
            f'border:1px solid rgba(255,255,255,.1);border-radius:20px;'
            f'padding:4px 12px;font-size:12px;color:#8890aa;margin:3px 3px 3px 0;">{f}</span>'
            for f in features
        ])
        st.html(f'<div style="line-height:2.4;">{pills}</div>')

# ── RIGHT: Quick SHAP overview ─────────────────────────────────────────────
with col_right:
    if not explanation:
        st.html("""
        <div class="mf-warn" style="margin-top:36px;">
            Explainability data not available for this model.<br>
            <span style="font-size:12px;color:#78350f;">
                Retrain the model to generate SHAP and LIME explanations automatically.
            </span>
        </div>
        """)
        st.page_link("pages/6_Train_Model.py", label="Go to Train Model")
    else:
        global_imp = explanation.get("global_importance")
        if global_imp:
            flat_imp = flatten_importance(global_imp)
            top5     = dict(sorted(flat_imp.items(), key=lambda x: x[1], reverse=True)[:5])

            st.html("""
            <div class="mf-sec" style="margin-top:4px;">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Top 5 Features (SHAP)</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            st.html(feat_bar_html(top5, "#4f8ef7", "blue", max_rows=5))


# ══════════════════════════════════════════════════════════
# FULL-WIDTH SECTIONS BELOW
# ══════════════════════════════════════════════════════════
if not explanation:
    st.stop()

st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:28px 0 24px;"></div>')

# ══════════════════════════════════════════════════════════
# GLOBAL SHAP — full width
# ══════════════════════════════════════════════════════════
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Global Feature Importance (SHAP)</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")
st.html("""
<div class="mf-info-card" style="margin-bottom:18px;">
    <span style="color:#7eb3f7;font-weight:600;">Global SHAP</span> shows the average impact
    of each feature across all predictions. Higher values mean the feature has a stronger
    influence on the model's output overall.
</div>
""")

global_imp = explanation.get("global_importance")

if not global_imp:
    st.html('<div class="mf-info-card">Global SHAP importance not available. Retrain to generate it.</div>')
else:
    flat_imp  = flatten_importance(global_imp)
    global_df = (
        pd.DataFrame.from_dict(flat_imp, orient="index", columns=["Importance"])
        .sort_values("Importance", ascending=False)
    )

    top_k = st.slider(
        "Show Top K Features",
        min_value=1,
        max_value=min(30, len(global_df)),
        value=min(10, len(global_df)),
        key="shap_topk",
    )

    top_data = dict(global_df.head(top_k)["Importance"])

    g_left, g_right = st.columns([1.2, 1], gap="large")

    with g_left:
        # Custom horizontal bar chart
        st.html(feat_bar_html(top_data, "#4f8ef7", "blue", max_rows=top_k))

    with g_right:
        # Matplotlib bar chart — styled
        fig, ax = plt.subplots(figsize=(5, max(3, top_k * 0.4)))
        features_list = list(top_data.keys())
        values_list   = list(top_data.values())
        colors = [
            "#4f8ef7" if v == max(values_list) else
            "#8b5cf6" if v > np.median(values_list) else
            "#525870"
            for v in values_list
        ]
        bars = ax.barh(features_list[::-1], values_list[::-1], color=colors[::-1],
                       height=0.6, edgecolor="none")
        ax.set_xlabel("SHAP Importance", fontsize=10, color="#7a82a6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close(fig)

    # Full table in expander
    with st.expander("View full SHAP importance table"):
        st.dataframe(global_df.round(6), use_container_width=True)


st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:28px 0 24px;"></div>')

# ══════════════════════════════════════════════════════════
# LOCAL SHAP + LIME — side by side
# ══════════════════════════════════════════════════════════
local = explanation.get("local_explanation")
has_shap = local and "shap" in local
has_lime = local and "lime" in local

if has_shap or has_lime:
    shap_col, lime_col = st.columns(2, gap="large")

    # ── Local SHAP ────────────────────────────────────────────────────────
    with shap_col:
        st.html("""
        <div class="mf-sec">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Local Explanation (SHAP)</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """)
        st.html("""
        <div class="mf-info-card" style="margin-bottom:14px;">
            <span style="color:#8b5cf6;font-weight:600;">Local SHAP</span> shows how each
            feature pushed the prediction higher or lower for a single representative sample.
        </div>
        """)

        if has_shap:
            flat_local = flatten_importance(local["shap"])
            shap_sorted = dict(sorted(flat_local.items(), key=lambda x: x[1], reverse=True))
            st.html(feat_bar_html(shap_sorted, "#8b5cf6", "purple"))

            base_val = local.get("base_value")
            if base_val is not None:
                st.html(f"""
                <div style="margin-top:10px;padding:8px 14px;background:rgba(12,14,22,.7);
                            border:1px solid rgba(255,255,255,.06);border-radius:8px;
                            font-size:12px;color:#7a82a6;">
                    Base value: <span style="color:#c8ccee;font-weight:600;">{round(float(base_val),6)}</span>
                </div>
                """)
        else:
            st.html('<div class="mf-info-card">Local SHAP not available. Retrain to generate it.</div>')

    # ── LIME ──────────────────────────────────────────────────────────────
    with lime_col:
        st.html("""
        <div class="mf-sec">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Local Explanation (LIME)</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """)
        st.html("""
        <div class="mf-info-card" style="margin-bottom:14px;">
            <span style="color:#10b981;font-weight:600;">LIME</span> approximates the model
            locally with a simpler interpretable model to explain individual predictions.
        </div>
        """)

        if has_lime:
            flat_lime  = flatten_importance(local["lime"])
            lime_sorted = dict(sorted(flat_lime.items(), key=lambda x: x[1], reverse=True))
            st.html(feat_bar_html(lime_sorted, "#10b981", "green"))

            with st.expander("View LIME table"):
                lime_df = pd.DataFrame.from_dict(lime_sorted, orient="index", columns=["Contribution"])
                st.dataframe(lime_df.round(6), use_container_width=True)
        else:
            st.html('<div class="mf-info-card">LIME not available. Retrain to generate it.</div>')

else:
    st.html("""
    <div class="mf-info-card">
        <span style="color:#f59e0b;font-weight:600;">Local explanations not available.</span><br>
        <span style="font-size:12px;color:#525870;">
            Retrain your model to generate SHAP local and LIME explanations automatically.
        </span>
    </div>
    """)
