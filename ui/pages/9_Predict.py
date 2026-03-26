# pages/9_Predict.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ModelForge • Predict", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)
API_BASE = "http://127.0.0.1:8000"

# ── Page styles ────────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

.mf-sec { display:flex;align-items:center;gap:10px;margin-bottom:16px; }
.mf-sec-line { height:1px;background:rgba(255,255,255,.1); }
.mf-sec-label {
    font-size:11px;font-weight:700;color:#7a82a6;
    letter-spacing:1.4px;text-transform:uppercase;white-space:nowrap;
}
.mf-subheading {
    font-family:'Syne',sans-serif;
    font-size:20px;font-weight:800;color:#e0e4ff;
    letter-spacing:-.4px;margin:0 0 4px;
}
.mf-detail-card {
    background:rgba(12,14,22,.85);border:1px solid rgba(255,255,255,.07);
    border-radius:12px;overflow:hidden;
}
.mf-detail-row {
    display:flex;align-items:center;padding:11px 18px;
    border-bottom:1px solid rgba(255,255,255,.04);
}
.mf-detail-row:last-child { border-bottom:none; }
.mf-detail-key { font-size:12px;color:#7a82a6;font-weight:500;width:160px;flex-shrink:0; }
.mf-detail-val { font-size:13px;color:#d0d4f0;font-weight:500; }

.mf-metric {
    background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.07);
    border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;
    backdrop-filter:blur(8px);transition:all .22s ease;text-align:center;
}
.mf-metric:hover { border-color:rgba(255,255,255,.13);transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,0,0,.5); }
.mf-metric-label { font-size:11px;font-weight:700;color:#7a82a6;letter-spacing:.8px;text-transform:uppercase;margin-bottom:10px; }
.mf-metric-value {
    font-family:'Syne',sans-serif;
    font-size:clamp(20px,3vw,30px);font-weight:800;letter-spacing:-.5px;
    line-height:1.1;white-space:nowrap;overflow:visible;
}

.mf-warn  { background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.5);border-radius:10px;padding:13px 18px;font-size:13px;color:#f59e0b; }
.mf-info  { background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.4);border-radius:10px;padding:13px 18px;font-size:13px;color:#8890aa;line-height:1.65; }
.mf-success { background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.5);border-radius:10px;padding:13px 18px;font-size:13px;color:#10b981; }

/* Mode tab switcher */
.pred-tab-wrap { display:flex;gap:0;background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:4px;width:fit-content;margin-bottom:24px; }
.pred-tab-sel > button {
    background:linear-gradient(135deg,rgba(13,25,41,.95),rgba(13,13,40,.95)) !important;
    border:1px solid rgba(79,142,247,.5) !important;color:#fff !important;
    border-radius:7px !important;padding:9px 22px !important;font-size:13px !important;
    font-weight:600 !important;white-space:nowrap !important;
    box-shadow:0 0 14px rgba(79,142,247,.15) !important;
    min-width:180px !important;
}
.pred-tab-unsel > button {
    background:transparent !important;border:1px solid transparent !important;
    color:#7a82a6 !important;border-radius:7px !important;padding:9px 22px !important;
    font-size:13px !important;font-weight:500 !important;white-space:nowrap !important;
    min-width:180px !important;
}
.pred-tab-unsel > button:hover { background:rgba(255,255,255,.04) !important;color:#c8ccee !important; }
</style>
""")

# ── Sidebar ────────────────────────────────────────────────────────────────
render_sidebar(current_page="Predict") 
# ── Auth ───────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Page header ────────────────────────────────────────────────────────────
st.html("""
<div style="padding:40px 0 28px;border-bottom:1px solid rgba(255,255,255,.07);
            margin-bottom:28px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-20px;right:0;width:240px;height:240px;border-radius:50%;
                background:radial-gradient(circle,rgba(16,185,129,.05),transparent 70%);pointer-events:none;"></div>
    <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.05);
                border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:5px 14px;margin-bottom:18px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#10b981;"></div>
        <span style="font-size:11px;color:#8890aa;font-weight:600;letter-spacing:.7px;">
            MODELLING &#183; PREDICT
        </span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.15;margin-bottom:14px;color:#fff;">
        Prediction &amp; Inference
    </div>
    <p style="font-size:15px;color:#8890aa;margin:0;font-weight:400;line-height:1.7;">
        Run single-row predictions via form input, or upload a CSV for batch inference.
    </p>
</div>
""")

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

# ── Load prep config ───────────────────────────────────────────────────────
try:
    prep = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=30).json()
except Exception:
    st.html('<div class="mf-warn">Could not load data preparation config.</div>')
    st.stop()

target       = prep.get("target", "")
problem_type = prep.get("problem_type", "Classification")
pt_color = "#7eb3f7" if problem_type == "Classification" else "#a78bfa"
pt_bg    = "rgba(6,14,29,.8)"   if problem_type == "Classification" else "rgba(13,8,25,.8)"
pt_bdr   = "rgba(29,58,110,.5)" if problem_type == "Classification" else "rgba(76,29,149,.5)"

# ── Load experiments ───────────────────────────────────────────────────────
try:
    experiments = requests.get(f"{API_BASE}/experiments/project/{project_id}", timeout=30).json()
except Exception:
    experiments = []

if not experiments:
    st.html("""
    <div style="background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:28px 24px;text-align:center;margin-top:8px;">
        <div style="font-size:32px;margin-bottom:12px;">🤖</div>
        <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No trained models found</div>
        <div style="font-size:13px;color:#8890aa;">Train a model first before running predictions.</div>
    </div>
    """)
    st.stop()

# ══════════════════════════════════════════════════════════
# MODEL SELECTOR
# ══════════════════════════════════════════════════════════
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Select Model</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
<div class="mf-subheading" style="margin-bottom:16px;">Choose a Trained Model</div>
""")

exp_labels = [
    f"{e['name']} — {e['algorithm']} (trained {e['created_at'][:10]})"
    for e in experiments
]
selected_label = st.selectbox("Model", exp_labels, label_visibility="collapsed")
selected_exp   = experiments[exp_labels.index(selected_label)]
model_id       = selected_exp.get("model_id")

if not model_id:
    st.html('<div class="mf-warn">Selected experiment has no associated model file.</div>')
    st.stop()

# ── Model info strip ───────────────────────────────────────────────────────
try:
    model_info_resp  = requests.get(f"{API_BASE}/predict/model-info/{model_id}", timeout=30)
    model_info       = model_info_resp.json() if model_info_resp.status_code == 200 else {}
    trained_features = model_info.get("trained_features", [])
except Exception:
    trained_features = []

if not trained_features:
    trained_features = prep.get("features", [])

info_items = [
    ("Algorithm",  selected_exp["algorithm"],        "#4f8ef7"),
    ("Tuning",     selected_exp["tuning_method"],     "#8b5cf6"),
    ("Target",     target,                            "#10b981"),
    ("Features",   str(len(trained_features)),        "#ec4899"),
    ("Problem",    problem_type,                      pt_color),
]
info_cells = "".join([
    f'<div style="display:flex;flex-direction:column;gap:4px;padding:14px 22px;'
    f'border-right:1px solid rgba(255,255,255,.05);flex-shrink:0;">'
    f'<span style="font-size:10px;font-weight:700;color:#7a82a6;letter-spacing:.8px;text-transform:uppercase;">{k}</span>'
    f'<span style="font-size:13px;font-weight:600;color:{c};white-space:nowrap;">{v}</span>'
    f'</div>'
    for k, v, c in info_items
])
st.html(f"""
<div style="background:rgba(12,14,22,.85);border:1px solid rgba(255,255,255,.07);
            border-radius:12px;overflow:hidden;margin:8px 0 28px;">
    <div style="height:2px;background:linear-gradient(90deg,transparent,rgba(16,185,129,.4),transparent);"></div>
    <div style="display:flex;align-items:stretch;flex-wrap:wrap;">
        {info_cells}
    </div>
</div>
""")

# ══════════════════════════════════════════════════════════
# PREDICTION MODE — tab switcher
# ══════════════════════════════════════════════════════════
st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:0 0 24px;"></div>')
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Prediction Mode</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")

if "pred_mode" not in st.session_state:
    st.session_state["pred_mode"] = "single"

tab_col1, tab_col2, _spacer = st.columns([1, 1, 4], gap="small")
with tab_col1:
    st.html(f'<div class="{"pred-tab-sel" if st.session_state["pred_mode"]=="single" else "pred-tab-unsel"}">')
    if st.button("✏️ Single Row (Form)", key="tab_single", use_container_width=True):
        st.session_state["pred_mode"] = "single"
        st.rerun()
    st.html('</div>')

with tab_col2:
    st.html(f'<div class="{"pred-tab-sel" if st.session_state["pred_mode"]=="batch" else "pred-tab-unsel"}">')
    if st.button("📁 Batch (CSV Upload)", key="tab_batch", use_container_width=True):
        st.session_state["pred_mode"] = "batch"
        st.rerun()
    st.html('</div>')

pred_mode = st.session_state["pred_mode"]

# ══════════════════════════════════════════════════════════
# MODE 1: SINGLE ROW FORM
# ══════════════════════════════════════════════════════════
if pred_mode == "single":
    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:8px 0 24px;"></div>')
    st.html("""
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Enter Feature Values</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    <div class="mf-subheading" style="margin-bottom:6px;">Input Features</div>
    <div style="font-size:13px;color:#8890aa;margin-bottom:20px;">
        Fill in values for each feature. Defaults are pulled from the most recent dataset row.
    </div>
    """)

    # Load sample + dtypes
    sample_values = {}
    try:
        preview = requests.get(
            f"{API_BASE}/dataset/{dataset_id}/preview", params={"n": 1}, timeout=10
        ).json()
        if preview.get("rows"):
            sample_values = preview["rows"][0]
    except Exception:
        pass

    dtypes = {}
    try:
        stats  = requests.get(f"{API_BASE}/dataset/{dataset_id}/stats", timeout=10).json()
        dtypes = stats.get("dtypes", {})
    except Exception:
        pass

    # Feature inputs — 3 columns for better use of full width
    input_values = {}
    num_features = len(trained_features)
    cols_per_row = 3
    for row_start in range(0, num_features, cols_per_row):
        row_features = trained_features[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="medium")
        for col, feature in zip(cols, row_features):
            dtype   = dtypes.get(feature, "object")
            default = sample_values.get(feature, "")
            with col:
                if dtype.startswith(("int", "float")):
                    try:
                        default_num = float(default) if default != "" else 0.0
                    except (ValueError, TypeError):
                        default_num = 0.0
                    input_values[feature] = st.number_input(
                        feature, value=default_num, key=f"input_{feature}"
                    )
                else:
                    input_values[feature] = st.text_input(
                        feature,
                        value=str(default) if default else "",
                        key=f"input_{feature}",
                    )

    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:22px 0 18px;"></div>')

    _pred_btn, _ = st.columns([1, 3])
    with _pred_btn:
        predict_clicked = st.button("Launch Prediction", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Running prediction..."):
            resp = requests.post(
                f"{API_BASE}/predict/single",
                json={"model_id": model_id, "features": input_values},
                timeout=30,
            )

        if resp.status_code == 200:
            result = resp.json()

            st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:24px 0 20px;"></div>')
            st.html("""
            <div class="mf-sec">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Prediction Result</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            <div style="background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.5);
                        border-radius:10px;padding:12px 18px;display:flex;align-items:center;
                        gap:10px;margin-bottom:20px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#10b981;
                            box-shadow:0 0 8px #10b981;flex-shrink:0;"></div>
                <span style="font-size:13px;font-weight:600;color:#10b981;">Prediction complete</span>
            </div>
            """)

            res_cols = st.columns(3 if "confidence" in result else 2, gap="small")
            with res_cols[0]:
                st.html(f"""
                <div class="mf-metric">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                                background:linear-gradient(90deg,transparent,#10b98155,transparent);
                                border-radius:12px 12px 0 0;"></div>
                    <div class="mf-metric-label">Predicted {target}</div>
                    <div class="mf-metric-value" style="color:#10b981;">{result['prediction']}</div>
                </div>
                """)
            if "confidence" in result:
                with res_cols[1]:
                    conf_pct = f"{result['confidence']*100:.1f}%"
                    st.html(f"""
                    <div class="mf-metric">
                        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                                    background:linear-gradient(90deg,transparent,#4f8ef755,transparent);
                                    border-radius:12px 12px 0 0;"></div>
                        <div class="mf-metric-label">Confidence</div>
                        <div class="mf-metric-value" style="color:#4f8ef7;">{conf_pct}</div>
                    </div>
                    """)

            if "probabilities" in result:
                st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
                st.html("""
                <div class="mf-sec">
                    <div class="mf-sec-line" style="width:16px;"></div>
                    <span class="mf-sec-label">Class Probabilities</span>
                    <div class="mf-sec-line" style="flex:1;"></div>
                </div>
                """)
                PROB_COLORS = ["#4f8ef7","#8b5cf6","#10b981","#ec4899","#f59e0b"]
                prob_items = list(result["probabilities"].items())
                prob_cols  = st.columns(len(prob_items), gap="small")
                for i, (cls, prob) in enumerate(prob_items):
                    c = PROB_COLORS[i % len(PROB_COLORS)]
                    with prob_cols[i]:
                        st.html(f"""
                        <div class="mf-metric">
                            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                                        background:linear-gradient(90deg,transparent,{c}55,transparent);
                                        border-radius:12px 12px 0 0;"></div>
                            <div class="mf-metric-label">Class {cls}</div>
                            <div class="mf-metric-value" style="color:{c};">{prob:.3f}</div>
                        </div>
                        """)
        else:
            detail = resp.json().get("detail", resp.text)
            st.html(f'<div class="mf-warn">Prediction failed: {detail}</div>')

# ══════════════════════════════════════════════════════════
# MODE 2: BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════
else:
    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:8px 0 24px;"></div>')
    st.html("""
    <div class="mf-sec">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Batch Inference</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    <div class="mf-subheading" style="margin-bottom:6px;">Upload CSV for Batch Predictions</div>
    <div style="font-size:13px;color:#8890aa;margin-bottom:20px;">
        Upload a CSV containing the required feature columns. The target column is not needed — it will be predicted.
    </div>
    """)

    # Required columns info card
    feature_pills = "".join([
        f'<span style="display:inline-flex;background:rgba(255,255,255,.04);'
        f'border:1px solid rgba(255,255,255,.09);border-radius:20px;'
        f'padding:3px 10px;font-size:11px;color:#8890aa;margin:2px 2px 2px 0;white-space:nowrap;">{f}</span>'
        for f in trained_features
    ])
    st.html(f"""
    <div style="background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.35);
                border-radius:12px;padding:16px 20px;margin-bottom:20px;">
        <div style="font-size:11px;font-weight:700;color:#7a82a6;letter-spacing:.8px;
                    text-transform:uppercase;margin-bottom:10px;">
            Required Columns ({len(trained_features)})
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:2px;">{feature_pills}</div>
    </div>
    """)

    # Template download
    template_df  = pd.DataFrame(columns=trained_features)
    csv_template = template_df.to_csv(index=False)
    _dl_col, _ = st.columns([1, 3])
    with _dl_col:
        st.download_button(
            "Download CSV Template",
            csv_template,
            "prediction_template.csv",
            "text/csv",
            use_container_width=True,
        )

    st.html('<div style="margin:16px 0 4px;"></div>')
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        preview_df = pd.read_csv(uploaded)

        st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
        st.html(f"""
        <div class="mf-sec">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">File Preview</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        <div style="font-size:13px;color:#8890aa;margin-bottom:12px;">
            {len(preview_df)} rows detected — showing first 10
        </div>
        """)
        st.dataframe(preview_df.head(10), use_container_width=True)

        missing_cols = [f for f in trained_features if f not in preview_df.columns]
        if missing_cols:
            missing_str = ", ".join(missing_cols)
            st.html(f'<div class="mf-warn" style="margin-top:12px;">Missing required columns: {missing_str}</div>')
        else:
            st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
            _run_col, _ = st.columns([1, 3])
            with _run_col:
                run_batch = st.button(
                    f"Run Batch Predictions ({len(preview_df)} rows)",
                    type="primary",
                    use_container_width=True,
                )

            if run_batch:
                with st.spinner(f"Running predictions on {len(preview_df)} rows..."):
                    uploaded.seek(0)
                    resp = requests.post(
                        f"{API_BASE}/predict/batch",
                        params={"model_id": model_id},
                        files={"file": ("data.csv", uploaded, "text/csv")},
                        timeout=120,
                    )

                if resp.status_code == 200:
                    result    = resp.json()
                    result_df = pd.DataFrame(result["predictions"])

                    st.html(f"""
                    <div class="mf-success" style="margin:16px 0;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <div style="width:8px;height:8px;border-radius:50%;background:#10b981;
                                        box-shadow:0 0 8px #10b981;flex-shrink:0;"></div>
                            <span style="font-weight:600;">Predictions complete for {result['total_rows']} rows</span>
                        </div>
                    </div>
                    """)

                    pred_col = "prediction"
                    if pred_col in result_df.columns:
                        st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
                        st.html("""
                        <div class="mf-sec">
                            <div class="mf-sec-line" style="width:16px;"></div>
                            <span class="mf-sec-label">Prediction Summary</span>
                            <div class="mf-sec-line" style="flex:1;"></div>
                        </div>
                        """)
                        _dist_col, _chart_col = st.columns(2, gap="large")
                        with _dist_col:
                            st.html('<div style="font-size:12px;color:#7a82a6;font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px;">Distribution</div>')
                            st.dataframe(
                                result_df[pred_col].value_counts().reset_index().astype(str),
                                use_container_width=True,
                            )
                        with _chart_col:
                            if problem_type == "Classification":
                                st.html('<div style="font-size:12px;color:#7a82a6;font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px;">Chart</div>')
                                st.bar_chart(result_df[pred_col].value_counts())

                    st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
                    st.html("""
                    <div class="mf-sec">
                        <div class="mf-sec-line" style="width:16px;"></div>
                        <span class="mf-sec-label">Full Results</span>
                        <div class="mf-sec-line" style="flex:1;"></div>
                    </div>
                    """)
                    st.dataframe(result_df, use_container_width=True)

                    _csv_col, _ = st.columns([1, 3])
                    with _csv_col:
                        st.download_button(
                            "Download Predictions CSV",
                            result_df.to_csv(index=False),
                            "predictions.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.html(f'<div class="mf-warn">Batch prediction failed: {detail}</div>')
