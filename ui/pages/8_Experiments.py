# pages/8_Experiments.py
import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="ModelForge • Experiments", layout="wide")
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
    font-size:clamp(20px,3vw,30px);
    font-weight:800;letter-spacing:-.5px;line-height:1.1;
    white-space:nowrap;overflow:visible;
}

.mf-exp-row {
    display:flex;align-items:center;padding:12px 18px;
    border-bottom:1px solid rgba(255,255,255,.04);transition:background .15s;gap:12px;
}
.mf-exp-row:last-child { border-bottom:none; }
.mf-exp-row:hover { background:rgba(255,255,255,.02); }

.mf-tag {
    display:inline-flex;align-items:center;
    background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
    border-radius:20px;padding:3px 10px;font-size:11px;color:#8890aa;
    white-space:nowrap;
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

.mf-warn {
    background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.5);
    border-radius:10px;padding:13px 18px;font-size:13px;color:#f59e0b;
}
.mf-info {
    background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.4);
    border-radius:10px;padding:13px 18px;font-size:13px;color:#8890aa;
}
</style>
""")

# ── Sidebar ────────────────────────────────────────────────────────────────
render_sidebar(current_page="Experiments") 
# ── Auth ───────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Page header ────────────────────────────────────────────────────────────
st.html("""
<div style="padding:40px 0 28px;border-bottom:1px solid rgba(255,255,255,.07);
            margin-bottom:28px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-20px;right:0;width:240px;height:240px;border-radius:50%;
                background:radial-gradient(circle,rgba(79,142,247,.05),transparent 70%);pointer-events:none;"></div>
    <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.05);
                border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:5px 14px;margin-bottom:18px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#4f8ef7;"></div>
        <span style="font-size:11px;color:#8890aa;font-weight:600;letter-spacing:.7px;">
            MODELLING &#183; EXPERIMENTS
        </span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.15;margin-bottom:14px;color:#fff;">
        Experiment Tracking
    </div>
    <p style="font-size:15px;color:#8890aa;margin:0;font-weight:400;line-height:1.7;">
        Compare training runs, inspect configurations, and track model performance over time.
    </p>
</div>
""")

if not st.session_state.get("project_id"):
    st.html("""
    <div style="background:rgba(6,14,29,.8);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:20px 24px;margin-top:20px;">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No project selected</div>
        <div style="font-size:13px;color:#8890aa;">Select a project first to view experiments.</div>
    </div>
    """)
    st.stop()

project_id = st.session_state["project_id"]
dataset_id = st.session_state.get("dataset_id")

# ── Load prep config ───────────────────────────────────────────────────────
problem_type = "Classification"
if dataset_id:
    try:
        prep = requests.get(f"{API_BASE}/dataset-preparation/{dataset_id}", timeout=10).json()
        problem_type = prep.get("problem_type", "Classification")
    except Exception:
        pass

# ── Load experiments ───────────────────────────────────────────────────────
try:
    resp = requests.get(f"{API_BASE}/experiments/project/{project_id}", timeout=30)
    experiments = resp.json() if resp.status_code == 200 else []
except Exception:
    experiments = []

if not experiments:
    st.html("""
    <div style="background:rgba(6,14,29,.7);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:28px 24px;text-align:center;margin-top:8px;">
        <div style="font-size:32px;margin-bottom:12px;">🧪</div>
        <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No experiments yet</div>
        <div style="font-size:13px;color:#8890aa;">Train a model to start tracking experiments.</div>
    </div>
    """)
    st.stop()

# ══════════════════════════════════════════════════════════
# SUMMARY STATS STRIP
# ══════════════════════════════════════════════════════════
pt_color = "#7eb3f7" if problem_type == "Classification" else "#a78bfa"
pt_bg    = "rgba(6,14,29,.8)"   if problem_type == "Classification" else "rgba(13,8,25,.8)"
pt_bdr   = "rgba(29,58,110,.5)" if problem_type == "Classification" else "rgba(76,29,149,.5)"

algos     = list({e["algorithm"] for e in experiments})
tunings   = list({e["tuning_method"] for e in experiments})
all_times = [e.get("training_time") for e in experiments if e.get("training_time")]
avg_time  = f"{round(sum(all_times)/len(all_times), 1)}s" if all_times else "—"

s1, s2, s3, s4 = st.columns(4, gap="small")
for col, (label, val, color) in zip([s1,s2,s3,s4], [
    ("Total Runs",       str(len(experiments)), "#4f8ef7"),
    ("Algorithms Tried", str(len(algos)),       "#8b5cf6"),
    ("Tuning Methods",   str(len(tunings)),     "#10b981"),
    ("Avg Train Time",   avg_time,              "#ec4899"),
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

st.html('<div style="margin-bottom:28px;"></div>')

# ══════════════════════════════════════════════════════════
# ALL EXPERIMENTS TABLE
# ══════════════════════════════════════════════════════════
st.html(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
    <div>
        <div class="mf-subheading">All Experiments</div>
        <div style="font-size:13px;color:#8890aa;margin-top:3px;">
            {len(experiments)} training run{"s" if len(experiments)!=1 else ""} recorded
            &nbsp;&#183;&nbsp;
            <span style="background:{pt_bg};border:1px solid {pt_bdr};border-radius:20px;
                         padding:2px 10px;font-size:11px;font-weight:600;color:{pt_color};">
                {problem_type}
            </span>
        </div>
    </div>
</div>
""")

rows = []
for e in experiments:
    m = e.get("metrics", {})
    row = {
        "ID":             e["id"],
        "Name":           e["name"],
        "Algorithm":      e["algorithm"],
        "Tuning":         e["tuning_method"],
        "CV Folds":       e["cv_folds"],
        "Train Time (s)": e.get("training_time") or "—",
        "Date":           e["created_at"][:16],
    }
    if problem_type == "Classification":
        row["Accuracy"]      = round(m.get("accuracy", 0), 4)
        row["F1 (weighted)"] = round(m.get("weighted avg", {}).get("f1-score", 0), 4)
        row["AUC"]           = round(m.get("roc_auc", 0), 4)
    else:
        row["R²"]   = round(m.get("r2",   0), 4)
        row["RMSE"] = round(m.get("rmse", 0), 4)
        row["MAE"]  = round(m.get("mae",  0), 4)
    rows.append(row)

exp_df = pd.DataFrame(rows).astype(str)
st.dataframe(exp_df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════
st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:32px 0 24px;"></div>')
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Model Comparison</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")
st.html('<div class="mf-subheading" style="margin-bottom:6px;">Compare Experiments</div>')
st.html('<div style="font-size:13px;color:#8890aa;margin-bottom:16px;">Select two or more runs to compare side by side.</div>')

exp_names = [f"{e['name']} — {e['algorithm']}" for e in experiments]
exp_ids   = [e["id"] for e in experiments]

selected_labels = st.multiselect(
    "Experiments",
    options=exp_names,
    default=exp_names[:min(3, len(exp_names))],
    label_visibility="collapsed",
)

if len(selected_labels) >= 2:
    selected_ids = [exp_ids[exp_names.index(lbl)] for lbl in selected_labels]
    ids_param    = ",".join(str(i) for i in selected_ids)

    try:
        cmp_resp = requests.get(
            f"{API_BASE}/experiments/compare/{project_id}",
            params={"ids": ids_param},
            timeout=30,
        )
        if cmp_resp.status_code == 200:
            cmp_data   = cmp_resp.json()
            comparison = cmp_data["comparison"]

            st.html('<div style="margin-top:12px;"></div>')
            st.dataframe(pd.DataFrame(comparison).astype(str), use_container_width=True)

            # ── Visual bar charts ──────────────────────────────────────────
            st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:24px 0 20px;"></div>')
            st.html("""
            <div class="mf-sec">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Visual Comparison</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            st.html('<div class="mf-subheading" style="margin-bottom:16px;">Metric Charts</div>')

            metrics_to_plot = (
                ["accuracy", "f1_weighted", "roc_auc"]
                if problem_type == "Classification"
                else ["r2", "rmse", "mae"]
            )
            available_metrics = [m for m in metrics_to_plot if m in comparison[0]]

            chart_cols = st.columns(len(available_metrics), gap="medium")
            CHART_COLORS = ["#4f8ef7","#8b5cf6","#10b981","#ec4899","#f59e0b"]

            for idx, (metric, col) in enumerate(zip(available_metrics, chart_cols)):
                with col:
                    names  = [c["name"] for c in comparison]
                    values = [float(c.get(metric, 0)) for c in comparison]
                    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(names))]

                    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#0c0e16")
                    ax.set_facecolor("#0c0e16")
                    bars = ax.barh(names, values, color=colors, height=0.5)
                    ax.bar_label(bars, fmt="%.4f", padding=4, color="#8890aa", fontsize=9)
                    ax.set_xlabel(metric.upper(), color="#525870", fontsize=9)
                    ax.set_title(metric.upper(), color="#c8ccee", fontsize=11, fontweight="bold", pad=10)
                    ax.set_xlim(0, max(values) * 1.25 if values else 1)
                    ax.tick_params(colors="#525870", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#1f2235")
                    ax.set_axisbelow(True)
                    ax.xaxis.grid(True, color="#1a1d2e", linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        else:
            st.html('<div class="mf-warn">Could not load comparison data from server.</div>')
    except Exception as e:
        st.html(f'<div class="mf-warn">Comparison failed: {e}</div>')

elif len(selected_labels) == 1:
    st.html('<div class="mf-info" style="margin-top:10px;">Select at least 2 experiments to compare.</div>')

# ══════════════════════════════════════════════════════════
# EXPERIMENT DETAIL
# ══════════════════════════════════════════════════════════
st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:32px 0 24px;"></div>')
st.html("""
<div class="mf-sec">
    <div class="mf-sec-line" style="width:16px;"></div>
    <span class="mf-sec-label">Experiment Detail</span>
    <div class="mf-sec-line" style="flex:1;"></div>
</div>
""")
st.html('<div class="mf-subheading" style="margin-bottom:6px;">Inspect a Run</div>')
st.html('<div style="font-size:13px;color:#8890aa;margin-bottom:16px;">Drill into configuration, hyperparameters, and per-fold CV scores.</div>')

selected_detail = st.selectbox("Experiment", options=exp_names, label_visibility="collapsed")

if selected_detail:
    detail_id  = exp_ids[exp_names.index(selected_detail)]
    detail_exp = next((e for e in experiments if e["id"] == detail_id), None)

    if detail_exp:
        cfg_col, param_col = st.columns(2, gap="large")

        # ── Config card ───────────────────────────────────────────────────
        with cfg_col:
            st.html("""
            <div class="mf-sec" style="margin-top:8px;">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Configuration</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            cfg_items = [
                ("Algorithm",    detail_exp["algorithm"]),
                ("Problem Type", detail_exp.get("problem_type", "—")),
                ("Tuning",       detail_exp["tuning_method"]),
                ("CV Folds",     str(detail_exp["cv_folds"])),
                ("Train Time",   f"{detail_exp.get('training_time', '—')}s"),
                ("Date",         detail_exp["created_at"][:16]),
            ]
            rows_html = "".join([
                f'<div class="mf-detail-row">'
                f'<span class="mf-detail-key">{k}</span>'
                f'<span class="mf-detail-val">{v}</span>'
                f'</div>'
                for k, v in cfg_items
            ])
            st.html(f"""
            <div class="mf-detail-card">
                <div style="height:2px;background:linear-gradient(90deg,transparent,rgba(79,142,247,.4),transparent);"></div>
                {rows_html}
            </div>
            """)

        # ── Hyperparameters card ──────────────────────────────────────────
        with param_col:
            st.html("""
            <div class="mf-sec" style="margin-top:8px;">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Hyperparameters</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            params = detail_exp.get("params", {})
            if params:
                param_rows = "".join([
                    f'<div class="mf-detail-row">'
                    f'<span class="mf-detail-key">{k}</span>'
                    f'<span class="mf-detail-val">{v}</span>'
                    f'</div>'
                    for k, v in params.items()
                ])
                st.html(f"""
                <div class="mf-detail-card">
                    <div style="height:2px;background:linear-gradient(90deg,transparent,rgba(139,92,246,.4),transparent);"></div>
                    {param_rows}
                </div>
                """)
            else:
                st.html('<div class="mf-info">No hyperparameters recorded for this run.</div>')

        # ── Best auto-tuned params ─────────────────────────────────────────
        best = detail_exp.get("metrics", {}).get("best_params")
        if best:
            st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
            st.html("""
            <div class="mf-sec">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Best Auto-Tuned Parameters</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            pills_html = "".join([
                f'<div style="background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.08);'
                f'border-radius:9px;padding:10px 14px;margin:4px;">'
                f'<div style="font-size:10px;color:#7a82a6;font-weight:600;letter-spacing:.5px;margin-bottom:4px;">{k.upper()}</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:#c8ccee;">{v}</div>'
                f'</div>'
                for k, v in best.items()
            ])
            st.html(f"""
            <div style="display:flex;flex-wrap:wrap;gap:6px;">
                {pills_html}
            </div>
            """)

        # ── CV results ─────────────────────────────────────────────────────
        cv = detail_exp.get("metrics", {}).get("cross_validation")
        if cv:
            st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
            st.html("""
            <div class="mf-sec">
                <div class="mf-sec-line" style="width:16px;"></div>
                <span class="mf-sec-label">Cross-Validation Results</span>
                <div class="mf-sec-line" style="flex:1;"></div>
            </div>
            """)
            cv_rows = []
            for metric_name, result in cv.items():
                cv_rows.append({
                    "Metric":      metric_name.upper(),
                    "Mean":        result["mean"],
                    "Std Dev (±)": result["std"],
                    "Per-Fold Scores": " | ".join(str(s) for s in result["scores"]),
                })
            st.dataframe(pd.DataFrame(cv_rows).astype(str), use_container_width=True)

        # ── Notes ──────────────────────────────────────────────────────────
        st.html('<div style="height:1px;background:rgba(255,255,255,.07);margin:20px 0 16px;"></div>')
        st.html("""
        <div class="mf-sec">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Notes</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """)
        current_notes = detail_exp.get("notes") or ""
        new_notes = st.text_area(
            "Add or edit notes for this experiment",
            value=current_notes,
            key=f"notes_{detail_id}",
            label_visibility="collapsed",
            placeholder="Add observations, findings, or any notes about this run…",
        )

        action_col, del_col, _ = st.columns([1, 1, 4], gap="small")
        with action_col:
            if st.button("Save Notes", type="primary", use_container_width=True, key=f"save_{detail_id}"):
                requests.patch(f"{API_BASE}/experiments/{detail_id}", json={"notes": new_notes})
                st.html("""
                <div style="background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.5);
                            border-radius:8px;padding:10px 14px;display:flex;align-items:center;gap:8px;margin-top:8px;">
                    <div style="width:6px;height:6px;border-radius:50%;background:#10b981;flex-shrink:0;"></div>
                    <span style="font-size:13px;color:#10b981;font-weight:500;">Notes saved successfully.</span>
                </div>
                """)
                st.rerun()

        with del_col:
            if st.button("Delete Run", type="secondary", use_container_width=True, key=f"del_{detail_id}"):
                requests.delete(f"{API_BASE}/experiments/{detail_id}")
                st.rerun()
