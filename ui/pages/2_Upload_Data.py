import bootstrap  # noqa: F401
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ModelForge • Upload & Explore", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# ── Page styles ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

[data-testid="stFileUploader"] {
    background: rgba(12,14,22,.8) !important;
    border: 1px dashed rgba(79,142,247,.25) !important;
    border-radius: 14px !important;
    transition: all .25s ease !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,142,247,.5) !important;
    background: rgba(79,142,247,.04) !important;
    box-shadow: 0 0 30px rgba(79,142,247,.06) !important;
}
.stSelectbox > div > div {
    background: rgba(12,14,22,.9) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}
.stSelectbox > div > div:hover { border-color: rgba(79,142,247,.3) !important; }

/* ── Stat card ── */
.mf-stat {
    background: rgba(12,14,22,.9);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: visible;          /* never clip the number */
    backdrop-filter: blur(8px);
    transition: all .22s ease;
    height: auto;               /* grow to fit */
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.mf-stat:hover {
    border-color: rgba(255,255,255,.14);
    transform: translateY(-2px);
    box-shadow: 0 10px 36px rgba(0,0,0,.5);
}
.mf-stat-label {
    font-size: 11px;
    font-weight: 600;
    color: #888eaa;             /* clearly visible muted label */
    letter-spacing: .8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
/* Number: never truncate, scale font to content */
.mf-stat-value {
    font-family: 'Syne', sans-serif;
    font-size: clamp(14px, 2vw, 22px);
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
    white-space: nowrap;
    overflow: visible;        /* 🔥 show full number */
}

/* ── Dataset row card ── */
.mf-ds-row {
    background: rgba(12,14,22,.85);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 11px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    transition: all .2s ease;
    backdrop-filter: blur(8px);
}
.mf-ds-row:hover {
    border-color: rgba(79,142,247,.2);
    background: rgba(13,25,41,.6);
    transform: translateX(3px);
}
.mf-ds-row.active {
    border-color: rgba(29,58,110,.6);
    background: linear-gradient(135deg,rgba(13,25,41,.9),rgba(13,13,32,.9));
}

/* ── Section divider label — VISIBLE ── */
.mf-sec {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.mf-sec-line { height: 1px; background: rgba(255,255,255,.1); }
.mf-sec-label {
    font-size: 11px;          /* up from 9px */
    font-weight: 700;
    color: #7a82a6;           /* clearly visible */
    letter-spacing: 1.4px;
    text-transform: uppercase;
    white-space: nowrap;
}

@keyframes mf-fade-up {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
.mf-s1 { animation: mf-fade-up .4s ease .05s both; }
.mf-s2 { animation: mf-fade-up .4s ease .12s both; }
.mf-s3 { animation: mf-fade-up .4s ease .19s both; }
.mf-s4 { animation: mf-fade-up .4s ease .26s both; }
.mf-s5 { animation: mf-fade-up .4s ease .33s both; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
render_sidebar(current_page="Upload")

# ── Auth ──────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

if not st.session_state.get("project_id"):
    st.markdown("""
    <div style="background:rgba(6,14,29,.8);border:1px solid rgba(29,58,110,.4);
                border-radius:12px;padding:20px 24px;margin-top:20px;backdrop-filter:blur(8px);">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                    color:#7eb3f7;margin-bottom:6px;">No project selected</div>
        <div style="font-size:13px;color:#8890aa;">Go to Projects and select or create one first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

project_id = st.session_state["project_id"]

# ── Fetch datasets ────────────────────────────────────────────────────────
def fetch_datasets():
    r = requests.get(f"{API_BASE}/dataset/project/{project_id}", timeout=30)
    r.raise_for_status()
    return r.json()

try:
    datasets = fetch_datasets()
except Exception as e:
    st.error(f"Failed to load datasets: {e}")
    datasets = []

# ══════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════
st.html("""
<div class="mf-s1" style="padding:40px 0 28px;border-bottom:1px solid rgba(255,255,255,.07);
            margin-bottom:28px;position:relative;overflow:hidden;">

    <div style="position:absolute;top:-20px;right:0;width:200px;height:200px;border-radius:50%;
                background:radial-gradient(circle,rgba(79,142,247,.05),transparent 70%);
                pointer-events:none;"></div>

    <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,.05);
                backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.1);
                border-radius:20px;padding:5px 14px;margin-bottom:18px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#4f8ef7;"></div>
        <span style="font-size:11px;color:#8890aa;font-weight:600;letter-spacing:.7px;">
            UPLOAD · EXPLORE
        </span>
    </div>

    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.15;margin-bottom:14px;
                background:linear-gradient(135deg,#ffffff 0%,#9099cc 40%,#ffffff 80%);
                background-size:200% auto;-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;background-clip:text;">
        Upload & Explore Data
    </div>

    <p style="font-size:15px;color:#8890aa;margin:0;font-weight:400;line-height:1.7;max-width:560px;">
        Upload your CSV dataset, inspect statistics, and preview rows
        before moving to EDA and feature engineering.
    </p>
</div>
""")

# ══════════════════════════════════════════════════════════
# TOP ROW — Datasets (left) + Upload form (right)
# ══════════════════════════════════════════════════════════
left, right = st.columns([1.1, 1], gap="large")

# ── LEFT: Dataset list + Stats ────────────────────────────
with left:
    st.markdown("""
    <div class="mf-sec mf-s2">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Your Datasets</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """, unsafe_allow_html=True)

    if datasets:
        default_idx = 0
        if "dataset_id" in st.session_state:
            for i, d in enumerate(datasets):
                if d["id"] == st.session_state["dataset_id"]:
                    default_idx = i
                    break

        for d in datasets:
            is_active  = st.session_state.get("dataset_id") == d["id"]
            active_cls = "active" if is_active else ""
            dot_color  = "#10b981" if is_active else "#3a3d55"
            name_color = "#7eb3f7" if is_active else "#c8ccee"
            glow       = "box-shadow:0 0 8px #10b981;" if is_active else ""
            badge      = (
                '<span style="font-size:10px;background:rgba(16,185,129,.1);'
                'border:1px solid rgba(20,83,45,.5);color:#10b981;'
                'padding:3px 10px;border-radius:20px;font-weight:600;">active</span>'
                if is_active else ""
            )
            st.markdown(f"""
            <div class="mf-ds-row {active_cls}">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:9px;height:9px;border-radius:50%;
                                background:{dot_color};flex-shrink:0;{glow}"></div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;
                                    color:{name_color};letter-spacing:-.2px;">{d['name']}</div>
                        <div style="font-size:11px;color:#525870;margin-top:2px;">ID #{d['id']}</div>
                    </div>
                </div>
                {badge}
            </div>
            """, unsafe_allow_html=True)

        ds = st.selectbox(
            "Select dataset",
            datasets,
            index=default_idx,
            format_func=lambda d: d["name"],
            label_visibility="collapsed",
        )
        dataset_id = ds["id"]
        st.session_state["dataset_id"] = dataset_id

        # ── Stats ──────────────────────────────────────────────────────────
        st.markdown('<div style="height:1px;background:rgba(255,255,255,.07);margin:22px 0 18px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="mf-sec mf-s3">
            <div class="mf-sec-line" style="width:16px;"></div>
            <span class="mf-sec-label">Dataset Statistics</span>
            <div class="mf-sec-line" style="flex:1;"></div>
        </div>
        """, unsafe_allow_html=True)

        try:
            stats_resp = requests.get(f"{API_BASE}/dataset/{dataset_id}/stats", timeout=30)
            stats_resp.raise_for_status()
            stats = stats_resp.json()

            s1, s2, s3 = st.columns(3, gap="small")
            stat_items = [
                (s1, "Rows",    f"{stats['rows_estimated']:,}", "#4f8ef7"),
                (s2, "Columns", str(stats["columns_count"]),    "#8b5cf6"),
                (s3, "Size MB", str(stats["file_size_mb"]),     "#10b981"),
            ]
            for col, label, val, color in stat_items:
                with col:
                    st.markdown(f"""
                    <div class="mf-stat">
                        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                                    background:linear-gradient(90deg,transparent,{color}55,transparent);
                                    border-radius:12px 12px 0 0;"></div>
                        <div class="mf-stat-label">{label}</div>
                        <div class="mf-stat-value" style="color:{color};">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div style="background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.4);
                        border-radius:10px;padding:14px 16px;font-size:13px;color:#f59e0b;">
                Could not load stats: {e}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:rgba(12,14,22,.8);border:1px dashed rgba(255,255,255,.08);
                    border-radius:14px;padding:36px 24px;text-align:center;backdrop-filter:blur(8px);">
            <div style="font-size:28px;margin-bottom:12px;opacity:.2;">&#8679;</div>
            <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
                        color:#7a82a6;margin-bottom:6px;">No datasets yet</div>
            <div style="font-size:13px;color:#525870;">Upload your first CSV using the form</div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Upload form ────────────────────────────────────
with right:
    st.markdown("""
    <div class="mf-sec mf-s2">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Upload New Dataset</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Top glow bar — self-contained div, no split tags
    st.markdown("""
    <div style="height:2px;background:linear-gradient(90deg,transparent,rgba(79,142,247,.4),transparent);
                border-radius:2px;margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    with st.form("upload_form", clear_on_submit=True):
        name = st.text_input(
            "Dataset name",
            placeholder="e.g. diabetes_raw, titanic_v1",
        )
        file = st.file_uploader("Drop your CSV here", type=["csv"])
        submitted = st.form_submit_button("Upload Dataset", type="primary")

    if submitted:
        if not file:
            st.markdown("""
            <div style="background:rgba(18,12,0,.7);border:1px solid rgba(120,53,15,.4);
                        border-radius:10px;padding:14px 16px;font-size:13px;color:#f59e0b;margin-top:8px;">
                Please select a CSV file first.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Uploading..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/dataset/upload",
                        data={"project_id": project_id, "name": name.strip() or file.name},
                        files={"file": (file.name, file, "text/csv")},
                        timeout=300,
                    )
                    r.raise_for_status()
                    payload = r.json()
                    st.session_state["dataset_id"] = payload["dataset_id"]
                    st.markdown(f"""
                    <div style="background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.5);
                                border-radius:10px;padding:14px 18px;margin-top:8px;
                                display:flex;align-items:center;gap:10px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:#10b981;
                                    flex-shrink:0;box-shadow:0 0 8px #10b981;"></div>
                        <div>
                            <div style="font-size:13px;font-weight:600;color:#10b981;">Upload successful</div>
                            <div style="font-size:12px;color:#8890aa;margin-top:2px;">
                                '{payload['name']}' is ready to use.
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.markdown(f"""
                    <div style="background:rgba(18,5,5,.7);border:1px solid rgba(127,29,29,.4);
                                border-radius:10px;padding:14px 16px;font-size:13px;
                                color:#f87171;margin-top:8px;">
                        Upload error: {e}
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# FULL-WIDTH DATA PREVIEW — below both columns
# ══════════════════════════════════════════════════════════
if datasets and st.session_state.get("dataset_id"):
    st.markdown("""
    <div style="height:1px;background:rgba(255,255,255,.07);margin:28px 0 24px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mf-sec mf-s4">
        <div class="mf-sec-line" style="width:16px;"></div>
        <span class="mf-sec-label">Data Preview</span>
        <div class="mf-sec-line" style="flex:1;"></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        preview_resp = requests.get(
            f"{API_BASE}/dataset/{dataset_id}/preview",
            params={"n": 50}, timeout=30,
        )
        preview_resp.raise_for_status()
        df = pd.DataFrame(preview_resp.json().get("rows", []))

        if df.empty:
            st.markdown("""
            <div style="background:rgba(12,14,22,.8);border:1px solid rgba(255,255,255,.07);
                        border-radius:10px;padding:20px;text-align:center;
                        font-size:13px;color:#7a82a6;">
                Dataset has no rows.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;" class="mf-s5">
                <span style="background:rgba(6,14,29,.8);border:1px solid rgba(29,58,110,.5);
                             border-radius:20px;padding:4px 13px;font-size:12px;
                             font-weight:600;color:#7eb3f7;">{len(df):,} rows</span>
                <span style="background:rgba(13,8,25,.8);border:1px solid rgba(76,29,149,.5);
                             border-radius:20px;padding:4px 13px;font-size:12px;
                             font-weight:600;color:#a78bfa;">{len(df.columns)} columns</span>
                <span style="background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.5);
                             border-radius:20px;padding:4px 13px;font-size:12px;
                             font-weight:600;color:#10b981;">first 50 rows</span>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=380)

    except Exception as e:
        st.markdown(f"""
        <div style="background:rgba(18,5,5,.7);border:1px solid rgba(127,29,29,.4);
                    border-radius:10px;padding:14px 16px;font-size:13px;color:#f87171;">
            Could not load preview: {e}
        </div>
        """, unsafe_allow_html=True)
