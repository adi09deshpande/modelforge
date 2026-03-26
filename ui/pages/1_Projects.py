import bootstrap  # noqa
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.chat_widget import render_chat_widget
from components.sidebar import render_sidebar
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="ModelForge • Projects", layout="wide", initial_sidebar_state="expanded")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)

# ── Auth gate ──────────────────────────────────────────────────────────────────
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar(current_page="Projects") 
# ── Page styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide sidebar collapse button */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Tab strip */
.mf-tabs { display:flex; gap:4px; margin-bottom:28px; }
.mf-tab {
    padding:8px 20px; border-radius:8px; font-size:13px; font-weight:500;
    border:1px solid rgba(255,255,255,.06); cursor:pointer;
    color:#525870; background:rgba(12,14,22,.8); transition:all .18s;
    font-family:'DM Sans',sans-serif; letter-spacing:.2px;
}
.mf-tab.active {
    background:rgba(79,142,247,.12); border-color:rgba(79,142,247,.35);
    color:#7eb3f7;
}

/* Project cards */
.proj-card {
    background:rgba(12,14,22,.9);
    border:1px solid rgba(255,255,255,.06);
    border-radius:14px; padding:20px 22px;
    transition:all .22s cubic-bezier(.4,0,.2,1);
    cursor:pointer; position:relative; overflow:hidden;
    animation: mf-card-in 0.4s ease both;
}
.proj-card::before {
    content:''; position:absolute; top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(79,142,247,.4),transparent);
    opacity:0; transition:opacity .22s;
}
.proj-card:hover { border-color:rgba(79,142,247,.25); transform:translateY(-3px);
    box-shadow:0 12px 40px rgba(0,0,0,.6); }
.proj-card:hover::before { opacity:1; }
.proj-card-num { font-size:10px;font-weight:800;color:rgba(79,142,247,.5);
    letter-spacing:1px;font-family:'Syne',sans-serif;margin-bottom:10px; }
.proj-card-title { font-size:17px;font-weight:700;color:#e0e4ff;
    font-family:'Syne',sans-serif;letter-spacing:-.3px;margin-bottom:6px; }
.proj-card-slug { font-size:12px;color:#525870;font-family:monospace;
    background:rgba(255,255,255,.04);border-radius:5px;padding:2px 8px;
    display:inline-block;margin-bottom:12px; }
.proj-card-arrow { position:absolute;right:18px;bottom:18px;font-size:14px;
    color:#2a2d3e;transition:all .2s;font-family:monospace; }
.proj-card:hover .proj-card-arrow { color:#4f8ef7; transform:translateX(3px); }
.proj-card.active-proj { border-color:rgba(16,185,129,.3) !important;
    background:rgba(5,18,9,.6) !important; }
.proj-card.active-proj::before {
    background:linear-gradient(90deg,transparent,rgba(16,185,129,.4),transparent) !important;
    opacity:1 !important;
}

/* Input styling */
.stTextInput input {
    background:rgba(12,14,22,.9) !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:10px !important; color:#e0e4ff !important;
    font-size:14px !important; padding:12px 16px !important;
    transition:all .18s !important;
}
.stTextInput input:focus {
    border-color:rgba(79,142,247,.4) !important;
    box-shadow:0 0 0 3px rgba(79,142,247,.08) !important;
}
.stTextInput label { font-size:12px !important; color:#6070a0 !important;
    font-weight:500 !important; letter-spacing:.3px !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;
    border:none !important; border-radius:10px !important;
    color:#fff !important; font-weight:600 !important;
    font-size:13px !important; padding:10px 24px !important;
    transition:all .2s !important; letter-spacing:.2px !important;
    box-shadow:0 4px 20px rgba(79,142,247,.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(79,142,247,.45) !important;
}
/* 🚫 Disable hover for disabled (active) buttons */
.stButton > button[kind="primary"]:disabled:hover {
    transform: none !important;
    box-shadow: none !important;
    background: linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;
}

/* Page link in main — Open button */
.block-container div[data-testid="stPageLink"] { display:none !important; }

/* 🔥 Disable hover for active project button */
.active-btn button {
    background: linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;
    color: #fff !important;
    cursor: default !important;
    pointer-events: none !important;
}

/* Remove hover effect completely */
.active-btn button:hover {
    background: linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────────
st.html("""
<div style="padding:36px 0 28px;border-bottom:1px solid rgba(255,255,255,.04);margin-bottom:32px;position:relative;">
    <div style="position:absolute;top:-20px;right:0;width:300px;height:200px;border-radius:50%;
                background:radial-gradient(circle,rgba(79,142,247,.05),transparent 70%);pointer-events:none;"></div>
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.04);
                border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:4px 12px;margin-bottom:16px;">
        <span style="font-size:14px;">◫</span>
        <span style="font-size:10px;color:#6070a0;font-weight:600;letter-spacing:.8px;text-transform:uppercase;">Projects</span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-1px;line-height:1.1;margin-bottom:10px;
                background:linear-gradient(135deg,#ffffff 0%,#9099cc 60%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        Your Workspaces
    </div>
    <p style="font-size:14px;color:#6070a0;margin:0;font-weight:300;line-height:1.6;">
        Select an existing project to continue your work, or create a new one to get started.
    </p>
</div>
""")

# ── Mode toggle ────────────────────────────────────────────────────────────────
# ── Mode toggle (clean buttons) ───────────────────────────────────────────────
# ── Mode toggle (clean professional) ─────────────────────────────────────────
st.markdown("""
<style>
.mf-toggle {
    display:flex;
    gap:12px;
    margin-bottom:28px;
}
.mf-toggle button {
    width:100%;
    height:44px;
    border-radius:10px;
    font-size:13.5px;
    font-weight:500;
    border:1px solid rgba(255,255,255,.08);
    background:rgba(12,14,22,.8);
    color:#70789a;
    transition:all .2s ease;
}
.mf-toggle button:hover {
    border-color:rgba(255,255,255,.15);
    color:#ccc;
}
.mf-toggle .active {
    background:rgba(79,142,247,.12);
    border-color:rgba(79,142,247,.4);
    color:#7eb3f7;
}
</style>
""", unsafe_allow_html=True)

if "mode" not in st.session_state:
    st.session_state["mode"] = "select"

col1, col2 = st.columns(2)

with col1:
    btn1 = st.button("Select existing project", key="btn_select", use_container_width=True)

with col2:
    btn2 = st.button("Create new project", key="btn_create", use_container_width=True)

if btn1:
    st.session_state["mode"] = "select"
if btn2:
    st.session_state["mode"] = "create"

mode = (
    "Select existing project"
    if st.session_state["mode"] == "select"
    else "Create new project"
)

# ══════════════════════════════════════════════════════════════════════════════
# SELECT EXISTING PROJECT
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Select existing project":

    resp = requests.get(
        f"{API_BASE}/projects",
        params={"user_id": st.session_state["user_id"]},
    )

    if resp.status_code != 200:
        st.markdown("""
        <div style="background:rgba(18,5,5,.8);border:1px solid rgba(127,29,29,.4);
                    border-radius:12px;padding:20px 24px;color:#f87171;font-size:14px;">
            ✕ &nbsp; Failed to load projects. Make sure the API is running.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    projects = resp.json()
    active   = st.session_state.get("project_id")

    if not projects:
        st.html("""
        <div style="text-align:center;padding:60px 0;">
            <div style="font-size:40px;margin-bottom:16px;">◫</div>
            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                        color:#e0e4ff;margin-bottom:8px;">No projects yet</div>
            <div style="font-size:13px;color:#525870;">Switch to "Create new project" to get started.</div>
        </div>
        """)
    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:18px;">
            <div style="height:1px;width:20px;background:rgba(255,255,255,.08);"></div>
            <span style="font-size:10px;font-weight:700;color:#6070a0;letter-spacing:1.5px;text-transform:uppercase;">
                {len(projects)} workspace{'s' if len(projects)!=1 else ''}
            </span>
            <div style="height:1px;flex:1;background:rgba(255,255,255,.04);"></div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3, gap="small")
        for idx, p in enumerate(projects):
            is_active = (p["id"] == active)
            with cols[idx % 3]:
                active_cls = "active-proj" if is_active else ""
                active_badge = (
                    '<span style="font-size:10px;background:rgba(16,185,129,.15);'
                    'border:1px solid rgba(16,185,129,.3);color:#10b981;'
                    'border-radius:20px;padding:2px 10px;font-weight:500;">● Active</span>'
                    if is_active else ""
                )

                html = f"""
                <div class="proj-card {active_cls}">
                    <div class="proj-card-num">#{str(p['id']).zfill(2)}</div>
                    <div class="proj-card-title">{p['title']}</div>
                    <div class="proj-card-slug">{p['slug']}</div>
                    {active_badge}
                    <div class="proj-card-arrow">→</div>
                </div>
                """

                st.html(html)
                btn_label = "✓ Active" if is_active else "Select"

                # Wrap active button
                clicked = st.button(
                    "✓ Active" if is_active else "Select",
                    key=f"sel_{p['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    disabled=is_active  # 🔥 THIS IS THE FIX
                )

                if clicked:
                    st.session_state["project_id"]   = p["id"]
                    st.session_state["project_slug"] = p["slug"]
                    st.rerun()


                    st.session_state["project_id"]   = p["id"]
                    st.session_state["project_slug"] = p["slug"]
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# CREATE NEW PROJECT
# ══════════════════════════════════════════════════════════════════════════════
else:
    col, _ = st.columns([1, 1])
    with col:
        st.html("""
        <div style="background:rgba(12,14,22,.9);border:1px solid rgba(255,255,255,.07);
                    border-radius:16px;padding:28px 28px 8px;">
            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                        color:#e0e4ff;margin-bottom:6px;">New Workspace</div>
            <div style="font-size:13px;color:#525870;margin-bottom:24px;">
                Give your project a name to get started.
            </div>
        </div>
        """)

        with st.container():
            name   = st.text_input("Project name", placeholder="e.g. Customer Churn Prediction")
            create = st.button("Create project →", type="primary", use_container_width=True)

        if create:
            if not name.strip():
                st.markdown("""
                <div style="background:rgba(18,12,0,.8);border:1px solid rgba(120,53,15,.4);
                            border-radius:10px;padding:12px 18px;color:#f59e0b;font-size:13px;margin-top:8px;">
                    ⚠ &nbsp; Please enter a project name.
                </div>
                """, unsafe_allow_html=True)
            else:
                resp = requests.post(
                    f"{API_BASE}/projects",
                    json={"title": name, "user_id": st.session_state["user_id"]},
                )
                if resp.status_code != 200:
                    st.markdown(f"""
                    <div style="background:rgba(18,5,5,.8);border:1px solid rgba(127,29,29,.4);
                                border-radius:10px;padding:12px 18px;color:#f87171;font-size:13px;margin-top:8px;">
                        ✕ &nbsp; {resp.text}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    p = resp.json()
                    st.session_state["project_id"]   = p["id"]
                    st.session_state["project_slug"] = p["slug"]
                    st.markdown(f"""
                    <div style="background:rgba(5,18,9,.8);border:1px solid rgba(20,83,45,.4);
                                border-radius:10px;padding:14px 18px;color:#10b981;font-size:13px;margin-top:8px;">
                        ✓ &nbsp; <strong>{p['title']}</strong> created successfully!
                        &nbsp;·&nbsp; Next: upload a dataset.
                    </div>
                    """, unsafe_allow_html=True)

# ── Status bar ─────────────────────────────────────────────────────────────────
proj_id   = st.session_state.get("project_id")
proj_slug = st.session_state.get("project_slug", "")

st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)

if proj_id:
    st.html(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
                background:rgba(5,18,9,.7);border:1px solid rgba(16,185,129,.2);
                border-radius:12px;padding:14px 20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#10b981;"></div>
            <span style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:#e0e4ff;">
                Project #{proj_id} active
            </span>
            <span style="font-size:12px;font-family:monospace;color:#525870;
                         background:rgba(255,255,255,.05);border-radius:5px;padding:2px 8px;">
                {proj_slug}
            </span>
        </div>
        <span style="font-size:12px;color:#10b981;font-weight:500;">Ready to continue →</span>
    </div>
    """)
else:
    st.html("""
    <div style="display:flex;align-items:center;gap:10px;
                background:rgba(12,14,22,.7);border:1px solid rgba(255,255,255,.06);
                border-radius:12px;padding:14px 20px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#445070;"></div>
        <span style="font-size:13px;color:#525870;">No active project — select or create one above.</span>
    </div>
    """)
