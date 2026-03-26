# mf_theme.py
# ModelForge Design System
# Inspired by: Vercel + Linear + Stripe
# Font: Syne (display) + DM Sans (body)

MF_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --mf-bg:     #07080d;
    --mf-bg2:    #0c0e16;
    --mf-bg3:    #11131e;
    --mf-border: rgba(255,255,255,0.05);
    --mf-border2:rgba(255,255,255,0.09);
    --mf-text:   #f0f2ff;
    --mf-muted:  #525870;
    --mf-dim:    #2a2d3e;
    --mf-blue:   #4f8ef7;
    --mf-purple: #8b5cf6;
    --mf-cyan:   #06b6d4;
    --mf-pink:   #ec4899;
    --mf-green:  #10b981;
    --mf-amber:  #f59e0b;
}

/* ---- KEYFRAMES ---- */
@keyframes mf-gradient {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes mf-shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes mf-pulse {
    0%,100% { opacity:.35; transform:scale(1);   }
    50%      { opacity:1;   transform:scale(1.6); }
}
@keyframes mf-float {
    0%,100% { transform:translateY(0);    }
    50%      { transform:translateY(-5px); }
}
@keyframes mf-card-in {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes mf-fade-up {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes mf-sidebar-bar {
    0%   { background-position: 0% 50%;   }
    100% { background-position: 300% 50%; }
}
@keyframes mf-orbit {
    from { transform:rotate(0deg)   translateX(80px) rotate(0deg);   }
    to   { transform:rotate(360deg) translateX(80px) rotate(-360deg);}
}

/* ---- BASE ---- */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
    animation: mf-fade-up 0.4s ease both;
}
.stApp { background: var(--mf-bg) !important; }
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--mf-text);
}

/* Hide default Streamlit chrome */
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
header    { visibility:hidden; }
.viewerBadge_container__1QSob { display:none !important; }

/* ---- HIDE sidebar collapse arrow completely ---- */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: var(--mf-bg2) !important;
    border-right: 1px solid var(--mf-border) !important;
    padding-top: 0 !important;
    position: relative !important;
    overflow: hidden !important;
}
/* Animated rainbow top bar */
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px; z-index: 100;
    background: linear-gradient(90deg,var(--mf-blue),var(--mf-purple),var(--mf-pink),var(--mf-cyan),var(--mf-blue));
    background-size: 300% 100%;
    animation: mf-sidebar-bar 3s linear infinite;
}
/* Subtle glow blob top-left */
section[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    top: -80px; left: -80px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,142,247,.07), transparent 70%);
    pointer-events: none;
}
section[data-testid="stSidebar"] > div { padding-top: 0.5rem !important; }

/* Nav links */
section[data-testid="stSidebar"] a[data-testid="stPageLink"] {
    display: flex !important;
    align-items: center !important;
    padding: 7px 10px !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--mf-muted) !important;
    text-decoration: none !important;
    transition: all 0.18s ease !important;
    margin: 1px 0 !important;
    border: 1px solid transparent !important;
    font-weight: 400 !important;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink"]:hover {
    background: var(--mf-bg3) !important;
    color: var(--mf-text) !important;
    border-color: var(--mf-border) !important;
    transform: translateX(3px) !important;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink"][aria-current="page"] {
    background: linear-gradient(135deg, rgba(13,25,41,.8), rgba(13,13,32,.8)) !important;
    color: #7eb3f7 !important;
    border-color: rgba(29,58,110,.4) !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--mf-text) !important;
    letter-spacing: -.2px !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--mf-border) !important;
    margin: 10px 0 !important;
}

/* ---- TYPOGRAPHY ---- */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    color: var(--mf-text) !important;
    letter-spacing: -0.6px !important;
    margin-bottom: 4px !important;
}
h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: var(--mf-text) !important;
    letter-spacing: -0.3px !important;
}
h3 {
    font-size: 9px !important;
    font-weight: 600 !important;
    color: var(--mf-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.4px !important;
}

/* ---- METRICS ---- */
[data-testid="metric-container"] {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,142,247,.3), transparent);
    opacity: 0; transition: opacity 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--mf-border2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,.5) !important;
}
[data-testid="metric-container"]:hover::before { opacity: 1; }
[data-testid="metric-container"] label {
    font-size: 10px !important;
    font-weight: 600 !important;
    color: var(--mf-muted) !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--mf-text) !important;
    letter-spacing: -0.5px !important;
}

/* ---- DATAFRAMES ---- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--mf-border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: var(--mf-bg2) !important;
    color: var(--mf-muted) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    border-bottom: 1px solid var(--mf-border) !important;
}
[data-testid="stDataFrame"] tbody tr td {
    background: var(--mf-bg) !important;
    color: #bbb !important;
    font-size: 12.5px !important;
    border-bottom: 1px solid rgba(255,255,255,.03) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: var(--mf-bg2) !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border2) !important;
    color: #bbb !important;
    border-radius: 9px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.22s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::after {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(79,142,247,.06), rgba(139,92,246,.06));
    opacity: 0; transition: opacity 0.22s;
}
.stButton > button:hover {
    background: var(--mf-bg3) !important;
    border-color: rgba(79,142,247,.3) !important;
    color: var(--mf-text) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.5), 0 0 0 1px rgba(79,142,247,.1) !important;
}
.stButton > button:hover::after { opacity: 1; }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button[kind="primary"] {
    background: var(--mf-text) !important;
    color: #000 !important;
    border-color: var(--mf-text) !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
}
.stButton > button[kind="primary"]:hover {
    background: #e8eaff !important;
    box-shadow: 0 6px 24px rgba(255,255,255,.15), 0 0 40px rgba(255,255,255,.05) !important;
}

/* ---- INPUTS ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border2) !important;
    border-radius: 9px !important;
    color: var(--mf-text) !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(79,142,247,.5) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,.08), 0 0 20px rgba(79,142,247,.05) !important;
}
.stSelectbox > div > div, .stMultiselect > div > div {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border2) !important;
    border-radius: 9px !important;
    color: var(--mf-text) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stNumberInput label, .stMultiselect label {
    font-size: 11px !important;
    color: var(--mf-muted) !important;
    font-weight: 500 !important;
    letter-spacing: .3px !important;
}

/* ---- RADIO ---- */
.stRadio > div { gap: 6px !important; }
.stRadio label {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border) !important;
    border-radius: 9px !important;
    padding: 9px 14px !important;
    color: var(--mf-muted) !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stRadio label:hover {
    border-color: var(--mf-border2) !important;
    color: var(--mf-text) !important;
    background: var(--mf-bg3) !important;
    transform: translateY(-1px) !important;
}
.stCheckbox label { color: var(--mf-muted) !important; font-size: 13px !important; }

/* ---- ALERTS ---- */
.stAlert { border-radius: 12px !important; border: 1px solid !important; font-size: 13px !important; }
[data-testid="stAlert"][kind="info"]    { background:rgba(6,14,29,.8) !important;  border-color:rgba(29,58,110,.5) !important;  color:#7eb3f7 !important; }
[data-testid="stAlert"][kind="success"] { background:rgba(5,18,9,.8)  !important;  border-color:rgba(20,83,45,.5)  !important;  color:var(--mf-green) !important; }
[data-testid="stAlert"][kind="warning"] { background:rgba(18,12,0,.8) !important;  border-color:rgba(120,53,15,.5) !important;  color:var(--mf-amber) !important; }
[data-testid="stAlert"][kind="error"]   { background:rgba(18,5,5,.8)  !important;  border-color:rgba(127,29,29,.5) !important;  color:#f87171 !important; }

/* ---- DIVIDER / HR ---- */
hr { border-color: var(--mf-border) !important; margin: 1.5rem 0 !important; }

/* ---- EXPANDER ---- */
.streamlit-expanderHeader {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border) !important;
    border-radius: 12px !important;
    color: var(--mf-muted) !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--mf-border2) !important;
    color: var(--mf-text) !important;
}
.streamlit-expanderContent {
    background: rgba(12,14,22,.8) !important;
    border: 1px solid var(--mf-border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    backdrop-filter: blur(8px) !important;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid var(--mf-border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--mf-muted) !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 9px 18px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #999 !important; }
.stTabs [aria-selected="true"] {
    color: var(--mf-text) !important;
    border-bottom-color: var(--mf-blue) !important;
    font-weight: 500 !important;
}

/* ---- PROGRESS BAR ---- */
.stProgress > div > div {
    background: var(--mf-border) !important;
    border-radius: 4px !important;
    height: 3px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--mf-blue), var(--mf-purple)) !important;
    border-radius: 4px !important;
    transition: width 0.4s ease !important;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--mf-bg); }
::-webkit-scrollbar-thumb { background: var(--mf-dim); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3a3d55; }

/* ---- CHAT ---- */
[data-testid="stChatMessage"] {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border2) !important;
    border-radius: 12px !important;
}

/* ---- MISC ---- */
[data-testid="stFileUploader"] {
    background: var(--mf-bg2) !important;
    border: 1px dashed var(--mf-border2) !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,142,247,.4) !important;
    box-shadow: 0 0 20px rgba(79,142,247,.05) !important;
}
[data-testid="stDownloadButton"] > button {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border2) !important;
    color: #bbb !important;
    border-radius: 9px !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
[data-testid="stToast"] {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border2) !important;
    border-radius: 12px !important;
    color: var(--mf-text) !important;
    font-size: 13px !important;
    backdrop-filter: blur(12px) !important;
}
.stTable table { border-collapse: collapse !important; width: 100% !important; }
.stTable th {
    background: var(--mf-bg2) !important;
    color: var(--mf-muted) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--mf-border) !important;
}
.stTable td {
    background: var(--mf-bg) !important;
    color: #bbb !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,.03) !important;
}
.stTable tr:hover td { background: var(--mf-bg2) !important; }
.stCaption, small { color: var(--mf-dim) !important; font-size: 11.5px !important; }
.stMarkdown p {
    color: var(--mf-muted) !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    font-weight: 300 !important;
}
.stMarkdown code {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border) !important;
    border-radius: 5px !important;
    color: #7eb3f7 !important;
    font-size: 12px !important;
    padding: 2px 7px !important;
}
[data-testid="stForm"] {
    background: var(--mf-bg2) !important;
    border: 1px solid var(--mf-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    backdrop-filter: blur(8px) !important;
}

/* 🚫 FINAL FIX: override Streamlit button hover properly */
.active-btn .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* also kill generic hover */
.active-btn .stButton > button:hover {
    background: linear-gradient(135deg,#4f8ef7,#8b5cf6) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* disable interaction */
.active-btn .stButton > button {
    pointer-events: none !important;
    cursor: default !important;
}
</style>
"""


# ---- Helper components -------------------------------------------------------

def page_header(title: str, subtitle: str = "") -> str:
    sub = (f'<p style="font-size:13px;color:#525870;margin:6px 0 0;'
           f'line-height:1.7;font-weight:300;">{subtitle}</p>') if subtitle else ""
    return f"""
    <div style="padding:26px 0 18px;border-bottom:1px solid rgba(255,255,255,.05);margin-bottom:22px;">
        <h1 style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                   color:#f0f2ff;letter-spacing:-.5px;margin:0;">{title}</h1>
        {sub}
    </div>"""


def badge(text: str, color: str = "gray") -> str:
    styles = {
        "gray":   "background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);color:#525870;",
        "blue":   "background:rgba(6,14,29,.9);border:1px solid rgba(29,58,110,.6);color:#7eb3f7;",
        "green":  "background:rgba(5,18,9,.9);border:1px solid rgba(20,83,45,.6);color:#10b981;",
        "amber":  "background:rgba(18,12,0,.9);border:1px solid rgba(120,53,15,.6);color:#f59e0b;",
        "red":    "background:rgba(18,5,5,.9);border:1px solid rgba(127,29,29,.6);color:#f87171;",
        "purple": "background:rgba(13,8,25,.9);border:1px solid rgba(76,29,149,.6);color:#a78bfa;",
        "pink":   "background:rgba(24,8,20,.9);border:1px solid rgba(131,24,67,.6);color:#f9a8d4;",
        "cyan":   "background:rgba(0,14,20,.9);border:1px solid rgba(8,145,178,.6);color:#22d3ee;",
    }
    s = styles.get(color, styles["gray"])
    return (f'<span style="{s}display:inline-flex;align-items:center;padding:2px 10px;'
            f'border-radius:20px;font-size:11px;font-weight:500;letter-spacing:.2px;">{text}</span>')


def section_label(text: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:22px 0 14px;">'
        f'<div style="height:1px;width:20px;background:rgba(255,255,255,.08);"></div>'
        f'<span style="font-size:9px;font-weight:700;color:#2a2d3e;'
        f'letter-spacing:1.5px;text-transform:uppercase;">{text}</span>'
        f'<div style="height:1px;flex:1;background:rgba(255,255,255,.04);"></div>'
        f'</div>'
    )
