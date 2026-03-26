# pages/0_Master.py
import bootstrap  # noqa: F401
import streamlit as st
import streamlit_authenticator as stauth

from db.db import get_session
from services.services_users import list_users_for_auth, create_user, get_user_by_username

st.set_page_config(
    page_title="ModelForge — Sign in",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { display:none !important; }
div[data-testid="collapsedControl"] { display:none !important; }
.stApp { background: #0a0a0a !important; }
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif !important;color:#ededed;}
.stButton>button{background:#fff !important;border:none !important;color:#000 !important;border-radius:6px !important;font-size:13px !important;font-weight:600 !important;padding:8px 16px !important;width:100% !important;}
.stButton>button:hover{background:#e5e5e5 !important;}
.stButton>button[kind="secondary"]{background:#111 !important;color:#ededed !important;border:0.5px solid #2a2a2a !important;}
.stTextInput>div>div>input{background:#111 !important;border:0.5px solid #2a2a2a !important;border-radius:6px !important;color:#ededed !important;font-size:13px !important;padding:8px 12px !important;}
.stTextInput>div>div>input:focus{border-color:#3a3a3a !important;box-shadow:0 0 0 3px rgba(255,255,255,0.05) !important;}
.stTextInput label{font-size:12px !important;color:#555 !important;font-weight:500 !important;}
[data-testid="stAlert"]{border-radius:6px !important;font-size:13px !important;}
[data-testid="stForm"]{background:transparent !important;border:none !important;padding:0 !important;}
hr{border-color:#1f1f1f !important;}
</style>
""", unsafe_allow_html=True)

db = get_session()

credentials = {"credentials": list_users_for_auth(db)}
authenticator = stauth.Authenticate(
    credentials=credentials["credentials"],
    cookie_name="modelforge_auth",
    key="REPLACE_A_LONG_RANDOM_KEY",
    cookie_expiry_days=30,
    preauthorized={"emails": []},
)

# ── Page layout ───────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:48px 0 40px 0;text-align:center;">
    <div style="display:inline-flex;align-items:center;justify-content:center;
                width:44px;height:44px;background:#fff;border-radius:10px;
                font-size:22px;font-weight:700;color:#000;margin-bottom:20px;">M</div>
    <h1 style="font-size:24px !important;font-weight:700 !important;color:#fff !important;
               letter-spacing:-0.5px !important;margin:0 0 8px 0 !important;">ModelForge</h1>
    <p style="font-size:14px;color:#555;margin:0;">Sign in to your account</p>
</div>
""", unsafe_allow_html=True)

tab_login, tab_register = st.tabs(["Sign in", "Create account"])

# ── Login ─────────────────────────────────────────────────────────────────
with tab_login:
    st.markdown('<div style="background:#111;border:0.5px solid #1f1f1f;border-radius:10px;padding:24px;">', unsafe_allow_html=True)

    authenticator.login(location="main")
    auth_status = st.session_state.get("authentication_status")

    if auth_status is True:
        username = st.session_state.get("username")
        name     = st.session_state.get("name")
        user     = get_user_by_username(db, username)

        if not user:
            st.error("Authenticated but user not found.")
            st.stop()

        st.session_state["user_id"]      = user.id
        st.session_state["username"]     = user.username
        st.session_state["display_name"] = name or user.username
        st.switch_page("Home.py")

    elif auth_status is False:
        st.error("Incorrect username or password.")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Register ──────────────────────────────────────────────────────────────
with tab_register:
    st.markdown('<div style="background:#111;border:0.5px solid #1f1f1f;border-radius:10px;padding:24px;">', unsafe_allow_html=True)

    with st.form("register_form"):
        u  = st.text_input("Username")
        nm = st.text_input("Full name")
        em = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        try:
            create_user(db, u, nm, em, pw)
            st.success("Account created. Sign in to continue.")
        except Exception as e:
            st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center;font-size:12px;color:#333;margin-top:24px;">
    ModelForge — End-to-End ML Platform
</p>
""", unsafe_allow_html=True)
