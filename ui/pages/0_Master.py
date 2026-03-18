# pages/0_Master.py
import bootstrap  # noqa: F401
import streamlit as st  # type: ignore
import streamlit_authenticator as stauth  # type: ignore

from db.db import get_session
from services.services_users import (
    list_users_for_auth,
    create_user,
    get_user_by_username,
)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="ModelForge • Sign in",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar on login page
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

db = get_session()

# -------------------------
# AUTH SETUP
# -------------------------
credentials = {
    "credentials": list_users_for_auth(db)
}

authenticator = stauth.Authenticate(
    credentials=credentials["credentials"],
    cookie_name="modelforge_auth",
    key="REPLACE_A_LONG_RANDOM_KEY",
    cookie_expiry_days=30,
    preauthorized={"emails": []},
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <div style="text-align:center;
                background: linear-gradient(135deg,#0a0d14,#16232d,#1a3848);
                padding:2rem 0;
                color:white;
                border-radius:10px;">
      <h1>⚡ ModelForge</h1>
      <h3>Sign in or create your account</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# -------------------------
# LOGIN (CORRECT WAY)
# -------------------------
with col1:
    st.subheader("Login")

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")

    if auth_status is True:
        username = st.session_state.get("username")
        name = st.session_state.get("name")

        user = get_user_by_username(db, username)
        if not user:
            st.error("Authenticated but user not found in database.")
            st.stop()

        st.session_state["user_id"] = user.id
        st.session_state["username"] = user.username
        st.session_state["display_name"] = name or user.username

        st.success(f"Welcome, {st.session_state['display_name']}")
        st.switch_page("Home.py")

    elif auth_status is False:
        st.error("Username/password incorrect")

# -------------------------
# REGISTER
# -------------------------
with col2:
    st.subheader("Register")

    with st.form("register_form"):
        u = st.text_input("Username")
        nm = st.text_input("Full name")
        em = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        try:
            create_user(db, u, nm, em, pw)
            st.success("User registered successfully. Please log in.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
