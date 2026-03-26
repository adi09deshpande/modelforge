"""
sidebar.py — Shared sidebar for all ModelForge pages.
Matches the Home.py sidebar exactly.

Usage (add to every page, replacing the existing with st.sidebar: block):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    from components.sidebar import render_sidebar

    render_sidebar(current_page="Upload_Data")

current_page values:
    Home | Projects | Upload_Data | EDA_Preprocessing | Feature_Engineering
    Feature_Selection | Data_Preparation | Train_Model | Experiments
    Model_Explainability | Predict | Chat
"""

import streamlit as st

NAV = {
    "Core":      [("Home",           "Home.py"),
                  ("Projects",        "pages/1_Projects.py")],
    "Data":      [("Upload",          "pages/2_Upload_Data.py"),
                  ("EDA",             "pages/3_EDA_Preprocessing.py"),
                  ("Features",        "pages/4_Feature_Engineering.py"),
                  ("Selection",       "pages/10_Feature_Selection.py"),
                  ("Preparation",     "pages/5_Data_Preparation.py")],
    "Modelling": [("Train Model",     "pages/6_Train_Model.py"),
                  ("Experiments",     "pages/8_Experiments.py"),
                  ("Explainability",  "pages/7_Model_Explainability.py"),
                  ("Predict",         "pages/9_Predict.py")],
    "Assistant": [("AI Chat",         "pages/11_Chat.py")],
}


def render_sidebar(
    current_page: str = None,
    authenticator=None,          # pass the stauth Authenticate instance for sign-out
    logout_callback=None,        # optional callable — called after sign-out
):
    """
    Renders the full premium sidebar.

    Parameters
    ----------
    current_page : str
        Name of the current page (for display only — Streamlit highlights active
        page link automatically).
    authenticator : streamlit_authenticator.Authenticate, optional
        Pass the authenticator so the Sign out button works.
    logout_callback : callable, optional
        Called after sign-out to clear session and redirect.
    """
    proj     = st.session_state.get("project_id")
    ds       = st.session_state.get("dataset_id")
    dn       = st.session_state.get("display_name", "User")
    initials = "".join([w[0].upper() for w in dn.split()[:2]]) if dn else "U"

    with st.sidebar:

        # ── Logo ──────────────────────────────────────────────────────────
        st.html(f"""
        <div style="display:flex;align-items:center;gap:11px;
                    padding:12px 14px 18px;
                    border-bottom:1px solid rgba(255,255,255,.05);
                    margin-bottom:12px;">
            <div style="width:32px;height:32px;border-radius:9px;flex-shrink:0;
                        background:linear-gradient(135deg,#4f8ef7,#8b5cf6);
                        display:flex;align-items:center;justify-content:center;
                        font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#fff;
                        box-shadow:0 0 22px rgba(79,142,247,.5);">M</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:800;
                            color:#f0f2ff;letter-spacing:-.3px;line-height:1.2;">ModelForge</div>
                <div style="font-size:9px;color:#4a5080;letter-spacing:.9px;
                            text-transform:uppercase;">ML Platform</div>
            </div>
        </div>
        """)

        # ── User card ─────────────────────────────────────────────────────
        st.html(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    background:rgba(255,255,255,.03);
                    border:1px solid rgba(255,255,255,.06);
                    border-radius:10px;padding:10px 12px;margin-bottom:14px;">
            <div style="width:30px;height:30px;border-radius:50%;flex-shrink:0;
                        background:linear-gradient(135deg,#1d3a6e,#4c1d95);
                        display:flex;align-items:center;justify-content:center;
                        font-family:'Syne',sans-serif;font-size:11px;
                        font-weight:800;color:#a78bfa;">{initials}</div>
            <div style="min-width:0;flex:1;">
                <div style="font-size:12.5px;font-weight:500;color:#d0d4f0;
                            white-space:nowrap;overflow:hidden;
                            text-overflow:ellipsis;">{dn}</div>
                <div style="font-size:9px;color:#4a5080;letter-spacing:.3px;">Signed in</div>
            </div>
        </div>
        """)

        # ── Sign out ──────────────────────────────────────────────────────
        if st.button("Sign out", type="secondary", key="sidebar_logout_btn"):
            if authenticator:
                try:
                    authenticator.logout("Logout", "sidebar", key="sidebar_auth_logout")
                except Exception:
                    pass
            if logout_callback:
                logout_callback()

        st.html('<div style="height:1px;background:rgba(255,255,255,.04);margin:6px 0 10px;"></div>')

        # ── Nav sections ──────────────────────────────────────────────────
        for section, items in NAV.items():
            st.html(
                f'<div style="font-size:10px;font-weight:700;color:#4a5080;'
                f'letter-spacing:1.4px;text-transform:uppercase;'
                f'padding:0 10px;margin:12px 0 5px;">{section}</div>'
            )
            for label, page in items:
                st.page_link(page, label=label)

        # ── Status strip ──────────────────────────────────────────────────
        st.html('<div style="height:1px;background:rgba(255,255,255,.04);margin:14px 0 10px;"></div>')
        pcolor = "#10b981" if proj else "#445070"
        dcolor = "#10b981" if ds   else "#445070"
        st.html(f"""
        <div style="padding:0 8px 4px;">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;padding:4px 0;">
                <span style="font-size:11px;color:#6070a0;">Project</span>
                <span style="font-size:10px;font-weight:600;color:{pcolor};">
                    {'#'+str(proj) if proj else 'none'}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;
                        align-items:center;padding:4px 0;">
                <span style="font-size:11px;color:#6070a0;">Dataset</span>
                <span style="font-size:10px;font-weight:600;color:{dcolor};">
                    {'#'+str(ds) if ds else 'none'}
                </span>
            </div>
        </div>
        """)

        # ── Chat widget ───────────────────────────────────────────────────
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parent))
            from chat_widget import render_chat_widget
            render_chat_widget(
                project_id=proj or 0,
                dataset_id=ds,
                current_page=current_page,
            )
        except Exception:
            pass
