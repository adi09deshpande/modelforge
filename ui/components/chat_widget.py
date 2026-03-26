"""
chat_widget.py — Groq-powered sidebar chat widget.
Appears on every page. Uses non-streaming for compact display.
API key loaded from .streamlit/secrets.toml — never exposed in UI.
"""

import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"


def _get_api_key() -> str:
    """Load Groq API key from secrets.toml only. Never from UI."""
    try:
        return st.secrets["groq"]["api_key"]
    except Exception:
        return ""


def render_chat_widget(
    project_id: int,
    dataset_id: int = None,
    current_page: str = None,
):
    st.divider()
    st.markdown("### 💬 AI Assistant")

    # ── Load API key from secrets.toml ────────────────────────────────────
    api_key = _get_api_key()

    if not api_key:
        st.error("❌ Groq API key missing")
        st.caption("Add to `.streamlit/secrets.toml`:\n```\n[groq]\napi_key = 'gsk_...'\n```")
        st.page_link("pages/11_Chat.py", label="Open full chat →", icon="💬")
        return

    # ── Model ─────────────────────────────────────────────────────────────
    model = st.session_state.get("full_chat_model", "llama-3.3-70b-versatile")

    # ── Init history ──────────────────────────────────────────────────────
    if "sidebar_chat_history" not in st.session_state:
        st.session_state["sidebar_chat_history"] = []

    history = st.session_state["sidebar_chat_history"]

    # ── Show last 4 messages ──────────────────────────────────────────────
    if history:
        for msg in history[-4:]:
            is_user = msg["role"] == "user"
            bg      = "#2d3748" if is_user else "#1a2535"
            border  = "#4a90d9" if is_user else "#48bb78"
            icon    = "🧑" if is_user else "🤖"
            preview = msg["content"][:180] + "..." if len(msg["content"]) > 180 else msg["content"]
            st.markdown(
                f'<div style="background:{bg};padding:6px 10px;border-radius:8px;'
                f'margin:3px 0;font-size:12px;border-left:3px solid {border};">'
                f'{icon} {preview}</div>',
                unsafe_allow_html=True,
            )

    # ── Suggestions ───────────────────────────────────────────────────────
    if not history and project_id:
        try:
            params = {"page": current_page} if current_page else {}
            sugg = requests.get(
                f"{API_BASE}/chat/suggestions/{project_id}",
                params=params, timeout=5,
            ).json().get("suggestions", [])
            if sugg:
                st.caption("💡 Quick questions:")
                for s in sugg[:3]:
                    if st.button(s, key=f"sugg_{current_page}_{s[:15]}", use_container_width=True):
                        st.session_state["sidebar_pending_q"] = s
                        st.rerun()
        except Exception:
            pass

    # ── Input form ────────────────────────────────────────────────────────
    with st.form(key=f"sidebar_form_{current_page}", clear_on_submit=True):
        user_input = st.text_input(
            "Ask anything...",
            placeholder="e.g. Which model is best?",
            label_visibility="collapsed",
        )
        col1, col2 = st.columns([4, 1])
        with col1:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)
        with col2:
            cleared = st.form_submit_button("🗑️", use_container_width=True)

    if cleared:
        st.session_state["sidebar_chat_history"] = []
        st.rerun()

    pending  = st.session_state.pop("sidebar_pending_q", None)
    question = pending or (user_input.strip() if submitted and user_input.strip() else None)

    if question and project_id:
        st.session_state["sidebar_chat_history"].append({
            "role": "user", "content": question
        })

        with st.spinner("⚡ Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat/ask",
                    json={
                        "project_id":   project_id,
                        "dataset_id":   dataset_id,
                        "question":     question,
                        "history":      st.session_state["sidebar_chat_history"][:-1],
                        "model":        model,
                        "current_page": current_page,
                        "api_key":      "",  # loaded server-side from secrets.toml
                    },
                    timeout=60,
                )
                answer = resp.json().get("answer", "Could not generate response.")
            except Exception as e:
                answer = f"❌ {str(e)}"

        st.session_state["sidebar_chat_history"].append({
            "role": "assistant", "content": answer
        })
        st.rerun()

    st.page_link("pages/11_Chat.py", label="Open full chat →", icon="💬")
