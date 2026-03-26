# pages/11_Chat.py
import bootstrap  # noqa: F401
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from components.sidebar import render_sidebar
import streamlit as st
import requests
import json

st.set_page_config(page_title="ModelForge • AI Assistant", layout="wide")
from mf_theme import MF_CSS
st.markdown(MF_CSS, unsafe_allow_html=True)
API_BASE = "http://127.0.0.1:8000"

# ---------------- SIDEBAR ----------------
render_sidebar(current_page="AI Chat") 
# ---------------- AUTH ----------------
if not st.session_state.get("user_id"):
    st.switch_page("pages/0_Master.py")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0a0d14,#16232d,#1a3848);
            padding:20px 24px;border-radius:12px;margin-bottom:16px;
            border:1px solid rgba(255,255,255,0.06);">
  <h2 style="margin:0;color:white;">💬 ModelForge AI Assistant</h2>
  <p style="margin:6px 0 0 0;color:#cfe7ff;font-size:14px;">
    Powered by Groq — responses in 1-3 seconds ⚡<br>
    Ask anything about your experiments, models, data science, or ML concepts.
  </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("project_id"):
    st.info("📁 Select a project first.")
    st.stop()

project_id = st.session_state["project_id"]
dataset_id = st.session_state.get("dataset_id")
api_key = ""

# ── Init history ──────────────────────────────────────────────────────────
if "full_chat_history" not in st.session_state:
    st.session_state["full_chat_history"] = []

# ── Suggestions ───────────────────────────────────────────────────────────
if not st.session_state["full_chat_history"]:
    st.subheader("💡 Suggested Questions")
    try:
        sugg = requests.get(
            f"{API_BASE}/chat/suggestions/{project_id}",
            timeout=5,
        ).json().get("suggestions", [])
        cols = st.columns(2)
        for i, s in enumerate(sugg[:6]):
            with cols[i % 2]:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["full_pending_q"] = s
                    st.rerun()
    except Exception:
        pass
    st.divider()

# ── Display history ───────────────────────────────────────────────────────
for msg in st.session_state["full_chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# STREAMING RESPONSE HELPER
# =====================================================
def stream_response(question: str):
    """Stream response from Groq via SSE and display word by word."""

    st.session_state["full_chat_history"].append({
        "role": "user", "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={
                    "project_id":   project_id,
                    "dataset_id":   dataset_id,
                    "question":     question,
                    "history":      st.session_state["full_chat_history"][:-1],
                    "model":        st.session_state.get("full_chat_model", "llama-3.3-70b-versatile"),
                    "current_page": "Chat",
                    "api_key":      api_key,
                },
                stream=True,
                timeout=60,
            ) as resp:
                for line in resp.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                token = json.loads(data).get("token", "")
                                full_response += token
                                placeholder.markdown(full_response + "▌")
                            except Exception:
                                pass

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.markdown(full_response)

    st.session_state["full_chat_history"].append({
        "role": "assistant", "content": full_response
    })


# ── Handle suggestion click ───────────────────────────────────────────────
pending = st.session_state.pop("full_pending_q", None)
if pending:
    stream_response(pending)
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about your experiments, models, data science..."):
    stream_response(question)
    st.rerun()
