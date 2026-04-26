import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from agent_system import (
    SongKnowledgeBase,
    BalancedMode,
    DeepDiveMode,
    VibeScoreAgent,
)

st.set_page_config(page_title="VibeScore 2.0", page_icon="🎵", layout="centered")
st.title("VibeScore 2.0")
st.caption("Agentic music recommendations powered by Gemini")

# ── API Key Gate ──────────────────────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if not st.session_state.api_key:
    st.subheader("Enter your Gemini API Key to get started")
    key_input = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    if st.button("Start Chatting", type="primary") and key_input.strip():
        st.session_state.api_key = key_input.strip()
        st.rerun()
    st.stop()

# ── Sidebar: Mode Selector + Session Controls ─────────────────────────────────
MODES = {"Balanced": BalancedMode(), "Deep Dive": DeepDiveMode()}

with st.sidebar:
    st.header("Settings")
    selected_mode_name = st.selectbox("Scoring Mode", list(MODES.keys()))
    st.caption(
        "**Balanced** — concise picks across genre, mood & energy (k=5)\n\n"
        "**Deep Dive** — rich analysis of valence, danceability & acousticness (k=10)"
    )
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Log Out"):
        for key in ("api_key", "agent", "active_mode", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

# ── Initialize Agent (once per session, or when mode changes) ─────────────────
if "agent" not in st.session_state or st.session_state.get("active_mode") != selected_mode_name:
    kb = SongKnowledgeBase(api_key=st.session_state.api_key)
    st.session_state.agent = VibeScoreAgent(
        api_key=st.session_state.api_key,
        knowledge_base=kb,
        mode=MODES[selected_mode_name],
    )
    st.session_state.active_mode = selected_mode_name

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Tell me what kind of music you're feeling..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass history without the current message — agent.chat() appends it internally
    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("Finding your vibe..."):
            reply = st.session_state.agent.chat(prompt, history)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
