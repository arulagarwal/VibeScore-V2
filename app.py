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
        with st.status("Curating your playlist...", expanded=True) as status:
            st.write("🔎 Step 1 — Semantic retrieval (ChromaDB)...")
            retrieved_songs = st.session_state.agent.knowledge_base.retrieve(
                prompt,
                k=st.session_state.agent.mode.retrieval_k,
                diversity_penalty=st.session_state.agent.mode.diversity_penalty,
            )

            st.write("🧠 Step 2 — Gemini 3.0 Flash planning playlist...")
            reply, is_clean = st.session_state.agent.chat(
                prompt, history, prefetched_songs=retrieved_songs
            )

            st.write("🛡️ Step 3 — HallucinationGuardrail validating titles...")
            status.update(label="Playlist ready", state="complete", expanded=False)

        st.markdown(reply)

        with st.expander("🔍 Developer Debug Info"):
            st.subheader("ChromaDB Retrieval")
            st.caption(
                f"Mode: **{st.session_state.active_mode}** | "
                f"k={st.session_state.agent.mode.retrieval_k} | "
                f"Diversity penalty: {st.session_state.agent.mode.diversity_penalty}"
            )
            for i, song in enumerate(retrieved_songs, 1):
                st.markdown(
                    f"**{i}. {song['title']}** by {song['artist']}  \n"
                    f"Genre: `{song['genre']}` | Mood: `{song['mood']}` | "
                    f"Energy: `{song['energy']}` | Valence: `{song.get('valence', 'N/A')}` | "
                    f"Danceability: `{song.get('danceability', 'N/A')}` | "
                    f"Acousticness: `{song.get('acousticness', 'N/A')}` | "
                    f"Tempo: `{song.get('tempo_bpm', 'N/A')} BPM`  \n"
                    f"Tags: `{song.get('mood_tags', 'N/A')}`"
                )

            st.divider()
            st.subheader("HallucinationGuardrail")
            if is_clean:
                st.success("PASS — All referenced titles verified against songs.csv")
            else:
                st.warning("FLAGGED — Response contained unverified title references. Warning note appended.")

    st.session_state.messages.append({"role": "assistant", "content": reply})
