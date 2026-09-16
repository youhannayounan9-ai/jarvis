"""
ui/dashboard.py
───────────────
Web UI Dashboard for JARVIS v0.6.0 using Streamlit.
Run with: streamlit run ui/dashboard.py
"""

import os
from pathlib import Path

import streamlit as st
from typing import Any

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.memory.vector_store import get_vector_store
from jarvis.tools.registry import ToolRegistry
from jarvis.tools import (
    CalculatorTool,
    GetCurrentDatetimeTool,
    ListDirectoryTool,
    ReadFileTool,
    RecallFactsTool,
    RememberFactTool,
    WebSearchTool,
    WikipediaSummaryTool,
    WriteFileTool,
    VisionAnalyzeTool,
    WebScrapeTool,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="JARVIS v0.6 Dashboard", page_icon="🧠", layout="wide")


# ── Initialization ────────────────────────────────────────────────────────────
@st.cache_resource
def get_jarvis_components():
    store = SessionStore()
    get_vector_store()  # initialize vector DB
    
    registry = ToolRegistry()
    registry.register(GetCurrentDatetimeTool())
    registry.register(WebSearchTool())
    registry.register(WikipediaSummaryTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(CalculatorTool())
    registry.register(RememberFactTool())
    registry.register(RecallFactsTool())
    registry.register(VisionAnalyzeTool())
    registry.register(WebScrapeTool())
    
    guard = PermissionGuard()
    orchestrator = Orchestrator(store, registry, guard)
    
    return orchestrator, store


orchestrator, store = get_jarvis_components()


# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = store.create_session()
    get_vector_store().set_session(st.session_state.session_id)


def start_new_session():
    st.session_state.session_id = store.create_session()
    get_vector_store().set_session(st.session_state.session_id)


# ── UI Layout ─────────────────────────────────────────────────────────────────
st.sidebar.title("JARVIS v0.6 Dashboard")
st.sidebar.button("New Session", on_click=start_new_session)
st.sidebar.caption(f"Current Session: `{st.session_state.session_id[:8]}...`")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.sidebar.file_uploader(
    "Upload an image for JARVIS to see", 
    type=["jpg", "png", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}"
)

saved_file_path = None
if uploaded_file is not None:
    uploads_dir = Path("jarvis_data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_file_path = (uploads_dir / uploaded_file.name).resolve()
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Image uploaded successfully.")

st.title("JARVIS Assistant")

history = store.load_history(st.session_state.session_id)

# Filter history for chat display
chat_messages = []
for msg in history:
    if msg.get("role") in ("user", "assistant"):
        # Skip assistant messages that were just tool calls (no content)
        if msg.get("role") == "assistant" and not msg.get("content"):
            continue
        chat_messages.append(msg)

for msg in chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))

# Chat Input
if user_input := st.chat_input("How can I help you?"):
    if saved_file_path is not None:
        user_input += f"\n[Attached Image: {saved_file_path}]"
        st.session_state.uploader_key += 1
        
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("JARVIS is planning and executing..."):
            try:
                response = orchestrator.chat(st.session_state.session_id, user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
                response = None

    # Load updated history to find the thought process for this turn
    updated_history = store.load_history(st.session_state.session_id)
    
    # Extract only the tools/thoughts from this specific interaction (after the user input)
    # Find the index of the last user message
    last_user_idx = -1
    for i in range(len(updated_history) - 1, -1, -1):
        if updated_history[i].get("role") == "user":
            last_user_idx = i
            break
            
    if last_user_idx != -1:
        turn_history = updated_history[last_user_idx + 1:]
        
        with st.expander("🧠 Agent Thought Process", expanded=False):
            for msg in turn_history:
                if msg.get("tool_calls"):
                    for call in msg["tool_calls"]:
                        fn = call.get("function", {})
                        st.info(f"**🛠️ Calling Tool:** `{fn.get('name')}`\n```json\n{fn.get('arguments')}\n```")
                elif msg.get("role") == "tool":
                    st.success(f"**📄 Tool Result ({msg.get('name')}):**\n```\n{msg.get('content')}\n```")
