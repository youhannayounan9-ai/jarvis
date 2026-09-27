"""
ui/dashboard.py
───────────────
JARVIS Web Dashboard (Streamlit).

v0.11: the dashboard is a *client* of the JARVIS REST API — it no longer
wires an in-process agent runtime. Set JARVIS_API_URL to point at a running
API server (see deploy/README.md); the dashboard can then run on a different
machine than the agent runtime.

Backward compatibility: if JARVIS_API_URL is empty, the dashboard falls back
to the legacy in-process runtime (same process builds build_runtime()) so
existing single-machine setups keep working.

Run with: streamlit run ui/dashboard.py
"""

import os
from typing import Any

import streamlit as st

from jarvis.config import settings

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="JARVIS Dashboard", page_icon="🧠", layout="wide")


# ── Backend abstraction (API client first, legacy runtime fallback) ──────────


class ApiBackend:
    """Talks to a running JARVIS API server (the service surface)."""

    mode = "api"

    def __init__(self) -> None:
        from jarvis.api.client import JarvisClient

        self._client = JarvisClient(
            base_url=settings.JARVIS_API_URL or "http://127.0.0.1:8000",
            api_key=settings.JARVIS_CLIENT_API_KEY or None,
        )

    def health(self) -> dict[str, Any]:
        return self._client.health()

    def start_session(self) -> str:
        return self._client.create_session()

    def chat(self, session_id: str, message: str) -> str:
        result = self._client.chat(message, session_id)
        return result.get("response", "")

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self._client.history(session_id)

    def message_count(self, session_id: str) -> int:
        return len(self.history(session_id))

    def pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        return self._client.get_confirmation(session_id)

    def resolve_confirmation(self, session_id: str, confirmed: bool) -> str:
        body = self._client.resolve_confirmation(session_id, confirmed)
        return body.get("response", "")


class LegacyBackend:
    """In-process runtime (single-machine fallback; pre-v0.11 behavior)."""

    mode = "legacy"

    def __init__(self) -> None:
        from jarvis.runtime import build_runtime

        self._runtime = build_runtime()

    def start_session(self) -> str:
        return self._runtime.start_session()

    def chat(self, session_id: str, message: str) -> str:
        return self._runtime.chat(session_id, message)

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self._runtime.store.load_history(session_id)

    def message_count(self, session_id: str) -> int:
        return self._runtime.store.message_count(session_id)

    def pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        return self._runtime.get_pending_confirmation(session_id)

    def resolve_confirmation(self, session_id: str, confirmed: bool) -> str:
        return self._runtime.handle_confirmation(session_id, confirmed)


@st.cache_resource
def get_backend() -> Any:
    """API mode when JARVIS_API_URL is set; legacy in-process runtime otherwise."""
    if settings.JARVIS_API_URL:
        return ApiBackend()
    return LegacyBackend()


backend = get_backend()

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = backend.start_session()


def start_new_session() -> None:
    st.session_state.session_id = backend.start_session()


# ── UI Layout ─────────────────────────────────────────────────────────────────
mode_badge = (
    f"🟢 API: `{settings.JARVIS_API_URL}`"
    if backend.mode == "api"
    else "🟡 legacy in-process runtime (set JARVIS_API_URL to use the API)"
)
st.sidebar.title("JARVIS Dashboard")
st.sidebar.caption(mode_badge)
st.sidebar.button("New Session", on_click=start_new_session)

st.sidebar.markdown("### Active Session")
st.sidebar.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
try:
    msg_count = backend.message_count(st.session_state.session_id)
except Exception as e:
    st.sidebar.error(f"Backend unreachable: {e}")
    msg_count = 0
st.sidebar.caption(f"Messages: {msg_count}")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.sidebar.file_uploader(
    "Upload an image for JARVIS to see",
    type=["jpg", "png", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}",
)

saved_file_path = None
if uploaded_file is not None:
    from pathlib import Path

    uploads_dir = Path("jarvis_data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_file_path = (uploads_dir / uploaded_file.name).resolve()
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Image uploaded successfully.")

st.title("JARVIS Assistant")

try:
    history = backend.history(st.session_state.session_id)
except Exception:
    history = []

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
                response = backend.chat(st.session_state.session_id, user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
                response = None

try:
    pending = backend.pending_confirmation(st.session_state.session_id)
except Exception:
    pending = None
if pending:
    st.error("⚠️ HIGH-RISK ACTION REQUIRES APPROVAL")
    st.markdown(f"**Tool:** `{pending['tool_name']}`\n\n**Risk Level:** `{pending['risk_level']}`")
    st.markdown("**Arguments:**")
    st.code(pending["tool_args"], language="json")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", type="primary"):
            response = backend.resolve_confirmation(st.session_state.session_id, True)
            st.success(response)
            st.rerun()
    with col2:
        if st.button("❌ Deny Action"):
            response = backend.resolve_confirmation(st.session_state.session_id, False)
            st.error(response)
            st.rerun()

    # Load updated history to find the thought process for this turn
    try:
        updated_history = backend.history(st.session_state.session_id)
    except Exception:
        updated_history = []

    # Extract only the tools/thoughts from this specific interaction (after the user input)
    last_user_idx = -1
    for i in range(len(updated_history) - 1, -1, -1):
        if updated_history[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx != -1:
        turn_history = updated_history[last_user_idx + 1 :]

        with st.expander("🧠 Agent Thought Process", expanded=False):
            for msg in turn_history:
                if msg.get("tool_calls"):
                    for call in msg["tool_calls"]:
                        fn = call.get("function", {})
                        st.info(
                            f"**🛠️ Calling Tool:** `{fn.get('name')}`\n```json\n{fn.get('arguments')}\n```"
                        )
                elif msg.get("role") == "tool":
                    st.success(f"**📄 Tool Result ({msg.get('name')}):**\n```\n{msg.get('content')}\n```")
