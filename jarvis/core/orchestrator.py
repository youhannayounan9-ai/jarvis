"""
jarvis/core/orchestrator.py
────────────────────────────
The Orchestrator — the brain of JARVIS.

Every user message flows through here. The orchestrator:
  1. Prepends the system prompt to the conversation.
  2. Calls the LLM with the current history + tool schemas.
  3. If the LLM requests tool calls, executes them (via PermissionGuard → ToolRegistry).
  4. Feeds tool results back to the LLM.
  5. Repeats until the LLM produces a plain text response (no more tool calls).
  6. Persists all messages to the session store.
  7. Returns the final text response.

This loop is the "ReAct" (Reason + Act) pattern:
  LLM thinks → decides to act → tool runs → LLM sees result → LLM thinks again.

Safety bounds:
  MAX_TOOL_ROUNDS caps how many tool-call / result cycles we allow per user
  message. This prevents infinite loops if the LLM gets stuck requesting tools.
"""

from typing import Any

from jarvis.config import settings
from jarvis.core.permissions import PermissionGuard
from jarvis.llm.client import chat_completion
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Maximum number of tool-call rounds per user message.
# If the LLM still hasn't given a plain text answer after this many rounds,
# we break the loop and return whatever content is available.
MAX_TOOL_ROUNDS = 5


class Orchestrator:
    """
    Central request handler. Receives user messages, runs the tool-calling
    loop, and returns the assistant's final text response.

    Args:
        session_store:   Persistence layer for conversation history.
        tool_registry:   Registry of available tools.
        permission_guard: Controls which tools may execute.
    """

    def __init__(
        self,
        session_store: SessionStore,
        tool_registry: ToolRegistry,
        permission_guard: PermissionGuard,
    ) -> None:
        self._store = session_store
        self._registry = tool_registry
        self._guard = permission_guard

    def chat(self, session_id: str, user_input: str) -> str:
        """
        Process one user message and return the assistant's response.

        Args:
            session_id:  The ID of the current conversation session.
            user_input:  The raw text typed by the user.

        Returns:
            The assistant's final plain-text response string.
        """
        # ── 1. Save the user's message ─────────────────────────────────────────
        user_message: dict[str, Any] = {"role": "user", "content": user_input}
        self._store.save_message(session_id, user_message)

        # ── 2. Build the full message list for the LLM ─────────────────────────
        # Load recent history (most-recent window — see SessionStore.load_history).
        history = self._store.load_history(session_id)

        # System prompt is injected at call time (not stored) so it can change
        # without a DB migration. A short memory cue reinforces follow-ups.
        memory_cue = (
            f"Short-term memory: {len(history)} message(s) from this session "
            "are included below. Use them to resolve references and follow-ups."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": settings.system_prompt},
            {"role": "system", "content": memory_cue},
            *history,
        ]

        tool_schemas = self._registry.get_schemas()

        # ── 3. Tool-calling loop ───────────────────────────────────────────────
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info("llm_call", session_id=session_id, round=round_num + 1)

            response = chat_completion(messages=messages, tools=tool_schemas)
            assistant_message = response.choices[0].message

            # Convert the response message to a dict for storage.
            assistant_dict = _message_to_dict(assistant_message)

            # Check if the LLM wants to call any tools.
            tool_calls = getattr(assistant_message, "tool_calls", None)

            if not tool_calls:
                # ── No tool calls → this is the final answer ──────────────────
                self._store.save_message(session_id, assistant_dict)
                final_text = assistant_message.content or ""
                log.info("response_ready", session_id=session_id, rounds=round_num + 1)
                return final_text

            # ── Tool calls requested → execute them all ────────────────────────
            log.info("tool_calls_requested", count=len(tool_calls), round=round_num + 1)

            # Save the assistant's tool-call message (no content, just tool_calls).
            self._store.save_message(session_id, assistant_dict)
            messages.append(assistant_dict)

            # Execute each requested tool and collect results.
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments  # JSON string
                tool_call_id = tool_call.id

                # ── Permission check (risk-aware) ─────────────────────────────
                risk_level = self._registry.get_tool_risk_level(tool_name)

                if self._guard.require_confirmation(tool_name, risk_level):
                    result = (
                        f"ERROR: Tool '{tool_name}' requires explicit user "
                        "confirmation, which is not yet supported in this "
                        "interface. Please inform the user that this action "
                        "cannot be performed automatically."
                    )
                    log.warning(
                        "tool_requires_confirmation",
                        tool=tool_name,
                        risk_level=risk_level,
                    )
                elif not self._guard.is_allowed(tool_name, risk_level):
                    result = (
                        f"ERROR: Tool '{tool_name}' is not permitted "
                        f"(Risk Level: {risk_level})."
                    )
                    log.warning(
                        "tool_blocked",
                        tool=tool_name,
                        risk_level=risk_level,
                    )
                else:
                    result = self._registry.dispatch(tool_name, tool_args)

                # Tool result message (OpenAI tool role format).
                tool_result_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
                self._store.save_message(session_id, tool_result_message)
                messages.append(tool_result_message)

        # ── 4. Safety: MAX_TOOL_ROUNDS exceeded ────────────────────────────────
        # Make one final call without tools to force a text response.
        log.warning("max_tool_rounds_exceeded", session_id=session_id)
        final_response = chat_completion(messages=messages, tools=None)
        final_message = final_response.choices[0].message
        final_dict = _message_to_dict(final_message)
        self._store.save_message(session_id, final_dict)
        return final_message.content or "I ran into an issue completing that request."


# ── Helpers ────────────────────────────────────────────────────────────────────

def _message_to_dict(message: Any) -> dict[str, Any]:
    """
    Convert a LiteLLM/OpenAI Message object to a plain dict.
    We need plain dicts for SQLite storage and for appending to the messages list.
    """
    d: dict[str, Any] = {"role": message.role}

    if message.content is not None:
        d["content"] = message.content

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        # Serialise tool_calls to a list of dicts (they may be objects).
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    return d
