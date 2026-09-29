"""
tests/test_context_management.py
────────────────────────────────
Integration tests ensuring context bounds and memory management work
end-to-end through the Orchestrator (LLM + tools mocked).
"""

from unittest.mock import patch, MagicMock

from jarvis.config import settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.core.permissions import PermissionGuard


# ── Shared mocks (mirror the LiteLLM response shape) ─────────────────────────

class Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, id, function):
        self.id = id
        self.function = function


class Message:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class Choice:
    def __init__(self, message):
        self.message = message


class Response:
    def __init__(self, choices):
        self.choices = choices


def _make_orchestrator():
    store = SessionStore()
    registry = ToolRegistry()
    guard = PermissionGuard()
    orchestrator = Orchestrator(store, registry, guard)
    return orchestrator, store, registry, guard


def test_active_tool_truncation():
    """Huge tool outputs are clamped before reaching the LLM; full payload stays in SQLite."""
    orchestrator, store, registry, guard = _make_orchestrator()

    huge_payload = "A" * 20000

    responses = [
        # Round 1: LLM calls web_search
        Response([Choice(Message("assistant", None, [ToolCall("call_1", Function("web_search", "{}"))]))]),
        # Round 2: LLM answers the user (stops the ReAct loop)
        Response([Choice(Message("assistant", "Final answer based on search."))]),
        # v0.25: the fast path then makes one evidence-grounded wrap-up call
        Response([Choice(Message("assistant", "Final answer based on search."))]),
    ]

    def mock_chat_completion(messages, tools=None, **kwargs):
        return responses.pop(0)

    import asyncio

    async def mock_dispatch_async(*args, **kwargs):
        return huge_payload

    with patch("jarvis.core.orchestrator.chat_completion", side_effect=mock_chat_completion) as mock_llm:
        with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
            with patch.object(guard, "is_allowed", return_value=True):
                with patch.object(guard, "require_confirmation", return_value=False):
                    with patch.object(orchestrator, "route_intent", return_value="simple"):
                        orchestrator.chat("session_123", "Search for something huge")

    # The LAST (evidence-grounded wrap-up) call must carry the clamped tool
    # result — the v0.25 evidence item is clamp-bounded like every raw copy.
    # The evidence block travels on the final synthesis message (v0.25: the
    # FINAL USER message — live-verified; scan all roles).
    last_call_messages = mock_llm.call_args[1]["messages"]
    evidence_blob = " ".join(
        str(m["content"]) for m in last_call_messages
    )

    assert len(evidence_blob) < len(huge_payload)
    # The evidence ITEM is clamp-bounded; the blob also carries the base
    # system prompt, so allow that overhead.
    assert len(evidence_blob) <= settings.max_tool_output_chars + 3500
    assert "characters omitted" in evidence_blob
    # Head is preserved (payload was uniform 'A's, so start must be present)
    assert "AAA" in evidence_blob

    # The database should have saved the full payload
    row = store._conn.execute("SELECT content FROM messages WHERE role='tool'").fetchone()
    assert len(row["content"]) == 20000


def test_long_session_context_stays_bounded():
    """After many turns, the LLM-facing history is windowed, anchored, and summarized."""
    orchestrator, store, registry, guard = _make_orchestrator()

    def mock_chat_completion(messages, tools=None, **kwargs):
        return Response([Choice(Message("assistant", f"ack-{len(messages)}"))])

    with patch("jarvis.core.orchestrator.chat_completion", side_effect=mock_chat_completion):
        with patch.object(orchestrator, "route_intent", return_value="simple"):
            orchestrator.chat("session_long", "Remember that my project codename is NIGHTINGALE")
            for i in range(30):
                orchestrator.chat("session_long", f"Follow-up question number {i}")

    # Inspect the messages the LLM saw on the LAST call — using the same wide
    # fetch horizon the orchestrator uses (narrow default would pre-drop the
    # first task before the ContextManager ever sees it).
    window = store.load_history(
        "session_long", limit=settings.max_context_messages * 3
    )

    # SQLite keeps everything (30 turns * 2 messages each + first turn)
    assert store.message_count("session_long") >= 60

    # But the context manager windows what the LLM sees
    cm = orchestrator._context
    built = cm.build_messages(
        system_prompts=[{"role": "system", "content": "SYS"}],
        history=window,
        user_input="another question",
    )
    system_msgs = [m for m in built if m["role"] == "system"]
    joined = "\n".join(str(m.get("content")) for m in system_msgs)

    # Anchor must re-inject the original task
    assert "NIGHTINGALE" in joined
    # Rolling summary must reference earlier requests
    assert "[Context summary]" in joined

    # The non-system message count stays within the window budget
    non_system = [m for m in built if m["role"] != "system"]
    assert len(non_system) <= cm.max_messages + 1  # +1 = current user input


def test_context_window_never_grows_unbounded():
    """Simulated long session: the LLM message list stays under a hard ceiling."""
    orchestrator, store, registry, guard = _make_orchestrator()

    def mock_chat_completion(messages, tools=None, **kwargs):
        return Response([Choice(Message("assistant", "ok"))])

    with patch("jarvis.core.orchestrator.chat_completion", side_effect=mock_chat_completion) as mock_llm:
        with patch.object(orchestrator, "route_intent", return_value="simple"):
            for i in range(40):
                orchestrator.chat("session_bound", f"question {i} about topic {i}")

    # Every LLM call must stay under a sane bound regardless of session length
    for call in mock_llm.call_args_list:
        msgs = call[1]["messages"]
        # system prompts (2) + anchor + summary + window + tool-policy block +
        # current user input. The v0.21 policy block is a single fixed system
        # message per call (never per round), so the bound stays constant.
        assert len(msgs) <= 2 + 2 + orchestrator._context.max_messages + 2


def test_route_intent_classification():
    """The heuristic router keeps single-intent queries fast and multi-step ones planned."""
    orchestrator, _, _, _ = _make_orchestrator()

    simple_cases = [
        "what time is it?",
        "hello",
        "calculate 25 * 4",
        "remember that my favorite color is blue",
    ]
    complex_cases = [
        "Search for AI news and then write a summary to news.md",
        "First scrape https://example.com, then after that summarize it",
        "Remember my name is Alex. Also calculate 12*12. Then search for weather today and write it to a file",
    ]

    for text in simple_cases:
        assert orchestrator.route_intent(text) == "simple", text

    for text in complex_cases:
        assert orchestrator.route_intent(text) == "complex", text
