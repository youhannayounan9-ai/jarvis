"""
tests/test_self_correction.py
──────────────────────────────
Tests for the orchestrator's self-correction / observation loop.

All LLM and tool calls are mocked — no Ollama or network required.
"""

from unittest.mock import MagicMock, patch, call

import pytest

from jarvis.core.orchestrator import (
    Orchestrator,
    MAX_SELF_CORRECTION_ATTEMPTS,
    MAX_TOOL_ROUNDS,
    _is_tool_error,
)
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry


# ── Shared test helpers ────────────────────────────────────────────────────────

def _make_tool_response(name: str, call_id: str = "call_1", args: str = '{"query": "test"}'):
    """Return a mocked LLM response that requests a single tool call."""
    class Fn:
        def __init__(self):
            self.name = name
            self.arguments = args
    class TC:
        def __init__(self):
            self.id = call_id
            self.function = Fn()
    class Msg:
        role = "assistant"
        content = None
        tool_calls = [TC()]
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]
    return Resp()


def _make_text_response(text: str):
    """Return a mocked LLM response with a text answer (no tool calls)."""
    class Msg:
        role = "assistant"
        content = text
        tool_calls = None
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]
    return Resp()


def _build_orchestrator():
    store = MagicMock(spec=SessionStore)
    store.load_history.return_value = []
    store.create_session.return_value = "sess"
    registry = ToolRegistry()
    guard = PermissionGuard()
    return Orchestrator(store, registry, guard), store


# ── _is_tool_error ─────────────────────────────────────────────────────────────

class TestIsToolError:
    def test_error_prefix_returns_true(self):
        assert _is_tool_error("ERROR: something went wrong") is True

    def test_success_returns_false(self):
        assert _is_tool_error("4") is False

    def test_empty_returns_false(self):
        assert _is_tool_error("") is False

    def test_confirmation_not_an_error(self):
        assert _is_tool_error("ACTION_REQUIRES_CONFIRMATION: ...") is False

    def test_partial_prefix_not_an_error(self):
        assert _is_tool_error("not an ERROR: msg") is False


# ── Self-correction loop ───────────────────────────────────────────────────────

class TestSelfCorrectionLoop:
    """Test _run_react directly, bypassing the full chat() flow."""

    def test_successful_tool_call_no_correction(self):
        """Normal path: tool succeeds → no correction logic triggered."""
        orc, store = _build_orchestrator()

        responses = [
            _make_tool_response("web_search"),
            _make_text_response("The answer is 42."),
        ]
        with patch("jarvis.core.orchestrator.chat_completion", side_effect=responses):
            with patch.object(orc._registry, "dispatch", return_value="result ok"):
                with patch.object(orc._guard, "is_allowed", return_value=True):
                    with patch.object(orc._guard, "require_confirmation", return_value=False):
                        with patch.object(orc._registry, "get_tool_risk_level", return_value="SAFE"):
                            text, rounds = orc._run_react(
                                session_id="s1",
                                messages=[{"role": "user", "content": "hello"}],
                                tool_schemas=[],
                                max_rounds=3,
                            )
        assert text == "The answer is 42."
        assert rounds == 1  # one tool round, then a text answer

    def test_tool_error_triggers_correction(self):
        """Single tool failure → LLM gets a second chance and recovers."""
        orc, store = _build_orchestrator()

        responses = [
            _make_tool_response("web_search", call_id="call_1"),  # round 1: tool called
            _make_text_response("I recovered and here is the answer."),  # round 2: text
        ]
        with patch("jarvis.core.orchestrator.chat_completion", side_effect=responses):
            with patch.object(orc._registry, "dispatch", return_value="ERROR: network failure"):
                with patch.object(orc._guard, "is_allowed", return_value=True):
                    with patch.object(orc._guard, "require_confirmation", return_value=False):
                        with patch.object(orc._registry, "get_tool_risk_level", return_value="SAFE"):
                            text, rounds = orc._run_react(
                                session_id="s2",
                                messages=[{"role": "user", "content": "search for X"}],
                                tool_schemas=[],
                                max_rounds=2,
                            )
        assert "recovered" in text
        assert rounds == 1  # only the failing round counts; recovery was a text answer

    def test_correction_limit_stops_loop(self):
        """When every tool call keeps failing, the loop stops after the limit."""
        orc, store = _build_orchestrator()

        # LLM always requests a tool, tool always errors
        # After MAX_SELF_CORRECTION_ATTEMPTS + 1 failures → wrap-up call
        tool_responses = [
            _make_tool_response("web_search", call_id=f"call_{i}")
            for i in range(MAX_SELF_CORRECTION_ATTEMPTS + 2)
        ]
        wrap_up = _make_text_response("I could not complete the task.")
        all_responses = tool_responses + [wrap_up]

        call_count = 0

        def llm_side_effect(**kwargs):
            nonlocal call_count
            resp = all_responses[min(call_count, len(all_responses) - 1)]
            call_count += 1
            return resp

        with patch("jarvis.core.orchestrator.chat_completion", side_effect=llm_side_effect):
            with patch.object(orc._registry, "dispatch", return_value="ERROR: permanent failure"):
                with patch.object(orc._guard, "is_allowed", return_value=True):
                    with patch.object(orc._guard, "require_confirmation", return_value=False):
                        with patch.object(orc._registry, "get_tool_risk_level", return_value="SAFE"):
                            text, rounds = orc._run_react(
                                session_id="s3",
                                messages=[{"role": "user", "content": "do something"}],
                                tool_schemas=[],
                                max_rounds=2,
                            )
        # Correction attempts must be capped
        assert rounds <= MAX_TOOL_ROUNDS + MAX_SELF_CORRECTION_ATTEMPTS

    def test_successful_round_resets_error_counter(self):
        """A successful tool call followed by a failure starts fresh (no leftover error count)."""
        orc, store = _build_orchestrator()

        # round1: tool ok, round2: tool error → should get 1 correction attempt (not 0)
        responses = [
            _make_tool_response("tool_a", call_id="call_a"),  # success
            _make_tool_response("tool_b", call_id="call_b"),  # failure
            _make_text_response("Recovered after fresh failure."),
        ]

        dispatch_results = ["good result", "ERROR: oops"]
        dispatch_iter = iter(dispatch_results)

        with patch("jarvis.core.orchestrator.chat_completion", side_effect=responses):
            with patch.object(orc._registry, "dispatch", side_effect=dispatch_iter):
                with patch.object(orc._guard, "is_allowed", return_value=True):
                    with patch.object(orc._guard, "require_confirmation", return_value=False):
                        with patch.object(orc._registry, "get_tool_risk_level", return_value="SAFE"):
                            text, rounds = orc._run_react(
                                session_id="s4",
                                messages=[{"role": "user", "content": "do two things"}],
                                tool_schemas=[],
                                max_rounds=3,
                            )
        assert "Recovered" in text

    def test_no_tool_calls_returns_immediately(self):
        """If the LLM answers with text on the first call, rounds_used stays 0."""
        orc, store = _build_orchestrator()

        with patch("jarvis.core.orchestrator.chat_completion",
                   return_value=_make_text_response("Direct answer.")):
            text, rounds = orc._run_react(
                session_id="s5",
                messages=[{"role": "user", "content": "what is 2+2"}],
                tool_schemas=[],
                max_rounds=3,
            )
        assert text == "Direct answer."
        assert rounds == 0

    def test_zero_budget_skips_tools(self):
        """max_rounds=0 → one tool-free call immediately, no tool dispatch possible."""
        orc, store = _build_orchestrator()

        with patch("jarvis.core.orchestrator.chat_completion",
                   return_value=_make_text_response("Best-effort answer.")):
            text, rounds = orc._run_react(
                session_id="s6",
                messages=[{"role": "user", "content": "anything"}],
                tool_schemas=[],
                max_rounds=0,
            )
        assert text == "Best-effort answer."
        assert rounds == 0

    def test_correction_hint_injected_into_messages(self):
        """When a tool fails, a recovery hint system message must be added to the context."""
        orc, store = _build_orchestrator()

        captured_messages = []

        def capture_llm(**kwargs):
            captured_messages.append(list(kwargs.get("messages", [])))
            if len(captured_messages) == 1:
                return _make_tool_response("bad_tool")
            return _make_text_response("recovered")

        with patch("jarvis.core.orchestrator.chat_completion", side_effect=capture_llm):
            with patch.object(orc._registry, "dispatch", return_value="ERROR: bad args"):
                with patch.object(orc._guard, "is_allowed", return_value=True):
                    with patch.object(orc._guard, "require_confirmation", return_value=False):
                        with patch.object(orc._registry, "get_tool_risk_level", return_value="SAFE"):
                            orc._run_react(
                                session_id="s7",
                                messages=[{"role": "user", "content": "do it"}],
                                tool_schemas=[],
                                max_rounds=2,
                            )

        # The second LLM call should have received the correction hint
        assert len(captured_messages) >= 2
        second_call_messages = captured_messages[1]
        hint_messages = [
            m for m in second_call_messages
            if m.get("role") == "system" and "recover" in m.get("content", "").lower()
        ]
        assert len(hint_messages) >= 1, "No recovery hint injected into second LLM call"
