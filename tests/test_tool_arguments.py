"""
tests/test_tool_arguments.py
────────────────────────────
v0.21 Part H: tool-argument reliability regression tests.

Locks in the dispatch-time validation boundary so "selection improved"
can never mean "validation loosened":

  - unknown tools are rejected with the real available surface
  - malformed JSON arguments are rejected
  - missing required fields are rejected (field named in the error)
  - extra/unknown fields are rejected (pydantic extra="forbid")
  - wrong-typed arguments are rejected (no silent coercion)
  - a validation failure flows into bounded self-correction and recovery

All errors are returned as "ERROR: ..." strings per the tool contract —
never exceptions, never fabricated success.
"""

from unittest.mock import patch

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.tools import CalculatorTool, WebSearchTool
from jarvis.tools.registry import ToolRegistry

from tests.test_tool_policy import (  # shared mock shapes
    Choice,
    Message,
    Response,
    ToolCall,
    Function,
    fake_text_response,
    fake_tool_response,
)


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool())
    return reg


def _is_error(result: str) -> bool:
    return isinstance(result, str) and result.startswith("ERROR:")


class TestUnknownToolRejection:
    def test_unknown_tool_lists_available_surface(self):
        reg = _registry()
        result = reg.dispatch("execute_python_code", "{}")
        assert _is_error(result)
        assert "Unknown tool" in result
        assert "calculator" in result  # the REAL available surface
        assert "web_search" in result

    def test_near_miss_tool_name_rejected(self):
        """A hallucinated near-miss must never fuzzy-match a real tool."""
        reg = _registry()
        for name in ("Calculator", "calc", "calculator ", "websearch", "calculate"):
            result = reg.dispatch(name, "{}")
            assert _is_error(result), name


class TestMalformedArguments:
    def test_non_json_arguments_rejected(self):
        reg = _registry()
        result = reg.dispatch("calculator", "not json at all")
        assert _is_error(result)
        assert "Could not parse tool arguments as JSON" in result

    def test_json_array_arguments_rejected(self):
        reg = _registry()
        result = reg.dispatch("calculator", '["expression", "2+2"]')
        assert _is_error(result)

    def test_missing_required_field_rejected_and_named(self):
        reg = _registry()
        result = reg.dispatch("calculator", "{}")
        assert _is_error(result)
        assert "expression" in result  # the missing field is named

    def test_extra_fields_rejected(self):
        """extra='forbid': injected/invented arguments never reach run()."""
        reg = _registry()
        result = reg.dispatch(
            "calculator", '{"expression": "2+2", "python_code": "import os"}'
        )
        assert _is_error(result)
        assert "python_code" in result

    def test_wrong_type_rejected_not_coerced(self):
        """A dict where a string belongs must fail loudly, not coerce."""
        reg = _registry()
        result = reg.dispatch("calculator", '{"expression": {"expr": "2+2"}}')
        assert _is_error(result)


class TestArgumentErrorsAreGraceful:
    def test_errors_are_strings_never_exceptions(self):
        reg = _registry()
        for name, args in (
            ("no_such_tool", "{}"),
            ("calculator", "{{{{"),
            ("calculator", "{}"),
            ("web_search", '{"query": ""}'),
        ):
            result = reg.dispatch(name, args)
            assert isinstance(result, str), (name, args)
            if name != "web_search":  # empty query is a tool-level guard
                assert _is_error(result), (name, args, result)

    def test_tool_level_empty_arg_guard(self):
        reg = _registry()
        result = reg.dispatch("web_search", '{"query": "   "}')
        assert _is_error(result)
        assert "query must not be empty" in result

    def test_valid_arguments_still_execute(self):
        reg = _registry()
        result = reg.dispatch("calculator", '{"expression": "12 * (4 + 3) / 2.5"}')
        assert not _is_error(result)
        assert "33.6" in result


class TestSelfCorrectionAfterValidationFailure:
    """Wrong tool arguments → error observation → corrected retry → answer."""

    def test_retry_after_invalid_arguments(self):
        store = SessionStore()
        reg = _registry()
        guard = PermissionGuard()
        orch = Orchestrator(store, reg, guard)
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return reg.dispatch(tool_name, tool_args)  # REAL validation

        responses = [
            # Round 1: model forgets the required field.
            fake_tool_response([("calculator", "{}")]),
            # Recovery round: corrected arguments.
            fake_tool_response([("calculator", '{"expression": "144 / 12"}')]),
            # Final grounded answer.
            fake_text_response("144 / 12 = 12."),
        ]

        with patch("jarvis.core.orchestrator.chat_completion", side_effect=responses):
            with patch.object(reg, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_args1", "What is 144 divided by 12?")

        assert dispatched == ["calculator", "calculator"]
        assert answer == "144 / 12 = 12."
        # The error observation must have been shown to the model (it is in
        # the persisted history as the tool result of the failed call).
        rows = store._conn.execute(
            "SELECT content FROM messages WHERE role='tool'"
        ).fetchall()
        assert any("Invalid arguments" in r["content"] for r in rows)
        store.close()

    def test_correction_is_bounded(self):
        """Repeated validation failures stop after the bounded budget."""
        from jarvis.core.orchestrator import MAX_SELF_CORRECTION_ATTEMPTS

        store = SessionStore()
        reg = _registry()
        guard = PermissionGuard()
        orch = Orchestrator(store, reg, guard)
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return reg.dispatch(tool_name, tool_args)

        # Always-invalid arguments; the loop must give up, not spin forever.
        bad_call = fake_tool_response([("calculator", "{}")])
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[bad_call] * (MAX_SELF_CORRECTION_ATTEMPTS + 3),
        ):
            with patch.object(reg, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_args2", "What is 144 divided by 12?")

        # Bounded: initial round + exactly MAX_SELF_CORRECTION_ATTEMPTS retries.
        assert len(dispatched) == 1 + MAX_SELF_CORRECTION_ATTEMPTS
        assert answer  # a wrap-up answer was still produced
        store.close()
