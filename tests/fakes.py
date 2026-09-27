"""
tests/fakes.py
──────────────
Shared offline test doubles for the LiteLLM response shape.

One definition instead of four: every module that fakes chat_completion
responses imports from here. Usage:

    from tests.fakes import fake_text_response, fake_tool_response

    with patch("jarvis.core.orchestrator.chat_completion",
               return_value=fake_text_response("hello")):
        ...
"""

from __future__ import annotations

from typing import Any


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id: str, function: FakeFunction) -> None:
        self.id = id
        self.function = function


class FakeMessage:
    def __init__(
        self,
        role: str = "assistant",
        content: str | None = None,
        tool_calls: list[FakeToolCall] | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeResponse:
    def __init__(self, choices: list[FakeChoice]) -> None:
        self.choices = choices


def fake_text_response(text: str = "ok") -> FakeResponse:
    """One assistant text answer (terminates a ReAct round)."""
    return FakeResponse([FakeChoice(FakeMessage("assistant", text))])


def fake_tool_response(
    calls: list[tuple[str, str]], call_id_prefix: str = "call"
) -> FakeResponse:
    """
    One assistant turn requesting tool calls.

    calls: [(tool_name, json_arguments), ...]
    """
    tool_calls = [
        FakeToolCall(f"{call_id_prefix}_{i + 1}", FakeFunction(name, args))
        for i, (name, args) in enumerate(calls)
    ]
    return FakeResponse([FakeChoice(FakeMessage("assistant", None, tool_calls))])


def make_llm_stub(script: list[Any]) -> Any:
    """
    A chat_completion side_effect that plays `script` in order.

    script entries: FakeResponse objects, or (tool_name, args) tuples that
    are auto-wrapped as tool-call turns, or plain strings as text answers.
    """
    queue = list(script)

    def _stub(messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
        if not queue:
            return fake_text_response("(script exhausted)")
        item = queue.pop(0)
        if isinstance(item, tuple):
            return fake_tool_response([item])
        if isinstance(item, str):
            return fake_text_response(item)
        return item

    return _stub


__all__ = [
    "FakeChoice",
    "FakeFunction",
    "FakeMessage",
    "FakeResponse",
    "FakeToolCall",
    "fake_text_response",
    "fake_tool_response",
    "make_llm_stub",
]
