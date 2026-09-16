"""
tests/test_planner.py
─────────────────────
Unit tests for the v0.3 Planner (mocked LLM — no Ollama).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from jarvis.core.planner import Planner, _strip_code_fences


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestStripCodeFences:
    def test_plain_json(self):
        raw = '[{"step_number": 1, "description": "Do it", "required_tools": []}]'
        assert _strip_code_fences(raw).startswith("[")

    def test_fenced_json(self):
        raw = '```json\n[{"step_number": 1, "description": "Do it", "required_tools": []}]\n```'
        assert _strip_code_fences(raw).startswith("[")


class TestPlanner:
    def test_parses_valid_json_array(self):
        llm = MagicMock(
            return_value=_fake_response(
                '[{"step_number": 1, "description": "Search news", '
                '"required_tools": ["web_search"]},'
                '{"step_number": 2, "description": "Summarise", '
                '"required_tools": []}]'
            )
        )
        planner = Planner(llm_client=llm)
        plan = planner.generate_plan("Get latest AI news", context="(none)")

        assert len(plan) == 2
        assert plan[0]["description"] == "Search news"
        assert plan[0]["required_tools"] == ["web_search"]
        llm.assert_called_once()

    def test_strips_markdown_fence(self):
        llm = MagicMock(
            return_value=_fake_response(
                '```json\n[{"step_number": 1, "description": "Greet", '
                '"required_tools": []}]\n```'
            )
        )
        planner = Planner(llm_client=llm)
        plan = planner.generate_plan("Hi", context="")
        assert len(plan) == 1
        assert plan[0]["description"] == "Greet"

    def test_fallback_on_invalid_json(self):
        llm = MagicMock(return_value=_fake_response("not json at all"))
        planner = Planner(llm_client=llm)
        plan = planner.generate_plan("Do something complex", context="")
        assert len(plan) == 1
        assert plan[0]["description"] == "Do something complex"
        assert plan[0]["required_tools"] == []

    def test_fallback_on_llm_error(self):
        llm = MagicMock(side_effect=RuntimeError("ollama down"))
        planner = Planner(llm_client=llm)
        plan = planner.generate_plan("Hello", context="")
        assert plan[0]["step_number"] == 1
        assert plan[0]["description"] == "Hello"
