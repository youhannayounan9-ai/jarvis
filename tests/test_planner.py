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

    def test_hallucinated_tools_filtered_against_registry(self):
        """required_tools that do not exist must not reach the executor."""
        llm = MagicMock(
            return_value=_fake_response(
                '[{"step_number": 1, "description": "Run code", '
                '"required_tools": ["execute_python_code", "web_search"]},'
                '{"step_number": 2, "description": "Summarize", '
                '"required_tools": ["list_files", "summarizer"]}]'
            )
        )
        planner = Planner(
            llm_client=llm, tool_names=["web_search", "calculator"]
        )
        plan = planner.generate_plan("Do impossible things", context="")
        assert plan[0]["required_tools"] == ["web_search"]  # hallucination dropped
        assert plan[1]["required_tools"] == []  # fully hallucinated → reasoning step

    def test_empty_tool_names_means_no_filtering_not_no_tools(self):
        """Empty tool_names = planner lacks registry info; plans pass through
        UNFILTERED (it means 'no filtering information', not 'no tools allowed')."""
        llm = MagicMock(
            return_value=_fake_response(
                '[{"step_number": 1, "description": "Search", '
                '"required_tools": ["web_search"]}]'
            )
        )
        planner = Planner(llm_client=llm, tool_names=[])  # standalone use
        plan = planner.generate_plan("x", context="")
        assert plan[0]["required_tools"] == ["web_search"]

    def test_plan_is_bounded_to_max_steps(self):
        """A runaway plan must be truncated to MAX_PLAN_STEPS and renumbered."""
        steps = ",".join(
            f'{{"step_number": {i}, "description": "Step {i}", "required_tools": []}}'
            for i in range(1, 13)
        )
        llm = MagicMock(return_value=_fake_response(f"[{steps}]"))
        planner = Planner(llm_client=llm)
        plan = planner.generate_plan("Do twelve things", context="")
        from jarvis.core.planner import MAX_PLAN_STEPS

        assert len(plan) == MAX_PLAN_STEPS
        assert [s["step_number"] for s in plan] == list(range(1, MAX_PLAN_STEPS + 1))
