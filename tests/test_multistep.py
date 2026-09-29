"""
tests/test_multistep.py
───────────────────────
v0.22 deterministic tests: plan validation, planner tool catalogue,
exact-evidence propagation across steps, per-step forced tool rounds,
multi-step routing, and the multi-step benchmark itself.

All LLM behavior is scripted; no Ollama. PermissionGuard and Pydantic
validation stay REAL.
"""

from typing import Any
from unittest.mock import patch

import pytest

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.core.plan_validator import validate_plan
from jarvis.core.planner import Planner, build_planner_user_prompt
from jarvis.memory.session_store import SessionStore
from jarvis.tools import (
    CalculatorTool,
    GetCurrentDatetimeTool,
    ListDirectoryTool,
    ReadFileTool,
    RecallFactsTool,
    RememberFactTool,
    SearchKnowledgeTool,
    ToolRegistry,
    VisionAnalyzeTool,
    WebScrapeTool,
    WebSearchTool,
    WikipediaSummaryTool,
    WriteFileTool,
)

from evaluation import multistep_benchmark as msb


# ── Shared fixtures/helpers ───────────────────────────────────────────────────


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


def fake_text_response(text: str) -> Response:
    return Response([Choice(Message("assistant", text))])


def fake_tool_response(calls: list[tuple[str, str]]) -> Response:
    return Response([
        Choice(Message("assistant", None, [ToolCall(f"call_{i}", Function(n, a))]))
        for i, (n, a) in enumerate(calls, start=1)
    ])


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for factory in (
        CalculatorTool,
        GetCurrentDatetimeTool,
        WebSearchTool,
        WikipediaSummaryTool,
        ReadFileTool,
        ListDirectoryTool,
        RememberFactTool,
        RecallFactsTool,
        SearchKnowledgeTool,
        WriteFileTool,
        VisionAnalyzeTool,
        WebScrapeTool,
    ):
        registry.register(factory())
    return registry


def _make_orchestrator():
    store = SessionStore()
    registry = _build_registry()
    guard = PermissionGuard()
    orch = Orchestrator(store, registry, guard)
    return orch, store, registry, guard


# ── 1. Plan validation (Part D) ───────────────────────────────────────────────


class TestPlanValidation:
    TOOLS = ["calculator", "remember_fact", "recall_facts", "search_knowledge", "web_search"]

    def test_valid_plan_passes_unchanged(self):
        plan = [
            {"step_number": 1, "description": "Calculate 2+2", "required_tools": ["calculator"]},
            {"step_number": 2, "description": "Remember the result", "required_tools": ["remember_fact"]},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok
        assert result.plan == plan
        assert result.issues == []

    def test_hallucinated_tools_removed(self):
        plan = [
            {"step_number": 1, "description": "Do it", "required_tools": ["execute_python_code", "calculator"]},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok
        assert result.plan[0]["required_tools"] == ["calculator"]
        assert any("unknown_tools" in i for i in result.issues)

    def test_forward_reference_dropped_but_plan_survives(self):
        """Step 2 references step 4 (later): the offending step is dropped
        (unwinding-safe), the valid prefix still executes."""
        plan = [
            {"step_number": 1, "description": "Do the thing", "required_tools": []},
            {"step_number": 2, "description": "Use the result of step 4", "required_tools": []},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok  # step 1 survives
        assert len(result.plan) == 1
        assert any("forward_reference" in i for i in result.issues)

    def test_backward_reference_accepted(self):
        plan = [
            {"step_number": 1, "description": "Calculate it", "required_tools": ["calculator"]},
            {"step_number": 2, "description": "Remember the result of step 1", "required_tools": ["remember_fact"]},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok
        assert len(result.plan) == 2

    def test_duplicate_steps_collapsed(self):
        plan = [
            {"step_number": 1, "description": "Search the web", "required_tools": ["web_search"]},
            {"step_number": 2, "description": "search the web", "required_tools": ["web_search"]},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok
        assert len(result.plan) == 1
        assert any("duplicate" in i for i in result.issues)

    def test_oversized_plan_truncated_to_max(self):
        from jarvis.core.planner import MAX_PLAN_STEPS

        plan = [
            {"step_number": i, "description": f"Step number {i}", "required_tools": []}
            for i in range(1, MAX_PLAN_STEPS + 3)
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok
        assert len(result.plan) == MAX_PLAN_STEPS
        assert any("truncated" in i for i in result.issues)

    def test_step_numbers_renumbered_contiguously(self):
        plan = [
            {"step_number": 1, "description": "First", "required_tools": []},
            {"step_number": 5, "description": "Second", "required_tools": []},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert [s["step_number"] for s in result.plan] == [1, 2]

    def test_empty_plan_rejected(self):
        result = validate_plan([], self.TOOLS)
        assert not result.ok

    def test_malformed_steps_skipped(self):
        plan = [
            "not a dict",
            {"step_number": 1, "description": "", "required_tools": []},
            {"step_number": 2, "description": "Valid step", "required_tools": ["calculator"]},
        ]
        result = validate_plan(plan, self.TOOLS)
        assert result.ok
        assert len(result.plan) == 1
        assert result.plan[0]["description"] == "Valid step"

    def test_not_a_list_rejected(self):
        assert not validate_plan({"nope": 1}, self.TOOLS).ok


# ── 2. Planner tool catalogue (Part C) ────────────────────────────────────────


class TestPlannerToolCatalogue:
    def test_prompt_includes_descriptions_not_just_names(self):
        prompt = build_planner_user_prompt(
            "Calculate 2+2 and remember it",
            "ctx",
            ["calculator", "remember_fact"],
            {"calculator": "Compute exact arithmetic.", "remember_fact": "Store a personal fact."},
        )
        assert "calculator: Compute exact arithmetic." in prompt
        assert "remember_fact: Store a personal fact." in prompt
        assert "Tool catalogue" in prompt

    def test_names_only_fallback_without_descriptions(self):
        prompt = build_planner_user_prompt("q", "ctx", ["calculator"], None)
        assert "Available tools:" in prompt
        assert "- calculator" in prompt

    def test_orchestrator_wires_descriptions_into_planner(self):
        orch, store, registry, guard = _make_orchestrator()
        assert orch._planner._tool_descriptions
        assert "calculator" in orch._planner._tool_descriptions
        assert "Compute" in orch._planner._tool_descriptions["calculator"]
        store.close()

    def test_planner_filters_via_registry_after_catalogue(self):
        """The catalogue cannot make hallucinated names executable."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        planner = Planner(
            llm_client=_llm_resp(
                '[{"step_number": 1, "description": "Run the code", "required_tools": ["execute_python_code"]},'
                ' {"step_number": 2, "description": "Use the calculator", "required_tools": ["calculator"]}]'
            ),
            tool_names=registry.list_tools(),
            tool_descriptions={t.name: t.description for t in registry._tools.values()},
        )
        plan = planner.generate_plan("run code and calc", "ctx")
        # execute_python_code does not exist in the registry → filtered out.
        assert [s["required_tools"] for s in plan] == [[], ["calculator"]]


def _llm_resp(content: str):
    """Return a planner-compatible llm_client callable (plain function)."""

    def _client(**kwargs):
        return Response([Choice(Message("assistant", content))])

    return _client


# ── 3. Exact-evidence propagation (Part E) ───────────────────────────────────


class TestObservationPropagation:
    def test_later_step_receives_exact_tool_result(self):
        """Step 2's prompt carries step 1's raw clamped tool result.

        The orchestrator is built INSIDE the chat_completion patch: the
        Planner captures its client at construction (the v0.20 finding).
        """
        plan = [
            {"step_number": 1, "description": "Calculate 893 * 47 with the calculator.", "required_tools": ["calculator"]},
            {"step_number": 2, "description": "Remember the calculated result.", "required_tools": ["remember_fact"]},
        ]

        def llm(messages, tools=None, **kwargs):
            system_text = " ".join(
                str(m.get("content") or "") for m in messages if m.get("role") == "system"
            )
            if "strategic planner" in system_text.lower():
                import json as _json

                return fake_text_response(_json.dumps(plan))
            # Identify the CURRENT step from the brief (it contains
            # 'executing step N'), because a step's later rounds carry its
            # brief in the conversation tail as well.
            brief = next(
                (str(m.get("content")) for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            names = {s["function"]["name"] for s in (tools or [])}
            if "step 1" in brief.lower() and tools and "calculator" in names:
                return fake_tool_response([("calculator", '{"expression": "893 * 47"}')])
            if tools:
                return fake_text_response("Step done.")
            return fake_text_response("The result is 41971 and remembered.")

        captured: list[list[dict[str, Any]]] = []

        def recording_llm(messages, tools=None, **kwargs):
            if tools and "strategic planner" not in " ".join(
                str(m.get("content") or "") for m in messages if m.get("role") == "system"
            ):
                captured.append(list(messages))
            return llm(messages, tools=tools, **kwargs)

        with patch("jarvis.core.orchestrator.chat_completion", side_effect=recording_llm):
            orch, store, registry, guard = _make_orchestrator()
            answer = orch.chat("s_obs1", "Calculate 893 * 47 and remember the result.")

        step2_prompts = [
            msgs for msgs in captured
            if any("Remember the calculated result" in str(m.get("content")) for m in msgs)
        ]
        assert step2_prompts, "step 2 was not executed"
        blob = " ".join(str(m.get("content")) for m in step2_prompts[0])
        assert "41971" in blob  # the EXACT calculator result reached step 2
        assert "41971" in answer
        store.close()

    def test_observations_are_clamped(self):
        """A huge tool result reaches later steps only in clamped form."""
        orch, store, registry, guard = _make_orchestrator()
        huge = "X" * 50000
        clamp_len = len(orch._context.clamp_tool_output(huge))
        assert clamp_len < 50000  # sanity: the clamp itself is the boundary
        store.close()


# ── 4. Per-step forced tool round (Part F) ───────────────────────────────────


class TestPerStepForcedRound:
    def test_tool_requiring_step_gets_recovery_round(self):
        """A step with required_tools whose model answers without a tool gets
        exactly one forced recovery round, then continues."""
        orch, store, registry, guard = _make_orchestrator()
        plan = [
            {"step_number": 1, "description": "Calculate 6 * 7 with the calculator.", "required_tools": ["calculator"]},
        ]

        calls = {"n": 0}

        def llm(messages, tools=None, **kwargs):
            system_text = " ".join(
                str(m.get("content") or "") for m in messages if m.get("role") == "system"
            )
            if "strategic planner" in system_text.lower():
                import json as _json

                return fake_text_response(_json.dumps(plan))
            if tools:
                calls["n"] += 1
                if calls["n"] == 1:
                    return fake_text_response("I know this one: 42.")  # refuses tools
                if calls["n"] == 2:
                    return fake_tool_response([("calculator", '{"expression": "6 * 7"}')])
                return fake_text_response("Step 1 complete: 42.")
            return fake_text_response("6 * 7 = 42.")

        dispatched: list[str] = []

        async def spy(tool_name, tool_args):
            # Full registry dispatch: PermissionGuard + validation + run()
            # (imported lazily to avoid a circular import at module load).
            from jarvis.tools.registry import ToolRegistry as _TR

            dispatched.append(tool_name)
            return await ToolRegistry.dispatch_async(registry, tool_name, tool_args)

        with patch("jarvis.core.orchestrator.chat_completion", side_effect=llm):
            with patch.object(registry, "dispatch_async", side_effect=spy):
                answer = orch.chat("s_force1", "Please calculate 6 * 7 with the calculator tool")

        assert dispatched == ["calculator"]  # the forced round produced the call
        assert "42" in answer
        store.close()


# ── 5. Multi-step routing (Part L) ───────────────────────────────────────────


class TestMultistepRouting:
    def test_calc_and_remember_goes_to_planner(self):
        orch, store, _, _ = _make_orchestrator()
        assert orch.route_intent("Calculate 893 * 47 and remember the result.") == "complex"
        assert orch.route_intent("Remember 144 and also calculate 12*12") == "complex"
        store.close()

    def test_search_and_compare_goes_to_planner(self):
        orch, store, _, _ = _make_orchestrator()
        assert (
            orch.route_intent(
                "Search my knowledge base for the orchestration plan and compare it with current web information."
            )
            == "complex"
        )
        store.close()

    def test_hyphenated_prose_not_caught_by_operator_keyword(self):
        """'plan-and-execute' contains '-'; it must not route as arithmetic."""
        orch, store, _, _ = _make_orchestrator()
        assert (
            orch.route_intent("Explain how plan-and-execute agents work, and compare them with ReAct loops.")
            == "complex"
        )
        store.close()

    def test_single_intent_asks_stay_simple(self):
        orch, store, _, _ = _make_orchestrator()
        assert orch.route_intent("what time is it?") == "simple"
        assert orch.route_intent("calculate 25 * 4") == "simple"
        assert orch.route_intent("What is 893 * 47?") == "simple"
        store.close()


# ── 6. The multi-step benchmark itself (Parts I/J/K/N) ───────────────────────


@pytest.fixture(scope="module")
def ms_registry():
    return msb._build_registry()


@pytest.fixture(scope="module")
def ms_results(ms_registry):
    return [msb.run_case(case, ms_registry) for case in msb.MULTISTEP_CASES]


class TestMultistepBenchmark:
    def test_all_required_categories_present(self):
        categories = {c["category"] for c in msb.MULTISTEP_CASES}
        assert {"calc_memory", "knowledge_multistep", "web_multistep", "correction", "no_tool"} <= categories

    def test_all_cases_pass(self, ms_results):
        failures = [r for r in ms_results if not r["passed"]]
        assert failures == [], f"multi-step benchmark failures: {failures}"

    def test_plan_creation_perfect(self, ms_results):
        assert all(r["plan_created"] for r in ms_results)

    def test_plan_structure_perfect(self, ms_results):
        assert all(r["plan_structure_ok"] for r in ms_results)

    def test_tool_order_perfect(self, ms_results):
        assert all(r["tool_order_ok"] for r in ms_results)

    def test_arguments_valid_perfect(self, ms_results):
        assert all(r["argument_valid"] for r in ms_results)

    def test_result_propagation_perfect(self, ms_results):
        assert all(r["result_propagation_ok"] for r in ms_results)

    def test_grounding_perfect(self, ms_results):
        assert all(r["grounded"] for r in ms_results)

    def test_correction_hint_injected_once_per_step(self, ms_results):
        case = next(r for r in ms_results if r["category"] == "correction")
        assert case["corrections"] == 1
        assert case["expected_corrections"] == 1

    def test_no_unexpected_corrections(self, ms_results):
        for r in ms_results:
            assert r["corrections"] == r.get("expected_corrections", 0), r["name"]

    def test_suppression_counts_match_expectations(self, ms_results):
        for r in ms_results:
            assert r["suppressed_count"] == r.get("expected_suppressed", 0), r["name"]

    def test_redundant_step_detected_when_expected(self, ms_results):
        case = next(
            (r for r in ms_results if r.get("expected_redundant_step")),
            None,
        )
        assert case is not None
        assert case["redundant_steps"] == [case["expected_redundant_step"]]
