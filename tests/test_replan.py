"""
tests/test_replan.py
────────────────────
v0.24 deterministic tests for ONE bounded replan (Part N, replanning): a
structurally failed plan triggers EXACTLY ONE validated replan within the
remaining budget; complete plans never replan; recursion is structurally
impossible; completed work is inherited, not repeated; and a second failure
produces a truthful incomplete answer.

All LLM behavior is scripted (multistep-benchmark pattern); no Ollama.
PermissionGuard, plan validation, and Pydantic validation stay REAL.
"""

import json
from typing import Any
from unittest.mock import patch

from jarvis.core.orchestrator import (
    MAX_TOOL_ROUNDS,
    Orchestrator,
    _build_replan_context,
    _format_incomplete_note,
    _is_tool_error,
)
from jarvis.core.permissions import PermissionGuard
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


class _Msg:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


def _text_resp(text: str) -> _Resp:
    return _Resp(_Msg("assistant", text))


def _tool_resp(calls: list[tuple[str, str]]) -> _Resp:
    tcs = []
    for i, (name, args) in enumerate(calls, start=1):
        fn = type("F", (), {"name": name, "arguments": args})()
        tcs.append(type("TC", (), {"id": f"call_{i}", "function": fn})())
    return _Resp(_Msg("assistant", None, tcs))


# ── Scripted scenario driver ──────────────────────────────────────────────────


class _Scenario:
    """Deterministic plan/replan driver.

    plan_sequence:  plan handed out per planner call (call 1 = original, call
                    2 = THE replan; a third call would mean recursion).
    fail_dispatch:  {(phase, tool_name)} → dispatch returns a scripted ERROR.
    step_scripts:   per phase, per step: round scripts (round = list of
                    (tool, args_json) calls or a str step result).
    step_finals:    per phase: result texts used when a step's script is
                    exhausted.
    phase_final:    per phase: the synthesis text.
    """

    def __init__(
        self,
        plan_sequence: list[list[dict[str, Any]]],
        step_scripts: list[list[list[Any]]],
        step_finals: list[list[str]],
        phase_final: list[str],
        fail_dispatch: set[tuple[int, str]] | None = None,
    ):
        self.plan_sequence = plan_sequence
        self.fail_dispatch = fail_dispatch or set()
        self.step_scripts = step_scripts
        self.step_finals = step_finals
        self.phase_final = phase_final
        self.phase = 0
        self.step = 0
        self.round = 0
        self.planner_calls = 0
        self.dispatch_log: list[tuple[int, str, str]] = []
        self.replan_events: list[dict[str, Any]] = []
        self.plan_complete_events: list[dict[str, Any]] = []

    # — LLM (jarvis.core.orchestrator.chat_completion) —
    def llm(self, messages, tools=None, **kwargs):
        system_text = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        if "strategic planner" in system_text.lower():
            plan = self.plan_sequence[min(self.phase, len(self.plan_sequence) - 1)]
            return _text_resp(json.dumps(plan))
        if tools:
            phase_scripts = self.step_scripts[min(self.phase, len(self.step_scripts) - 1)]
            step_idx = min(self.step, len(phase_scripts) - 1)
            rounds = phase_scripts[step_idx] if phase_scripts else []
            if self.round < len(rounds):
                round_spec = rounds[self.round]
                self.round += 1
                if isinstance(round_spec, str):
                    return _text_resp(round_spec)
                return _tool_resp(list(round_spec))
            # Script exhausted → this step's scripted result text; next step.
            finals = self.step_finals[min(self.phase, len(self.step_finals) - 1)]
            default = finals[min(step_idx, len(finals) - 1)]
            self.step += 1
            self.round = 0
            return _text_resp(default)
        # Tool-free synthesis call for the CURRENT phase.
        return _text_resp(self.phase_final[min(self.phase, len(self.phase_final) - 1)])

    # — Planner (orch._planner.generate_plan) —
    def planner(self, user_input: str, context: str) -> list[dict[str, Any]]:
        if self.planner_calls == 0:
            # FIRST call: the original plan; execution stays in phase 0.
            self.planner_calls = 1
            return [dict(step) for step in self.plan_sequence[0]]
        # ANY later call is THE replan: switch to the replan phase's plan and
        # scripts. A third call would mean recursive replanning.
        self.planner_calls += 1
        self.phase = 1
        self.step = 0
        self.round = 0
        plan = self.plan_sequence[min(1, len(self.plan_sequence) - 1)]
        return [dict(step) for step in plan]

    # — Registry dispatch —
    def dispatch(self, tool_name: str, tool_args: str) -> str:
        phase = 1 if self.planner_calls > 1 else 0
        self.dispatch_log.append((phase, tool_name, tool_args))
        if (phase, tool_name) in self.fail_dispatch:
            return f"ERROR: scripted {tool_name} failure (phase {phase})"
        return f"{tool_name} ok: {tool_args}"

    # — Event observer —
    def on_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "replan":
            self.replan_events.append(event)
        if event.get("type") == "plan_complete":
            self.plan_complete_events.append(event)


def _run(scenario: _Scenario, session_id: str) -> tuple[Orchestrator, SessionStore, str]:
    orch, store, registry, guard = _make_orchestrator()
    with patch.object(orch._planner, "generate_plan", side_effect=scenario.planner):
        with patch("jarvis.core.orchestrator.chat_completion", side_effect=scenario.llm):
            with patch.object(registry, "dispatch_async", side_effect=scenario.dispatch):
                answer = orch.chat(
                    session_id,
                    "Calculate 893 * 47 and remember the result.",
                    on_event=scenario.on_event,
                )
    return orch, store, answer


PLAN_CALC_REMEMBER = [
    {"step_number": 1, "description": "Calculate 893 * 47 with the calculator.", "required_tools": ["calculator"]},
    {"step_number": 2, "description": "Remember the calculated result as a fact.", "required_tools": ["remember_fact"]},
]
PLAN_CALC_RETRY = [
    {"step_number": 1, "description": "Retry the calculation with the calculator.", "required_tools": ["calculator"]},
]


# ── 1. Trigger conditions (Part I1) ──────────────────────────────────────────


class TestReplanTrigger:
    def test_failed_step_triggers_exactly_one_replan(self):
        """Calculator fails in the original plan, remember succeeds → ONE
        replan; the replan's retry succeeds → plan completes, replans=1."""
        scenario = _Scenario(
            plan_sequence=[PLAN_CALC_REMEMBER, PLAN_CALC_RETRY],
            fail_dispatch={(0, "calculator")},
            step_scripts=[
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                    [[("remember_fact", '{"fact": "893*47 = 41971"}')]],
                ],
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                ],
            ],
            step_finals=[
                ["ERROR: calculator unavailable this attempt", "Step 2 complete: remembered."],
                ["Replan step complete: 41971."],
            ],
            phase_final=[
                "(must never be reached — the replan must intervene)",
                "893 * 47 = 41971 and it is remembered.",
            ],
        )
        _, store, answer = _run(scenario, "s_replan1")

        assert scenario.planner_calls == 2  # original + EXACTLY ONE replan
        assert len(scenario.replan_events) == 1
        assert scenario.plan_complete_events == [{
            "type": "plan_complete", "complete": True, "failed_steps": [], "replans": 1,
        }]
        assert "41971" in answer
        assert "must never be reached" not in answer
        # The retry actually dispatched the calculator again (phase 1):
        assert any(p == 1 and t == "calculator" for p, t, _ in scenario.dispatch_log)
        store.close()

    def test_complete_plan_never_replans(self):
        scenario = _Scenario(
            plan_sequence=[PLAN_CALC_REMEMBER],
            step_scripts=[
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                    [[("remember_fact", '{"fact": "893*47 = 41971"}')]],
                ],
            ],
            step_finals=[["Step 1 complete: 41971.", "Step 2 complete: remembered."]],
            phase_final=["893 * 47 = 41971 and it is remembered."],
        )
        _, store, answer = _run(scenario, "s_replan2")

        assert scenario.planner_calls == 1
        assert scenario.replan_events == []
        assert scenario.plan_complete_events == [{
            "type": "plan_complete", "complete": True, "failed_steps": [], "replans": 0,
        }]
        assert "41971" in answer
        store.close()

    def test_replan_failure_does_not_replan_again(self):
        """The replan ALSO fails: still exactly ONE replan total (recursion is
        structurally impossible), then a truthful incomplete answer."""
        scenario = _Scenario(
            plan_sequence=[PLAN_CALC_REMEMBER, PLAN_CALC_RETRY],
            fail_dispatch={(0, "calculator"), (1, "calculator")},
            step_scripts=[
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                    [[("remember_fact", '{"fact": "893*47 = 41971"}')]],
                ],
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                ],
            ],
            step_finals=[
                ["ERROR: calculator attempt 1 failed", "Step 2 complete: remembered."],
                ["ERROR: calculator retry failed too"],
            ],
            phase_final=[
                "(unused)",
                "I could not complete the calculation; it remains unfinished.",
            ],
        )
        _, store, answer = _run(scenario, "s_replan3")

        assert scenario.planner_calls == 2  # NOT 3 — no recursive replanning
        assert len(scenario.replan_events) == 1
        assert scenario.plan_complete_events == [{
            "type": "plan_complete", "complete": False, "failed_steps": [1], "replans": 1,
        }]
        assert "unfinished" in answer.lower()
        store.close()

    def test_no_tool_error_means_no_replan(self):
        """A plan whose steps all succeed produces zero replans — even when a
        step used multiple legitimate tool rounds."""
        scenario = _Scenario(
            plan_sequence=[PLAN_CALC_REMEMBER],
            step_scripts=[
                [
                    # Step 1: ONE round with two legitimate (different-arg) calls.
                    [[
                        ("calculator", '{"expression": "800 * 47"}'),
                        ("calculator", '{"expression": "93 * 47"}'),
                    ]],
                    # Step 2: one round remembering the totals.
                    [[
                        ("remember_fact", '{"fact": "sum remembered"}'),
                    ]],
                ],
            ],
            step_finals=[["Both parts calculated.", "Step 2 complete: remembered."]],
            phase_final=["Totals calculated and remembered."],
        )
        _, store, answer = _run(scenario, "s_replan4")
        assert scenario.planner_calls == 1
        assert scenario.replan_events == []
        assert scenario.plan_complete_events[0]["complete"] is True
        store.close()


# ── 2. Budget + reuse guarantees (Parts I2/I5) ───────────────────────────────


class TestReplanBudgetAndReuse:
    def test_replan_inherits_remaining_budget_no_reset(self):
        """The replan execution starts from the budget LEFT OVER after the
        original plan — tool rounds are never reset by replanning."""
        orch, store, registry, guard = _make_orchestrator()
        observed_budgets: list[int] = []
        original_execute = Orchestrator._execute_plan_steps

        def budget_spy(inner_self, **kwargs):
            observed_budgets.append(kwargs["remaining_rounds"])
            return original_execute(inner_self, **kwargs)

        scenario = _Scenario(
            plan_sequence=[PLAN_CALC_REMEMBER, PLAN_CALC_RETRY],
            fail_dispatch={(0, "calculator")},
            step_scripts=[
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                    [[("remember_fact", '{"fact": "893*47 = 41971"}')]],
                ],
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                ],
            ],
            step_finals=[
                ["ERROR: calculator unavailable", "Step 2 complete: remembered."],
                ["Replan ok: 41971."],
            ],
            phase_final=["(unused)", "41971 via replan."],
        )
        with patch.object(Orchestrator, "_execute_plan_steps", budget_spy):
            with patch.object(orch._planner, "generate_plan", side_effect=scenario.planner):
                with patch("jarvis.core.orchestrator.chat_completion", side_effect=scenario.llm):
                    with patch.object(registry, "dispatch_async", side_effect=scenario.dispatch):
                        orch.chat("s_replan5", "Calculate 893 * 47 and remember the result.")

        # Original execution saw the FULL budget; the replan saw what remained
        # after the original plan's two rounds (one failed + one successful).
        # Nothing was reset.
        assert observed_budgets[0] == MAX_TOOL_ROUNDS
        assert observed_budgets[1] == MAX_TOOL_ROUNDS - 2
        store.close()

    def test_replan_context_lists_completed_and_failed_compactly(self):
        """The replan prompt is compact: completed steps as DO-NOT-REPEAT,
        failed steps as re-plan-only, and the remaining budget stated."""
        ctx = _build_replan_context(
            "Calculate 893 * 47 and remember the result.",
            completed_steps=[{
                "step_number": 2, "description": "Remember the calculated result as a fact.",
                "result": "Step 2 complete: remembered.",
            }],
            failed_steps=[{
                "step_number": 1, "description": "Calculate 893 * 47 with the calculator.",
                "required_tools": ["calculator"], "result": "ERROR: x",
            }],
            remaining_rounds=3,
        )
        assert "do NOT re-plan these" in ctx
        assert "Remember the calculated result" in ctx
        assert "re-plan ONLY these requirements" in ctx
        assert "Calculate 893 * 47 with the calculator" in ctx
        assert "Remaining tool-round budget for the WHOLE replan: 3" in ctx
        assert "REPLAN CONTEXT" in ctx

    def test_incomplete_note_forces_truthful_reporting(self):
        note = _format_incomplete_note([{
            "step_number": 1, "description": "Calculate 893 * 47 with the calculator.",
            "required_tools": ["calculator"], "result": "ERROR: provider down",
        }])
        assert "INCOMPLETENESS NOTICE" in note
        assert "never claim full success" in note
        assert "Calculate 893 * 47" in note
        assert "calculator" in note

    def test_synthesis_receives_incomplete_note_after_failed_replan(self):
        """The post-replan synthesis call carries the INCOMPLETENESS NOTICE in
        its system context (Part I6: truthful final synthesis)."""
        orch, store, registry, guard = _make_orchestrator()
        scenario = _Scenario(
            plan_sequence=[PLAN_CALC_REMEMBER, PLAN_CALC_RETRY],
            fail_dispatch={(0, "calculator"), (1, "calculator")},
            step_scripts=[
                [
                    [[("calculator", '{"expression": "893 * 47"}')]],
                    [[("remember_fact", '{"fact": "893*47 = 41971"}')]],
                ],
                [[[("calculator", '{"expression": "893 * 47"}')]]],
            ],
            step_finals=[
                ["ERROR: attempt 1 failed", "Step 2 complete: remembered."],
                ["ERROR: retry failed"],
            ],
            phase_final=["(unused)", "The calculation remains incomplete."],
        )
        synthesis_system: list[str] = []

        real_llm = scenario.llm

        def llm_recorder(messages, tools=None, **kwargs):
            if tools is None:
                # v0.25: the synthesis payload (evidence + incompleteness
                # notice) rides the FINAL USER message — live-verified as the
                # layout small models actually obey — so scan ALL messages.
                all_text = " ".join(
                    str(m.get("content") or "") for m in messages
                )
                if "INCOMPLETENESS NOTICE" in all_text:
                    synthesis_system.append(all_text)
            return real_llm(messages, tools=tools, **kwargs)

        with patch.object(orch._planner, "generate_plan", side_effect=scenario.planner):
            with patch("jarvis.core.orchestrator.chat_completion", side_effect=llm_recorder):
                with patch.object(registry, "dispatch_async", side_effect=scenario.dispatch):
                    orch.chat("s_replan6", "Calculate 893 * 47 and remember the result.")

        assert synthesis_system, "synthesis never received the incompleteness note"
        assert "Retry the calculation" in synthesis_system[0]
        store.close()


# ── 3. Failure detection is structural ───────────────────────────────────────


class TestFailureDetection:
    def test_is_tool_error_convention(self):
        assert _is_tool_error("ERROR: anything")
        assert not _is_tool_error("Result: 4")
        assert not _is_tool_error("ACTION_REQUIRES_CONFIRMATION: waiting for user")
        assert not _is_tool_error("")
        assert not _is_tool_error(None)
