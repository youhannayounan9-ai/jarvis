"""
tests/test_synthesis.py
───────────────────────
v0.25 (Part C) deterministic synthesis-grounding tests.

The v0.24 live failure: execution produced the correct tool result, but the
final synthesis repeated the model's earlier incorrect prose. These tests pin
the v0.25 mechanism that prevents it:

  - every successful tool observation enters a bounded AUTHORITATIVE evidence
    ledger that is injected into the synthesis prompt with an explicit
    contract (tool evidence WINS over earlier assistant prose);
  - failed tool results never enter the ledger (a failed later call cannot
    overwrite a successful earlier one);
  - replan observations join the same ledger (newest-last, authoritative);
  - incompleteness is disclosed; citations and clamped outputs are preserved.

The model itself is scripted — the DETERMINISTIC assertion is what the
synthesis PROMPT contains (the evidence the real model will be grounded on),
never a hardcoded final number. PermissionGuard, plan validation and Pydantic
validation stay REAL; no Ollama.
"""

import json
from typing import Any
from unittest.mock import patch

from jarvis.core.orchestrator import (
    _diff_plans,
    _EVIDENCE_CONTRACT,
    _format_evidence_ledger,
    _MAX_EVIDENCE_ITEM_CHARS,
    _MAX_EVIDENCE_ITEMS,
    _tool_from_observation,
    Orchestrator,
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

EVIDENCE_BEGIN = "BEGIN AUTHORITATIVE TOOL EVIDENCE"
EVIDENCE_END = "END AUTHORITATIVE TOOL EVIDENCE"


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


class _SynthScenario:
    """Scripted plan-path driver that CAPTURES every synthesis prompt.

    plan:        list of step dicts; each may carry a private ``_calls`` key —
                 the list of (tool, args_json) tool calls the model makes in
                 that step's first round. ``required_tools`` stays real.
    tool_results: tool name → scripted successful dispatch result.
    step_finals: per-step assistant prose AFTER the tool round (may contradict
                 the tool result — that is the point).
    fail:        {(phase, tool_name)} → dispatch returns a scripted ERROR.
    final_answer: the scripted final synthesis text.

    Every tool-free chat_completion call (i.e., every _synthesize call) is
    recorded in ``synthesis_prompts`` for exact assertions.
    """

    def __init__(
        self,
        plan: list[dict[str, Any]],
        tool_results: dict[str, str],
        step_finals: list[str],
        final_answer: str,
        fail: set[tuple[int, str]] | None = None,
        replan_plan: list[dict[str, Any]] | None = None,
    ):
        self.plan = plan
        self.replan_plan = replan_plan
        self.tool_results = tool_results
        self.step_finals = step_finals
        self.final_answer = final_answer
        self.fail = fail or set()
        self.phase = 0
        self.step = 0
        self.round = 0
        self.planner_calls = 0
        self.dispatch_log: list[tuple[int, str, str]] = []
        self.synthesis_prompts: list[list[dict[str, Any]]] = []
        self.replan_events: list[dict[str, Any]] = []

    # — LLM (jarvis.core.orchestrator.chat_completion) —
    def llm(self, messages, tools=None, **kwargs):
        if tools:
            active = self.plan if self.phase == 0 else (self.replan_plan or self.plan)
            step_calls = active[min(self.step, len(active) - 1)].get("_calls") or []
            if self.round < len(step_calls):
                spec = step_calls[self.round]
                self.round += 1
                if isinstance(spec, str):
                    return _text_resp(spec)
                if isinstance(spec, tuple):
                    return _tool_resp([spec])
                return _tool_resp(list(spec))
            final = self.step_finals[min(self.step, len(self.step_finals) - 1)]
            self.step += 1
            self.round = 0
            return _text_resp(final)
        # Tool-free call == synthesis: capture the EXACT prompt.
        self.synthesis_prompts.append([dict(m) for m in messages])
        return _text_resp(self.final_answer)

    # — Planner (orch._planner.generate_plan) —
    def planner(self, user_input: str, context: str) -> list[dict[str, Any]]:
        self.planner_calls += 1
        if self.planner_calls > 1:
            # The ONE bounded replan: switch phase, reset step cursor.
            self.phase = 1
            self.step = 0
            self.round = 0
            if self.replan_plan is not None:
                self.plan = self.replan_plan
        return [dict(step) for step in self.plan]

    # — Registry dispatch —
    def dispatch(self, tool_name: str, tool_args: str) -> str:
        self.dispatch_log.append((self.phase, tool_name, tool_args))
        if (self.phase, tool_name) in self.fail:
            return f"ERROR: scripted {tool_name} failure (phase {self.phase})"
        return self.tool_results.get(tool_name, f"{tool_name} ok: {tool_args}")

    # — Event observer —
    def on_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "replan":
            self.replan_events.append(event)

    # — Assertions helpers —
    def evidence_blocks(self) -> list[str]:
        blocks = []
        for prompt in self.synthesis_prompts:
            for message in prompt:
                content = str(message.get("content") or "")
                if EVIDENCE_BEGIN in content:
                    blocks.append(
                        content.split(EVIDENCE_BEGIN, 1)[1].split(EVIDENCE_END, 1)[0]
                    )
        return blocks

    def synthesis_block(self) -> str:
        """The full system content of the LAST synthesis call."""
        assert self.synthesis_prompts, "no synthesis call captured"
        return str(self.synthesis_prompts[-1][-1].get("content") or "")


def _run(scenario: _SynthScenario, session_id: str = "s-synth") -> str:
    orch, store, registry, guard = _make_orchestrator()
    with patch.object(orch._planner, "generate_plan", side_effect=scenario.planner):
        with patch(
            "jarvis.core.orchestrator.chat_completion", side_effect=scenario.llm
        ):
            with patch.object(
                registry, "dispatch_async", side_effect=scenario.dispatch
            ):
                return orch.chat(
                    session_id,
                    "Calculate 893 * 47 and remember the result.",
                    on_event=scenario.on_event,
                )


def _calc_step(step_number: int, expression: str, description: str) -> dict[str, Any]:
    return {
        "step_number": step_number,
        "description": description,
        "required_tools": ["calculator"],
        "_calls": [("calculator", json.dumps({"expression": expression}))],
    }


# ── 1. Tool evidence overrides earlier model prose (the live v0.24 failure) ──


class TestEvidenceOverridesProse:
    def test_tool_result_reaches_synthesis_as_authoritative(self):
        """Model prose says 42071; the calculator said 41971. The synthesis
        prompt MUST contain the tool result as AUTHORITATIVE evidence, with
        the explicit contract that tool evidence wins. (Deterministic: the
        final answer is scripted — what we pin is what the model is shown.)"""
        scenario = _SynthScenario(
            plan=[_calc_step(1, "893 * 47", "Calculate 893 * 47.")],
            tool_results={"calculator": "Result: 41971"},
            step_finals=["My mental estimate says the answer is 42071."],
            final_answer="The calculator returned 41971.",
        )
        _run(scenario)

        assert scenario.planner_calls == 1  # success → no replan
        blocks = scenario.evidence_blocks()
        assert blocks, "synthesis prompt carried no evidence block"
        block = blocks[-1]
        assert "41971" in block, "tool value missing from evidence"
        assert "step 1" in block, "evidence lacks source-step provenance"
        # The contract must travel WITH the evidence.
        prompt = scenario.synthesis_block()
        assert "TOOL EVIDENCE WINS" in prompt
        assert _EVIDENCE_CONTRACT.splitlines()[0].startswith(
            "AUTHORITATIVE TOOL EVIDENCE"
        )

    def test_evidence_block_positioned_after_step_prose(self):
        """The evidence block must come AFTER 'Executed steps and results' —
        position and framing both signal 'this is the measured truth'."""
        scenario = _SynthScenario(
            plan=[_calc_step(1, "893 * 47", "Calculate 893 * 47.")],
            tool_results={"calculator": "Result: 41971"},
            step_finals=["done"],
            final_answer="ok",
        )
        _run(scenario)
        prompt = scenario.synthesis_block()
        assert prompt.index(EVIDENCE_BEGIN) > prompt.index(
            "Executed steps and results"
        )


# ── 2. Replan-corrected result reaches synthesis ──────────────────────────────


class TestReplanEvidence:
    def test_replan_corrected_numeric_reaches_synthesis(self):
        """Original calculator dispatch fails → ONE replan → retry succeeds
        with the corrected value. The synthesis prompt must contain the
        REPLAN's successful result as evidence."""
        scenario = _SynthScenario(
            plan=[_calc_step(1, "777 * 111", "Calculate 777 * 111.")],
            replan_plan=[_calc_step(1, "777 * 111", "Retry the calculation with the calculator.")],
            tool_results={"calculator": "Result: 86247"},
            step_finals=["Step complete."],
            final_answer="86247 per the calculator.",
            fail={(0, "calculator")},
        )
        answer = _run(scenario)

        assert scenario.planner_calls == 2, "exactly one replan expected"
        assert len(scenario.replan_events) == 1
        block = scenario.evidence_blocks()[-1]
        assert "86247" in block, "replan result missing from synthesis evidence"
        assert "step 1" in block
        # The failed original attempt must NOT appear as evidence.
        assert "ERROR:" not in block
        assert answer == "86247 per the calculator."


# ── 3. Failed later tool never enters the ledger ──────────────────────────────


class TestFailedToolExcluded:
    def test_failed_later_tool_cannot_overwrite_successful_evidence(self):
        """Step 1 succeeds (Result: 55); step 2's tool call fails. The ledger
        keeps step 1's successful result and contains no ERROR entries —
        a failed later call cannot overwrite successful evidence."""
        scenario = _SynthScenario(
            plan=[
                _calc_step(1, "5 * 11", "Calculate 5 * 11."),
                {
                    "step_number": 2,
                    "description": "Look up a corroborating source.",
                    "required_tools": [],  # not structural → no replan path
                    "_calls": [("wikipedia_summary", json.dumps({"topic": "probe-five-times-eleven"}))],
                },
            ],
            tool_results={"calculator": "Result: 55"},
            step_finals=["Step one done.", "Step two finished."],
            final_answer="55.",
            fail={(0, "wikipedia_summary")},
        )
        _run(scenario)

        assert scenario.planner_calls == 1
        block = scenario.evidence_blocks()[-1]
        assert "Result: 55" in block, "successful evidence was lost"
        assert "ERROR:" not in block, "a failed tool entered the ledger"
        assert "step 1" in block


# ── 4. Two successful tools disagree — provenance preserved ───────────────────


class TestDisagreeingEvidence:
    def test_both_results_present_with_distinct_provenance(self):
        """Two successful tools return conflicting values: the ledger must
        carry BOTH, each labeled with its own step — never a silent
        reconciliation and never a dropped side."""
        scenario = _SynthScenario(
            plan=[
                _calc_step(1, "10 * 10", "Calculate 10 * 10."),
                {
                    "step_number": 2,
                    "description": "Cross-check the value against knowledge.",
                    "required_tools": ["wikipedia_summary"],
                    "_calls": [("wikipedia_summary", json.dumps({"topic": "probe-disagreement-check"}))],
                },
            ],
            tool_results={
                "calculator": "Result: 100",
                "wikipedia_summary": "Draft note: the value is 200.",
            },
            step_finals=["Step one done.", "Step two done."],
            final_answer="Sources disagree.",
        )
        _run(scenario)

        block = scenario.evidence_blocks()[-1]
        assert "Result: 100" in block
        assert "the value is 200" in block
        assert "step 1" in block and "step 2" in block


# ── 5. Incomplete plan → truthful synthesis instructions ──────────────────────


class TestIncompleteTruthfulness:
    def test_second_failure_appends_incompleteness_notice(self):
        """Calculator fails, replan retries, fails again → the synthesis
        prompt MUST carry the INCOMPLETENESS NOTICE (never claim success)."""
        scenario = _SynthScenario(
            plan=[_calc_step(1, "121 * 13", "Calculate 121 * 13.")],
            replan_plan=[_calc_step(1, "121 * 13", "Retry the calculation once more.")],
            tool_results={"calculator": "Result: 1573"},
            step_finals=["..."],
            final_answer="...",
            fail={(0, "calculator"), (1, "calculator")},
        )
        _run(scenario)

        assert scenario.planner_calls == 2
        prompt = scenario.synthesis_block()
        assert "INCOMPLETENESS NOTICE" in prompt
        assert "Unfinished steps:" in prompt
        assert "121 * 13" in prompt  # the failed step is named
        # And the failed tool produced NO evidence item.
        assert scenario.evidence_blocks() == []


# ── 6. Citation preserved in evidence ─────────────────────────────────────────


class TestCitationPreserved:
    def test_source_url_survives_into_evidence_ledger(self):
        """A knowledge result with a source URL keeps the citation attached to
        the evidence item — provenance must not be stripped."""
        url = "https://en.wikipedia.org/wiki/Eiffel_Tower"
        scenario = _SynthScenario(
            plan=[
                {
                    "step_number": 1,
                    "description": "Summarize the Eiffel Tower.",
                    "required_tools": ["wikipedia_summary"],
                    "_calls": [("wikipedia_summary", json.dumps({"topic": "Eiffel Tower v025 citation probe"}))],
                }
            ],
            tool_results={
                "wikipedia_summary": (
                    "The Eiffel Tower is a wrought-iron lattice tower in Paris "
                    f"(source: {url})."
                )
            },
            step_finals=["Summary retrieved."],
            final_answer="The tower is in Paris.",
        )
        _run(scenario)

        block = scenario.evidence_blocks()[-1]
        assert url in block, "citation was stripped from evidence"


# ── 7. Bounded ledger: clamping and item cap ──────────────────────────────────


class TestLedgerBounds:
    def test_huge_output_clamped_per_item(self):
        """A tool result longer than the per-item cap is clamped in the
        rendered ledger (ContextManager clamping happens upstream; this is the
        ledger's own final bound)."""
        huge = "X" * (_MAX_EVIDENCE_ITEM_CHARS * 4)
        blob = _format_evidence_ledger(
            [{"step_number": 1, "tool": "calculator", "status": "ok", "result": huge}]
        )
        assert EVIDENCE_BEGIN in blob and EVIDENCE_END in blob
        payload = blob.split(EVIDENCE_BEGIN, 1)[1].split(EVIDENCE_END, 1)[0]
        assert "X" * _MAX_EVIDENCE_ITEM_CHARS in payload
        assert "X" * (_MAX_EVIDENCE_ITEM_CHARS + 1) not in payload

    def test_item_cap_keeps_newest(self):
        """More items than the cap → the OLDEST are dropped, newest (which
        include replan corrections) are kept."""
        items = [
            {"step_number": i, "tool": "t", "status": "ok", "result": f"item-{i}"}
            for i in range(_MAX_EVIDENCE_ITEMS + 5)
        ]
        blob = _format_evidence_ledger(items)
        assert "item-0" not in blob, "oldest item should be dropped"
        assert f"item-{_MAX_EVIDENCE_ITEMS + 4}" in blob, "newest kept"

    def test_empty_ledger_renders_nothing(self):
        assert _format_evidence_ledger(None) == ""
        assert _format_evidence_ledger([]) == ""


# ── 8. Helper units: _diff_plans / _tool_from_observation ─────────────────────


class TestHelperUnits:
    def test_diff_plans_reports_removed_added_changed(self):
        original = [
            {"step_number": 1, "description": "Calculate the sum.", "required_tools": ["calculator"]},
            {"step_number": 2, "description": "Remember the result.", "required_tools": ["remember_fact"]},
        ]
        replanned = [
            {"step_number": 1, "description": "Calculate the sum.", "required_tools": ["calculator", "web_search"]},
            {"step_number": 2, "description": "Search for the answer instead.", "required_tools": ["web_search"]},
        ]
        diff = _diff_plans(original, replanned)
        assert diff["steps_removed"] == 1
        assert diff["steps_added"] == 1
        assert "remember_fact" in diff["removed_tools"]
        assert "web_search" in diff["added_tools"]
        assert diff["tools_changed"] == 1  # shared step: calculator → +web_search
        assert diff["capabilities_changed"] is True

    def test_diff_plans_identical_plans_are_empty(self):
        plan = [{"step_number": 1, "description": "Do the thing.", "required_tools": ["calculator"]}]
        diff = _diff_plans(plan, [dict(step) for step in plan])
        assert diff["steps_removed"] == 0 and diff["steps_added"] == 0
        assert diff["tools_changed"] == 0 and diff["capabilities_changed"] is False

    def test_tool_from_observation_best_effort(self):
        assert _tool_from_observation("calculator ok: {}") == "calculator"
        assert _tool_from_observation("Result: 41971") == "Result"
        assert _tool_from_observation("some long sentence with spaces") == "tool"
        assert _tool_from_observation("") == "tool"
        long_prefix = "a" * 40
        assert _tool_from_observation(f"{long_prefix}: x") == "tool"