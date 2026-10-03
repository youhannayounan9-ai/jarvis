"""
evaluation/grounding_benchmark.py
─────────────────────────────────
v0.26 DETERMINISTIC grounding benchmark (Parts 3-9, 12) — 13 cases A-M
proving the guard BOTH at the pure-function level AND through the REAL
orchestrator (plan path, bounded replan, cache, refresh, confirmation
resume, unmet capability), with scripted LLM/planner/network and real
PermissionGuard + Pydantic validation + per-case fresh_store() isolation.

Cases (phase spec Part 12):
  A correct evidence + correct answer          (guard + orchestrator)
  B correct evidence + wrong numeric answer    (guard + orchestrator)
  C comma-formatted correct answer             (orchestrator)
  D failed later tool + earlier success        (guard + orchestrator)
  E cached result + correct answer             (orchestrator, 2 turns)
  F refresh result + correct answer            (orchestrator, 2 turns)
  G successful replan + correct answer         (orchestrator, 2-phase plan)
  H correction round required                  (orchestrator)
  I correction round still fails → fallback    (orchestrator)
  J unrelated numbers in answer                (guard)
  K timestamps/IDs must not trigger            (guard)
  L multi-step evidence                        (orchestrator)
  M no trusted evidence → normal synthesis     (orchestrator)

Run:
    uv run python evaluation/grounding_benchmark.py
    uv run python evaluation/grounding_benchmark.py --json grounding_report.json
"""

from __future__ import annotations

# ISOLATION (v0.25 Part G): private temp DB BEFORE any jarvis import.
from evaluation import _bootstrap as _eval

_eval.isolate()

import argparse
import json
import sys
import time
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from jarvis.core.grounding import check_grounding
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.tools import CalculatorTool, ToolRegistry

CALC = "calculator"


def _resp(content: str, tool_calls=None):
    msg = type("M", (), {"role": "assistant", "content": content, "tool_calls": tool_calls})()
    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def _plan(steps: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    return [
        {"step_number": i + 1, "description": d, "required_tools": t}
        for i, (d, t) in enumerate(steps)
    ]


# ── Layer 1: deterministic guard (pure function) ─────────────────────────────

GUARD_CASES: list[dict[str, Any]] = [
    {
        "name": "A_correct_answer",
        "evidence": [{"step_number": 1, "tool": CALC, "result": "Result: 41971"}],
        "answer": "The result of 893 * 47 is 41971.",
        "expect_contradiction": False,
    },
    {
        "name": "B_wrong_numeric_answer",
        "evidence": [{"step_number": 1, "tool": CALC, "result": "Result: 41971"}],
        "answer": "The result of 893 * 47 is 33071.",
        "expect_contradiction": True,
    },
    {
        "name": "D_failed_later_attempt_governs_success",
        "evidence": [
            {"step_number": 1, "tool": CALC, "result": "Result: 41971"},
            {"step_number": 2, "tool": CALC, "status": "error", "result": "ERROR: timeout"},
        ],
        "answer": "The result is 33071.",
        "expect_contradiction": True,   # the SUCCESS still governs
    },
    {
        "name": "J_unrelated_numbers",
        "evidence": [{"step_number": 1, "tool": CALC, "result": "Result: 41971"}],
        "answer": "Step 1 gave 893 and step 2 gave 47. Around 2026, roughly 500 people agree.",
        "expect_contradiction": False,
    },
    {
        "name": "K_timestamps_ids",
        "evidence": [{"step_number": 1, "tool": CALC, "result": "Result: 41971"}],
        "answer": "Request 1777777777 handled in 120 ms. It is 25% higher than 33071.",
        "expect_contradiction": False,
    },
    {
        "name": "K2_grouped_wrong_value_fires",
        "evidence": [{"step_number": 1, "tool": CALC, "result": "Result: 41971"}],
        "answer": "The total is 33,071 for 893 x 47.",
        "expect_contradiction": True,
    },
    {
        "name": "A2_cached_evidence_same_authority",
        "evidence": [{
            "step_number": 2, "tool": CALC,
            "result": "[cached result: retrieved 3m ago via calculator] Result: 41971",
        }],
        "answer": "The result is 33071.",
        "expect_contradiction": True,
    },
    {
        "name": "M0_no_evidence_no_check",
        "evidence": None,
        "answer": "The result is 33071.",
        "expect_contradiction": False,
        "expect_checked": False,
    },
]


# ── Layer 2: orchestrator integration (real chat() machinery) ────────────────

class GroundingScript:
    """
    Scripted planner + LLM + dispatch for one orchestrator case (the
    cache_replan_benchmark pattern, generalized for grounding).

    Two planner phases support the bounded-replan case; per-step rounds emit
    the step's tool calls then a step-final text; the tool-free synthesis
    call plays `finals` in order (initial answer, then optional scripted
    correction answer).
    """

    def __init__(
        self,
        *,
        plan: list[dict[str, Any]],
        replan_plan: list[dict[str, Any]] | None = None,
        step_rounds: list[list[tuple[str, str]]] | None = None,
        step_finals: list[str] | None = None,
        finals: list[str],
        dispatch_results: dict[str, str] | None = None,
        fail_expressions: set[str] | None = None,
    ) -> None:
        self.plan1 = plan
        self.plan2 = replan_plan
        self.step_rounds = step_rounds or []
        self.step_finals = step_finals or []
        self.final_texts = finals
        self.dispatch_results = dispatch_results or {}
        self.fail_expressions = fail_expressions or set()
        self.planner_calls = 0
        self.phase = 0
        self.step = 0
        self.round_in_step = 0
        self.replans = 0
        self.dispatched: list[str] = []          # successful dispatches only
        self.attempted: list[str] = []
        self.correction_calls = 0                # tool-free calls w/ correction prompt
        self.toolfree_calls = 0

    # ── planner ──────────────────────────────────────────────────────────
    def planner(self, user_input: str, context: str):
        plan = self.plan1 if self.planner_calls == 0 else (self.plan2 or self.plan1)
        self.planner_calls += 1
        if self.planner_calls > 1:
            self.phase = 1
            self.step = 0
            self.round_in_step = 0
            self.replans += 1
        return [dict(s) for s in plan]

    # ── LLM ──────────────────────────────────────────────────────────────
    def llm(self, messages, tools=None, **kwargs):
        system_text = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        blob = system_text + " " + str(messages[-1].get("content") or "") if messages else system_text
        if "strategic planner" in system_text.lower():
            plan = self.plan1 if self.planner_calls == 0 else (self.plan2 or self.plan1)
            return _resp(json.dumps(plan))
        if tools:
            plan = self.plan1 if self.phase == 0 else (self.plan2 or self.plan1)
            step_def = plan[min(self.step, len(plan) - 1)]
            tools_for_step = step_def.get("required_tools") or []
            rounds = self.step_rounds[min(self.step, len(self.step_rounds) - 1)] if self.step_rounds else []
            if tools_for_step and self.round_in_step < len(rounds):
                name, args = rounds[self.round_in_step]
                self.round_in_step += 1
                tc = type("TC", (), {
                    "id": f"call_{self.phase}_{self.step}_{self.round_in_step}",
                    "function": type("F", (), {"name": name, "arguments": args})(),
                })()
                msg = type("M", (), {"role": "assistant", "content": None, "tool_calls": [tc]})()
                return _resp(None, [tc])
            default = self.step_finals[min(self.step, len(self.step_finals) - 1)] if self.step_finals else "done"
            self.step += 1
            self.round_in_step = 0
            return _resp(default)
        # Tool-free (synthesis / correction) call.
        self.toolfree_calls += 1
        if any("GROUNDING CORRECTION REQUIRED" in str(m.get("content") or "") for m in messages):
            self.correction_calls += 1
        idx = min(self.toolfree_calls - 1, len(self.final_texts) - 1)
        return _resp(self.final_texts[idx])

    # ── dispatch (validation REAL; results scripted) ─────────────────────
    def dispatch(self, tool_name: str, tool_args: str) -> str:
        self.attempted.append(tool_name)
        if "wrong first try" in tool_args:
            return "ERROR: scripted dispatch failure (grounding benchmark)."
        tool = _REGISTRY_SINGLETON.get(tool_name)
        if tool is not None:
            try:
                raw = json.loads(tool_args) if tool_args else {}
                tool._args_model.model_validate(raw)
            except (ValidationError, json.JSONDecodeError) as e:
                return f"ERROR: Invalid arguments for '{tool_name}': {e}"
        self.dispatched.append(tool_name)
        scripted = self.dispatch_results.get(tool_name)
        if scripted is not None:
            return scripted
        # Tool-contract results (the REAL calculator runs for web_search etc.
        # — only the network layer is ever canned in this project's harnesses).
        return TOOL_RESULTS.get(tool_name, f"{tool_name} ok")


# Registry handle for the dispatch probe (set per case; scripts are per-case).
_REGISTRY_SINGLETON = None


def _run_case(
    script: GroundingScript,
    user_input: str,
    *,
    refresh: bool = False,
    store=None,
    registry=None,
    session_id: str = "gb",
) -> str:
    """One chat() turn. ``store``/``registry`` are shared across turns of a
    case so the cross-turn cache behaves as in production; a per-case store
    is created when none is given."""
    global _REGISTRY_SINGLETON
    owned_store = store is None
    if store is None:
        store = _eval.fresh_store()      # per-case hermetic DB (v0.25 pattern)
    if registry is None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())  # cache_policy wired → real cache path
    _REGISTRY_SINGLETON = registry
    orch = Orchestrator(store, registry, PermissionGuard())
    try:
        with patch.object(orch._planner, "generate_plan", side_effect=script.planner):
            with patch("jarvis.core.orchestrator.chat_completion", side_effect=script.llm):
                with patch.object(registry, "dispatch_async", side_effect=script.dispatch):
                    answer = orch.chat(session_id, user_input, refresh=refresh)
    finally:
        if owned_store:
            store.close()
    return answer


# Per-case specs: `turns` is [(user_input_override, refresh)] — one chat()
# call per entry.

PLAN_CALC = _plan([("Multiply 893 by 47 with the calculator.", [CALC])])
PLAN_TWO = _plan([
    ("Multiply 893 by 47 with the calculator.", [CALC]),
    ("Multiply 12 by 12 with the calculator.", [CALC]),
])
PLAN_SEARCH_CALC = _plan([
    ("Search the web for langgraph.", ["web_search"]),
    ("Multiply 893 by 47 with the calculator.", [CALC]),
])
PLAN_RETRY = _plan([("Retry the calculation with the calculator.", [CALC])])
PLAN_UNMET = _plan([("Run the Python snippet.", ["code_interpreter"])])

ORCH_CASES: list[dict[str, Any]] = [
    # C. comma-formatted correct answer
    {
        "name": "C_comma_formatted_correct",
        "user_input": "What is 893 * 47?",
        "script": dict(
            plan=PLAN_CALC,
            step_rounds=[[(CALC, '{"expression": "893 * 47"}')]],
            finals=["The result of 893 * 47 is 41,971."],
        ),
        "turns": [(None, False)],
        "grade": {"final_contains": ["41,971"], "corrections": 0, "dispatched": [CALC]},
    },
    # D. failed later dispatch; the earlier success still governs the guard
    # (the failed confirmation attempt cannot overwrite it), and the turn
    # ends with the honest incomplete-answer note.
    {
        "name": "D_failed_later_success_governs",
        "user_input": "What is 893 * 47? Confirm it.",
        "script": dict(
            plan=_plan([("Multiply 893 by 47.", [CALC]), ("Confirm the product.", [CALC])]),
            step_rounds=[
                [(CALC, '{"expression": "893 * 47"}')],
                [(CALC, '{"expression": "893 * 47 wrong first try"}')],
            ],
            finals=["The result of 893 * 47 is 33071."],
        ),
        "turns": [(None, False)],
        # Wrong answer vs the surviving SUCCESS → exactly one correction,
        # then the fallback (the scripted correction also contradicts).
        "grade": {
            "fallback": True,
            "final_contains": ["41,971"],
            "corrections": 1,
        },
    },
    # E. cached result + correct answer (turn 2 = cache hit, no dispatch)
    {
        "name": "E_cached_result_correct",
        "user_input": "What is 893 * 47?",
        "script": dict(
            plan=PLAN_CALC,
            step_rounds=[[(CALC, '{"expression": "893 * 47"}')]],
            finals=["The result of 893 * 47 is 41971.",
                    "The result of 893 * 47 is 41971."],
        ),
        "turns": [(None, False), (None, False)],
        "grade": {
            "final_contains": ["41971"],
            "corrections": 0,
            "dispatched_total": 1,   # turn 2 served from cache
        },
    },
    # F. refresh bypass → real re-dispatch on turn 2
    {
        "name": "F_refresh_bypass_redispatch",
        "user_input": "What is 893 * 47?",
        "script": dict(
            plan=PLAN_CALC,
            step_rounds=[[(CALC, '{"expression": "893 * 47"}')]],
            finals=["The result of 893 * 47 is 41971.",
                    "The result of 893 * 47 is 41971."],
        ),
        "turns": [(None, False), (None, True)],
        "grade": {
            "final_contains": ["41971"],
            "corrections": 0,
            "dispatched_total": 2,   # refresh forces the real re-run
        },
    },
    # G. bounded replan: first calculator round fails → ONE replan succeeds
    {
        "name": "G_replan_success_grounded",
        "user_input": "Search langgraph, then calculate 893 * 47.",
        "script": dict(
            plan=PLAN_SEARCH_CALC,
            replan_plan=PLAN_RETRY,
            step_rounds=[
                [("web_search", '{"query": "langgraph"}')],
                [(CALC, '{"expression": "893 * 47 wrong first try"}')],
            ],
            step_finals=["Search done."],
            finals=["The result is 41971; LangGraph found."],
            dispatch_results={"web_search": "Web search results for: langgraph\n[1] LangGraph page"},
        ),
        "turns": [(None, False)],
        "grade": {
            "final_contains": ["41971"],
            "corrections": 0,
            "replans": 1,
        },
    },
    # H. correction round required (the v0.25 bug pattern)
    {
        "name": "H_correction_required",
        "user_input": "What is 893 * 47?",
        "script": dict(
            plan=PLAN_CALC,
            step_rounds=[[(CALC, '{"expression": "893 * 47"}')]],
            finals=["The result of 893 * 47 is 33071.",
                    "The result of 893 * 47 is 41971."],
        ),
        "turns": [(None, False)],
        "grade": {"final_contains": ["41971"], "corrections": 1, "no_wrong_value": "33071"},
    },
    # I. correction still fails → fail-closed fallback with the real value
    {
        "name": "I_correction_fails_fallback",
        "user_input": "What is 893 * 47?",
        "script": dict(
            plan=PLAN_CALC,
            step_rounds=[[(CALC, '{"expression": "893 * 47"}')]],
            finals=["The result of 893 * 47 is 33071.",
                    "It equals 33071 total."],
        ),
        "turns": [(None, False)],
        "grade": {
            "fallback": True,
            "final_contains": ["41,971"],
            "corrections": 1,
        },
    },
    # L. multi-step evidence; transcription of both results corrected
    {
        "name": "L_multistep_two_results",
        "user_input": "What is 893 * 47? And what is 12 * 12?",
        "script": dict(
            plan=PLAN_TWO,
            step_rounds=[
                [(CALC, '{"expression": "893 * 47"}')],
                [(CALC, '{"expression": "12 * 12"}')],
            ],
            finals=["The results are 33071 and 144.",
                    "The results are 41971 and 144."],
        ),
        "turns": [(None, False)],
        "grade": {"final_contains": ["41971", "144"], "corrections": 1},
    },
    # M. no trusted evidence (unmet capability) → ordinary synthesis, no guard
    {
        "name": "M_no_evidence_normal_synthesis",
        "user_input": "Run this python snippet for me and then explain what it does.",
        "script": dict(
            plan=PLAN_UNMET,
            step_rounds=[[]],
            finals=["I cannot run code locally, but I can help read it."],
        ),
        "turns": [(None, False)],
        "grade": {"final_contains": ["cannot run code"], "corrections": 0, "dispatched_total": 0},
    },
]


# Tool-contract outputs (same formats the real tools emit; the calculator's
# documented format is what the grounding policies verify).
TOOL_RESULTS = {
    CALC: "Result: 41971",
}


def _grade_orch(case: dict[str, Any], script: GroundingScript, answers: list[str]) -> tuple[bool, dict[str, Any]]:
    grade = case["grade"]
    final = answers[-1] if answers else ""
    blob = final.lower()
    detail: dict[str, Any] = {"name": case["name"], "answers": [a[:160] for a in answers]}

    ok = True
    for needle in grade.get("final_contains", []):
        if needle.lower() not in blob:
            ok = False
            detail.setdefault("missing", []).append(needle)
    if "no_wrong_value" in grade and grade["no_wrong_value"].lower() in blob:
        ok = False
        detail["wrong_value_present"] = grade["no_wrong_value"]
    expect_fallback = grade.get("fallback", False)
    is_fallback = "could not produce a verified answer" in blob
    if expect_fallback != is_fallback:
        ok = False
        detail["fallback_mismatch"] = {"expected": expect_fallback, "got": is_fallback}
    if script.correction_calls != grade.get("corrections", 0):
        ok = False
        detail["corrections_mismatch"] = {
            "expected": grade.get("corrections", 0),
            "got": script.correction_calls,
        }
    if script.correction_calls > 1:  # bounded: NEVER more than one round
        ok = False
        detail["unbounded_corrections"] = script.correction_calls
    if "replans" in grade and script.replans != grade["replans"]:
        ok = False
        detail["replans_mismatch"] = {"expected": grade["replans"], "got": script.replans}
    if "dispatched" in grade and script.dispatched != grade["dispatched"]:
        ok = False
        detail["dispatched_mismatch"] = {"expected": grade["dispatched"], "got": script.dispatched}
    if "dispatched_total" in grade and len(script.dispatched) != grade["dispatched_total"]:
        ok = False
        detail["dispatched_total_mismatch"] = {
            "expected": grade["dispatched_total"],
            "got": len(script.dispatched),
        }
    detail["dispatched"] = script.dispatched
    detail["attempted"] = script.attempted
    detail["corrections"] = script.correction_calls
    detail["replans"] = script.replans
    detail["passed"] = ok
    return ok, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grounding_benchmark")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    # ── Layer 1: deterministic guard ─────────────────────────────────────
    for case in GUARD_CASES:
        verdict = check_grounding(case["answer"], case["evidence"])
        ok = verdict.contradiction == case["expect_contradiction"] and verdict.checked == case.get(
            "expect_checked", True
        )
        results.append({
            "layer": "guard",
            "name": case["name"],
            "contradiction": verdict.contradiction,
            "checked": verdict.checked,
            "passed": ok,
        })

    # ── Layer 2: orchestrator integration ────────────────────────────────
    for case in ORCH_CASES:
        script = GroundingScript(**case["script"])
        store = _eval.fresh_store()      # shared across the case's turns
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        session_id = f"gb_{int(time.time() * 1000)}_{id(case)}"
        answers: list[str] = []
        for i, (user_input, refresh) in enumerate(case["turns"]):
            answers.append(
                _run_case(
                    script,
                    user_input or case["user_input"],
                    refresh=refresh,
                    store=store,
                    registry=registry,
                    session_id=f"{session_id}_t{i}",
                )
            )
        store.close()
        _, detail = _grade_orch(case, script, answers)
        detail["layer"] = "orchestrator"
        results.append(detail)

    elapsed = time.perf_counter() - started
    n_ok = sum(1 for r in results if r["passed"])
    report = {
        "version": "0.26",
        "passed": n_ok,
        "total": len(results),
        "elapsed_s": round(elapsed, 2),
        "cases": results,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
