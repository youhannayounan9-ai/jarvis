"""
evaluation/cache_replan_benchmark.py
────────────────────────────────────
v0.24 DETERMINISTIC cache + replan benchmark (Part N).

Runs through the REAL orchestrator (plan validation, PermissionGuard, the
v0.23 dispatch ledger, the v0.24 result cache and the ONE bounded replan all
real); only the LLM is scripted:

  CACHE       — cross-turn hit (real dispatch once), argument sensitivity,
                session isolation, freshness bypass, KB invalidation,
                kill switch, provenance framing.
  REPLAN      — failed plan triggers exactly ONE replan, complete plans
                never replan, recursion impossible, second failure yields a
                truthful incomplete answer.
  NORMALIZE   — calculator whitespace/paren/value equivalence; verbatim
                keys for path-like args; unsafe strings stay distinct.

Run:
    uv run python evaluation/cache_replan_benchmark.py
    uv run python evaluation/cache_replan_benchmark.py --json report.json
Exit code 0 iff every case passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any
from unittest.mock import patch

# ISOLATION (v0.25 Part G): shared bootstrap — private temp DB BEFORE any
# jarvis import (the real jarvis.db would serve legitimate cache hits across
# processes and invalidate the measurement).
from evaluation import _bootstrap as _eval

_eval.isolate()

from jarvis.config import settings
from jarvis.core.orchestrator import Orchestrator
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


def _new_turn(orch: Orchestrator) -> None:
    """chat() replaces the ledger at entry; simulate that between turns."""
    from jarvis.core.dispatch_guard import DispatchLedger

    orch._dispatch_ledger = DispatchLedger()


async def _counting_dispatch(calls: dict[str, int], payload: str = "payload"):
    async def _dispatch(tool_name: str, tool_args_json: str) -> str:
        calls[tool_name] = calls.get(tool_name, 0) + 1
        return f"{payload} #{calls[tool_name]}"

    return _dispatch


# ── Cache cases ───────────────────────────────────────────────────────────────


async def case_cross_turn_hit(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls, "web results")):
        await orch._dispatch_with_permissions_async("t1", "web_search", '{"query": "q"}', "c1")
        _new_turn(orch)
        second = await orch._dispatch_with_permissions_async("t1", "web_search", '{"query": "q"}', "c2")
    hit = "cached result" in second
    passed = calls["web_search"] == 1 and hit and "provenance" not in second
    store.close()
    return {
        "name": "cache: cross-turn hit",
        "category": "cache",
        "passed": passed,
        "detail": {"real_dispatches": calls["web_search"], "hit": hit},
    }


async def case_different_args_distinct(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls)):
        await orch._dispatch_with_permissions_async("t1", "web_search", '{"query": "a"}', "c1")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "web_search", '{"query": "b"}', "c2")
    passed = calls["web_search"] == 2
    store.close()
    return {
        "name": "cache: different arguments not reused",
        "category": "cache",
        "passed": passed,
        "detail": {"real_dispatches": calls["web_search"]},
    }


async def case_calculator_equivalence(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls, "Result: 4")):
        await orch._dispatch_with_permissions_async("t1", "calculator", '{"expression": "2+2"}', "c1")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "calculator", '{"expression": "2 + 2"}', "c2")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "calculator", '{"expression": "(2+2)"}', "c3")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "calculator", '{"expression": "5-1"}', "c4")
    passed = calls["calculator"] == 1  # all four share ONE entry (value "4")
    store.close()
    return {
        "name": "normalize: calculator value equivalence",
        "category": "normalization",
        "passed": passed,
        "detail": {"real_dispatches": calls["calculator"], "expected": 1},
    }


async def case_path_verbatim(registry: ToolRegistry) -> dict[str, Any]:
    """Two path spellings that differ by inner spacing must NOT share an entry."""
    from jarvis.core.result_cache import ResultCache

    policy = ReadFileTool.cache_policy
    k1 = ResultCache.cache_key("read_file", policy, '{"path": "my file.txt"}')
    k2 = ResultCache.cache_key("read_file", policy, '{"path": "my  file.txt"}')
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls)):
        await orch._dispatch_with_permissions_async("t1", "read_file", '{"path": "my file.txt"}', "c1")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "read_file", '{"path": "my  file.txt"}', "c2")
    passed = k1 != k2 and calls["read_file"] == 2
    store.close()
    return {
        "name": "normalize: paths stay verbatim",
        "category": "normalization",
        "passed": passed,
        "detail": {"distinct_keys": k1 != k2, "real_dispatches": calls["read_file"]},
    }


async def case_session_isolation(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls)):
        await orch._dispatch_with_permissions_async("sessA", "list_directory", '{"path": "."}', "c1")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("sessA", "list_directory", '{"path": "."}', "c2")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("sessB", "list_directory", '{"path": "."}', "c3")
    passed = calls["list_directory"] == 2  # same session reuses; other session re-runs
    store.close()
    return {
        "name": "cache: session isolation for user files",
        "category": "cache",
        "passed": passed,
        "detail": {"real_dispatches": calls["list_directory"], "expected": 2},
    }


async def case_kb_invalidation(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls, "kb evidence")):
        await orch._dispatch_with_permissions_async("t1", "search_knowledge", '{"query": "roadmap"}', "c1")
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "search_knowledge", '{"query": "roadmap"}', "c2")
        store.upsert_knowledge_document(
            document_id="d1", source_path="x.md", filename="x.md",
            media_type="text/markdown", size_bytes=4, content_hash="h",
            chunk_count=2, parser_version="v1",
        )
        _new_turn(orch)
        await orch._dispatch_with_permissions_async("t1", "search_knowledge", '{"query": "roadmap"}', "c3")
    passed = calls["search_knowledge"] == 2  # ingest invalidated the cache
    store.close()
    return {
        "name": "cache: KB ingest invalidates knowledge retrieval",
        "category": "cache",
        "passed": passed,
        "detail": {"real_dispatches": calls["search_knowledge"], "expected": 2},
    }


async def case_freshness_bypass(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls, "fresh")):
        await orch._dispatch_with_permissions_async("t1", "web_search", '{"query": "q"}', "c1")
        _new_turn(orch)
        orch._freshness_request = True  # chat() sets this for "latest ..." asks
        await orch._dispatch_with_permissions_async("t1", "web_search", '{"query": "q"}', "c2")
    passed = calls["web_search"] == 2
    store.close()
    return {
        "name": "cache: freshness request forces real retrieval",
        "category": "cache",
        "passed": passed,
        "detail": {"real_dispatches": calls["web_search"], "expected": 2},
    }


async def case_kill_switch(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}
    with patch.object(settings, "JARVIS_DISABLE_RESULT_CACHE", True):
        with patch.object(registry, "dispatch_async", side_effect=await _counting_dispatch(calls)):
            await orch._dispatch_with_permissions_async("t1", "calculator", '{"expression": "6*7"}', "c1")
            _new_turn(orch)
            await orch._dispatch_with_permissions_async("t1", "calculator", '{"expression": "6*7"}', "c2")
    passed = calls["calculator"] == 2 and store.cache_stats()["entries"] == 0
    store.close()
    return {
        "name": "cache: kill switch disables reuse",
        "category": "cache",
        "passed": passed,
        "detail": {"real_dispatches": calls["calculator"], "entries": store.cache_stats()["entries"] if not store._conn else 0},
    }


async def case_never_cache_side_effects(registry: ToolRegistry) -> dict[str, Any]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    calls: dict[str, int] = {}

    async def stateful(tool_name: str, tool_args_json: str) -> str:
        calls[tool_name] = calls.get(tool_name, 0) + 1
        return f"ok #{calls[tool_name]}"

    with patch.object(registry, "dispatch_async", side_effect=stateful):
        for tool, args in (("get_current_datetime", "{}"), ("recall_facts", '{"query": "x"}')):
            await orch._dispatch_with_permissions_async("t1", tool, args, "c1")
            _new_turn(orch)
            await orch._dispatch_with_permissions_async("t1", tool, args, "c2")
    passed = calls["get_current_datetime"] == 2 and calls["recall_facts"] == 2
    store.close()
    return {
        "name": "cache: state-dependent tools never reused",
        "category": "cache",
        "passed": passed,
        "detail": {"calls": calls},
    }


# ── Replan cases (scripted planner + LLM, real execution machinery) ──────────


class _ReplanScript:
    """Two-phase scripted LLM: original plan fails → replan succeeds/fails."""

    def __init__(self, plan1, plan2, dispatch_fail_phase: set[int], step_finals, final_texts):
        self.plan1 = plan1
        self.plan2 = plan2
        self.fail_phases = dispatch_fail_phase
        self.step_finals = step_finals
        self.final_texts = final_texts
        self.planner_calls = 0
        self.phase = 0
        self.step = 0
        self.round_in_step = 0
        self.replans = 0

    def planner(self, user_input: str, context: str):
        plan = self.plan1 if self.planner_calls == 0 else self.plan2
        self.planner_calls += 1
        if self.planner_calls > 1:
            self.phase = 1
            self.step = 0
            self.round_in_step = 0
            self.replans += 1
        return [dict(s) for s in plan]

    def llm(self, messages, tools=None, **kwargs):
        system_text = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        if "strategic planner" in system_text.lower():
            plan = self.plan1 if self.planner_calls == 0 else self.plan2
            return _resp(json.dumps(plan))
        if tools:
            # Two calls per step: ROUND 1 emits the tool call the step needs
            # (the dispatch result — including the scripted ERROR — becomes
            # the observation); ROUND 2 wraps up with that step's result text.
            plan = self.plan1 if self.phase == 0 else self.plan2
            step_def = plan[min(self.step, len(plan) - 1)]
            tools_for_step = step_def.get("required_tools") or []
            if self.round_in_step == 0 and tools_for_step:
                self.round_in_step = 1
                calls = [
                    type("TC", (), {
                        "id": f"call_{self.phase}_{self.step}_{i}",
                        "function": type("F", (), {
                            "name": t,
                            "arguments": "{\"expression\": \"893 * 47\"}" if t == "calculator" else "{\"fact\": \"893*47 = 41971\"}"},
                        )(),
                    })()
                    for i, t in enumerate(tools_for_step, start=1)
                ]
                msg = type("M", (), {"role": "assistant", "content": None, "tool_calls": calls})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
            finals = self.step_finals[self.phase]
            default = finals[min(self.step, len(finals) - 1)]
            self.step += 1
            self.round_in_step = 0
            return _resp(default)
        return _resp(self.final_texts[min(self.phase, len(self.final_texts) - 1)])

    def dispatch(self, tool_name: str, tool_args: str) -> str:
        if self.phase in self.fail_phases and tool_name == "calculator":
            return f"ERROR: scripted calculator failure (phase {self.phase})"
        return f"{tool_name} ok"


def _resp(content: str):
    msg = type("M", (), {"role": "assistant", "content": content, "tool_calls": None})()
    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


PLAN_A = [
    {"step_number": 1, "description": "Calculate 893 * 47 with the calculator.", "required_tools": ["calculator"]},
    {"step_number": 2, "description": "Remember the calculated result as a fact.", "required_tools": ["remember_fact"]},
]
PLAN_RETRY = [
    {"step_number": 1, "description": "Retry the calculation with the calculator.", "required_tools": ["calculator"]},
]


def _run_replan_case(script: _ReplanScript, registry: ToolRegistry) -> tuple[Orchestrator, SessionStore, str]:
    store = _eval.fresh_store()  # per-case hermetic DB (v0.25 benchmark isolation)
    orch = Orchestrator(store, registry, PermissionGuard())
    with patch.object(orch._planner, "generate_plan", side_effect=script.planner):
        with patch("jarvis.core.orchestrator.chat_completion", side_effect=script.llm):
            with patch.object(registry, "dispatch_async", side_effect=script.dispatch):
                answer = orch.chat("rpbench", "Calculate 893 * 47 and remember the result.")
    return orch, store, answer


def case_replan_on_failure(registry: ToolRegistry) -> dict[str, Any]:
    script = _ReplanScript(
        PLAN_A, PLAN_RETRY, {0},
        [["ERROR: calculator unavailable", "Step 2 complete: remembered."], ["Replan ok: 41971."]],
        ["(unreachable)", "41971 via replan."],
    )
    _, store, answer = _run_replan_case(script, registry)
    passed = script.planner_calls == 2 and script.replans == 1 and "41971" in answer
    store.close()
    return {
        "name": "replan: failed plan triggers exactly ONE replan",
        "category": "replan",
        "passed": passed,
        "detail": {"planner_calls": script.planner_calls, "replans": script.replans},
    }


def case_complete_no_replan(registry: ToolRegistry) -> dict[str, Any]:
    script = _ReplanScript(
        PLAN_A, [], set(),
        [["Step 1 complete: 41971.", "Step 2 complete: remembered."]],
        ["893 * 47 = 41971 and remembered."],
    )
    _, store, answer = _run_replan_case(script, registry)
    passed = script.planner_calls == 1 and script.replans == 0 and "41971" in answer
    store.close()
    return {
        "name": "replan: complete plan never replans",
        "category": "replan",
        "passed": passed,
        "detail": {"planner_calls": script.planner_calls, "replans": script.replans},
    }


def case_no_recursive_replan(registry: ToolRegistry) -> dict[str, Any]:
    script = _ReplanScript(
        PLAN_A, PLAN_RETRY, {0, 1},
        [["ERROR: attempt 1 failed", "Step 2 complete: remembered."], ["ERROR: retry failed"]],
        ["(unused)", "The calculation remains incomplete; I could not finish it."],
    )
    _, store, answer = _run_replan_case(script, registry)
    passed = (
        script.planner_calls == 2 and script.replans == 1
        and "incomplete" in answer.lower()
    )
    store.close()
    return {
        "name": "replan: recursion impossible, truthful incomplete answer",
        "category": "replan",
        "passed": passed,
        "detail": {"planner_calls": script.planner_calls, "replans": script.replans},
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def case_replan_on_failure_sync(registry: ToolRegistry) -> dict[str, Any]:
    return case_replan_on_failure(registry)


def case_complete_no_replan_sync(registry: ToolRegistry) -> dict[str, Any]:
    return case_complete_no_replan(registry)


def case_no_recursive_replan_sync(registry: ToolRegistry) -> dict[str, Any]:
    return case_no_recursive_replan(registry)


_CASES = [
    case_cross_turn_hit,
    case_different_args_distinct,
    case_calculator_equivalence,
    case_path_verbatim,
    case_session_isolation,
    case_kb_invalidation,
    case_freshness_bypass,
    case_kill_switch,
    case_never_cache_side_effects,
    case_replan_on_failure_sync,
    case_complete_no_replan_sync,
    case_no_recursive_replan_sync,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cache_replan_benchmark")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    records = []
    for case_fn in _CASES:
        # v0.25: fresh registry per case (stateless today, but keeps every
        # case hermetic if a tool ever gains in-memory state) alongside
        # the per-case hermetic store from _eval.fresh_store().
        started = time.perf_counter()
        result = case_fn(_build_registry())
        record = asyncio.run(result) if asyncio.iscoroutine(result) else result
        record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        records.append(record)
        status = "PASS" if record["passed"] else "FAIL"
        print(f"  {record['name']:<58} {record['duration_ms']:>7}ms  {status}", flush=True)
        if not record["passed"]:
            print(f"    -> {record['detail']}")

    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    by_category: dict[str, dict[str, float]] = {}
    for r in records:
        cat = by_category.setdefault(r["category"], {"cases": 0, "passed": 0})
        cat["cases"] += 1
        cat["passed"] += 1 if r["passed"] else 0
    print(f"CACHE+REPLAN BENCHMARK: {passed}/{total} passed")
    print(f"summary: {json.dumps({'total': total, 'passed': passed, 'by_category': by_category})}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": {"total": total, "passed": passed, "by_category": by_category}, "cases": records}, fh, indent=2)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
