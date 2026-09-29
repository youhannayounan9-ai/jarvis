"""
live_cache_replan_eval.py
────────────────────────
v0.24 LIVE verification against the REAL Ollama model (Part O/V) —
manual-only, requires Ollama + qwen2.5:7b, NOT part of pytest.

What is real:  the model (planner + step execution + synthesis), the
               orchestrator's full v0.24 path — cross-turn result cache,
               freshness bypass, the ONE bounded replan, truthful
               incomplete-answer synthesis, PermissionGuard, validation.
What is canned: ONLY the network layer of web_search / wikipedia_summary
               (deterministic payloads; the calculator is REAL), so cache
               metrics measure REUSE, not network noise.

Cases (multiple reps; small sample — never presented as universal):
  1. cross-turn knowledge question   → turn-2 cache HIT (no real retrieval)
  2. cross-turn web query            → turn-2 cache HIT
  3. calculator across turns         → turn-2 cache HIT (value-equal expr)
  4. freshness wording ("latest")    → turn-2 cache BYPASS (real re-run)
  5. multi-step incomplete plan      → exactly ONE replan, then completion
  6. complete plan                   → NO replan
  7. no-tool explanation             → no cache interaction
  8. explicit refresh=True (v0.25)   → turn-2 cache BYPASS without any
                                        freshness wording (Part D/P)
  9. grounding (v0.25 Part B/P)      → hard multiplication: the model cannot
                                        know the value; the FINAL synthesis
                                        must carry the calculator's exact
                                        result (evidence beats guesswork)
 10. knowledge-step replan (v0.25)   → wikipedia step fails once → ONE replan
                                        → grounded answer

Run:
    uv run python live_cache_replan_eval.py --reps 2 --json live_cache_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ISOLATION (v0.25 Part G): explicit, not incidental via the benchmark import.
from evaluation import _bootstrap as _eval

_eval.isolate()

import evaluation.cache_replan_benchmark as _crb  # reuse the registry builder

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.tools import (
    CalculatorTool,
    WebSearchTool,
    WikipediaSummaryTool,
)

WEB_PAYLOAD = (
    'Web search results for: "LangGraph news"\n'
    "[1] LangGraph adoption grows for agent orchestration\n"
    "    URL: https://example.com/langgraph-news\n"
    "    Excerpt: LangGraph provides graph-based orchestration with checkpointing."
)
WIKI_PAYLOAD = (
    "Wikipedia summary for Alan Turing: English mathematician and computer "
    "scientist, father of theoretical computer science and artificial intelligence."
)


class CannedNetwork:
    """Patch ONLY the network layer of web tools; validation is real."""

    def __init__(self, record: dict[str, Any], registry=None):
        self.record = record
        self.registry = registry
        self._web_search = WebSearchTool()
        self._wiki = WikipediaSummaryTool()
        self._calc = CalculatorTool()

    async def dispatch(self, tool_name: str, tool_args: str) -> str:
        self.record["attempts"].append(tool_name)
        import json as _json

        from pydantic import ValidationError

        # Only the NETWORK layer is canned; every other registered tool runs
        # for real through its own instance (remember_fact writes to the
        # store, etc.).
        tool = {
            "web_search": self._web_search,
            "wikipedia_summary": self._wiki,
            "calculator": self._calc,
        }.get(tool_name) or (self.registry.get(tool_name) if self.registry else None)
        if tool is None:
            return f"ERROR: Unknown tool '{tool_name}'."
        try:
            raw = _json.loads(tool_args) if tool_args else {}
            tool._args_model.model_validate(raw)
        except (ValidationError, _json.JSONDecodeError) as e:
            return f"ERROR: Invalid arguments for '{tool_name}': {e}"
        self.record["dispatched"].append(tool_name)
        self.record["live_tool_calls"] += 1
        if tool_name == "calculator":
            return self._calc.run(**raw)  # REAL calculator
        if tool_name == "web_search":
            return WEB_PAYLOAD
        return WIKI_PAYLOAD


def _run_turn(
    orch: Orchestrator,
    registry,
    session_id: str,
    user_input: str,
    record: dict[str, Any],
    dispatch,
    replans: list[int],
    events: list[dict[str, Any]],
    refresh: bool = False,
) -> str:
    replans_before = len(replans)

    def on_event(e: dict[str, Any]) -> None:
        events.append(e)
        if e.get("type") == "replan":
            replans.append(1)

    started = time.time()
    with patch.object(registry, "dispatch_async", side_effect=dispatch):
        response = orch.chat(
            session_id, user_input, on_event=on_event, refresh=refresh
        ) or ""
    record["turn_latency_s"] = round(time.time() - started, 1)
    record["replans_this_turn"] = len(replans) - replans_before
    return response


def run_case(case: dict[str, Any], registry, reps: int) -> list[dict[str, Any]]:
    records = []
    for rep in range(reps):
        store = SessionStore()
        orch = Orchestrator(store, registry, PermissionGuard())
        record: dict[str, Any] = {
            "case": case["name"], "rep": rep + 1,
            "attempts": [], "dispatched": [], "live_tool_calls": 0,
            "turns": [], "cache": {},
        }
        canned = CannedNetwork(record, registry)
        session_id = f"live_cache_{case['name']}_{rep}_{int(time.time() * 1000)}"
        # ``fresh_session_each_turn`` (v0.25): for refresh/freshness cases the
        # second turn runs in a NEW session — the GLOBAL cache is what's under
        # test, and session history answering from memory would pre-empt the
        # measurement (no tool round → no bypass to observe).
        fresh_each_turn = bool(case.get("fresh_session_each_turn"))
        # Scripted-failure cases need the target tool's first attempt to REALLY
        # dispatch: an earlier case in the same process may have cached the
        # same retrieval (global scope), and a cache hit never reaches the
        # dispatcher — so the scripted failure would never inject. Purge the
        # target tool's entries per rep (harness-side only; v0.25 Part G keeps
        # the DB isolated anyway).
        fail_tool_prep = case.get("scripted_fail_tool")
        if fail_tool_prep:
            with store._lock:
                store._conn.execute(
                    "DELETE FROM result_cache WHERE tool_name = ?", (fail_tool_prep,)
                )
                store._conn.commit()
        replans: list[int] = []
        events: list[dict[str, Any]] = []
        answers: list[str] = []
        ok = True
        notes: list[str] = []
        for t, turn in enumerate(case["turns"], start=1):
            record["attempts"].clear()
            record["dispatched"].clear()
            if fresh_each_turn and t > 1:
                session_id = f"{session_id}_t{t}"
            stats_before = store.cache_stats()
            answer = _run_turn(
                orch, registry, session_id, turn["input"], record,
                canned.dispatch, replans, events,
                refresh=bool(turn.get("refresh")),
            )
            answers.append(answer)
            stats_after = store.cache_stats()
            record["turns"].append({
                "input": turn["input"],
                "dispatched": list(record["dispatched"]),
                "latency_s": record["turn_latency_s"],
                "replans": record["replans_this_turn"],
                "answer": answer[:300],
            })
            # First tool-bearing turn is a MISS (nothing cached yet);
            # later identical turns should HIT unless freshness-bypassed.
            record["cache"][f"turn{t}"] = {
                "entries": stats_after["entries"],
                "hits": stats_after["hits"],
            }
        # Metrics.
        if case.get("expect_hit_second_turn"):
            # The MEASURE is "no new network work on the repeat": zero real
            # dispatches on turn 2 AND a live answer that is not a refusal.
            # (Session history can legitimately answer the restated question
            # without ANY tool round — the expensive retrieval was still
            # avoided; a hit counter bump additionally proves the CACHE
            # specifically served it.)
            second_turn_calls = len(record["turns"][1]["dispatched"]) if len(record["turns"]) > 1 else 0
            second_answer = record["turns"][1]["answer"] if len(record["turns"]) > 1 else ""
            refused = (
                "haven't" in second_answer.lower()
                or "do not have" in second_answer.lower()
                or "unable to" in second_answer.lower()
            )
            hit = second_turn_calls == 0 and not refused
            record["cache"]["turn2_hit"] = hit
            record["cache"]["turn2_dispatches"] = second_turn_calls
            if not hit:
                ok = False
                notes.append(f"expected turn-2 reuse (no re-retrieval); dispatched={second_turn_calls}")
        if case.get("expect_bypass_second_turn"):
            second_turn_calls = len(record["turns"][1]["dispatched"]) if len(record["turns"]) > 1 else 0
            bypass = second_turn_calls >= 1
            record["cache"]["turn2_bypassed"] = bypass
            if not bypass:
                ok = False
                notes.append("expected freshness-word turn-2 to bypass cache and re-run")
        if case.get("expect_replans") is not None:
            got = sum(t["replans"] for t in record["turns"])
            record["replans_total"] = got
            if got != case["expect_replans"]:
                ok = False
                notes.append(f"expected {case['expect_replans']} replan(s), got {got}")
        if case.get("expect_no_tools"):
            if any(t["dispatched"] for t in record["turns"]):
                ok = False
                notes.append("expected a tool-free answer")
        if case.get("expect_tool_used"):
            required = case["expect_tool_used"]
            if required not in record["dispatched"]:
                ok = False
                notes.append(f"expected the {required} tool to run for real")
        grader = case.get("grader")
        if grader:
            grounded = all(grader(a) for a in answers)
            record["grounded"] = grounded
            if not grounded:
                ok = False
                notes.append("grounding grader failed on at least one turn")
        record["ok"] = ok
        record["notes"] = notes
        records.append(record)
        store.close()
    return records


CASES: list[dict[str, Any]] = [
    {
        "name": "wiki_cross_turn_hit",
        "turns": [
            {"input": "Give me a short summary of Alan Turing from Wikipedia."},
            {"input": "Give me a short summary of Alan Turing from Wikipedia."},
        ],
        "expect_hit_second_turn": True,
        "grader": lambda a: "turing" in a.lower(),
    },
    {
        "name": "web_cross_turn_hit",
        "turns": [
            {"input": "Search the web for LangGraph news and summarize."},
            {"input": "Search the web for LangGraph news and summarize."},
        ],
        "expect_hit_second_turn": True,
        "grader": lambda a: "langgraph" in a.lower(),
    },
    {
        "name": "calculator_cross_turn_hit",
        "turns": [
            {"input": "Calculate 893 * 47 with the calculator."},
            {"input": "Calculate 893 * 47 with the calculator."},
        ],
        "expect_hit_second_turn": True,
        "grader": lambda a: "41971" in a.replace(",", ""),
    },
    {
        "name": "freshness_request_bypasses_cache",
        "turns": [
            {"input": "Search the web for LangGraph news and summarize."},
            {"input": "Search the web for the LATEST LangGraph news and summarize."},
        ],
        "expect_bypass_second_turn": True,
        "grader": lambda a: "langgraph" in a.lower(),
    },
    {
        "name": "incomplete_plan_one_bounded_replan",
        "turns": [
            {
                "input": (
                    "Search the web for LangGraph orchestration news, then "
                    "calculate 893 * 47 with the calculator, and remember the result."
                ),
            },
        ],
        # Canned dispatcher fails the FIRST calculator attempt only →
        # structural step failure → the ONE bounded replan must retry it.
        "scripted_fail_tool": "calculator",
        "expect_replans": 1,
        "grader": lambda a: "41971" in a.replace(",", ""),
    },
    {
        "name": "complete_plan_no_replan",
        "turns": [
            {
                "input": (
                    "Calculate 893 * 47 with the calculator and then remember "
                    "the result."
                ),
            },
        ],
        "expect_replans": 0,
        "grader": lambda a: "41971" in a.replace(",", ""),
    },
    {
        "name": "explicit_refresh_flag_bypasses_cache",
        "fresh_session_each_turn": True,  # global-cache measurement, not memory
        "turns": [
            {"input": "Search the web for LangGraph news and summarize."},
            {
                "input": "Search the web for LangGraph news and summarize.",
                "refresh": True,  # v0.25 programmatic refresh — no freshness wording
            },
        ],
        "expect_bypass_second_turn": True,
        "grader": lambda a: "langgraph" in a.lower(),
    },
    {
        "name": "grounding_tool_value_overrides_guess",
        "turns": [
            {
                "input": (
                    "Use the calculator tool to compute 8947 * 123457. You "
                    "MUST call the calculator tool for this — first say your "
                    "mental guess, then report the calculator's exact result "
                    "as the final answer."
                ),
            },
        ],
        "expect_replans": 0,
        # 8947 * 123457 = 1,104,569,779 — beyond confident mental arithmetic.
        # Grounding contract: the FINAL answer must carry the TOOL's value.
        "expect_tool_used": "calculator",
        "grader": lambda a: "1104569779" in "".join(ch for ch in a if ch.isdigit()),
    },
    {
        "name": "knowledge_step_one_bounded_replan",
        "turns": [
            {
                "input": (
                    "Summarize Alan Turing from Wikipedia, then calculate "
                    "12 * 12 with the calculator and remember the result."
                ),
            },
        ],
        # Scripted: the FIRST wikipedia_summary attempt fails (structural
        # step failure) → exactly ONE replan → both steps complete.
        "scripted_fail_tool": "wikipedia_summary",
        "expect_replans": 1,
        "expect_tool_used": "calculator",
        "grader": lambda a: "turing" in a.lower() and "144" in a,
    },
    {
        "name": "no_tool_explanation",
        "turns": [
            {
                "input": (
                    "Without using any tools: explain in two sentences how "
                    "plan-and-execute agents differ from ReAct loops."
                ),
            },
        ],
        "expect_no_tools": True,
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live_cache_replan_eval")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    cases = [c for c in CASES if not args.filter or args.filter in c["name"]]
    registry = _crb._build_registry()

    all_records: list[dict[str, Any]] = []
    for case in cases:
        # Per-case dispatcher tweak (structural, deterministic): a case with
        # ``scripted_fail_tool`` fails the FIRST attempt of that tool so the
        # plan produces genuine structural failure evidence for the replan.
        fail_tool = case.get("scripted_fail_tool")
        if fail_tool:
            original_dispatch = CannedNetwork.dispatch

            def make_failing(base, tool_to_fail):
                async def failing_dispatch(self, tool_name, tool_args):
                    result = await base(self, tool_name, tool_args)
                    marker = f"{tool_to_fail}_failed_once"
                    if tool_name == tool_to_fail and not self.record.get(marker):
                        self.record[marker] = True
                        return f"ERROR: {tool_to_fail} service temporarily unavailable (scripted)."
                    return result

                return failing_dispatch

            CannedNetwork.dispatch = make_failing(original_dispatch, fail_tool)
            try:
                records = run_case(case, registry, args.reps)
            finally:
                CannedNetwork.dispatch = original_dispatch
        else:
            records = run_case(case, registry, args.reps)
        all_records.extend(records)
        for r in records:
            status = "PASS" if r["ok"] else "FAIL"
            print(
                f"  {r['case']:<36} rep{r['rep']} tools={sum(len(t['dispatched']) for t in r['turns']):<3} "
                f"replans={r.get('replans_total', sum(t['replans'] for t in r['turns']))} "
                f"{r.get('turns', [{}])[-1].get('latency_s', '?')}s  {status}",
                flush=True,
            )
            if not r["ok"]:
                print(f"    -> {r['notes']}")

    total = len(all_records)
    passed = sum(1 for r in all_records if r["ok"])
    # Aggregate (small sample; NOT statistically universal).
    cache_hits = sum(1 for r in all_records if r.get("cache", {}).get("turn2_hit"))
    bypasses = sum(1 for r in all_records if r.get("cache", {}).get("turn2_bypassed"))
    replan_total = sum(sum(t["replans"] for t in r["turns"]) for r in all_records)
    live_calls = sum(r["live_tool_calls"] for r in all_records)
    latencies = [t["latency_s"] for r in all_records for t in r["turns"]]
    summary = {
        "cases_run": total,
        "passed": passed,
        "cache_turn2_hit_rate": f"{cache_hits}/{sum(1 for c in cases if c.get('expect_hit_second_turn')) * args.reps}",
        "freshness_bypass_rate": f"{bypasses}/{sum(1 for c in cases if c.get('expect_bypass_second_turn')) * args.reps}",
        "replans_total": replan_total,
        "live_tool_calls": live_calls,
        "latency_avg_s": round(sum(latencies) / max(1, len(latencies)), 1),
        "note": "small-sample live check; not statistically universal",
    }
    print(f"LIVE CACHE+REPLAN EVAL: {passed}/{total} passed")
    print(f"summary: {json.dumps(summary)}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "records": all_records}, fh, indent=2)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
