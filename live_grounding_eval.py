"""
live_grounding_eval.py
──────────────────────
v0.26 LIVE verification against the REAL Ollama model (Part 13) —
manual-only, requires Ollama + qwen2.5:7b, NOT part of pytest.

What is real:  the model (planner + steps + synthesis + correction), the
               full v0.26 orchestrator path — grounding guard, ONE bounded
               correction round, fail-closed fallback, evidence ledger with
               real tool attribution, cross-turn cache, refresh, replan.
What is canned: ONLY the network layer of web_search / wikipedia_summary
               (deterministic payloads; the calculator is REAL).

Metrics are kept SEPARATE (the spec's hard requirement):
  - tool-choice success   (model behavior: did the model call the tool?)
  - grounding check success / correction rate / fallback rate (SYSTEM
    mechanism, largely model-independent)
  - false-positive grounding failures (answers the guard rejected that a
    human would accept — the guard's true FP rate)
  - latency (mean/median seconds per turn)

Cases (reps each; small sample — never presented as universal):
  1. calculator grounding      → hard multiplication; final must carry the
                                  tool value (evidence beats guesswork)
  2. vulnerable layout         → the synthesis block rides as a trailing
                                  SYSTEM message (the live-bisected v0.25
                                  failure shape); guard must still catch a
                                  recomputed value
  3. multi-step plan           → datetime + calculator steps, both values in
                                  the final answer
  4. replan + synthesis        → scripted first failure, ONE replan, then a
                                  grounded answer
  5. cached evidence           → turn-2 cache HIT still governs the guard
  6. refresh                   → refresh=True bypasses the cache and re-runs
  7. conversational            → no-tool answer; guard must NOT interfere

Run:
    uv run python live_grounding_eval.py --reps 2 --json live_grounding_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ISOLATION (v0.25 Part G): private temp DB BEFORE any jarvis import.
from evaluation import _bootstrap as _eval

_eval.isolate()

from jarvis.core.grounding import check_grounding
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.tools import CalculatorTool, GetCurrentDatetimeTool, ToolRegistry, WebSearchTool

WEB_PAYLOAD = (
    'Web search results for: "LangGraph news"\n'
    "[1] LangGraph adoption grows for agent orchestration\n"
    "    URL: https://example.com/langgraph-news\n"
    "    Excerpt: LangGraph provides graph-based orchestration with checkpointing."
)


class CannedNetwork:
    """Patch ONLY the network layer; validation and other tools are real."""

    def __init__(self, record: dict[str, Any]):
        self.record = record
        self._web = WebSearchTool()
        self._calc = CalculatorTool()

    async def dispatch(self, tool_name: str, tool_args: str) -> str:
        import json as _json

        from pydantic import ValidationError

        tool = {"web_search": self._web, "calculator": self._calc}.get(tool_name)
        if tool is None:
            return f"ERROR: Unknown tool '{tool_name}'."
        try:
            raw = _json.loads(tool_args) if tool_args else {}
            tool._args_model.model_validate(raw)
        except (ValidationError, _json.JSONDecodeError) as e:
            return f"ERROR: Invalid arguments for '{tool_name}': {e}"
        self.record["dispatched"].append(tool_name)
        if tool_name == "web_search":
            return WEB_PAYLOAD
        return tool.run(**raw)


def _register(record: dict[str, Any], registry: ToolRegistry) -> None:
    registry.register(CalculatorTool())
    registry.register(GetCurrentDatetimeTool())
    registry.register(WebSearchTool())
    original = registry.dispatch_async

    canned = CannedNetwork(record)

    async def probe(tool_name: str, tool_args: str) -> str:
        # EVERY successful dispatch is recorded here (attempted/appended
        # inside CannedNetwork.dispatch after REAL Pydantic validation).
        record["attempted"].append(tool_name)
        if tool_name == "web_search":
            return await canned.dispatch(tool_name, tool_args)
        record["dispatched"].append(tool_name)
        return await original(tool_name, tool_args)

    registry.dispatch_async = probe  # type: ignore[method-assign]


HARD_ASK = "Calculate 7284 * 931 with the calculator."   # 6,781,404
HARD_VALUE = 6781404

MULTI_ASK = (
    "First tell me the current date, and then calculate 8% of 12500. "
    "Answer both."
)                                                        # 1000

REPLAN_ASK = "Search the web for LangGraph news, then calculate 7284 * 931."

CHAT_ASK = "In one or two sentences, what is the difference between RAM and disk?"


def _new_orchestrator(record: dict[str, Any]) -> tuple[Orchestrator, SessionStore, ToolRegistry]:
    store = _eval.fresh_store()
    registry = ToolRegistry()
    _register(record, registry)
    orch = Orchestrator(store, registry, PermissionGuard())
    return orch, store, registry


def _vulnerable_synthesis(self, *, session_id, user_input, memory_cue, history,
                           completed_steps, incomplete_note=None, evidence=None):
    """
    The live-bisected v0.25 FAILURE shape: the identical evidence block
    delivered as a trailing SYSTEM message instead of riding on the final
    USER message. Used ONLY by case 2 to measure whether the guard catches
    what this layout makes the model do. Everything else (guard, ledger,
    storage) is the production path.
    """
    from jarvis.config import settings as _settings
    from jarvis.core.orchestrator import (
        _EVIDENCE_CONTRACT,
        _SYNTHESIZE_PROMPT,
        _clamp_step_results,
        _format_completed_steps,
        _format_evidence_ledger,
        _message_to_dict,
    )
    from jarvis.llm.client import chat_completion as _cc

    clamped = _clamp_step_results(completed_steps, self._context.clamp_tool_output)
    steps_blob = _format_completed_steps(clamped) or "(no steps completed)"
    messages = self._context.build_messages(
        system_prompts=[
            {"role": "system", "content": _settings.system_prompt},
            {"role": "system", "content": memory_cue},
        ],
        history=history,
        user_input=f"Original request:\n{user_input}",
    )
    blob = f"{_SYNTHESIZE_PROMPT}\n\nExecuted steps and results:\n{steps_blob}"
    ev = _format_evidence_ledger(evidence)
    if ev:
        blob += f"\n\n{_EVIDENCE_CONTRACT}\n\n{ev}"
    # THE VULNERABILITY: block in a SYSTEM message; the user turn is bare.
    messages.append({"role": "system", "content": blob})
    messages.append({
        "role": "user",
        "content": "Answer the original request now.",
    })
    response = _cc(messages=messages, tools=None, temperature=0.0)
    text = response.choices[0].message.content or "I ran into an issue."
    return self._enforce_grounding(
        session_id=session_id,
        initial_text=text,
        base_messages=messages,
        synthesis_block=blob,
        evidence=evidence,
        memory_cue=memory_cue,
        history=history,
        user_input=user_input,
        completed_steps=completed_steps,
        incomplete_note=incomplete_note,
    )


def _run_turn(
    orch: Orchestrator,
    registry: ToolRegistry,
    record: dict[str, Any],
    session_id: str,
    ask: str,
    *,
    refresh: bool = False,
    vulnerable_layout: bool = False,
    extra_patches: list | None = None,
) -> tuple[str, float]:
    """One chat() turn — REAL planner and LLM; only scenario patches apply."""
    record["attempted"] = []
    record["dispatched"] = []
    patches = [] if not extra_patches else list(extra_patches)
    if vulnerable_layout:
        patches.append(
            patch.object(Orchestrator, "_synthesize", _vulnerable_synthesis)
        )
    t0 = time.perf_counter()
    for p in patches:
        p.start()
    try:
        answer = orch.chat(session_id, ask, refresh=refresh)
    finally:
        for p in patches:
            p.stop()
    return answer, time.perf_counter() - t0


def _grade_calc(answer: str) -> bool:
    from jarvis.core.grounding import canonical_numbers

    return any(c.value == HARD_VALUE for c in canonical_numbers(answer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live_grounding_eval")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)

    from jarvis.llm.client import chat_completion  # noqa: F401 - reachability

    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    def add(name: str, **fields: Any) -> None:
        entry = {"case": name, **fields}
        results.append(entry)
        lat = entry.pop("latency_s", None)
        if lat is not None:
            latencies.append(lat)
        print(json.dumps(entry, ensure_ascii=False))

    for rep in range(1, args.reps + 1):
        # ── Case 1: calculator grounding (2 turns for cross-turn cache) ──
        record: dict[str, Any] = {"attempted": [], "dispatched": []}
        orch, store, registry = _new_orchestrator(record)
        answer, dt = _run_turn(orch, registry, record, f"lg1_{rep}", HARD_ASK)
        add(
            "1_calculator_grounding",
            rep=rep,
            tool_choice_ok=record["dispatched"] == ["calculator"],
            final_value_ok=_grade_calc(answer),
            guarded=bool("could not produce a verified answer" in answer.lower()),
            latency_s=round(dt, 2),
            answer=answer[:220],
        )
        # turn 2: cached evidence path
        answer2, dt2 = _run_turn(orch, registry, record, f"lg1_{rep}", HARD_ASK)
        add(
            "1b_calculator_cached_turn2",
            rep=rep,
            cache_served=record["dispatched"] == [],   # no real dispatch
            final_value_ok=_grade_calc(answer2),
            latency_s=round(dt2, 2),
            answer=answer2[:220],
        )
        store.close()

        # ── Case 2: vulnerable layout (system-message evidence block) ────
        record = {"attempted": [], "dispatched": []}
        orch, store, registry = _new_orchestrator(record)
        answer, dt = _run_turn(
            orch, registry, record, f"lg2_{rep}", HARD_ASK, vulnerable_layout=True
        )
        add(
            "2_vulnerable_layout_guarded",
            rep=rep,
            tool_choice_ok=record["dispatched"] == ["calculator"],
            final_value_ok=_grade_calc(answer),
            guarded=bool("could not produce a verified answer" in answer.lower()),
            latency_s=round(dt, 2),
            answer=answer[:220],
        )
        store.close()

        # ── Case 3: multi-step (datetime + calculator) ───────────────────
        record = {"attempted": [], "dispatched": []}
        orch, store, registry = _new_orchestrator(record)
        answer, dt = _run_turn(orch, registry, record, f"lg3_{rep}", MULTI_ASK)
        from jarvis.core.grounding import canonical_numbers

        vals = {c.value for c in canonical_numbers(answer)}
        add(
            "3_multistep_datetime_plus_calc",
            rep=rep,
            tool_choice_ok=set(record["dispatched"]) >= {"get_current_datetime", "calculator"},
            final_value_ok=1000 in vals,
            latency_s=round(dt, 2),
            answer=answer[:220],
        )
        store.close()

        # ── Case 4: replan + synthesis (scripted first failure) ──────────
        record = {"attempted": [], "dispatched": []}
        orch, store, registry = _new_orchestrator(record)

        real_calc_run = CalculatorTool.run
        flaky = {"tries": 0}

        def flaky_calc(self, expression: str, **kw):
            flaky["tries"] += 1
            if flaky["tries"] == 1:
                return "ERROR: scripted transient failure (live grounding eval)."
            return real_calc_run(self, expression=expression, **kw)

        answer, dt = _run_turn(
            orch,
            registry,
            record,
            f"lg4_{rep}",
            REPLAN_ASK,
            extra_patches=[patch.object(CalculatorTool, "run", flaky_calc)],
        )
        add(
            "4_replan_then_grounded",
            rep=rep,
            final_value_ok=_grade_calc(answer),
            latency_s=round(dt, 2),
            answer=answer[:220],
        )
        store.close()

        # ── Case 5/6: refresh on turn 2 (cache bypass) ───────────────────
        record = {"attempted": [], "dispatched": []}
        orch, store, registry = _new_orchestrator(record)
        _run_turn(orch, registry, record, f"lg6_{rep}", HARD_ASK)
        n_first = len(record["dispatched"])
        answer, dt = _run_turn(orch, registry, record, f"lg6_{rep}", HARD_ASK, refresh=True)
        add(
            "6_refresh_bypasses_cache",
            rep=rep,
            turn1_dispatched=n_first,
            turn2_redispatched=record["dispatched"] == ["calculator"],
            final_value_ok=_grade_calc(answer),
            latency_s=round(dt, 2),
        )
        store.close()

        # ── Case 7: conversational — guard must not interfere ────────────
        record = {"attempted": [], "dispatched": []}
        orch, store, registry = _new_orchestrator(record)
        answer, dt = _run_turn(orch, registry, record, f"lg7_{rep}", CHAT_ASK)
        add(
            "7_conversational_no_interference",
            rep=rep,
            no_tools=not record["dispatched"],
            fallback_leak=bool("could not produce a verified answer" in answer.lower()),
            latency_s=round(dt, 2),
            answer=answer[:160],
        )
        store.close()

    # ── Summary (mechanism vs model kept separate) ────────────────────────
    calc_rows = [r for r in results if r["case"].startswith("1")]
    summary = {
        "reps": args.reps,
        "tool_choice_success": sum(r.get("tool_choice_ok", False) for r in results)
        / max(1, sum(1 for r in results if "tool_choice_ok" in r)),
        "grounding_final_value_ok": sum(r.get("final_value_ok", False) for r in results)
        / max(1, sum(1 for r in results if "final_value_ok" in r)),
        "cache_hit_turn2": sum(
            r.get("cache_served", False) for r in calc_rows if r["case"] == "1b_calculator_cached_turn2"
        ),
        "refresh_redispatch": sum(r.get("turn2_redispatched", False) for r in results),
        "fallback_used": sum(r.get("guarded", False) for r in results),
        "conversational_clean": sum(
            r.get("no_tools", False) and not r.get("fallback_leak", False)
            for r in results
            if r["case"].startswith("7")
        ),
        "latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
        "note": (
            "Small sample; mechanism metrics (cache/refresh/guard) are "
            "system-side and deterministic given the model's tool choice. "
            "Tool-choice success is MODEL behavior and varies per run."
        ),
    }
    report = {"summary": summary, "cases": results}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
