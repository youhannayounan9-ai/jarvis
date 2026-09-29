"""
evaluation/plan_judge.py
───────────────────────
v0.23 OPTIONAL semantic plan judge (Part H) — EVALUATION-ONLY.

Not part of the production execution loop: production safety remains
plan_validator (structural) + PermissionGuard (authorization). The judge:

  - runs only when invoked manually,
  - only READS (no tool execution, no state mutation),
  - can never authorize a tool or override the guard,
  - grades plan SENSIBILITY: coverage, necessity, order, efficiency.

It asks the LIVE model (default OLLAMA_MODEL) to score a generated plan
1-5 on four axes and returns structured, conservative output: any LLM
error degrades to `{"error": ...}` rather than fabricating a verdict.
No tool calls are offered to the judge model (tools=None) and its answer
is never fed back into any orchestrator path.

Run:
    uv run python evaluation/plan_judge.py --input plan.json
    uv run python evaluation/plan_judge.py --request "Calculate 5*5 and remember it"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jarvis.config import settings
from jarvis.llm.client import chat_completion
from jarvis.runtime import build_runtime

JUDGE_SYSTEM_PROMPT = (
    "You are a plan-quality reviewer. You will see a user request and a "
    "JSON plan (steps with descriptions and required_tools). Score the plan "
    "1-5 on each axis and be conservative; reserve 5 for excellent plans.\n"
    "Axes:\n"
    "  coverage    - do the steps cover everything the request needs?\n"
    "  necessity   - is every step needed (no busywork/redundant steps)?\n"
    "  order       - are dependencies in a workable order?\n"
    "  efficiency  - could the same outcome be reached with fewer steps/tools?\n"
    "Return ONLY a JSON object: "
    '{"coverage": n, "necessity": n, "order": n, "efficiency": n, '
    '"unnecessary_steps": [numbers], "comment": "one short sentence"}. '
    "No markdown fences."
)


def judge_plan(request: str, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask the live model to grade one plan. Read-only; never authorizes."""
    plan_blob = json.dumps(plan, indent=2)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User request:\n{request}\n\nPlan to review:\n{plan_blob}",
        },
    ]
    try:
        response = chat_completion(messages=messages, tools=None)
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - judge must never break anything
        return {"error": f"{type(e).__name__}: {e}"}

    # Extract the JSON object (models sometimes add prose/fences).
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"error": "judge returned no JSON object", "raw": raw[:300]}
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"error": "judge JSON unparseable", "raw": raw[:300]}

    axes = ("coverage", "necessity", "order", "efficiency")
    scores = {a: verdict.get(a) for a in axes}
    if not all(isinstance(v, (int, float)) and 1 <= v <= 5 for v in scores.values()):
        return {"error": "judge scores missing/out of range", "raw": raw[:300]}
    return {
        "scores": {a: float(scores[a]) for a in axes},
        "mean": round(sum(float(scores[a]) for a in axes) / len(axes), 2),
        "unnecessary_steps": verdict.get("unnecessary_steps") or [],
        "comment": str(verdict.get("comment") or "")[:200],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plan_judge")
    parser.add_argument("--input", help="JSON file {request, plan}")
    parser.add_argument("--request", help="generate a plan for this request and judge it")
    args = parser.parse_args(argv)

    from jarvis.api.health import check_ollama

    if not check_ollama(settings.ollama_base_url):
        print(f"ERROR: Ollama unreachable at {settings.ollama_base_url}.")
        return 2

    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            data = json.load(fh)
        request, plan = data["request"], data["plan"]
    elif args.request:
        runtime = build_runtime()
        try:
            # Ask the real planner for a plan WITHOUT executing anything.
            plan = runtime.orchestrator._planner.generate_plan(args.request, "")
        finally:
            runtime.close()
        request = args.request
    else:
        parser.error("provide --input or --request")
        return 1

    verdict = judge_plan(request, plan)
    print(json.dumps({"request": request, "plan": plan, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
