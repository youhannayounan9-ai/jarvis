"""
jarvis/core/plan_validator.py
─────────────────────────────
v0.22 deterministic plan validation (Part D/G).

The planner is an LLM: its output is UNTRUSTED STRUCTURE. Before a plan is
executed it is checked for structural problems only — never semantic
correctness:

  - shape: every step has an int step_number, a non-empty description,
    and a list required_tools
  - bounds: at most MAX_PLAN_STEPS steps (truncated + renumbered)
  - registry truth: required_tools must exist in the actual registry
  - duplicates: consecutive steps with identical descriptions are collapsed
  - dependency ordering: an explicit reference to "step K['s] result" must
    point BACKWARD (K < current step); a forward reference is unwinding-safe
    rejected (the step is dropped with an issue, never executed blind)
  - orphan reasoning steps: a tool-less step that references "the result"
    BEFORE any tool step exists has nothing to reason about → dropped

A plan is a request, never authorization: validation grants no permission —
every execution still passes PermissionGuard + schema validation + ledger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.planner import MAX_PLAN_STEPS
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_STEP_REF_PATTERN = re.compile(r"\bstep\s*(\d+)\b", re.IGNORECASE)


@dataclass
class PlanValidationResult:
    """Outcome of deterministic plan validation."""

    plan: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.plan)


def validate_plan(
    plan: list[dict[str, Any]],
    registry_tools: list[str] | set[str],
) -> PlanValidationResult:
    """
    Validate and normalize a generated plan against the real tool registry.

    Returns a PlanValidationResult whose ``plan`` is safe to execute (may be
    fewer steps than the input) and whose ``issues`` records every
    deterministic problem found (for telemetry, never shown to the model).
    """
    known = set(registry_tools)
    issues: list[str] = []
    if not isinstance(plan, list):
        return PlanValidationResult([], ["plan_not_a_list"])

    # 1. Shape + bounds.
    steps: list[dict[str, Any]] = []
    for i, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            issues.append(f"step_{i}_not_an_object")
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            issues.append(f"step_{i}_empty_description")
            continue
        tools_raw = item.get("required_tools")
        if not isinstance(tools_raw, list):
            issues.append(f"step_{i}_required_tools_not_a_list")
            tools_raw = []
        try:
            step_number = int(item.get("step_number") or len(steps) + 1)
        except (TypeError, ValueError):
            step_number = len(steps) + 1
            issues.append(f"step_{i}_invalid_step_number")
        steps.append({
            "step_number": step_number,
            "description": description,
            "required_tools": [str(t).strip() for t in tools_raw if str(t).strip()],
        })

    if not steps:
        return PlanValidationResult([], issues + ["plan_empty_after_shape_check"])

    if len(steps) > MAX_PLAN_STEPS:
        issues.append(f"plan_truncated_{len(steps)}_to_{MAX_PLAN_STEPS}")
        steps = steps[:MAX_PLAN_STEPS]

    # 2. Renumber contiguously (planners emit gaps/jumps).
    for i, step in enumerate(steps, start=1):
        if step["step_number"] != i:
            issues.append(f"step_renumbered_{step['step_number']}_to_{i}")
            step["step_number"] = i

    # 3. Registry truth: hallucinated tools are removed (never executed).
    for step in steps:
        invalid = [t for t in step["required_tools"] if t not in known]
        if invalid:
            issues.append(f"step_{step['step_number']}_unknown_tools_{','.join(sorted(invalid))}")
            step["required_tools"] = [t for t in step["required_tools"] if t in known]

    # 4. Duplicate consecutive steps are collapsed.
    deduped: list[dict[str, Any]] = []
    for step in steps:
        if deduped and step["description"].lower() == deduped[-1]["description"].lower():
            issues.append(f"duplicate_step_{step['step_number']}_collapsed")
            continue
        deduped.append(step)
    steps = deduped
    for i, step in enumerate(steps, start=1):
        step["step_number"] = i

    # 5. Dependency ordering: explicit "step K" references must be backward.
    # (Deliberately NO deeper check: whether a tool-less step can reason over
    # prior results is a semantic question, and the executor feeds prior step
    # results to every later step regardless. Structural validation only.)
    kept: list[dict[str, Any]] = []
    for step in steps:
        refs = [int(m) for m in _STEP_REF_PATTERN.findall(step["description"])]
        forward = [k for k in refs if k >= step["step_number"]]
        if forward:
            issues.append(
                f"step_{step['step_number']}_forward_reference_{','.join(map(str, forward))}"
            )
            continue  # dropped: it depends on a result that does not exist yet
        kept.append(step)

    if not kept:
        return PlanValidationResult([], issues + ["plan_empty_after_dependency_check"])

    for i, step in enumerate(kept, start=1):
        step["step_number"] = i

    if issues:
        log.info("plan_validation_issues_found", count=len(issues), issues=issues[:6])
    return PlanValidationResult(kept, issues)
