"""
jarvis/core/plan_quality.py
───────────────────────────
v0.23 deterministic structural plan-quality report (Part G).

An EVALUATION/REPORTING layer: it measures every generated plan and logs
``plan_quality_report`` with safe metadata only (counts and tool NAMES —
never arguments or user text). It NEVER rejects a plan by itself and never
authorizes anything; the production safety layer remains plan_validator
(structural) + PermissionGuard (authorization).

Measured per plan:
  steps                 number of planned steps
  toolless_steps        pure-reasoning steps
  unique_tools          distinct required tools
  repeated_tools        tool -> count, only where count > 1 (suspicious,
                        not automatically wrong)
  forward_references    references to later steps (validator drops these)
  duplicate_steps       steps whose normalized description repeats
  estimated_llm_calls   steps + synthesis (cost transparency)
  issues                compact flag list for log-based analysis
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_STEP_REF = re.compile(r"\bstep\s*(\d+)\b", re.IGNORECASE)


def report_plan_quality(
    plan: list[dict[str, Any]],
    known_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Build the deterministic quality report for one plan (read-only)."""
    steps = [s for s in (plan or []) if isinstance(s, dict)]
    descriptions = [
        " ".join(str(s.get("description") or "").lower().split()) for s in steps
    ]
    tool_seq = [
        t for s in steps for t in (s.get("required_tools") or []) if t
    ]
    tool_counts = Counter(tool_seq)
    desc_counts = Counter(descriptions)

    issues: list[str] = []
    for tool, count in sorted(tool_counts.items()):
        if count > 1:
            issues.append(f"repeated_tool_{tool}_x{count}")
    duplicate_descs = sum(1 for c in desc_counts.values() if c > 1)
    if duplicate_descs:
        issues.append(f"duplicate_step_descriptions_x{duplicate_descs}")

    forward_refs = 0
    for s in steps:
        try:
            n = int(s.get("step_number") or 0)
        except (TypeError, ValueError):
            continue
        desc = str(s.get("description") or "")
        forward_refs += sum(1 for m in _STEP_REF.findall(desc) if int(m) >= n and n > 0)
    if forward_refs:
        issues.append(f"forward_references_x{forward_refs}")

    unknown = []
    if known_tools:
        unknown = sorted({t for t in tool_seq if t not in known_tools})
        if unknown:
            issues.append(f"unknown_tools_{','.join(unknown)}")

    report: dict[str, Any] = {
        "steps": len(steps),
        "toolless_steps": sum(1 for t in (s.get("required_tools") or [] for s in steps) if not t),
        "unique_tools": len(set(tool_seq)),
        "repeated_tools": {t: c for t, c in tool_counts.items() if c > 1},
        "forward_references": forward_refs,
        "duplicate_steps": duplicate_descs,
        "estimated_llm_calls": len(steps) + 1,
        "issues": issues,
    }
    return report
