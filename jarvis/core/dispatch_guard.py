"""
jarvis/core/dispatch_guard.py
─────────────────────────────
v0.23 repeat-call semantics (Parts B/C/E).

Categories (Part B):
  1. DUPLICATE   — same tool + same meaningful arguments within ONE turn
                   with a successful result already in hand → suppressed
                   (the previous observation is reattached; nothing runs).
  2. LEGITIMATE  — same tool, meaningfully different arguments → allowed
                   (the fingerprint differs).
  3. STATE-DEPENDENT — same tool + same arguments but the previous attempt
                   FAILED, or the tool is in the state-dependent allowlist
                   (its result depends on state that legitimately changes:
                   clock, other people's websites, the user's own memory),
                   or the caller explicitly marks the repeat intentional
                   (recovery rounds) → allowed.

Scoping is per TURN (one chat() call, both paths) and per STEP is refused:
the fingerprint ledger lives in the orchestrator for the duration of a
single user request, so a resumed plan rebuilds it from the persisted tool
messages instead of carrying stale state across processes.

The fingerprint (Part C) is deterministic and safe to log:
    sha1( tool_name + "\x1f" + canonical_json(sorted args) ) [:12]
No secrets beyond what the arguments already carry are introduced, and
telemetry logs only the SHORT fingerprint plus tool name — never the
arguments themselves.

This layer NEVER authorizes anything: suppression only avoids a repeat
execution; every unsuppressed dispatch still passes PermissionGuard,
schema validation, and (when applicable) the action ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Part B category 3: tools whose identical-argument repeat is legitimately
# meaningful because their OUTPUT depends on mutable state within one turn
# (the clock; the user's own memory, which remember_fact mutates). Repeats
# are never suppressed for these.
#
# Deliberately NOT exempt: web_search / web_scrape / wikipedia_summary /
# search_knowledge. External pages and the user's indexed documents do not
# meaningfully change SECONDS apart within one task, so an identical-argument
# re-call inside one turn is the redundancy pattern v0.23 targets (live v0.22
# evidence: web_search×3 with the same query). Different arguments remain
# fully allowed, and a FAILED call never blocks retry.
_STATE_DEPENDENT_TOOLS = frozenset({
    "get_current_datetime",
    "recall_facts",
    "remember_fact",
})

_ARG_VALUE_NORMALIZER = re.compile(r"\s+")


def canonical_arguments(tool_name: str, tool_args_json: str) -> str:
    """
    Deterministic canonical form of a tool's arguments (Part C).

    - parses JSON (invalid JSON passes through marked, so a malformed
      repeat is never confused with a valid one),
    - sorts object keys, collapses insignificant whitespace inside string
      values, strips type-coercion ambiguity by stringifying scalars,
    - rejects nothing: this is a fingerprint, not a validator.
    """
    if not tool_args_json or not tool_args_json.strip():
        return "{}"
    try:
        raw = json.loads(tool_args_json)
    except (json.JSONDecodeError, TypeError):
        return f"__unparsed__:{tool_args_json.strip()}"
    return json.dumps(_normalize(raw), sort_keys=True, separators=(",", ":"))


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, str):
        return _ARG_VALUE_NORMALIZER.sub(" ", value).strip()
    return value


def fingerprint(tool_name: str, tool_args_json: str) -> str:
    """Short, log-safe dispatch identity: tool + canonical args."""
    canonical = canonical_arguments(tool_name, tool_args_json)
    digest = hashlib.sha1(f"{tool_name}\x1f{canonical}".encode("utf-8")).hexdigest()[:12]
    return digest


@dataclass
class DispatchLedger:
    """
    Per-turn ledger of successful dispatch fingerprints (Part E).

    ``is_duplicate`` decides suppression. FAILED results never enter the
    ledger, so retry-after-failure stays legitimate (Part K), and
    state-dependent tools are exempt by class (Part B category 3).
    """

    seen: dict[str, str] = field(default_factory=dict)  # fingerprint -> tool
    suppressed_count: int = 0

    def is_duplicate(self, tool_name: str, tool_args_json: str) -> bool:
        if tool_name in _STATE_DEPENDENT_TOOLS:
            return False
        return fingerprint(tool_name, tool_args_json) in self.seen

    def record_success(self, tool_name: str, tool_args_json: str) -> None:
        if tool_name in _STATE_DEPENDENT_TOOLS:
            return
        self.seen[fingerprint(tool_name, tool_args_json)] = tool_name

    def record_suppressed(self, tool_name: str, tool_args_json: str) -> None:
        self.suppressed_count += 1
        log.info(
            "duplicate_tool_call_suppressed",
            tool=tool_name,
            fingerprint=fingerprint(tool_name, tool_args_json),
        )
