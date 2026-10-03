"""
jarvis/browser/verification.py
──────────────────────────────
v0.28 Parts 16/17 — OBSERVE-ACT-VERIFY outcome statuses and trusted
action-evidence formatting.

Contract:
  - Every browser action's result string begins with EXACTLY ONE status
    token, computed by the RUNTIME (never by the model):

      ACTION_EXECUTED     — the driver ran the operation successfully.
      ACTION_VERIFIED     — executed AND a deterministic post-check
                            observed the expected state change.
      ACTION_NOT_VERIFIED — executed but the post-check could not confirm
                            the expected change (honest uncertainty).
      ACTION_BLOCKED      — a runtime gate refused the action (URL policy,
                            stale observation, pacing, risk confirmation,
                            duplicate suppression). Never executed.
      ACTION_INTERRUPTED  — the emergency stop (or a timeout) interrupted
                            the action; state is unknown-but-bounded.

  - The final answer must respect these statuses; ACTION_NOT_VERIFIED and
    ACTION_INTERRUPTED mean the model may NOT claim success.
  - ``build_action_result`` produces the structured, deterministic result
    string (safe fields only). These strings are TRUSTED ACTION EVIDENCE:
    they ride the v0.25 evidence ledger like any other successful tool
    observation, because they were produced by the runtime, not narrated
    by the model. Page text inside them is already redacted and framed.
"""

from __future__ import annotations

from enum import Enum

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class VerificationStatus(str, Enum):
    EXECUTED = "ACTION_EXECUTED"
    VERIFIED = "ACTION_VERIFIED"
    NOT_VERIFIED = "ACTION_NOT_VERIFIED"
    BLOCKED = "ACTION_BLOCKED"
    INTERRUPTED = "ACTION_INTERRUPTED"


VERIFICATION_STATUSES: frozenset[str] = frozenset(s.value for s in VerificationStatus)

# Fields allowed in a structured action result — safe metadata only.
# Values are redacted by the caller (redaction.redact) before formatting.
_RESULT_FIELDS_ORDER = (
    "action", "target", "requested_url", "final_url", "expected", "observed",
    "detail", "observation_id",
)


def build_action_result(
    status: VerificationStatus,
    *,
    action: str,
    target: str | None = None,
    requested_url: str | None = None,
    final_url: str | None = None,
    expected: str | None = None,
    observed: str | None = None,
    detail: str | None = None,
    observation_id: str | None = None,
    extra_fields: dict[str, str] | None = None,
) -> str:
    """
    Deterministic, structured action-result line. The FIRST token is the
    status so downstream consumers (grounding, telemetry, tests) can parse
    outcomes without trusting any model prose.
    """
    fields: dict[str, str] = {
        "action": action,
        "target": target or "",
        "requested_url": requested_url or "",
        "final_url": final_url or "",
        "expected": expected or "",
        "observed": observed or "",
        "detail": detail or "",
        "observation_id": observation_id or "",
    }
    if extra_fields:
        for k, v in extra_fields.items():
            if str(k).isalnum() or "_" in str(k):
                fields[str(k)] = str(v)
    parts = [status.value]
    parts.extend(
        f"{key}={fields[key]}" for key in _RESULT_FIELDS_ORDER if fields.get(key)
    )
    parts.extend(
        f"{key}={value}" for key, value in fields.items()
        if key not in _RESULT_FIELDS_ORDER and value
    )
    return " ".join(parts)


def status_of(result: str) -> VerificationStatus | None:
    """Parse the leading status token of an action result (or None)."""
    first = str(result or "").strip().split(" ", 1)[0]
    try:
        return VerificationStatus(first)
    except ValueError:
        return None


def is_verified(result: str) -> bool:
    """True only when the result carries ACTION_VERIFIED."""
    return status_of(result) is VerificationStatus.VERIFIED
