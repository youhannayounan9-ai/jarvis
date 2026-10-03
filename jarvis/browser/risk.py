"""
jarvis/browser/risk.py
──────────────────────
v0.28 Parts 9/10 — action risk classification.

Contract:
  - Exactly four action-risk levels: LOW / MEDIUM / HIGH / CRITICAL.
  - Each maps onto an EXISTING PermissionGuard tier (no second permission
    system): LOW→SAFE, MEDIUM→NETWORK, HIGH→SYSTEM, CRITICAL→DESTRUCTIVE.
  - Classification is DETERMINISTIC and computed from VALIDATED arguments
    (element label text, field names, URL) — never from model self-report.
  - HIGH/CRITICAL route into the existing confirmation parking flow by the
    orchestrator consulting ``risk_to_permission_tier``; nothing here
    confirms, denies, or executes anything.
"""

from __future__ import annotations

import re
from enum import Enum

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class ActionRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# The ONLY mapping into the existing permission tiers. Orchestration asks
# "which tier does this call behave like?" and the existing PermissionGuard
# answers allow / confirm / block exactly as it always has.
_RISK_TO_TIER: dict[ActionRisk, str] = {
    ActionRisk.LOW: "SAFE",
    ActionRisk.MEDIUM: "NETWORK",
    ActionRisk.HIGH: "SYSTEM",
    ActionRisk.CRITICAL: "DESTRUCTIVE",
}

def risk_to_permission_tier(risk: ActionRisk) -> str:
    """Map an action risk onto the existing PermissionGuard tier."""
    return _RISK_TO_TIER[risk]


# ── Deterministic word classifiers ────────────────────────────────────────────
# Word-boundary matching so 'submitting' matches submit-class but 'terms'
# never matches 'submit'. These lists are policy vocabulary — extend by
# category, never by quoting a specific site's phrases.

_SUBMIT_WORDS: tuple[str, ...] = (
    "submit", "send", "publish", "purchase", "pay", "checkout", "buy",
    "order", "confirm order", "place order", "delete", "remove", "sign out",
    "signout", "log out", "logout", "post", "send message", "transfer",
    "agree", "accept", "authorize", "authorise", "verify account",
    "unsubscribe", "cancel account", "deactivate",
)

_SENSITIVE_FIELD_WORDS: tuple[str, ...] = (
    "password", "passwd", "pwd", "passphrase", "card", "cvv", "cvc",
    "security code", "ssn", "social security", "otp", "pin", "token",
    "secret", "api key", "apikey", "credit card", "expiry", "cvnum",
)

_DOWNLOAD_WORDS: tuple[str, ...] = ("download", "export", "save as")

_SUBMIT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _SUBMIT_WORDS) + r")\b",
    re.IGNORECASE,
)
_SENSITIVE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _SENSITIVE_FIELD_WORDS) + r")\b",
    re.IGNORECASE,
)
_DOWNLOAD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _DOWNLOAD_WORDS) + r")\b",
    re.IGNORECASE,
)


def _text_of(*values: object) -> str:
    return " ".join(str(v) for v in values if v)


def classify_click_risk(label: str | None = None, selector: str | None = None, **_: object) -> ActionRisk:
    """
    Risk of clicking an element named by ``label`` (accessible name/text)
    and/or ``selector`` (CSS). A submit/destroy-worded target is HIGH —
    the runtime will require the existing user confirmation.
    """
    text = _text_of(label, selector)
    if _SUBMIT_RE.search(text):
        return ActionRisk.HIGH
    if _DOWNLOAD_RE.search(text):
        return ActionRisk.HIGH
    return ActionRisk.MEDIUM


def classify_fill_risk(field: str | None = None, value: str | None = None, **_: object) -> ActionRisk:
    """
    Risk of filling a form field. Sensitive-looking fields (password/card
    surfaces) are HIGH: the confirmation context shows the target and the
    masked value so the user sees exactly what would be typed where.
    """
    text = _text_of(field)
    if _SENSITIVE_RE.search(text):
        return ActionRisk.HIGH
    if value and len(value) > 2000:
        # Very large fills are unusual; make them visible to the user.
        return ActionRisk.HIGH
    return ActionRisk.MEDIUM


def classify_risk(tool_name: str, args: dict[str, object]) -> ActionRisk:
    """
    Runtime dispatch point: classify any browser tool call by name+args.
    Unknown tool names are OUT OF SCOPE here (the registry never dispatches
    them); this defaults to the most conservative MEDIUM so a future tool
    that forgets its own classifier cannot silently become LOW.
    """
    if tool_name == "click_element":
        return classify_click_risk(
            label=args.get("label"), selector=args.get("selector")
        )
    if tool_name == "fill_input":
        return classify_fill_risk(field=args.get("field"), value=args.get("value"))
    if tool_name == "open_url":
        return ActionRisk.MEDIUM
    if tool_name == "select_option":
        return classify_fill_risk(field=args.get("field"), value=args.get("value"))
    return ActionRisk.MEDIUM
