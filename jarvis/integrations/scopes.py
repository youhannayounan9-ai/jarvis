"""
jarvis/integrations/scopes.py
─────────────────────────────
v0.29 Part 5 — exact, minimal authorization scopes for integrations.

Contract:
  - Scopes are FINE-GRAINED and capability-specific (CALENDAR_READ vs
    CALENDAR_WRITE vs CALENDAR_DELETE). A broad "FULL_*_ACCESS" scope is
    deliberately absent from the vocabulary.
  - READ / WRITE / DELETE are separate scopes. A future email integration
    would add a separate SEND scope (never reuse WRITE for sending).
  - The MODEL can never expand scopes: scopes are granted only by the
    user through the explicit connect flow, validated at registration,
    and re-checked on every provider operation (deny-by-default).
  - Scope sets are immutable (frozenset); any request for an unknown or
    ungranted scope fails with InsufficientScopeError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class ScopeError(ValueError):
    """Base class for scope vocabulary/enforcement errors."""


class UnknownScopeError(ScopeError):
    """A scope name outside the declared vocabulary was requested."""


class InsufficientScopeError(ScopeError):
    """An operation needs a scope the connected account was never granted."""


# ── The scope vocabulary (exact, capability-named) ────────────────────────────

CALENDAR_READ: Final = "CALENDAR_READ"
CALENDAR_WRITE: Final = "CALENDAR_WRITE"
CALENDAR_DELETE: Final = "CALENDAR_DELETE"

TASKS_READ: Final = "TASKS_READ"
TASKS_WRITE: Final = "TASKS_WRITE"
TASKS_DELETE: Final = "TASKS_DELETE"

# Email is DESIGNED (v0.29 Part 11) but NOT implemented; its SEND scope is
# declared so the vocabulary is complete and future adapters cannot invent
# a broader one. No v0.29 tool or provider grants or uses it.
EMAIL_READ: Final = "EMAIL_READ"
EMAIL_SEND: Final = "EMAIL_SEND"

ALL_SCOPES: Final[frozenset[str]] = frozenset(
    {
        CALENDAR_READ, CALENDAR_WRITE, CALENDAR_DELETE,
        TASKS_READ, TASKS_WRITE, TASKS_DELETE,
        EMAIL_READ, EMAIL_SEND,
    }
)

# Provider → scopes that provider is CAPABLE of granting (its scope menu).
# An account can hold only a subset of its provider's menu. The menu never
# contains a wildcard.
PROVIDER_SCOPE_MENUS: Final[dict[str, frozenset[str]]] = {
    "calendar": frozenset({CALENDAR_READ, CALENDAR_WRITE, CALENDAR_DELETE}),
    "tasks": frozenset({TASKS_READ, TASKS_WRITE, TASKS_DELETE}),
    "email": frozenset({EMAIL_READ, EMAIL_SEND}),  # designed, not implemented
}


def validate_scope_set(provider: str, scopes: frozenset[str] | set[str]) -> frozenset[str]:
    """
    Validate a scope grant for a provider: every scope must be in the global
    vocabulary AND in the provider's menu. Raises UnknownScopeError with a
    safe message (never echoes anything beyond the offending scope name —
    scope names are not secrets, but keep the message bounded).
    """
    menu = PROVIDER_SCOPE_MENUS.get(provider)
    if menu is None:
        raise UnknownScopeError(f"unknown provider: {provider!r}")
    bad = sorted(set(scopes) - ALL_SCOPES)
    if bad:
        raise UnknownScopeError(f"unknown scope(s): {bad[:3]}")
    off_menu = sorted(set(scopes) - menu)
    if off_menu:
        raise UnknownScopeError(
            f"scope(s) {off_menu[:3]} are not valid for provider {provider!r}"
        )
    return frozenset(scopes)


def require_scope(granted: frozenset[str], needed: str) -> None:
    """
    Enforce one exact scope. Deny-by-default: an empty grant or a missing
    scope raises InsufficientScopeError (never a silent degradation).
    """
    if needed not in ALL_SCOPES:
        raise UnknownScopeError(f"unknown scope: {needed!r}")
    if needed not in granted:
        raise InsufficientScopeError(
            f"operation requires scope {needed}, which the connected "
            "account was not granted"
        )
