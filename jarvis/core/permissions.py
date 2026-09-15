"""
jarvis/core/permissions.py
───────────────────────────
Permission Guard — controls which tools may be called in which contexts.

Risk levels:
  - SAFE:        No side-effects. Read-only, no network. (e.g. get_datetime)
  - NETWORK:     Makes external HTTP requests. (e.g. web_search)
  - FILE_READ:   Reads local files. (e.g. read_file)
  - FILE_WRITE:  Modifies local files. Requires explicit user confirmation.
  - SYSTEM:      Runs shell commands or OS automation. Requires confirmation.
  - DESTRUCTIVE: Irreversible actions. Always requires user confirmation.

v0.1 hardened policy:
  SAFE / NETWORK / FILE_READ are allowed automatically.
  FILE_WRITE / SYSTEM / DESTRUCTIVE are blocked until an interactive
  confirmation UI exists in a later version.
"""

from typing import Literal

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

RiskLevel = Literal[
    "SAFE",
    "NETWORK",
    "FILE_READ",
    "FILE_WRITE",
    "SYSTEM",
    "DESTRUCTIVE",
]

# Auto-allowed without user confirmation.
_ALLOWED_RISKS: frozenset[str] = frozenset({"SAFE", "NETWORK", "FILE_READ"})

# Blocked until confirmation UI lands; also flagged by require_confirmation().
_CONFIRMATION_RISKS: frozenset[str] = frozenset(
    {"FILE_WRITE", "SYSTEM", "DESTRUCTIVE"}
)


class PermissionGuard:
    """
    Enforces which tools may be called based on declared risk level.
    """

    def is_allowed(self, tool_name: str, risk_level: str) -> bool:
        """
        Return True if the tool is allowed to execute without confirmation.

        Args:
            tool_name:  The tool identifier from the registry.
            risk_level: Declared risk tier from the tool class.

        Returns:
            True for SAFE / NETWORK / FILE_READ.
            False for FILE_WRITE / SYSTEM / DESTRUCTIVE (blocked for now).
        """
        allowed = risk_level in _ALLOWED_RISKS

        if allowed:
            log.debug(
                "permission_allowed",
                tool=tool_name,
                risk_level=risk_level,
            )
        else:
            log.warning(
                "permission_blocked",
                tool=tool_name,
                risk_level=risk_level,
                reason="risk_requires_confirmation_ui",
            )

        return allowed

    def require_confirmation(self, tool_name: str, risk_level: str) -> bool:
        """
        Return True if this tool requires explicit user confirmation
        before execution (FILE_WRITE / SYSTEM / DESTRUCTIVE).

        Args:
            tool_name:  The tool identifier from the registry.
            risk_level: Declared risk tier from the tool class.
        """
        needs_confirmation = risk_level in _CONFIRMATION_RISKS

        if needs_confirmation:
            log.info(
                "permission_confirmation_required",
                tool=tool_name,
                risk_level=risk_level,
            )

        return needs_confirmation
