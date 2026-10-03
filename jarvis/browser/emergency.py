"""
jarvis/browser/emergency.py
───────────────────────────
v0.28 Part 11 — external emergency stop.

Contract:
  - ONE process-global stop. Triggered ONLY by the human (CLI Ctrl+X or
    /stop command, API POST /browser/emergency-stop, dashboard button).
    The model has NO tool that can trigger or reset it — external to model
    reasoning by construction.
  - ``check()`` is called by the controller before EVERY driver operation
    and between observe-act-verify cycles; it raises
    :class:`EmergencyStopTriggered` while active.
  - ``trigger`` returns a monotonically increasing token; ``is_stale``
    lets a caller detect that a NEW stop fired during its own operation
    (double-stop during one action still counts).
  - ``reset()`` is an explicit operator action and returns the state to
    normal; it is idempotent.
  - Triggering is safe from any thread; checks are lock-cheap.
"""

from __future__ import annotations

import threading
import time

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class EmergencyStopTriggered(RuntimeError):
    """Raised inside browser operations while the emergency stop is active."""

    def __init__(self, token: int, reason: str) -> None:
        super().__init__(f"EMERGENCY STOP (#{token}): {reason}")
        self.token = token
        self.reason = reason


class EmergencyStop:
    """Process-global emergency-stop gate (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._token = 0
        self._reason = ""
        self._triggered_at: float | None = None

    # ── Human-facing ───────────────────────────────────────────────────────

    def trigger(self, reason: str = "user requested emergency stop") -> int:
        """Activate the stop; returns the trigger token (monotonic)."""
        with self._lock:
            self._token += 1
            self._active = True
            self._reason = str(reason)[:200]
            self._triggered_at = time.time()
            token = self._token
        log.warning("browser_emergency_stop_triggered", token=token, reason=str(reason)[:200])
        return token

    def reset(self) -> bool:
        """
        Deactivate the stop (explicit operator action). Returns True when a
        stop was actually cleared, False when none was active (idempotent).
        """
        with self._lock:
            was_active = self._active
            self._active = False
            self._reason = ""
        if was_active:
            log.info("browser_emergency_stop_reset")
        return was_active

    # ── Runtime-facing ─────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def check(self) -> None:
        """Raise EmergencyStopTriggered when active; cheap no-op otherwise."""
        with self._lock:
            if self._active:
                raise EmergencyStopTriggered(self._token, self._reason or "stopped")

    def status(self) -> dict[str, object]:
        """Safe status snapshot for API/dashboard/CLI display."""
        with self._lock:
            return {
                "active": self._active,
                "token": self._token,
                "reason": self._reason,
                "triggered_at": self._triggered_at,
            }


# ── Process-global singleton ──────────────────────────────────────────────────
_STOP = EmergencyStop()


def get_emergency_stop() -> EmergencyStop:
    """The process-global stop used by every controller, CLI, API surface."""
    return _STOP
