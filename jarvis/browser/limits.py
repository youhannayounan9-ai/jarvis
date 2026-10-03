"""
jarvis/browser/limits.py
────────────────────────
v0.28 Part 12 — deterministic action pacing and rate limits.

Contract:
  - Every browser ACTION passes ``check_and_count`` BEFORE the driver runs.
    The check is pure accounting: no LLM, no heuristics, no model input.
  - Bounds (all fail closed):
      * max actions per TURN (one chat() call)
      * max actions per BROWSER SESSION (cumulative, survives turns)
      * max duration of one TURN (a turn that runs longer is refused
        further actions — observation stays allowed so the model can see
        where it stopped)
      * max repeated IDENTICAL side-effecting actions (same fingerprint)
      * max navigation DEPTH (chain of open_url/go_back per session)
      * max retries of the same fingerprint after a FAILURE
  - ``new_turn()`` resets per-turn counters (turn scoping mirrors the v0.23
    dispatch ledger's per-turn lifetime).
  - Observation (read-only) actions share the per-turn count but NOT the
    repeated-side-effect cap: SAFE repeated observation is legitimate.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class PacingLimitError(RuntimeError):
    """Raised when a deterministic bound is exceeded. Message is safe."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Action limit reached: {reason}")
        self.reason = reason


@dataclass
class BrowserLimits:
    """Immutable-by-convention limit configuration (settings-driven)."""

    max_actions_per_turn: int = 24
    max_actions_per_session: int = 120
    max_turn_duration_seconds: float = 300.0
    max_repeated_identical_actions: int = 1
    max_navigation_depth: int = 12
    max_retries_after_failure: int = 2


@dataclass
class _TurnState:
    started_at: float = field(default_factory=time.monotonic)
    actions: int = 0
    nav_depth: int = 0


class PacingLedger:
    """
    Per-browser-session pacing accountant. Thread-safe. Pure accounting —
    the caller decides what a denial means (the controller converts it to
    an ACTION_BLOCKED result string).
    """

    def __init__(self, limits: BrowserLimits | None = None) -> None:
        self._limits = limits or BrowserLimits()
        self._lock = threading.Lock()
        self._turn = _TurnState()
        self._session_actions = 0
        self._nav_depth = 0
        self._fingerprint_successes: dict[str, int] = {}
        self._fingerprint_failures: dict[str, int] = {}

    # ── Turn lifecycle ─────────────────────────────────────────────────────

    def new_turn(self) -> None:
        """Reset per-turn counters (chat() entry)."""
        with self._lock:
            self._turn = _TurnState()

    # ── The gate ───────────────────────────────────────────────────────────

    def check_and_count(
        self,
        *,
        is_observation: bool,
        is_navigation: bool,
        fingerprint: str | None,
    ) -> None:
        """
        Validate one proposed action against every bound and record it.
        Raises PacingLimitError when a bound is exceeded.
        """
        with self._lock:
            turn = self._turn
            # 1. Per-turn duration (actions only; observation is free).
            if not is_observation:
                elapsed = time.monotonic() - turn.started_at
                if elapsed > self._limits.max_turn_duration_seconds:
                    raise PacingLimitError(
                        f"turn duration exceeded "
                        f"({int(elapsed)}s > {int(self._limits.max_turn_duration_seconds)}s)"
                    )
                # 2. Per-turn action count.
                if turn.actions >= self._limits.max_actions_per_turn:
                    raise PacingLimitError(
                        f"max actions per turn ({self._limits.max_actions_per_turn})"
                    )
                # 3. Per-session cumulative count.
                if self._session_actions >= self._limits.max_actions_per_session:
                    raise PacingLimitError(
                        f"max actions per browser session "
                        f"({self._limits.max_actions_per_session})"
                    )
                # 4. Navigation depth.
                if is_navigation:
                    if self._nav_depth >= self._limits.max_navigation_depth:
                        raise PacingLimitError(
                            f"max navigation depth ({self._limits.max_navigation_depth})"
                        )
                # 5. Repeated identical SIDE-EFFECTING actions (observation
                #    repeats are SAFE and never hit this cap).
                if fingerprint is not None:
                    successes = self._fingerprint_successes.get(fingerprint, 0)
                    if successes >= self._limits.max_repeated_identical_actions:
                        raise PacingLimitError(
                            "this identical side-effecting action already "
                            "succeeded; repeating it is suppressed"
                        )
                    failures = self._fingerprint_failures.get(fingerprint, 0)
                    if failures > self._limits.max_retries_after_failure:
                        raise PacingLimitError(
                            f"too many failed attempts for this action "
                            f"(max retries {self._limits.max_retries_after_failure})"
                        )

            # ── Record (past every check) ──
            turn.actions += 1
            self._session_actions += 1
            if is_navigation:
                turn.nav_depth += 1
                self._nav_depth += 1

    def record_outcome(self, *, fingerprint: str | None, success: bool) -> None:
        """Record success/failure for repeated-side-effect accounting."""
        if fingerprint is None:
            return
        with self._lock:
            bucket = (
                self._fingerprint_successes if success else self._fingerprint_failures
            )
            bucket[fingerprint] = bucket.get(fingerprint, 0) + 1
            if success:
                self._fingerprint_failures.pop(fingerprint, None)

    # ── Introspection (bounded, safe metadata only) ────────────────────────

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "turn_actions": self._turn.actions,
                "session_actions": self._session_actions,
                "navigation_depth": self._nav_depth,
                "distinct_side_effect_fingerprints": len(self._fingerprint_successes)
                + len(self._fingerprint_failures),
            }
