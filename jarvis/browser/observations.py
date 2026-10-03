"""
jarvis/browser/observations.py
─────────────────────────────
v0.28 Part 8 — observation identity and freshness.

Contract:
  - Every observation (page state, visible text, screenshot) carries an
    ``observation_id`` — a random, unguessable handle bound to (a) the
    session, (b) the URL observed, (c) a monotonic sequence number.
  - Every ACTION (click/fill/select/navigate) must reference the
    observation the model based its decision on.
  - The store validates actions: unknown IDs, foreign-session IDs, and
    IDs older than ``max_age_seconds`` are REJECTED (fail closed). A new
    navigation invalidates all previous observations of the session —
    content the model saw no longer describes the page.
  - IDs are 128-bit random hex: not enumerable, not forgeable by a page.

The store is per-session and lives inside the controller; it holds no
page content — only identity metadata (bounded ring buffer).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Bounded identity history per session (oldest evicted first).
_MAX_OBSERVATIONS = 32


@dataclass(frozen=True)
class Observation:
    """Identity metadata of one observation (never page content)."""

    observation_id: str
    session_id: str
    url: str
    seq: int
    created_at: float          # monotonic clock
    kind: str                  # "page_state" | "visible_text" | "screenshot"
    content_chars: int = 0


class StaleObservationError(RuntimeError):
    """Raised when an action references an unknown/stale/foreign observation."""


class ObservationStore:
    """
    Per-session registry of observation identity. Thread-safe. Deliberately
    tiny: identity only, no content, bounded size.
    """

    def __init__(self, session_id: str, max_age_seconds: float = 120.0) -> None:
        self._session_id = session_id
        self._max_age = float(max_age_seconds)
        self._lock = threading.Lock()
        self._seq = 0
        self._items: dict[str, Observation] = {}

    # ── Recording (controller-side, after each observation) ────────────────

    def record(self, *, url: str, kind: str, content_chars: int = 0) -> Observation:
        with self._lock:
            self._seq += 1
            obs = Observation(
                observation_id=uuid.uuid4().hex,
                session_id=self._session_id,
                url=url,
                seq=self._seq,
                created_at=time.monotonic(),
                kind=kind,
                content_chars=int(content_chars),
            )
            self._items[obs.observation_id] = obs
            # Bound: evict oldest (lowest seq) when over cap.
            if len(self._items) > _MAX_OBSERVATIONS:
                oldest = min(self._items.values(), key=lambda o: o.seq)
                del self._items[oldest.observation_id]
            return obs

    # ── Validation (action-side) ───────────────────────────────────────────

    def validate_action_reference(self, observation_id: str) -> Observation:
        """
        Return the referenced observation or raise StaleObservationError.

        Rejects: unknown IDs, IDs from another session (cannot happen when
        stores are per-session, but defended anyway), IDs older than the
        freshness window, and IDs superseded by a NEWER observation with a
        different URL (the page moved; the old view is stale regardless of
        age).
        """
        with self._lock:
            obs = self._items.get(str(observation_id or ""))
        if obs is None:
            raise StaleObservationError(
                "unknown observation_id: the action must reference a fresh "
                "observation from this session"
            )
        if obs.session_id != self._session_id:
            raise StaleObservationError("observation belongs to another session")
        age = time.monotonic() - obs.created_at
        if age > self._max_age:
            raise StaleObservationError(
                f"observation is {int(age)}s old (freshness limit "
                f"{int(self._max_age)}s); re-observe before acting"
            )
        # URL-change staleness: any observation whose seq is NOT the latest
        # for its URL is stale only when a LATER observation exists. The
        # controller invalidates on navigation; the seq check here is the
        # defensive backstop for direct store users.
        with self._lock:
            latest = max(self._items.values(), key=lambda o: o.seq)
        if latest.seq > obs.seq and latest.url != obs.url:
            raise StaleObservationError(
                "observation is stale: the browser has navigated to a "
                "different page since this observation was taken"
            )
        return obs

    def invalidate_all(self, *, reason: str) -> None:
        """Drop every observation (navigation, emergency stop, session end)."""
        with self._lock:
            self._items.clear()
        log.info("observations_invalidated", session_id=self._session_id, reason=reason)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
