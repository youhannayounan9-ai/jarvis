"""
jarvis/api/ratelimit.py
───────────────────────
Sliding-window rate limiter for the JARVIS API.

v0.17: two interchangeable backends behind one class:

  - SQLite-backed (``store=``): hits are recorded in the shared JARVIS
    database, so multiple JARVIS *processes* pointing at the same file
    enforce ONE limit per client. Each ``check`` is a single write
    transaction (BEGIN IMMEDIATE), so two processes cannot both admit a
    boundary request — SQLite's file lock serializes them.
  - In-memory (default when no store is given): the original per-process
    deque implementation, still used by tools/embeddings without a store.

Both keep the original semantics:
  - at most ``max_requests`` per ``window_seconds`` per client key
    (API key when auth is on, else client IP)
  - 429 → (False, retry_after); 0 max_requests disables
  - ``/health`` is exempt (wired in app.py)

Bounded growth: events older than the window are deleted inside the same
transaction, so the table holds at most ~clients × max_requests rows.

SQLite notes (documented, not hidden):
  - Writes serialize process-wide on the file; a burst of concurrent
    requests waits on ``busy_timeout`` (5 s) rather than failing.
  - Throughput is far below any dedicated rate-limit store — this is a
    correctness/back-compat feature for small deployments, not a
    high-scale solution (see deploy/README.md).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from jarvis.config import settings
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from jarvis.memory.session_store import SessionStore

log = get_logger(__name__)

# Cross-process consistency requires WALL-CLOCK time (monotonic is
# per-process). Tests may inject explicit ``now`` values in either scheme;
# only differences are compared.
_DEFAULT_EPOCH = time.time


class SlidingWindowRateLimiter:
    """Sliding-window counter: at most ``max_requests`` per ``window_seconds``."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        store: "SessionStore | None" = None,
    ) -> None:
        self.max_requests = max(0, int(max_requests))
        self.window_seconds = max(0.0, float(window_seconds))
        self._store = store
        # In-memory fallback state (used when no store is provided).
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        if store is not None:
            self._init_table()

    @property
    def durable(self) -> bool:
        """True when hits are recorded in the shared database."""
        return self._store is not None

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0 and self.window_seconds > 0

    # ── Public API (unchanged contract) ────────────────────────────────────

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """
        Record one hit for ``key`` and decide whether it is allowed.

        Returns:
            (allowed, retry_after_seconds) — retry_after is 0 when allowed.
        """
        if not self.enabled:
            return True, 0
        if self._store is not None:
            return self._check_db(key, now)
        return self._check_memory(key, now)

    def cleanup(self, *, now: float | None = None) -> int:
        """Delete expired events (durable backend) / empty buckets (memory)."""
        if self._store is not None:
            return self._cleanup_db(now)
        return self._cleanup_memory()

    # ── SQLite backend ─────────────────────────────────────────────────────

    def _init_table(self) -> None:
        assert self._store is not None
        with self._store._lock:
            self._store._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    key TEXT NOT NULL,
                    ts  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rate_limit_key_ts
                    ON rate_limit_events (key, ts);
                """
            )
            self._store._conn.commit()

    def _check_db(self, key: str, now: float | None) -> tuple[bool, int]:
        store = self._store
        assert store is not None
        t = _DEFAULT_EPOCH() if now is None else float(now)
        cutoff = t - self.window_seconds

        with store._lock:
            conn = store._conn
            try:
                # One write transaction per decision: the file lock makes
                # the count+insert atomic ACROSS processes.
                conn.execute("BEGIN IMMEDIATE")
                # Bound growth: drop expired events for every key.
                conn.execute("DELETE FROM rate_limit_events WHERE ts <= ?", (cutoff,))
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM rate_limit_events WHERE key = ? AND ts > ?",
                    (key, cutoff),
                ).fetchone()
                count = int(row["n"]) if row else 0
                if count >= self.max_requests:
                    oldest = conn.execute(
                        "SELECT MIN(ts) AS oldest FROM rate_limit_events WHERE key = ? AND ts > ?",
                        (key, cutoff),
                    ).fetchone()
                    oldest_ts = float(oldest["oldest"]) if oldest and oldest["oldest"] is not None else t
                    retry_after = max(1, int(self.window_seconds - (t - oldest_ts)) + 1)
                    conn.commit()  # nothing was inserted
                    log.warning(
                        "rate_limit_exceeded", key=_redact(key), retry_after=retry_after
                    )
                    return False, retry_after
                conn.execute(
                    "INSERT INTO rate_limit_events (key, ts) VALUES (?, ?)",
                    (key, t),
                )
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:  # pragma: no cover - rollback is best-effort
                    pass
                raise
        return True, 0

    def _cleanup_db(self, now: float | None) -> int:
        store = self._store
        assert store is not None
        t = _DEFAULT_EPOCH() if now is None else float(now)
        with store._lock:
            cursor = store._conn.execute(
                "DELETE FROM rate_limit_events WHERE ts <= ?",
                (t - self.window_seconds,),
            )
            store._conn.commit()
            removed = cursor.rowcount
        return max(0, removed)

    # ── In-memory backend (original implementation) ─────────────────────────

    def _check_memory(self, key: str, now: float | None) -> tuple[bool, int]:
        t = time.monotonic() if now is None else now
        cutoff = t - self.window_seconds

        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (t - hits[0])) + 1)
                log.warning("rate_limit_exceeded", key=_redact(key), retry_after=retry_after)
                return False, retry_after
            hits.append(t)
            # Opportunistic pruning of dead buckets.
            if len(self._hits) > 10_000:
                for k in [k for k, q in self._hits.items() if not q]:
                    del self._hits[k]
            return True, 0

    def _cleanup_memory(self) -> int:
        with self._lock:
            dead = [k for k, q in self._hits.items() if not q]
            for k in dead:
                del self._hits[k]
            return len(dead)


def _redact(key: str) -> str:
    """Never log full API keys; show a stable short prefix."""
    if len(key) <= 6:
        return key
    return f"{key[:6]}…"


def client_key(request: Any) -> str:
    """Rate-limit identity: API key when present, else client host."""
    presented = request.headers.get("x-api-key")
    if not presented:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    if presented:
        return f"key:{presented}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


# Process-wide limiter used by the app (swappable in tests / by set_runtime).
limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


def set_limiter(new_limiter: SlidingWindowRateLimiter | None) -> None:
    """Replace the process limiter (tests / runtime swap).

    ``None`` restores the default in-memory limiter; a store-backed limiter
    is installed by the API layer when a runtime (and therefore a shared
    database) is available.
    """
    global limiter
    if new_limiter is None:
        limiter = SlidingWindowRateLimiter(
            max_requests=settings.RATE_LIMIT_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )
    else:
        limiter = new_limiter


def make_durable_limiter(store: "SessionStore") -> SlidingWindowRateLimiter:
    """Build the database-backed limiter used by the API service layer."""
    return SlidingWindowRateLimiter(
        max_requests=settings.RATE_LIMIT_REQUESTS,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        store=store,
    )


__all__ = [
    "SlidingWindowRateLimiter",
    "client_key",
    "limiter",
    "make_durable_limiter",
    "set_limiter",
]
