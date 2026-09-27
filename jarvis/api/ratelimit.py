"""
jarvis/api/ratelimit.py
───────────────────────
In-process sliding-window rate limiter for the JARVIS API.

- No new dependencies (sorted timestamps per client key).
- Keyed by API key when auth is configured, else by client IP.
- /health is exempt (wired in app.py).
- Thread-safe: uvicorn may serve requests from multiple threads.

Limits are per-process. For multi-replica deployments enforce limits at the
reverse proxy and keep this as defense in depth.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Sliding-window counter: at most ``max_requests`` per ``window_seconds``."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max(0, int(max_requests))
        self.window_seconds = max(0.0, float(window_seconds))
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0 and self.window_seconds > 0

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """
        Record one hit for ``key`` and decide whether it is allowed.

        Returns:
            (allowed, retry_after_seconds) — retry_after is 0 when allowed.
            Expired timestamps are pruned opportunistically; empty buckets
            are dropped so the map cannot grow without bound.
        """
        if not self.enabled:
            return True, 0

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


def _redact(key: str) -> str:
    """Never log full API keys; show a stable short prefix."""
    if len(key) <= 6:
        return key
    return f"{key[:6]}…"


def client_key(request) -> str:
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


# Process-wide limiter used by the app (swappable in tests).
limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


def set_limiter(new_limiter: SlidingWindowRateLimiter | None) -> None:
    """Replace the process limiter (tests / reconfiguration)."""
    global limiter
    if new_limiter is None:
        limiter = SlidingWindowRateLimiter(
            max_requests=settings.RATE_LIMIT_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )
    else:
        limiter = new_limiter


__all__ = [
    "SlidingWindowRateLimiter",
    "client_key",
    "limiter",
    "set_limiter",
]
