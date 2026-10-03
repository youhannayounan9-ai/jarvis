"""
jarvis/browser/registry.py
──────────────────────────
v0.28 Part 7/25 — bounded ownership of browser controllers.

One BrowserController per agent session, created on demand and closed
explicitly. Bounds and cleanup:

  - Max concurrently open controllers (settings.BROWSER_MAX_SESSIONS);
    opening beyond the cap force-closes the LEAST-RECENTLY-USED one.
  - Idle TTL: ``close_idle()`` sweeps controllers idle beyond
    ``BROWSER_SESSION_IDLE_TTL_SECONDS`` (called opportunistically on
    registry access and at graceful API shutdown).
  - ``close_all()`` at process exit / emergency stop.

The registry never holds page content; only controller handles.
"""

from __future__ import annotations

import threading
import time

from jarvis.browser.controller import BrowserController
from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class BrowserSessionRegistry:
    """Thread-safe, bounded registry of open browser controllers."""

    def __init__(self, max_sessions: int | None = None) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[BrowserController, float]] = {}
        self._max = int(max_sessions or settings.BROWSER_MAX_SESSIONS)

    def get(self, session_id: str) -> BrowserController | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            controller, _ = entry
            self._sessions[session_id] = (controller, time.monotonic())
            return controller

    def create(self, session_id: str, **kwargs) -> BrowserController:
        """Create a controller for ``session_id`` (LRU-capped)."""
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                existing[0].close()
                del self._sessions[session_id]
            while len(self._sessions) >= self._max:
                oldest_key = min(self._sessions, key=lambda k: self._sessions[k][1])
                controller, _ = self._sessions.pop(oldest_key)
                log.info("browser_session_evicted_lru", session=oldest_key[:8])
                controller.close()
            controller = BrowserController(session_id, **kwargs)
            self._sessions[session_id] = (controller, time.monotonic())
            log.info(
                "browser_session_created", session=session_id[:8],
                open=len(self._sessions),
            )
            return controller

    def get_or_create(self, session_id: str, **kwargs) -> BrowserController:
        with self._lock:
            found = self.get(session_id)
        if found is not None and not found.status()["closed"]:
            return found
        return self.create(session_id, **kwargs)

    def close(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is None:
            return False
        entry[0].close()
        return True

    def close_idle(self) -> int:
        """Close controllers idle beyond the TTL; returns how many."""
        ttl = float(settings.BROWSER_SESSION_IDLE_TTL_SECONDS)
        now = time.monotonic()
        with self._lock:
            stale = [k for k, (_, at) in self._sessions.items() if now - at > ttl]
            victims = [self._sessions.pop(k) for k in stale]
        for controller, _ in victims:
            controller.close()
        if victims:
            log.info("browser_sessions_idle_closed", count=len(victims))
        return len(victims)

    def close_all(self) -> int:
        with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()
        for controller, _ in entries:
            controller.close()
        return len(entries)

    def open_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def controllers(self) -> list[BrowserController]:
        """Snapshot of open controllers (for status sweeps / shutdown)."""
        with self._lock:
            return [controller for controller, _ in self._sessions.values()]

    def status(self) -> dict[str, object]:
        with self._lock:
            return {"open_sessions": len(self._sessions), "max": self._max}


# Process-global registry (one per JARVIS process).
_REGISTRY: BrowserSessionRegistry | None = None


def get_browser_registry() -> BrowserSessionRegistry:
    """
    The process-global browser-session registry. Lazily created so tests
    can reset isolation state between cases (see reset_for_tests).
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = BrowserSessionRegistry()
    return _REGISTRY


def close_browser_registry() -> int:
    """Close every open controller and drop the registry (process exit)."""
    global _REGISTRY
    if _REGISTRY is None:
        return 0
    registry, _REGISTRY = _REGISTRY, None
    return registry.close_all()


def reset_for_tests() -> None:
    """Hard reset: close everything and force a fresh registry (tests only)."""
    close_browser_registry()
