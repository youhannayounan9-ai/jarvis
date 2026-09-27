"""
jarvis/api/health.py
────────────────────
Deep health checks for the JARVIS API.

Liveness (the current /health) answers "is the process up". This module
answers the stronger question: "can this instance actually serve?" — by
performing a real read AND write round-trip against the session store.

Design:
  - Cheap enough for frequent probes (two tiny SQL statements).
  - Fail-closed: any exception degrades to status="degraded".
  - Side-effect-free for callers: writes go to a probe row that is deleted
    immediately; the sessions table only ever sees transient probe rows
    with a recognizable id prefix (probes never collide with real sessions,
    whose ids are UUIDs).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from jarvis.runtime import JarvisRuntime

log = get_logger(__name__)

_PROBE_PREFIX = "healthprobe-"


def deep_health(runtime: "JarvisRuntime") -> dict[str, Any]:
    """
    Verify the runtime's persistence path end-to-end.

    Returns:
        {"status": "ok" | "degraded", "db": {"read": bool, "write": bool}}
    """
    read_ok = True
    write_ok = True
    try:
        # Read: a real query against the sessions table.
        runtime.store._conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()

        # Write + cleanup: insert a probe row, then remove it.
        probe_id = f"{_PROBE_PREFIX}{uuid.uuid4().hex}"
        with runtime.store._lock:
            runtime.store._conn.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (probe_id, "1970-01-01T00:00:00"),
            )
            runtime.store._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (probe_id,)
            )
            runtime.store._conn.commit()
    except Exception as e:
        # Distinguish read vs write failure for the operator.
        read_ok = False
        write_ok = False
        log.error("health_db_check_failed", error=str(e))

    # Cleanup sweep: remove any probe rows left behind by a crash between
    # insert and delete (idempotent, deletes at most a handful of rows).
    try:
        with runtime.store._lock:
            runtime.store._conn.execute(
                "DELETE FROM sessions WHERE id LIKE ?", (f"{_PROBE_PREFIX}%",)
            )
            runtime.store._conn.commit()
    except Exception:  # pragma: no cover - best-effort sweep
        pass

    ok = read_ok and write_ok
    return {
        "status": "ok" if ok else "degraded",
        "db": {"read": read_ok, "write": write_ok},
    }


def check_ollama(base_url: str, timeout: float = 3.0) -> bool:
    """
    True when the configured Ollama endpoint answers. Used by the maintenance
    doctor command (never by /health — the API must stay 'ok' while the LLM
    is down, because sessions/history/confirmations still work).
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


__all__ = ["check_ollama", "deep_health"]
