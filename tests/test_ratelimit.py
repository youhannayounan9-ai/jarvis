"""
tests/test_ratelimit.py
───────────────────────
v0.17 Track C: rate limiting without single-process assumptions.

Proves:
  - the in-memory backend keeps the original sliding-window semantics
  - the SQLite backend (store=) enforces the SAME semantics through the
    shared database, so the limit holds across processes
  - window expiry and cleanup bound table growth
  - the opt-in wiring surface: make_durable_limiter / set_limiter(None)
"""

from concurrent.futures import ProcessPoolExecutor

import pytest

from jarvis.api.ratelimit import (
    SlidingWindowRateLimiter,
    make_durable_limiter,
    set_limiter,
)
from jarvis.config import settings
from jarvis.memory.session_store import SessionStore


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "v017_ratelimit.db")
    monkeypatch.setattr(settings, "db_path", path)
    return path


def _store() -> SessionStore:
    return SessionStore()


# ── In-memory backend (original semantics preserved) ──────────────────────────


class TestInMemoryBackend:
    def test_basic_window_semantics(self):
        rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        assert all(rl.check("k")[0] for _ in range(3))
        allowed, retry_after = rl.check("k")
        assert not allowed
        assert retry_after >= 1
        assert rl.durable is False

    def test_window_slides(self):
        rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
        assert rl.check("k", now=100.0)[0]
        assert rl.check("k", now=101.0)[0]
        assert not rl.check("k", now=102.0)[0]
        # Both hits expired: the window has slid past them.
        assert rl.check("k", now=115.0)[0]

    def test_disabled_allows_everything(self):
        rl = SlidingWindowRateLimiter(max_requests=0, window_seconds=60)
        assert all(rl.check("k")[0] for _ in range(100))

    def test_keys_are_independent(self):
        rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        assert rl.check("a")[0]
        assert rl.check("b")[0]
        assert not rl.check("a")[0]


# ── SQLite backend (same semantics through the shared database) ───────────────


class TestSqliteBackend:
    def test_basic_window_semantics_db(self, db_path):
        store = _store()
        try:
            rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60, store=store)
            assert rl.durable is True
            assert all(rl.check("k")[0] for _ in range(3))
            allowed, retry_after = rl.check("k")
            assert not allowed
            assert retry_after >= 1
        finally:
            store.close()

    def test_hits_recorded_in_shared_table(self, db_path):
        store = _store()
        try:
            rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=60, store=store)
            for _ in range(3):
                rl.check("client-a")
            with store._lock:
                rows = store._conn.execute(
                    "SELECT COUNT(*) AS n FROM rate_limit_events WHERE key = 'client-a'"
                ).fetchone()["n"]
            assert rows == 3
        finally:
            store.close()

    def test_window_expiry_uses_wall_clock(self, db_path):
        store = _store()
        try:
            rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=10, store=store)
            assert rl.check("k", now=100.0)[0]
            assert rl.check("k", now=101.0)[0]
            assert not rl.check("k", now=102.0)[0]
            assert rl.check("k", now=115.0)[0]
        finally:
            store.close()

    def test_cleanup_removes_only_expired_events(self, db_path):
        """cleanup() is the background-maintenance path: it purges expired
        events when no check has intervened (checks self-purge in-transaction)."""
        store = _store()
        try:
            rl = SlidingWindowRateLimiter(max_requests=10, window_seconds=10, store=store)
            rl.check("old")  # real clock
            # Age the event past the window WITHOUT another check running.
            with store._lock:
                store._conn.execute("UPDATE rate_limit_events SET ts = ts - 100")
                store._conn.commit()
            removed = rl.cleanup()
            assert removed == 1
            # Table is empty afterwards; the limiter still admits normally.
            assert rl.check("old")[0]
        finally:
            store.close()

    def test_check_deletes_expired_rows_for_all_keys(self, db_path):
        """Growth is bounded: any check purges every key's expired events."""
        store = _store()
        try:
            rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=10, store=store)
            rl.check("ghost", now=0.0)
            rl.check("live", now=100.0)
            rl.check("unrelated", now=100.5)  # purges on its own transaction
            with store._lock:
                keys = {
                    r["key"]
                    for r in store._conn.execute("SELECT key FROM rate_limit_events")
                }
            assert "ghost" not in keys
        finally:
            store.close()

    def test_boundary_atomicity_under_threads(self, db_path):
        """Within one process, threads still cannot exceed the limit."""
        import threading

        store = _store()
        try:
            rl = SlidingWindowRateLimiter(max_requests=4, window_seconds=60, store=store)
            barrier = threading.Barrier(8)
            outcomes: list[bool] = []
            lock = threading.Lock()

            def hit():
                barrier.wait()
                allowed, _ = rl.check("racer")
                with lock:
                    outcomes.append(allowed)

            threads = [threading.Thread(target=hit) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert sum(outcomes) == 4  # exactly max_requests admitted
        finally:
            store.close()


# ── Cross-process: the whole point of the durable backend ─────────────────────


def _rl_worker(args):
    path, n = args
    import jarvis.config as cfg

    cfg.settings.db_path = path
    from jarvis.api.ratelimit import SlidingWindowRateLimiter as _RL
    from jarvis.memory.session_store import SessionStore as _Store

    store = _Store()
    try:
        rl = _RL(max_requests=2, window_seconds=60, store=store)
        return [rl.check("shared-client")[0] for _ in range(n)]
    finally:
        store.close()


class TestCrossProcessLimiter:
    def test_limit_holds_across_processes(self, db_path):
        """4 attempts (2 processes × 2 checks) against a limit of 2: exactly
        2 admitted overall — the database is the coordination point."""
        # Deployment shape: the first process creates the database; worker
        # processes JOIN an existing file (two fresh-DB creations racing is
        # an anti-pattern SQLite does not guarantee on any platform).
        seed = _store()
        seed.close()
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_rl_worker, [(db_path, 2)] * 2))

        admitted = sum(sum(r) for r in results)
        assert admitted == 2
        # No process saw more than its own share of admits.
        assert all(sum(r) <= 2 for r in results)


# ── Opt-in wiring surface ─────────────────────────────────────────────────────


class TestWiringSurface:
    def test_make_durable_limiter_uses_settings(self, db_path):
        store = _store()
        try:
            rl = make_durable_limiter(store)
            assert rl.durable
            assert rl.max_requests == settings.RATE_LIMIT_REQUESTS
            assert rl.window_seconds == settings.RATE_LIMIT_WINDOW_SECONDS
        finally:
            store.close()

    def test_set_limiter_none_restores_memory_default(self, db_path):
        store = _store()
        try:
            set_limiter(make_durable_limiter(store))
            from jarvis.api import ratelimit

            assert ratelimit.limiter.durable
            set_limiter(None)
            assert not ratelimit.limiter.durable
        finally:
            set_limiter(None)
            store.close()
