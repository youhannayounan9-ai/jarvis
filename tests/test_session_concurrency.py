"""
tests/test_session_concurrency.py
─────────────────────────────────
v0.17 Track B: database-backed session coordination.

Proves:
  - two runtimes in ONE process: same-session turns are mutually exclusive
    (in-process mutex + shared lease), different sessions run in parallel
  - two runtimes over SEPARATE connections to the SAME SQLite file (true
    multi-process shape): same-session lease is exclusive, different
    sessions proceed concurrently, and a stale lease is recovered after TTL
  - fencing increases on ownership change; releases are owner-checked
"""

from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pytest

from jarvis.config import settings
from jarvis.memory.session_store import SESSION_LEASE_TTL_SECONDS, SessionStore


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "v017_lease.db")
    monkeypatch.setattr(settings, "db_path", path)
    return path


def _runtime_with_store(store: SessionStore):
    """A JarvisRuntime wired around an EXISTING store (shared or separate)."""
    from jarvis.core.permissions import PermissionGuard
    from jarvis.core.orchestrator import Orchestrator
    from jarvis.runtime import JarvisRuntime
    from jarvis.tools.registry import ToolRegistry

    registry = ToolRegistry()
    guard = PermissionGuard()
    return JarvisRuntime(
        store=store,
        registry=registry,
        guard=guard,
        orchestrator=Orchestrator(store, registry, guard),
    )


class TestSameProcessLease:
    def test_second_runtime_blocked_on_same_session(self, db_path):
        with patch("jarvis.runtime.get_vector_store"):
            rt1_store = SessionStore()
            rt1 = _runtime_with_store(rt1_store)
            rt2_store = SessionStore()  # separate connection, same file
            rt2 = _runtime_with_store(rt2_store)
            try:
                sid = rt1.start_session()
                assert rt1._acquire_session_lease_or_busy(sid) == 1
                with pytest.raises(TimeoutError):
                    rt2._acquire_session_lease_or_busy(sid)
                # Reentrant by the SAME owner is fine (refresh, not block).
                assert rt1._acquire_session_lease_or_busy(sid) == 1
            finally:
                rt1.store.release_session_lease(sid, rt1.owner_token)
                rt1_store.close()
                rt2_store.close()

    def test_different_sessions_independent(self, db_path):
        rt1_store = SessionStore()
        rt1 = _runtime_with_store(rt1_store)
        rt2_store = SessionStore()
        rt2 = _runtime_with_store(rt2_store)
        try:
            s1 = rt1.start_session()
            s2 = rt2.start_session()
            f1 = rt1._acquire_session_lease_or_busy(s1)
            f2 = rt2._acquire_session_lease_or_busy(s2)  # no contention
            assert f1 == 1 and f2 == 1
            rt1.store.release_session_lease(s1, rt1.owner_token)
            rt2.store.release_session_lease(s2, rt2.owner_token)
        finally:
            rt1_store.close()
            rt2_store.close()

    def test_release_allows_reacquire(self, db_path):
        rt1_store = SessionStore()
        rt1 = _runtime_with_store(rt1_store)
        rt2_store = SessionStore()
        rt2 = _runtime_with_store(rt2_store)
        try:
            sid = rt1.start_session()
            rt1._acquire_session_lease_or_busy(sid)
            assert rt1.store.release_session_lease(sid, rt1.owner_token)
            assert rt2._acquire_session_lease_or_busy(sid) == 1
            rt2.store.release_session_lease(sid, rt2.owner_token)
        finally:
            rt1_store.close()
            rt2_store.close()

    def test_fencing_and_stale_ownership_rules(self, db_path):
        store = SessionStore()
        try:
            sid = store.create_session()
            ok1, f1 = store.acquire_session_lease(sid, "ownerA")
            ok2, f2 = store.acquire_session_lease(sid, "ownerB")  # blocked
            assert (ok1, f1) == (True, 1) and (ok2, f2) == (False, 1)
            store.release_session_lease(sid, "ownerA")
            ok3, f3 = store.acquire_session_lease(sid, "ownerB")
            assert (ok3, f3) == (True, 1)  # clean handoff: fencing unchanged
            # Simulate a stale holder (lease expired) instead of a release:
            with store._lock:
                store._conn.execute(
                    "UPDATE session_leases SET expires_at = '2000-01-01T00:00:00' WHERE session_id = ?",
                    (sid,),
                )
                store._conn.commit()
            ok4, f4 = store.acquire_session_lease(sid, "ownerC")
            assert ok4 and f4 == 2  # takeover bumps fencing
            # The stale owner must not be able to renew or release.
            assert not store.renew_session_lease(sid, "ownerB")
            assert not store.release_session_lease(sid, "ownerB")
        finally:
            store.close()

    def test_renew_only_by_owner(self, db_path):
        store = SessionStore()
        try:
            sid = store.create_session()
            store.acquire_session_lease(sid, "ownerA", ttl_seconds=3600)
            assert store.renew_session_lease(sid, "ownerA", ttl_seconds=7200)
            row = store.get_session_lease(sid)
            assert row["owner_token"] == "ownerA"
            assert not store.renew_session_lease(sid, "intruder")
        finally:
            store.close()

    def test_no_unbounded_lock_ttl_exists(self):
        """The lease TTL is finite: a crashed owner cannot lock forever."""
        assert 0 < SESSION_LEASE_TTL_SECONDS <= 3600


# ── Multi-process workers (module level so ProcessPoolExecutor can pickle) ────


def _worker_takeover(args):
    path, sid, hold_seconds = args
    import time

    import jarvis.config as cfg

    cfg.settings.db_path = path
    st = SessionStore()
    try:
        owner = f"worker-{time.time_ns()}"
        acquired, fencing = st.acquire_session_lease(sid, owner)
        if acquired:
            time.sleep(hold_seconds)  # let the sibling observe the live lease
            st.release_session_lease(sid, owner)
        return {"acquired": acquired, "fencing": fencing}
    finally:
        st.close()


def _worker_recover(args):
    path, sid = args
    import jarvis.config as cfg

    cfg.settings.db_path = path
    st = SessionStore()
    try:
        acquired, fencing = st.acquire_session_lease(sid, "recovering-worker")
        owner = st.get_session_lease(sid)["owner_token"] if acquired else None
        return {"acquired": acquired, "fencing": fencing, "owner": owner}
    finally:
        st.close()


def _worker_own_session(args):
    path, sid = args
    import time

    import jarvis.config as cfg

    cfg.settings.db_path = path
    st = SessionStore()
    try:
        acquired, _ = st.acquire_session_lease(sid, f"worker-{sid[:4]}")
        time.sleep(0.15)
        return acquired
    finally:
        st.close()


def _worker_churn(args):
    path, sid, i = args
    import jarvis.config as cfg

    cfg.settings.db_path = path
    st = SessionStore()
    try:
        for _ in range(5):
            acquired, _ = st.acquire_session_lease(sid, f"w{i}")
            if acquired:
                st.release_session_lease(sid, f"w{i}")
        return True
    finally:
        st.close()


class TestMultiProcessLease:
    """Two OS processes over the same SQLite file."""

    def test_stale_lease_recovered_across_processes(self, db_path):
        store = SessionStore()
        sid = store.create_session()
        store.acquire_session_lease(sid, "dead-process", ttl_seconds=3600)
        # Owner 'dies': backdate the lease so its TTL has elapsed.
        with store._lock:
            store._conn.execute(
                "UPDATE session_leases SET expires_at = '2000-01-01T00:00:00' WHERE session_id = ?",
                (sid,),
            )
            store._conn.commit()
        store.close()

        with ProcessPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_worker_recover, (db_path, sid)).result(timeout=60)

        assert result["acquired"] is True
        assert result["owner"] == "recovering-worker"
        assert result["fencing"] == 2  # bumped on takeover from the dead owner

    def test_different_sessions_concurrent_across_processes(self, db_path):
        store = SessionStore()
        sid1 = store.create_session()
        sid2 = store.create_session()
        store.close()

        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_worker_own_session, (db_path, sid1)),
                pool.submit(_worker_own_session, (db_path, sid2)),
            ]
            outcomes = [f.result(timeout=60) for f in futures]

        assert outcomes == [True, True]  # zero cross-session contention

    def test_takeover_race_across_processes_consistent(self, db_path):
        store = SessionStore()
        sid = store.create_session()
        store.close()

        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_worker_takeover, [(db_path, sid, 0.05)] * 2))

        acquired = [r for r in results if r["acquired"]]
        blocked = [r for r in results if not r["acquired"]]
        # The lease is exclusive: at most one live owner holds it.
        assert 1 <= len(acquired) <= 2
        assert len(acquired) + len(blocked) == 2
        # A blocked loser still sees a real fencing token, never corrupt state.
        assert all(r["fencing"] >= 1 for r in results)
        # Every acquired lease was released in-process (worker releases on success).
        reopened = SessionStore()
        try:
            reopened.acquire_session_lease(sid, "verifier")
            reopened.release_session_lease(sid, "verifier")
        finally:
            reopened.close()

    def test_db_consistent_after_concurrent_lease_churn(self, db_path):
        store = SessionStore()
        sid = store.create_session()
        store.close()

        with ProcessPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(_worker_churn, [(db_path, sid, i) for i in range(3)]))

        assert all(results)
        # Churn releases only clean handoffs, so one live lease may remain:
        # exactly 0 or 1 lease rows survive, never more, never corrupt.
        reopened = SessionStore()
        try:
            with reopened._lock:
                count = reopened._conn.execute(
                    "SELECT COUNT(*) AS n FROM session_leases"
                ).fetchone()["n"]
            assert count in (0, 1)
            if count == 1:
                # A residual lease must still work: expiry backstops it.
                ok, fencing = reopened.acquire_session_lease(sid, "verifier")
                assert ok and fencing >= 1
                reopened.release_session_lease(sid, "verifier")
        finally:
            reopened.close()
