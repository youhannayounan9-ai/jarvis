"""
tests/test_action_idempotency.py
────────────────────────────────
v0.17 Track A: durable action identity + execution ledger.

Proves:
  - every parked confirmation gets a server-generated durable action id
    paired with a PENDING ledger row (same transaction)
  - the ledger claim is atomic: two claimants → one winner
  - duplicate approvals report the recorded outcome, never re-dispatch
  - concurrent approvals dispatch exactly once
  - crash ambiguity (RUNNING at startup) → UNKNOWN, never auto-retried
  - UNKNOWN surfaces a clear user-facing status and blocks resume
  - denial creates no execution attempt
  - nested confirmations keep unique action identity
  - confirmation continuation (approve → remaining steps → synthesis)
    still works end-to-end through the ledger
"""

import threading
from unittest.mock import patch

import pytest

from tests.fakes import fake_text_response, make_llm_stub

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import (
    ACTION_STATE_FAILED,
    ACTION_STATE_PENDING,
    ACTION_STATE_RUNNING,
    ACTION_STATE_SUCCEEDED,
    ACTION_STATE_UNKNOWN,
    SessionStore,
)
from jarvis.tools.registry import ToolRegistry


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from jarvis.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "v017_idem.db"))
    s = SessionStore()
    yield s
    s.close()


@pytest.fixture()
def orch(store):
    return Orchestrator(store, ToolRegistry(), PermissionGuard())


def _park(store, sid, tool="execute_python_code", args='{"code": "print(1)"}', ctx=None):
    """Park a protected action the way the orchestrator does."""
    return store.save_pending_confirmation(
        session_id=sid,
        tool_name=tool,
        tool_args=args,
        tool_call_id="call_test",
        risk_level="SYSTEM",
        context=ctx if ctx is not None else {"original_request": "run my code"},
    )


class TestActionIdentity:
    def test_park_creates_pending_ledger_row_with_durable_id(self, store):
        sid = store.create_session()
        confirmation_id = _park(store, sid)
        assert confirmation_id  # server-generated, not from model output

        action = store.get_action_execution_by_confirmation(confirmation_id)
        assert action is not None
        assert action.state == ACTION_STATE_PENDING
        assert action.action_id
        assert action.session_id == sid
        assert action.tool_name == "execute_python_code"
        assert action.attempt == 0
        assert action.result is None

    def test_action_ids_unique_and_survive_reopen(self, store, tmp_path):
        sid1 = store.create_session()
        cid1 = _park(store, sid1)
        store.complete_pending_confirmation(sid1)
        sid2 = store.create_session()
        cid2 = _park(store, sid2)
        assert cid1 != cid2
        a1 = store.get_action_execution_by_confirmation(cid1)
        a2 = store.get_action_execution_by_confirmation(cid2)
        assert a1.action_id != a2.action_id

        # Durable: a fresh store instance over the same file sees the rows.
        store.close()
        from jarvis.config import settings

        reopened = SessionStore()
        try:
            assert reopened.get_action_execution(a1.action_id).action_id == a1.action_id
            assert reopened.get_action_execution(a2.action_id).state == ACTION_STATE_PENDING
        finally:
            reopened.close()


class TestAtMostOnceClaim:
    def test_second_claim_cannot_run(self, store):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)

        assert store.claim_action_execution(action.action_id, "ownerA") == "claimed"
        assert store.claim_action_execution(action.action_id, "ownerB") == "already_running"
        row = store.get_action_execution(action.action_id)
        assert row.state == ACTION_STATE_RUNNING
        assert row.attempt == 1  # never incremented twice

    def test_terminal_states_block_reclaim(self, store):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)
        store.claim_action_execution(action.action_id, "ownerA")
        store.finish_action_execution(action.action_id, ACTION_STATE_SUCCEEDED, "Result: 1")

        result = store.claim_action_execution(action.action_id, "ownerB")
        assert result == "already_terminal:SUCCEEDED"
        assert store.get_action_execution(action.action_id).attempt == 1

    def test_concurrent_claims_single_winner(self, store):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)

        results: list[str] = []
        barrier = threading.Barrier(8)

        def claim(i: int) -> None:
            barrier.wait()
            results.append(store.claim_action_execution(action.action_id, f"owner{i}"))

        threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count("claimed") == 1
        assert results.count("already_running") == 7
        assert store.get_action_execution(action.action_id).attempt == 1

    def test_concurrent_pop_yields_one_confirmation(self, store):
        sid = store.create_session()
        _park(store, sid)
        pops: list[object] = []
        barrier = threading.Barrier(6)

        def pop() -> None:
            barrier.wait()
            pops.append(store.complete_pending_confirmation(sid))

        threads = [threading.Thread(target=pop) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(1 for p in pops if p is not None) == 1


class TestDuplicateApprovalBehavior:
    def _approve_once(self, orch, sid, tool_result="Result: 42"):
        with (
            patch.object(orch._registry, "dispatch", return_value=tool_result) as disp,
            patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("done")),
        ):
            first = orch.handle_confirmation(sid, True)
        assert disp.call_count == 1
        return first

    def test_duplicate_approval_does_not_redispatch(self, orch, store):
        sid = store.create_session()
        _park(store, sid)
        self._approve_once(orch, sid)
        row = store.get_last_action_execution(sid)
        assert row.state == ACTION_STATE_SUCCEEDED
        assert row.result == "Result: 42"

        # The confirmation is consumed; a repeat approval must NOT dispatch.
        with patch.object(orch._registry, "dispatch") as disp:
            second = orch.handle_confirmation(sid, True)
        disp.assert_not_called()
        assert "No pending actions" in second
        assert "already succeeded" in second
        assert "Result: 42" in second  # recorded outcome is reported

    def test_triple_approval_stable(self, orch, store):
        sid = store.create_session()
        _park(store, sid)
        self._approve_once(orch, sid)
        for _ in range(2):
            with patch.object(orch._registry, "dispatch") as disp:
                orch.handle_confirmation(sid, True)
            disp.assert_not_called()

    def test_approval_after_failure_not_rerun(self, orch, store):
        sid = store.create_session()
        _park(store, sid)
        self._approve_once(orch, sid, tool_result="ERROR: sandbox exploded")
        assert store.get_last_action_execution(sid).state == ACTION_STATE_FAILED

        with patch.object(orch._registry, "dispatch") as disp:
            second = orch.handle_confirmation(sid, True)
        disp.assert_not_called()
        assert "did not execute successfully" in second
        assert "was not re-executed" in second
        row = store.get_last_action_execution(sid)
        assert row.state == ACTION_STATE_FAILED

    def test_success_recorded_in_ledger(self, orch, store):
        sid = store.create_session()
        _park(store, sid)
        self._approve_once(orch, sid)
        row = store.get_last_action_execution(sid)
        assert row.state == ACTION_STATE_SUCCEEDED
        assert row.result == "Result: 42"
        assert row.attempt == 1
        assert row.owner  # durable traceability


class TestCrashSafety:
    def test_startup_sweep_marks_running_as_unknown(self, store, tmp_path, monkeypatch):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)
        store.claim_action_execution(action.action_id, "crashed-process")
        assert store.get_action_execution(action.action_id).state == ACTION_STATE_RUNNING

        # Simulate restart: fresh store over the same DB file.
        store.close()
        from jarvis.config import settings

        monkeypatch.setattr(settings, "db_path", str(tmp_path / "v017_idem.db"))
        reopened = SessionStore()
        try:
            recovered = reopened.recover_unknown_action_executions()
            assert recovered == 1
            row = reopened.get_action_execution(action.action_id)
            assert row.state == ACTION_STATE_UNKNOWN

            # Sweep is idempotent.
            assert reopened.recover_unknown_action_executions() == 0
        finally:
            reopened.close()

    def test_unknown_action_never_auto_reruns(self, orch, store):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)
        # Crash window: claimed (dispatch may have happened) but no result.
        store.claim_action_execution(action.action_id, "dead-owner")
        store.mark_action_unknown(action.action_id, "simulated crash")
        # No pending confirmation remains (it was consumed by the crash).
        store.complete_pending_confirmation(sid)

        with patch.object(orch._registry, "dispatch") as disp:
            reply = orch.handle_confirmation(sid, True)
        disp.assert_not_called()  # THE critical assertion
        assert reply.startswith("ACTION_EXECUTION_STATE_UNKNOWN")
        assert "automatic retry was prevented" in reply

    def test_unknown_report_via_duplicate_path(self, orch, store):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)
        store.claim_action_execution(action.action_id, "o")
        store.mark_action_unknown(action.action_id, "crash")
        store.complete_pending_confirmation(sid)

        reply = orch.handle_confirmation(sid, True)
        assert reply.startswith("ACTION_EXECUTION_STATE_UNKNOWN")
        assert action.action_id in reply

    def test_explicitly_marked_unknown_wins_over_claim(self, store):
        sid = store.create_session()
        cid = _park(store, sid)
        action = store.get_action_execution_by_confirmation(cid)
        store.claim_action_execution(action.action_id, "o")
        store.mark_action_unknown(action.action_id, "crash")
        assert (
            store.claim_action_execution(action.action_id, "other")
            == "already_terminal:UNKNOWN"
        )


class TestDenialAndNested:
    def test_denial_creates_no_execution_attempt(self, orch, store):
        sid = store.create_session()
        cid = _park(store, sid)
        with (
            patch.object(orch._registry, "dispatch") as disp,
            patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("ok")),
        ):
            orch.handle_confirmation(sid, False)
        disp.assert_not_called()
        row = store.get_action_execution_by_confirmation(cid)
        assert row.state == ACTION_STATE_FAILED
        assert "denied" in (row.result or "").lower()
        assert row.attempt == 0  # no execution attempt was ever made

    def test_denial_then_reapprove_reports_denial(self, orch, store):
        sid = store.create_session()
        _park(store, sid)
        with (
            patch.object(orch._registry, "dispatch") as disp,
            patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("ok")),
        ):
            orch.handle_confirmation(sid, False)
        disp.assert_not_called()
        with (
            patch.object(orch._registry, "dispatch") as disp2,
            patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("ok")),
        ):
            reply = orch.handle_confirmation(sid, True)
        disp2.assert_not_called()
        assert "No pending actions" in reply
        assert "did not execute successfully" in reply

    def test_nested_parks_have_unique_identity_and_superseded_row_fails(self, orch, store):
        sid = store.create_session()
        ctx = {"original_request": "multi risk"}
        cid1 = _park(store, sid, tool="execute_python_code", args='{"code": "a"}', ctx=ctx)
        a1 = store.get_action_execution_by_confirmation(cid1)
        # A second park replaces the first pending confirmation (one per
        # session); the superseded ledger row must never be claimable.
        cid2 = _park(store, sid, tool="execute_python_code", args='{"code": "b"}', ctx=ctx)
        a1_after = store.get_action_execution(a1.action_id)
        assert a1_after.state == ACTION_STATE_FAILED
        a2 = store.get_action_execution_by_confirmation(cid2)
        assert a2.action_id != a1.action_id
        assert a2.state == ACTION_STATE_PENDING
        assert store.claim_action_execution(a1.action_id, "x") == "already_terminal:FAILED"


class TestContinuationThroughLedger:
    def test_approve_resumes_plan_and_synthesizes(self, orch, store):
        sid = store.create_session()
        plan = [
            {"step_number": 1, "description": "do risky thing", "required_tools": ["dangerous_tool"]},
            {"step_number": 2, "description": "summarize", "required_tools": []},
        ]
        _park(store, sid, ctx={"original_request": "run the plan", "pending_plan": plan, "step_number": 1})

        script = [
            ("dangerous_tool", "{}"),           # step 2's ReAct round
            fake_text_response("step 2 done"),  # step 2 completes
            fake_text_response("FINAL ANSWER"),  # synthesis
        ]
        with (
            patch("jarvis.core.orchestrator.chat_completion", side_effect=make_llm_stub(script)),
            patch.object(orch._registry, "dispatch", return_value="Result: risky ok"),
        ):
            reply = orch.handle_confirmation(sid, True)

        assert reply == "FINAL ANSWER"
        row = store.get_last_action_execution(sid)
        assert row.state == ACTION_STATE_SUCCEEDED
        assert row.attempt == 1

    def test_legacy_row_without_ledger_still_dispatches(self, orch, store):
        """A confirmation parked without a ledger pair (simulated pre-v0.17)
        degrades to the historical direct dispatch — no crash, no block."""
        sid = store.create_session()
        _park(store, sid)
        action = store.get_last_action_execution(sid)
        # Simulate legacy: remove the ledger row entirely.
        with store._lock:
            store._conn.execute(
                "DELETE FROM action_executions WHERE action_id = ?", (action.action_id,)
            )
            store._conn.commit()

        with (
            patch.object(orch._registry, "dispatch", return_value="legacy ok") as disp,
            patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("done")),
        ):
            reply = orch.handle_confirmation(sid, True)
        assert disp.call_count == 1  # historical behavior: dispatched exactly once
        assert isinstance(reply, str) and reply

    def test_restart_between_park_and_approval(self, store, tmp_path, monkeypatch):
        """Park in one process/store; approve via a fresh store (restart)."""
        sid = store.create_session()
        cid = _park(store, sid)
        store.close()

        from jarvis.config import settings

        monkeypatch.setattr(settings, "db_path", str(tmp_path / "v017_idem.db"))
        fresh = SessionStore()
        try:
            orch = Orchestrator(fresh, ToolRegistry(), PermissionGuard())
            with (
                patch.object(orch._registry, "dispatch", return_value="post-restart ok") as disp,
                patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("done")),
            ):
                reply = orch.handle_confirmation(sid, True)
            assert disp.call_count == 1
            row = fresh.get_action_execution_by_confirmation(cid)
            assert row.state == ACTION_STATE_SUCCEEDED
        finally:
            fresh.close()

    def test_concurrent_handle_confirmation_single_dispatch(self, store):
        sid = store.create_session()
        _park(store, sid)
        orch = Orchestrator(store, ToolRegistry(), PermissionGuard())

        replies: list[str] = []
        start = threading.Event()

        def approve() -> None:
            start.wait()
            replies.append(orch.handle_confirmation(sid, True))

        threads = [threading.Thread(target=approve) for _ in range(4)]
        with (
            patch.object(orch._registry, "dispatch", return_value="RAN") as disp,
            patch("jarvis.core.orchestrator.chat_completion", return_value=fake_text_response("done")),
        ):
            for t in threads:
                t.start()
            start.set()
            for t in threads:
                t.join()

        assert disp.call_count == 1  # exactly one dispatch across all resolvers
        row = store.get_last_action_execution(sid)
        assert row.state == ACTION_STATE_SUCCEEDED
        assert row.result == "RAN"  # the winner's result, durably recorded
        # Non-winners get a duplicate/absent-pending report, never a dispatch.
        assert all(isinstance(r, str) and r for r in replies)


class TestLedgerBounded:
    def test_cleanup_removes_only_old_terminal_rows(self, store):
        sid = store.create_session()
        cid = _park(store, sid)
        a = store.get_action_execution_by_confirmation(cid)
        store.finish_action_execution(a.action_id, ACTION_STATE_SUCCEEDED, "r")
        # Fresh row (PENDING) must survive cleanup.
        cid2 = _park(store, sid)
        # Backdate the terminal row beyond the retention window.
        with store._lock:
            store._conn.execute(
                "UPDATE action_executions SET created_at = '2000-01-01T00:00:00' WHERE action_id = ?",
                (a.action_id,),
            )
            store._conn.commit()
        removed = store.cleanup_old_action_executions(max_age_days=30)
        assert removed == 1
        assert store.get_action_execution(a.action_id) is None
        assert store.get_action_execution_by_confirmation(cid2) is not None

    def test_migration_backfills_ledger_for_legacy_confirmations(self, tmp_path, monkeypatch):
        """A v0.15-era DB (pending_confirmations without confirmation_id)
        gains identity + ledger rows without losing data."""
        import sqlite3

        from jarvis.config import settings

        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
                content TEXT, tool_call_id TEXT, name TEXT, tool_calls_json TEXT,
                created_at TEXT NOT NULL);
            CREATE TABLE pending_confirmations (
                session_id TEXT PRIMARY KEY, tool_name TEXT, tool_args TEXT,
                tool_call_id TEXT, risk_level TEXT, created_at TEXT,
                expires_at TEXT, completed_at TEXT, context_json TEXT);
            INSERT INTO sessions VALUES ('s1', '2026-01-01');
            INSERT INTO sessions VALUES ('s2', '2026-01-01');
            INSERT INTO pending_confirmations VALUES
                ('s1', 'old_tool', '{}', 'c1', 'SYSTEM', '2026-01-01', '2099-01-01', NULL, '{}'),
                ('s2', 'done_tool', '{}', 'c2', 'SAFE', '2026-01-01', '2099-01-01', '2026-01-02', '{}');
            """
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(settings, "db_path", str(db))
        store = SessionStore()  # migration runs here
        try:
            pending = store.load_pending_confirmation("s1")
            assert pending is not None
            assert pending["confirmation_id"]
            ledger = store.get_action_execution_by_confirmation(pending["confirmation_id"])
            assert ledger is not None
            assert ledger.state == ACTION_STATE_PENDING
            # The completed legacy row maps to UNKNOWN (may have executed).
            with store._lock:
                rows = store._conn.execute(
                    "SELECT state FROM action_executions"
                ).fetchall()
            states = sorted(r["state"] for r in rows)
            assert states == ["PENDING", "UNKNOWN"]
        finally:
            store.close()

    def test_migration_preserves_existing_v015_context(self, tmp_path, monkeypatch):
        """The v0.15 context_json column and data survive the v0.17 migration."""
        import json as _json
        import sqlite3

        from jarvis.config import settings

        db = tmp_path / "v015.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
                content TEXT, tool_call_id TEXT, name TEXT, tool_calls_json TEXT,
                created_at TEXT NOT NULL);
            CREATE TABLE pending_confirmations (
                session_id TEXT PRIMARY KEY, tool_name TEXT, tool_args TEXT,
                tool_call_id TEXT, risk_level TEXT, created_at TEXT,
                expires_at TEXT, completed_at TEXT, context_json TEXT);
            INSERT INTO sessions VALUES ('s1', '2026-01-01');
            """
        )
        ctx = _json.dumps({"original_request": "keep me", "pending_plan": [{"step_number": 1}]})
        conn.execute(
            "INSERT INTO pending_confirmations VALUES ('s1','t','{}','c','SYSTEM','2026-01-01','2099-01-01',NULL,?)",
            (ctx,),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(settings, "db_path", str(db))
        store = SessionStore()
        try:
            pending = store.load_pending_confirmation("s1")
            assert pending["context"]["original_request"] == "keep me"
            assert pending["context"]["pending_plan"][0]["step_number"] == 1
        finally:
            store.close()
