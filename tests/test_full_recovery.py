"""
tests/test_full_recovery.py
───────────────────────────
v0.19 Tracks B/C/D: full-context UNKNOWN recovery, session timelines, and
operational-record retention.

Proves:
  - pause context is captured on the ledger row at park time (same
    transaction) and survives the confirmation pop + crash window
  - reissue copies the ORIGINAL task context onto the new confirmation
    (transitively across a reissue chain), with recovery lineage markers
  - approval of a recovered action dispatches exactly once, then resumes
    the remaining plan and synthesizes a final answer (real orchestrator)
  - denial dispatches nothing, records the denial, original stays UNKNOWN
  - nested confirmations during recovery keep context and are themselves
    recoverable; a second UNKNOWN mid-recovery reissues safely
  - malformed context degrades to the pre-v0.19 raw-result reply
  - unknown session: timeline is empty, reissue preview honest
  - timeline: causal ordering, bounded, session-isolated, JSON-safe, no
    tool_args/result/message-content/owner-token leakage
  - retention: old terminal rows removed; PENDING/RUNNING/UNKNOWN and
    active-confirmation FAILED rows protected; reissue-referenced rows
    kept; dry-run mutates nothing; audit chains stay intact
All offline (LLM mocked, in-memory SQLite).
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tests.fakes import make_llm_stub

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
from jarvis.tools.base import BaseTool
from jarvis.tools.registry import ToolRegistry


class HighRiskTool(BaseTool):
    name = "execute_python_code"
    description = "test high-risk tool"
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}}
    risk_level = "SYSTEM"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, code: str = "", **kwargs) -> str:
        self.calls += 1
        return f"OK:{code}"


class SafeTool(BaseTool):
    name = "calculator"
    description = "test safe tool"
    parameters = {"type": "object", "properties": {"expression": {"type": "string"}}}
    risk_level = "SAFE"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, expression: str = "", **kwargs) -> str:
        self.calls += 1
        return "42"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from jarvis.config import settings
    from tests.fakes import fake_text_response

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "v019_recovery.db"))
    store = SessionStore()
    reg = ToolRegistry()
    hr, sf = HighRiskTool(), SafeTool()
    reg._tools["execute_python_code"] = hr
    reg._tools["calculator"] = sf
    # The Planner captures its LLM client at construction. Patch the name
    # BEFORE constructing the Orchestrator so the planner never reaches the
    # real model (its calls would otherwise consume the ReAct script and
    # shift every subsequent mock response — the offline guard would then
    # refuse it loudly anyway). The deterministic TWO-step plan gives every
    # recovery test a real remaining step to resume after approval.
    two_step_plan = (
        '[{"step_number": 1, "description": "Run the side effect and then '
        'summarize the plan output", "required_tools": []}, '
        '{"step_number": 2, "description": "Summarize the results and write '
        'the final answer", "required_tools": []}]'
    )
    with patch(
        "jarvis.core.orchestrator.chat_completion",
        return_value=fake_text_response(two_step_plan),
    ):
        orch = Orchestrator(store, reg, PermissionGuard())
    env = type("Env", (), {"store": store, "reg": reg, "hr": hr, "sf": sf, "orch": orch})()
    yield env
    store.close()


def _park_plan_turn(env, sid, llm_script):
    """Drive a real chat turn that parks a confirmation mid-plan."""
    with patch(
        "jarvis.core.orchestrator.chat_completion",
        side_effect=make_llm_stub(llm_script),
    ):
        out = env.orch.chat(sid, "Run the side effect and then summarize the plan output")
    return out


def _crash_parked_action(env, sid):
    """Pop the confirmation, claim, and mark UNKNOWN exactly like a crash."""
    s = env.store
    pend = s.load_pending_confirmation(sid)
    assert pend is not None
    aid = s.get_action_execution_by_confirmation(pend["confirmation_id"]).action_id
    s.complete_pending_confirmation(sid)
    assert s.claim_action_execution(aid, "host:1:t:o") == "claimed"
    s.mark_action_unknown(aid, "simulated crash")
    return aid


# ── Track B: context capture + full-context reissue ──────────────────────────


class TestContextCapture:
    def test_pause_context_captured_on_ledger_row(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s = env.store
        pend = s.load_pending_confirmation("s1")
        aid = s.get_action_execution_by_confirmation(
            pend["confirmation_id"]
        ).action_id
        row = s.get_action_execution(aid)
        assert row.pause_context_json, "context must be captured at park time"
        ctx = json.loads(row.pause_context_json)
        assert ctx["original_request"] == (
            "Run the side effect and then summarize the plan output"
        )
        assert ctx["step_number"] == 1
        assert ctx["remaining_rounds"] > 0

    def test_context_survives_pop_and_crash(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s = env.store
        aid = _crash_parked_action(env, "s1")
        # Confirmation is gone (popped); the ledger row still has the task.
        assert s.load_pending_confirmation("s1") is None
        preview = s.get_action_recovery_preview(aid)
        assert preview["recoverable"] is True
        assert "summarize" in preview["original_request"]

    def test_legacy_row_reports_honestly(self, env):
        """A row without captured context must not pretend to be recoverable."""
        s = env.store
        cid = s.save_pending_confirmation("sL", "calculator", "{}", "c1", "SAFE")
        # Simulate a pre-v0.19 row: strip the captured context.
        s._conn.execute(
            "UPDATE action_executions SET pause_context_json=NULL WHERE confirmation_id=?",
            (cid,),
        )
        s._conn.commit()
        aid = s.get_action_execution_by_confirmation(cid).action_id
        s.complete_pending_confirmation("sL")
        s.claim_action_execution(aid, "host:1:t:o")
        s.mark_action_unknown(aid, "crash")
        preview = s.get_action_recovery_preview(aid)
        assert preview["recoverable"] is False
        assert "no durable resume context" in preview["recoverable_reason"]

    def test_corrupt_context_reports_corrupt(self, env):
        s = env.store
        cid = s.save_pending_confirmation("sC", "calculator", "{}", "c1", "SAFE")
        aid = s.get_action_execution_by_confirmation(cid).action_id
        s._conn.execute(
            "UPDATE action_executions SET pause_context_json='{broken' WHERE action_id=?",
            (aid,),
        )
        s._conn.commit()
        s.complete_pending_confirmation("sC")
        s.claim_action_execution(aid, "host:1:t:o")
        s.mark_action_unknown(aid, "crash")
        preview = s.get_action_recovery_preview(aid)
        assert preview["recoverable"] is False
        assert preview.get("corrupt") is True


class TestFullContextReissue:
    def test_reissue_copies_context_to_new_confirmation(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s = env.store
        aid = _crash_parked_action(env, "s1")
        new_id = s.request_action_reissue(aid, "req-1")
        pend = s.load_pending_confirmation("s1")
        assert pend is not None
        rc = pend["context"]
        assert rc["original_request"] == (
            "Run the side effect and then summarize the plan output"
        )
        assert rc["recovered_from_action"] == aid
        assert rc["recovered_request_id"] == "req-1"
        assert rc["remaining_rounds"] > 0
        assert s.get_action_execution(aid).state == ACTION_STATE_UNKNOWN

    def test_approval_resumes_plan_and_synthesizes(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s, orch = env.store, env.orch
        aid = _crash_parked_action(env, "s1")
        new_id = s.request_action_reissue(aid, "req-1")
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=make_llm_stub(["step result", "FINAL ANSWER"]),
        ) as m:
            out = orch.handle_confirmation("s1", True)
        assert env.hr.calls == 1, "recovered action dispatches exactly once"
        assert s.get_action_execution(new_id).state == ACTION_STATE_SUCCEEDED
        assert s.get_action_execution(aid).state == ACTION_STATE_UNKNOWN
        assert "FINAL ANSWER" in out, "remaining work must complete with synthesis"
        assert m.call_count >= 2, "remaining step + synthesis calls"

    def test_denial_does_not_dispatch_and_records_denial(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s, orch = env.store, env.orch
        aid = _crash_parked_action(env, "s1")
        new_id = s.request_action_reissue(aid, "req-1")
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=make_llm_stub(["denial synthesis"]),
        ):
            out = orch.handle_confirmation("s1", False)
        assert env.hr.calls == 0, "denial must never dispatch"
        assert s.get_action_execution(new_id).state == ACTION_STATE_FAILED
        assert s.get_action_execution(aid).state == ACTION_STATE_UNKNOWN
        hist = s.load_history("s1")
        assert any(
            m.get("role") == "tool"
            and "denied" in str(m.get("content", "")).lower()
            for m in hist
        ), "denial must be recorded in history"
        assert isinstance(out, str) and out

    def test_nested_confirmation_during_recovery(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s, orch = env.store, env.orch
        aid = _crash_parked_action(env, "s1")
        s.request_action_reissue(aid, "req-1")
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=make_llm_stub([
                ("execute_python_code", '{"code": "nested()"}'),
                "unused",
            ]),
        ):
            orch.handle_confirmation("s1", True)
        assert env.hr.calls == 1
        pend2 = s.load_pending_confirmation("s1")
        assert pend2 is not None, "nested action re-parks"
        nested_aid = s.get_action_execution_by_confirmation(
            pend2["confirmation_id"]
        ).action_id
        # Nested park captures its own full pause context on the ledger row.
        nested_preview = s.get_action_recovery_preview(nested_aid)
        assert nested_preview["recoverable"] is True
        assert nested_preview["step_number"] == 2
        # Approving the nested action completes the turn.
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=make_llm_stub(["FINAL NESTED ANSWER"]),
        ):
            out = orch.handle_confirmation("s1", True)
        assert env.hr.calls == 2
        assert "FINAL NESTED ANSWER" in out

    def test_second_unknown_mid_recovery_transitive_context(self, env):
        s = env.store
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        aid = _crash_parked_action(env, "s1")
        new1 = s.request_action_reissue(aid, "req-1")
        # The recovered action crashes the same way.
        s.complete_pending_confirmation("s1")
        assert s.claim_action_execution(new1, "host:1:t:o2") == "claimed"
        s.mark_action_unknown(new1, "second crash")
        new2 = s.request_action_reissue(new1, "req-2")
        assert s.get_reissue_origin(new2) == new1
        # Context must survive TRANSITIVELY (original row holds the task).
        pend = s.load_pending_confirmation("s1")
        assert pend is not None
        ctx = pend["context"]
        assert ctx["recovered_from_action"] == new1
        assert ctx["original_request"] == (
            "Run the side effect and then summarize the plan output"
        )

    def test_budget_respected_on_recovery(self, env):
        s = env.store
        # Hand-build a recovery with a nearly exhausted budget.
        cid = s.save_pending_confirmation(
            "sB", "execute_python_code", '{"code": "x()"}', "cb", "SYSTEM",
            context={
                "original_request": "budgeted task",
                "memory_cue": "",
                "step_number": 1,
                "pending_plan": [],
                "completed_steps": [],
                "remaining_rounds": 1,
                "mode": "simple",
            },
        )
        aid = s.get_action_execution_by_confirmation(cid).action_id
        s.complete_pending_confirmation("sB")
        s.claim_action_execution(aid, "host:1:t:o")
        s.mark_action_unknown(aid, "crash")
        new_id = s.request_action_reissue(aid, "req-b")
        pend = s.load_pending_confirmation("sB")
        assert pend["context"]["remaining_rounds"] == 1, "budget copied verbatim"
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=make_llm_stub(["FINAL BUDGETED"]),
        ):
            out = orch_handle(env, "sB", True)
        assert env.hr.calls == 1
        assert s.get_action_execution(new_id).state == ACTION_STATE_SUCCEEDED
        assert "FINAL BUDGETED" in out

    def test_malformed_context_degrades_gracefully(self, env):
        s, orch = env.store, env.orch
        cid = s.save_pending_confirmation("sM", "calculator", "{}", "cm", "SAFE")
        aid = s.get_action_execution_by_confirmation(cid).action_id
        s._conn.execute(
            "UPDATE action_executions SET pause_context_json='{oops' WHERE action_id=?",
            (aid,),
        )
        s._conn.commit()
        s.complete_pending_confirmation("sM")
        s.claim_action_execution(aid, "host:1:t:o")
        s.mark_action_unknown(aid, "crash")
        s.request_action_reissue(aid, "req-m")
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=make_llm_stub([]),
        ):
            out = orch.handle_confirmation("sM", True)
        # Legacy raw-result reply (no resume attempt, no crash).
        assert "Executed" in out or "Result" in out, out
        assert env.sf.calls == 1

    def test_missing_session_timeline_empty_and_reissue_refused(self, env):
        s = env.store
        assert s.get_session_timeline("no-such-session") == []
        with pytest.raises(ValueError, match="unknown action_id"):
            s.request_action_reissue("no-such-action", "req-x")

    def test_duplicate_reissue_and_ceiling_still_enforced(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s = env.store
        aid = _crash_parked_action(env, "s1")
        first = s.request_action_reissue(aid, "req-1")
        s.complete_pending_confirmation("s1")  # resolve the parked reissue
        again = s.request_action_reissue(aid, "req-1")
        assert first == again, "same request id → same new action"
        # Ceiling (MAX_REISSUES_PER_ACTION from v0.18) unchanged.
        from jarvis.memory.session_store import MAX_REISSUES_PER_ACTION

        for i in range(MAX_REISSUES_PER_ACTION - 1):
            s.request_action_reissue(aid, f"req-more-{i}")
            s.complete_pending_confirmation("s1")
        with pytest.raises(ValueError, match="reissue limit reached"):
            s.request_action_reissue(aid, "req-over")


def orch_handle(env, sid, confirmed):
    return env.orch.handle_confirmation(sid, confirmed)


# ── Track C: session timeline ────────────────────────────────────────────────


class TestSessionTimeline:
    def test_event_ordering_and_kinds(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s = env.store
        aid = _crash_parked_action(env, "s1")
        s.request_action_reissue(aid, "req-t")
        tl = s.get_session_timeline("s1")
        kinds = [e["kind"] for e in tl]
        assert "message" in kinds
        assert "confirmation_parked" in kinds
        assert "action_state" in kinds
        assert "reissue" in kinds
        ts = [e["ts"] for e in tl]
        assert ts == sorted(ts), "chronological order"

    def test_causal_relationships(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "x()"}'),
            "done",
        ])
        s = env.store
        aid = _crash_parked_action(env, "s1")
        new_id = s.request_action_reissue(aid, "req-c")
        tl = s.get_session_timeline("s1")
        actions = {e["action_id"]: e for e in tl if e["kind"] == "action_state"}
        reissues = [e for e in tl if e["kind"] == "reissue"]
        assert actions[aid]["state"] == ACTION_STATE_UNKNOWN
        reissue = next(e for e in reissues if e["request_id"] == "req-c")
        assert reissue["original_action_id"] == aid
        assert reissue["new_action_id"] == new_id
        # The reissued action exists as PENDING (the UNKNOWN original is
        # never mutated, the new identity is present and claimable).
        assert actions[new_id]["state"] == ACTION_STATE_PENDING
        # Causal ordering by timestamp: the original action row exists
        # before/at the reissue event; the new PENDING row shares the
        # reissue's timestamp (same transaction) and is not older.
        assert actions[aid]["ts"] <= reissue["ts"]
        assert actions[new_id]["ts"] == reissue["ts"]

    def test_bounded_result_size(self, env):
        s = env.store
        for i in range(30):
            s.save_message("sB", {"role": "user", "content": f"m{i}"})
        tl = s.get_session_timeline("sB", limit=10)
        assert len(tl) <= 10

    def test_session_isolation(self, env):
        s = env.store
        s.save_message("sA", {"role": "user", "content": "a"})
        s.save_message("sZ", {"role": "user", "content": "z"})
        tl_a = s.get_session_timeline("sA")
        assert all("sZ" not in json.dumps(e) for e in tl_a)
        assert len(tl_a) == 1
        assert len(s.get_session_timeline("sZ")) == 1

    def test_no_sensitive_leakage(self, env):
        _park_plan_turn(env, "s1", [
            ("execute_python_code", '{"code": "TOP_SECRET_PAYLOAD"}'),
            "done",
        ])
        s = env.store
        aid = _crash_parked_action(env, "s1")
        s.finish_action_execution(aid, ACTION_STATE_FAILED, "SECRET_RESULT_BODY")
        s.acquire_session_lease("s1", "host:123:runtime:abcd1234")
        tl = s.get_session_timeline("s1")
        blob = json.dumps(tl)
        assert "TOP_SECRET_PAYLOAD" not in blob, "tool_args must never leak"
        assert "SECRET_RESULT_BODY" not in blob, "result bodies must never leak"
        assert "Run the side effect" not in blob, "message content must never leak"
        assert "host:123" not in blob, "raw owner tokens must never leak"
        assert any(e.get("owner") == "runtime:abcd1234" for e in tl), (
            "redacted owner IS present"
        )

    def test_unknown_session_empty(self, env):
        assert env.store.get_session_timeline("ghost") == []


def kinds_index(tl, kind):
    return next(i for i, e in enumerate(tl) if e["kind"] == kind)


# ── Track D: retention ───────────────────────────────────────────────────────


class TestRetention:
    def _backdate(self, s, action_id, days=60):
        old = (
            datetime.now(tz=timezone.utc) - timedelta(days=days)
        ).isoformat()
        with s._lock:
            s._conn.execute(
                "UPDATE action_executions SET created_at=? WHERE action_id=?",
                (old, action_id),
            )
            s._conn.commit()

    def _parked(self, s, sid, call):
        cid = s.save_pending_confirmation(sid, "calculator", "{}", call, "SAFE")
        aid = s.get_action_execution_by_confirmation(cid).action_id
        s.complete_pending_confirmation(sid)
        return aid

    def test_old_terminal_removed_recent_kept(self, env):
        s = env.store
        old = self._parked(s, "s1", "c1")
        s.finish_action_execution(old, ACTION_STATE_SUCCEEDED, "r")
        self._backdate(s, old)
        recent = self._parked(s, "s2", "c2")
        s.finish_action_execution(recent, ACTION_STATE_SUCCEEDED, "r")
        report = s.cleanup_operational_records(terminal_actions_days=30)
        assert report["actions_removed"] == 1
        assert s.get_action_execution(old) is None
        assert s.get_action_execution(recent) is not None

    def test_unknown_pending_running_protected(self, env):
        s = env.store
        unk = self._parked(s, "s1", "c1")
        s.claim_action_execution(unk, "h:1:t:o")
        s.mark_action_unknown(unk, "crash")
        self._backdate(s, unk)
        cid = s.save_pending_confirmation("s2", "calculator", "{}", "c2", "SAFE")
        pending = s.get_action_execution_by_confirmation(cid).action_id
        self._backdate(s, pending)
        run = self._parked(s, "s3", "c3")
        s.claim_action_execution(run, "h:1:t:o")
        self._backdate(s, run)
        report = s.cleanup_operational_records(terminal_actions_days=30)
        assert report["actions_removed"] == 0
        for aid in (unk, pending, run):
            assert s.get_action_execution(aid) is not None

    def test_failed_with_active_confirmation_protected(self, env):
        s = env.store
        cid = s.save_pending_confirmation("s1", "calculator", "{}", "c1", "SAFE")
        aid = s.get_action_execution_by_confirmation(cid).action_id
        s.claim_action_execution(aid, "h:1:t:o")
        s.finish_action_execution(aid, ACTION_STATE_FAILED, "boom")
        self._backdate(s, aid)
        report = s.cleanup_operational_records(terminal_actions_days=30)
        assert report["actions_removed"] == 0
        assert s.get_action_execution(aid) is not None

    def test_failed_with_resolved_confirmation_removable(self, env):
        s = env.store
        aid = self._parked(s, "s1", "c1")
        s.claim_action_execution(aid, "h:1:t:o")
        s.finish_action_execution(aid, ACTION_STATE_FAILED, "boom")
        self._backdate(s, aid)
        report = s.cleanup_operational_records(terminal_actions_days=30)
        assert report["actions_removed"] == 1
        assert s.get_action_execution(aid) is None

    def test_reissue_referenced_rows_kept(self, env):
        s = env.store
        aid = self._parked(s, "s1", "c1")
        s.claim_action_execution(aid, "h:1:t:o")
        s.mark_action_unknown(aid, "crash")
        new_id = s.request_action_reissue(aid, "req-k")
        s.complete_pending_confirmation("s1")
        s.finish_action_execution(new_id, ACTION_STATE_SUCCEEDED, "r")
        self._backdate(s, new_id, days=400)  # far past any retention
        report = s.cleanup_operational_records(
            terminal_actions_days=30, reissues_days=90
        )
        # Both ends survive (chain integrity), and the audit row survives.
        assert s.get_action_execution(new_id) is not None
        assert s.get_action_execution(aid) is not None
        assert len(s.get_reissue_chain(aid)) == 1

    def test_dry_run_mutates_nothing(self, env):
        s = env.store
        aid = self._parked(s, "s1", "c1")
        s.finish_action_execution(aid, ACTION_STATE_SUCCEEDED, "r")
        self._backdate(s, aid)
        s.acquire_session_lease("sD", "h:1:t:o", ttl_seconds=-1)
        before = s._conn.execute(
            "SELECT COUNT(*) FROM action_executions"
        ).fetchone()[0]
        report = s.cleanup_operational_records(
            terminal_actions_days=30, dry_run=True
        )
        assert report["actions_removed"] == 1
        assert report["dry_run"] == 1
        assert s._conn.execute(
            "SELECT COUNT(*) FROM action_executions"
        ).fetchone()[0] == before
        assert s.get_action_execution(aid) is not None
        assert s.get_session_lease("sD") is not None
        # Real run then removes exactly the candidates dry-run reported.
        report2 = s.cleanup_operational_records(terminal_actions_days=30)
        assert report2["actions_removed"] == 1
        assert s.get_action_execution(aid) is None

    def test_stale_leases_purged(self, env):
        s = env.store
        # Sessions must exist: leases of dead sessions are also purgeable,
        # and we want expiry-age to be the discriminator under test.
        with s._lock:
            for sid in ("sOld", "sFresh"):
                s._conn.execute(
                    "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                    (sid, datetime.now(tz=timezone.utc).isoformat()),
                )
            s._conn.commit()
        s.acquire_session_lease("sOld", "h:1:t:o", ttl_seconds=-1)
        with s._lock:
            s._conn.execute(
                "UPDATE session_leases SET expires_at=? WHERE session_id='sOld'",
                ((datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat(),),
            )
            s._conn.commit()
        s.acquire_session_lease("sFresh", "h:1:t:o")
        report = s.cleanup_operational_records(leases_days=30)
        assert report["stale_leases_removed"] == 1
        assert s.get_session_lease("sOld") is None
        assert s.get_session_lease("sFresh") is not None

    def test_audit_integrity_after_cleanup(self, env):
        s = env.store
        aid = self._parked(s, "s1", "c1")
        s.claim_action_execution(aid, "h:1:t:o")
        s.mark_action_unknown(aid, "crash")
        new_id = s.request_action_reissue(aid, "req-a")
        s.complete_pending_confirmation("s1")
        s.finish_action_execution(new_id, ACTION_STATE_SUCCEEDED, "r")
        s.cleanup_operational_records(terminal_actions_days=30, reissues_days=90)
        chain = s.get_reissue_chain(aid)
        assert len(chain) == 1
        assert chain[0]["new_action_id"] == new_id
        assert s.get_reissue_origin(new_id) == aid
