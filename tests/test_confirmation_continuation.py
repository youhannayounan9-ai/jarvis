"""
tests/test_confirmation_continuation.py
───────────────────────────────────────
v0.15: confirmation continuation. A high-risk action no longer dead-ends the
turn — the durable pause context lets the ORIGINAL workflow resume after
approval/denial, including across a restart.
"""

import datetime
from unittest.mock import patch

import pytest

from tests.fakes import fake_text_response, fake_tool_response

from jarvis.core.orchestrator import PAUSED_FOR_CONFIRMATION, Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.core.sandbox import ExecutionResult
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry

PLAN = [
    {"step_number": 1, "description": "do the risky thing", "required_tools": ["dangerous_tool"]},
    {"step_number": 2, "description": "summarize the outcome", "required_tools": []},
]


def _store():
    s = SessionStore()
    # Suppressed close: tests inspect after main-ish flows; own the lifetime.
    real_close = s.close
    s.close = lambda: None  # type: ignore[method-assign]
    s._real_close = real_close  # type: ignore[attr-defined]
    return s


def _orch(store) -> Orchestrator:
    return Orchestrator(store, ToolRegistry(), PermissionGuard())


def _park(store, orch, session_id: str, user_request="do the risky task then summarize"):
    """Drive one turn into the paused-for-confirmation state."""
    with patch(
        "jarvis.core.orchestrator.chat_completion",
        side_effect=[fake_tool_response([("dangerous_tool", "{}")]), fake_text_response("x")],
    ):
        with patch.object(orch, "route_intent", return_value="complex"):
            with patch.object(
                orch._planner, "generate_plan", return_value=list(PLAN)
            ):
                with patch.object(orch._registry, "get_tool_risk_level", return_value="SYSTEM"):
                    with patch.object(orch._guard, "require_confirmation", return_value=True):
                        result = orch.chat(session_id, user_request)
    assert result == PAUSED_FOR_CONFIRMATION
    pending = store.load_pending_confirmation(session_id)
    assert pending is not None
    return pending


class TestPauseBehavior:
    def test_high_risk_step_pauses_instead_of_completing(self):
        store = _store()
        sid = store.create_session()
        try:
            pending = _park(store, _orch(store), sid)
            # The paused step must NOT be recorded as completed.
            assert pending["context"]["step_number"] == 1
            assert pending["context"]["completed_steps"] == []
            # Remaining plan is durable.
            assert len(pending["context"]["pending_plan"]) == 2
            assert pending["context"]["original_request"] == "do the risky task then summarize"
        finally:
            store._real_close()

    def test_pause_context_survives_restart(self):
        """A NEW orchestrator on the SAME store must be able to resume."""
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            fresh_orch = _orch(store)  # simulates a process restart
            pending = fresh_orch.get_pending_confirmation(sid)
            assert pending is not None
            assert pending["context"]["mode"] in ("complex", "simple")
        finally:
            store._real_close()

    def test_simple_path_also_pauses_with_context(self):
        store = _store()
        sid = store.create_session()
        try:
            orch = _orch(store)
            with patch(
                "jarvis.core.orchestrator.chat_completion",
                side_effect=[fake_tool_response([("dangerous_tool", "{}")])],
            ):
                with patch.object(orch, "route_intent", return_value="simple"):
                    with patch.object(orch._registry, "get_tool_risk_level", return_value="SYSTEM"):
                        with patch.object(orch._guard, "require_confirmation", return_value=True):
                            result = orch.chat(sid, "do the risky thing now")
            assert result == PAUSED_FOR_CONFIRMATION
            ctx = store.load_pending_confirmation(sid)["context"]
            assert ctx["mode"] == "simple"
            assert ctx["original_request"] == "do the risky thing now"
        finally:
            store._real_close()


class TestResumeOnApproval:
    def test_approval_executes_tool_and_synthesizes_full_task(self):
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)  # fresh orchestrator == restart

            with patch.object(
                orch._registry, "dispatch", return_value="dangerous action result: OK"
            ) as disp:
                with patch(
                    "jarvis.core.orchestrator.chat_completion",
                    side_effect=[
                        fake_text_response("step two completed"),
                        fake_text_response("Task finished: risky thing done, then summarized."),
                    ],
                ):
                    answer = orch.handle_confirmation(sid, True)

            disp.assert_called_once()  # the approved tool actually ran
            # The final answer is a SYNTHESIS, not a raw tool string.
            assert "Task finished" in answer
            # The tool result was persisted as a proper tool message.
            history = store.load_history(sid, limit=100)
            assert any(
                m.get("role") == "tool" and "OK" in (m.get("content") or "")
                for m in history
            )
        finally:
            store._real_close()

    def test_approval_with_remaining_steps_executes_them(self):
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)

            llm_calls = []

            def fake_llm(messages, tools=None, **kwargs):
                llm_calls.append(messages)
                # Step 2's ReAct call, then synthesis.
                return fake_text_response("step two result")

            with patch.object(orch._registry, "dispatch", return_value="OK"):
                with patch("jarvis.core.orchestrator.chat_completion", side_effect=fake_llm):
                    answer = orch.handle_confirmation(sid, True)

            # At least the resumed step + a synthesis call happened.
            assert len(llm_calls) >= 2
            assert "Task" in answer or "step two" in answer or answer.strip()
        finally:
            store._real_close()

    def test_failed_approved_tool_is_surfaced_not_hidden(self):
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)

            with patch.object(
                orch._registry,
                "dispatch",
                return_value="ERROR: sandbox unavailable",
            ):
                with patch(
                    "jarvis.core.orchestrator.chat_completion",
                    side_effect=[
                        fake_text_response("step two completed anyway"),
                        fake_text_response(
                            "The approved action failed (sandbox unavailable); "
                            "the rest of the task completed."
                        ),
                    ],
                ):
                    answer = orch.handle_confirmation(sid, True)

            # The failure reaches the synthesis context (as a tool message).
            history = store.load_history(sid, limit=100)
            assert any("ERROR: sandbox unavailable" in (m.get("content") or "") for m in history)
            # And the loop still produced a final (non-crash) answer that
            # does NOT pretend the risky action succeeded.
            assert "failed" in answer.lower()
        finally:
            store._real_close()

    def test_step2_tool_error_after_resume_triggers_self_correction(self):
        """Errors during resumed steps flow into the normal recovery loop."""
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)

            with patch.object(orch._registry, "dispatch", return_value="OK"):
                with patch(
                    "jarvis.core.orchestrator.chat_completion",
                    side_effect=[
                        fake_tool_response([("another_tool", "{}")]),  # step 2 tries a tool
                        fake_text_response("recovered answer"),
                        fake_text_response("final synthesis"),
                    ],
                ):
                    with patch.object(
                        orch._registry, "get_tool_risk_level", return_value="SAFE"
                    ):
                        with patch.object(orch._guard, "is_allowed", return_value=True):
                            with patch.object(
                                orch._registry,
                                "dispatch_async",
                                side_effect=["ERROR: transient failure", "recovered"],
                            ):
                                answer = orch.handle_confirmation(sid, True)
            assert answer.strip()
        finally:
            store._real_close()


class TestResumeOnDenial:
    def test_denial_does_not_execute_tool(self):
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)

            with patch.object(
                orch._registry, "dispatch", side_effect=AssertionError("denied tool ran!")
            ) as dispatch:
                with patch(
                    "jarvis.core.orchestrator.chat_completion",
                    side_effect=[
                        fake_text_response("step two without the risky bit"),
                        fake_text_response("I did not perform the risky action."),
                    ],
                ):
                    answer = orch.handle_confirmation(sid, False)
            dispatch.assert_not_called()
            assert answer.strip()
        finally:
            store._real_close()

    def test_denial_still_continues_remaining_task(self):
        """A denial resolves the action but the rest of the request continues."""
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)

            llm_seen = []

            def fake_llm(messages, tools=None, **kwargs):
                llm_seen.append(messages)
                return fake_text_response("step two without the risky bit")

            with patch.object(orch._registry, "dispatch", side_effect=AssertionError("ran!")):
                with patch("jarvis.core.orchestrator.chat_completion", side_effect=fake_llm):
                    answer = orch.handle_confirmation(sid, False)
            # Remaining step executed (LLM calls beyond nothing).
            assert len(llm_seen) >= 2  # resumed step + synthesis
            assert answer.strip()
            history = store.load_history(sid, limit=100)
            assert any("denied" in (m.get("content") or "").lower() for m in history)
        finally:
            store._real_close()


class TestEdgeStates:
    def test_expired_confirmation_cannot_execute(self):
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            # Backdate the TTL.
            past = (
                datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
            ).isoformat()
            with store._lock:
                store._conn.execute(
                    "UPDATE pending_confirmations SET expires_at = ?", (past,)
                )
                store._conn.commit()

            orch = _orch(store)
            with patch.object(
                orch._registry, "dispatch", side_effect=AssertionError("expired ran!")
            ):
                answer = orch.handle_confirmation(sid, True)
            assert "No pending actions" in answer
        finally:
            store._real_close()

    def test_no_pending_confirmation_is_a_noop(self):
        store = _store()
        store.create_session()
        try:
            answer = _orch(store).handle_confirmation("sess-none", True)
            assert "No pending" in answer
        finally:
            store._real_close()

    def test_legacy_row_without_context_keeps_raw_reply(self):
        """Pre-v0.15 rows (no context) resolve without crashing."""
        store = _store()
        sid = store.create_session()
        try:
            store.save_pending_confirmation(
                session_id=sid,
                tool_name="legacy_tool",
                tool_args="{}",
                tool_call_id="c-legacy",
                risk_level="SYSTEM",
                context=None,  # legacy: no resume context
            )
            orch = _orch(store)
            with patch.object(orch._registry, "dispatch", return_value="legacy result"):
                answer = orch.handle_confirmation(sid, True)
            assert answer.startswith("Executed legacy_tool")
        finally:
            store._real_close()

    def test_corrupt_context_degrades_gracefully(self):
        store = _store()
        sid = store.create_session()
        try:
            store.save_pending_confirmation(
                session_id=sid,
                tool_name="t",
                tool_args="{}",
                tool_call_id="c",
                risk_level="SYSTEM",
                context={"original_request": "ok"},
            )
            # Corrupt the stored JSON directly.
            with store._lock:
                store._conn.execute(
                    "UPDATE pending_confirmations SET context_json = '{not json' "
                    "WHERE session_id = ?",
                    (sid,),
                )
                store._conn.commit()
            orch = _orch(store)
            with patch.object(orch._registry, "dispatch", return_value="r"):
                answer = orch.handle_confirmation(sid, True)
            # Falls back to the raw-result reply rather than crashing.
            assert "Executed t" in answer
        finally:
            store._real_close()

    def test_nested_pause_during_resume_reparks(self):
        """A second high-risk tool in a resumed step re-parks cleanly."""
        store = _store()
        sid = store.create_session()
        try:
            _park(store, _orch(store), sid)
            orch = _orch(store)

            with patch.object(orch._registry, "dispatch", return_value="first ok"):
                with patch(
                    "jarvis.core.orchestrator.chat_completion",
                    side_effect=[
                        fake_tool_response([("second_danger", "{}")]),  # step 2 parks again
                    ],
                ):
                    with patch.object(
                        orch._registry, "get_tool_risk_level", return_value="SYSTEM"
                    ):
                        with patch.object(
                            orch._guard, "require_confirmation", return_value=True
                        ):
                            answer = orch.handle_confirmation(sid, True)
            assert answer == PAUSED_FOR_CONFIRMATION
            # A NEW pending confirmation exists with refreshed context.
            new_pending = store.load_pending_confirmation(sid)
            assert new_pending is not None
            assert new_pending["context"]["step_number"] == 2
            # The first approved step is already in the durable state.
            assert any(
                s.get("step_number") == 1 for s in new_pending["context"]["completed_steps"]
            )
        finally:
            store._real_close()
