"""
tests/test_maintenance.py
────────────────────────
Tests for the operational maintenance CLI (offline, in-memory DB).
"""

import datetime
from unittest.mock import patch

from jarvis.maintenance import main


def _seed_session(store, age_days: int | None = None):
    sid = store.create_session()
    store.save_message(sid, {"role": "user", "content": "hello"})
    if age_days is not None:
        old = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=age_days)
        ).isoformat()
        with store._lock:
            store._conn.execute(
                "UPDATE sessions SET created_at = ? WHERE id = ?", (old, sid)
            )
            store._conn.commit()
    return sid


def _pinned_store():
    """
    A SessionStore whose close() is suppressed so tests can inspect the DB
    after main() returns (main closes the store in its finally block).
    The test owns closing.
    """
    from jarvis.memory.session_store import SessionStore

    store = SessionStore()
    store.close = lambda: None  # type: ignore[method-assign]
    return store


def test_stats_reports_counts(capsys):
    with patch("jarvis.maintenance._open_store") as p:
        store = _pinned_store()
        p.return_value = store
        _seed_session(store)
        code = main(["stats"])
    out = capsys.readouterr().out
    assert code == 0
    assert "sessions:" in out and "messages:" in out
    store.close()


def test_cleanup_removes_old_sessions_only(capsys):
    with patch("jarvis.maintenance._open_store") as p:
        store = _pinned_store()
        p.return_value = store
        old_sid = _seed_session(store, age_days=90)
        new_sid = _seed_session(store)
        code = main(["cleanup", "--days", "30"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 session(s)" in out
    old_gone = (
        store._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (old_sid,)
        ).fetchone()[0]
        == 0
    )
    still_there = (
        store._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (new_sid,)
        ).fetchone()[0]
        == 1
    )
    assert old_gone and still_there
    store.close()


def test_expire_confirmations_purges_expired_rows(capsys):
    with patch("jarvis.maintenance._open_store") as p:
        store = _pinned_store()
        p.return_value = store
        sid = store.create_session()
        store.save_pending_confirmation(
            session_id=sid,
            tool_name="write_file",
            tool_args="{}",
            tool_call_id="c1",
            risk_level="SYSTEM",
        )
        # Backdate the expiry so the row is purgeable now.
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)).isoformat()
        with store._lock:
            store._conn.execute(
                "UPDATE pending_confirmations SET expires_at = ?", (past,)
            )
            store._conn.commit()
        code = main(["expire-confirmations"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 expired/completed confirmation row(s)" in out
    assert store.load_pending_confirmation(sid) is None
    store.close()


def test_error_maps_to_exit_code_1(capsys):
    with patch("jarvis.maintenance._open_store") as p:
        p.side_effect = RuntimeError("db locked")
        code = main(["stats"])
    assert code == 1
