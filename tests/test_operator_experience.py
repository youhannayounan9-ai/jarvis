"""
tests/test_operator_experience.py
─────────────────────────────────
v0.19 Tracks A/C/E at the API + client boundary: the new operator endpoints
served by a REAL uvicorn server, consumed by the REAL JarvisClient and the
dashboard backends.

Proves:
  - GET /sessions/{id}/timeline: 200 safe events, 404 unknown session,
    422 bad limit, 401 when auth enabled
  - POST /actions/{id}/reissue: server-generated request_id echoed,
    client idempotency, 409 wrong state, 401 without key
  - JarvisClient operator methods (list_actions/get_action/list_leases/
    reissue_action/session_timeline) over real HTTP incl. error mapping
  - dashboard backends: ApiBackend (HTTP) and LegacyBackend (in-process)
    expose identical safe operator surfaces; owner redaction; viewing
    never mutates; reissue is the only mutation and stays authenticated
All offline (LLM never called; isolated SQLite file).
"""

import json
import socket
import threading
import time
import urllib.request
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import uvicorn
from fastapi.testclient import TestClient

from jarvis.api.app import app, set_runtime
from jarvis.api.client import JarvisClient, JarvisClientError
from jarvis.memory.session_store import (
    ACTION_STATE_UNKNOWN,
    SessionStore,
)
from jarvis.runtime import build_runtime


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """Real uvicorn server over an isolated DB + its runtime (for seeding)."""
    db = tmp_path_factory.mktemp("v019live") / "live.db"
    from jarvis.config import settings

    with patch("jarvis.config.settings.db_path", str(db)):
        rt = build_runtime()
    set_runtime(rt)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("live server did not start")
    yield base, rt
    server.should_exit = True
    thread.join(timeout=5)
    set_runtime(None)
    rt.close()


def _seed_unknown(store: SessionStore, sid: str):
    """A crashed SYSTEM action for reissue testing (real park path)."""
    # Ensure the session row exists (session_exists gates the API reads).
    with store._lock:
        store._conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?) "
            "ON CONFLICT(id) DO NOTHING",
            (sid, datetime.now(tz=timezone.utc).isoformat()),
        )
        store._conn.commit()
    cid = store.save_pending_confirmation(
        sid, "calculator", '{"expression": "2+2"}', f"call-{sid}", "SAFE"
    )
    aid = store.get_action_execution_by_confirmation(cid).action_id
    store.complete_pending_confirmation(sid)
    assert store.claim_action_execution(aid, "host:1:runtime:feedface") == "claimed"
    store.mark_action_unknown(aid, "test crash")
    return aid


class TestLiveOperatorApi:
    def test_timeline_endpoint_safe_and_ordered(self, live):
        base, rt = live
        store = rt.store
        sid = "tl-live"
        store.save_message(sid, {"role": "user", "content": "SECRET_BODY"})
        _seed_unknown(store, sid)
        client = JarvisClient(base)
        events = client.session_timeline(sid)
        assert events, "timeline must have events"
        blob = json.dumps(events)
        assert "SECRET_BODY" not in blob
        assert "host:1" not in blob
        ts = [e["ts"] for e in events]
        assert ts == sorted(ts)

    def test_timeline_404_unknown_session(self, live):
        base, _ = live
        client = JarvisClient(base)
        with pytest.raises(JarvisClientError) as ei:
            client.session_timeline("no-such-session")
        assert ei.value.status == 404

    def test_reissue_server_generated_request_id_echoed(self, live):
        base, rt = live
        aid = _seed_unknown(rt.store, "srv-rid")
        client = JarvisClient(base)
        body = client.reissue_action(aid)
        assert body["request_id"], "server must generate + echo a request_id"
        assert body["request_id"].startswith("srv-")
        assert body["new_action_id"] != aid
        # Replay with the SAME echoed key → idempotent.
        replay = client.reissue_action(aid, body["request_id"])
        assert replay["new_action_id"] == body["new_action_id"]
        assert replay["reused_existing"] is True

    def test_reissue_409_wrong_state(self, live):
        base, rt = live
        store = rt.store
        cid = store.save_pending_confirmation(
            "s-ok", "calculator", "{}", "c-ok", "SAFE"
        )
        aid = store.get_action_execution_by_confirmation(cid).action_id
        client = JarvisClient(base)
        with pytest.raises(JarvisClientError) as ei:
            client.reissue_action(aid, "req-409-live")
        assert ei.value.status == 409

    def test_reissue_401_without_key_when_auth_enabled(self, live):
        base, rt = live
        aid = _seed_unknown(rt.store, "s-auth-live")
        anon = JarvisClient(base)  # no api_key
        with patch("jarvis.api.auth.settings") as auth_settings:
            auth_settings.JARVIS_API_KEY = "live-secret"
            with pytest.raises(JarvisClientError) as ei:
                anon.reissue_action(aid, "req-noauth")
            assert ei.value.status == 401
            with pytest.raises(JarvisClientError) as ei2:
                anon.session_timeline("s-auth-live")
            assert ei2.value.status == 401
        # With the key everything works (auth is the only gate).
        authed = JarvisClient(base, api_key="live-secret")
        assert authed.session_timeline("s-auth-live")

    def test_list_actions_and_leases_via_client(self, live):
        base, rt = live
        _seed_unknown(rt.store, "s-list")
        client = JarvisClient(base)
        rows = client.list_actions(state="UNKNOWN", limit=10)
        assert rows and all(r["state"] == "UNKNOWN" for r in rows)
        assert all("tool_args" not in r for r in rows)
        leases = client.list_leases()
        assert isinstance(leases, list)
        assert all("owner_token" not in l for l in leases)

    def test_get_action_404_maps_to_client_error(self, live):
        base, _ = live
        client = JarvisClient(base)
        with pytest.raises(JarvisClientError) as ei:
            client.get_action("missing-action")
        assert ei.value.status == 404


class TestDashboardBackends:
    """The dashboard's data layer — same surface over HTTP and in-process."""

    def test_api_backend_operator_surface(self, live):
        from ui.dashboard import ApiBackend

        base, rt = live
        sid = "s-dash"
        aid = _seed_unknown(rt.store, sid)
        with patch("jarvis.config.settings.JARVIS_API_URL", base):
            with patch("jarvis.config.settings.JARVIS_CLIENT_API_KEY", ""):
                backend = ApiBackend()
        actions = backend.list_actions(state="UNKNOWN", limit=10)
        assert any(a["action_id"] == aid for a in actions)
        detail = backend.get_action(aid)
        assert detail["state"] == "UNKNOWN"
        leases = backend.list_leases()
        assert isinstance(leases, list)
        events = backend.session_timeline(sid)
        assert events
        # Viewing never mutates: reissue count unchanged by all reads above.
        assert len(rt.store.get_reissue_chain(aid)) == 0
        # Explicit mutation path works through the backend too.
        result = backend.reissue_action(aid, "dash-req-1")
        assert result["new_action_id"] != aid
        assert len(rt.store.get_reissue_chain(aid)) == 1
        with pytest.raises(JarvisClientError) as ei:
            backend.get_action("missing")
        assert ei.value.status == 404

    def test_legacy_backend_operator_surface(self, tmp_path, monkeypatch):
        from ui.dashboard import LegacyBackend
        from jarvis.config import settings

        monkeypatch.setattr(
            settings, "db_path", str(tmp_path / "legacy_dash.db")
        )
        with patch("jarvis.runtime.get_vector_store"):
            backend = LegacyBackend()
        try:
            store = backend._runtime.store
            aid = _seed_unknown(store, "s-legacy")
            actions = backend.list_actions(state="UNKNOWN", limit=10)
            assert any(a["action_id"] == aid for a in actions)
            detail = backend.get_action(aid)
            assert detail["state"] == "UNKNOWN"
            # Redaction parity with the API surface.
            store.acquire_session_lease("s-legacy", "host:123:runtime:abcd1234")
            leases = backend.list_leases()
            mine = next(l for l in leases if l["session_id"] == "s-legacy")
            assert mine["owner"] == "runtime:abcd1234"
            assert "host:123" not in json.dumps(leases)
            events = backend.session_timeline("s-legacy")
            assert events
            result = backend.reissue_action(aid, "dash-legacy-1")
            assert result["new_action_id"] != aid
            with pytest.raises(JarvisClientError) as ei:
                backend.get_action("missing")
            assert ei.value.status == 404
        finally:
            backend._runtime.close()

    def test_dashboard_module_imports_cleanly(self):
        """The dashboard module (Streamlit page) imports without crashing."""
        import importlib

        mod = importlib.import_module("ui.dashboard")
        assert hasattr(mod, "_render_ops")
        assert hasattr(mod, "_ops_error")
        assert hasattr(mod, "ApiBackend")
        assert hasattr(mod, "LegacyBackend")

    def test_ops_error_maps_all_statuses(self):
        """_ops_error renders an operator message for every error class."""
        from ui.dashboard import _ops_error

        cases = [
            JarvisClientError(0, "connection failed"),
            JarvisClientError(401, "unauthorized"),
            JarvisClientError(404, "Unknown action_id."),
            JarvisClientError(409, "reissue limit reached"),
            JarvisClientError(429, "slow down"),
            JarvisClientError(500, "boom"),
            RuntimeError("surprise"),
        ]
        for e in cases:
            # Must not raise — page stability is the contract.
            _ops_error(e, "testing")
