"""
tests/test_operational.py
────────────────────────
v0.13 operational readiness: WAL persistence, deep health probes, error
sanitization, and the maintenance doctor command.
"""

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jarvis.api.app import app, set_runtime
from jarvis.api.health import check_ollama, deep_health
from jarvis.memory.session_store import SessionStore


@pytest.fixture()
def client():
    """TestClient over a locally-built runtime (same pattern as test_api.py)."""
    with patch("jarvis.runtime.get_vector_store"):
        from jarvis.runtime import build_runtime

        rt = build_runtime()
    set_runtime(rt)
    with TestClient(app) as c:
        yield c
    set_runtime(None)
    rt.close()


# ── SQLite WAL mode ───────────────────────────────────────────────────────────


class TestWalMode:
    def test_file_db_runs_wal(self, tmp_path):
        db_file = tmp_path / "wal.db"
        with patch("jarvis.config.settings.db_path", str(db_file)):
            store = SessionStore()
        try:
            mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
            sync = store._conn.execute("PRAGMA synchronous").fetchone()[0]
            assert sync == 1  # NORMAL
        finally:
            store.close()

    def test_memory_db_uses_default_mode(self):
        store = SessionStore()
        try:
            mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "memory"
        finally:
            store.close()

    def test_wal_db_survives_reopen(self, tmp_path):
        """A real file DB must persist across connection cycles."""
        db_file = tmp_path / "persist.db"
        with patch("jarvis.config.settings.db_path", str(db_file)):
            store = SessionStore()
            sid = store.create_session()
            store.save_message(sid, {"role": "user", "content": "durable?"})
            store.close()

            store2 = SessionStore()
        try:
            history = store2.load_history(sid, limit=10)
            assert history and history[0]["content"] == "durable?"
        finally:
            store2.close()


# ── Deep health ───────────────────────────────────────────────────────────────


class TestDeepHealth:
    def test_healthy_store_reports_ok(self):
        store = SessionStore()
        try:
            class Holder:
                pass

            rt = Holder()
            rt.store = store
            report = deep_health(rt)
            assert report["status"] == "ok"
            assert report["db"] == {"read": True, "write": True}
            # Probe rows are cleaned up — no residue.
            n = store._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id LIKE 'healthprobe-%'"
            ).fetchone()[0]
            assert n == 0
        finally:
            store.close()

    def test_broken_store_degrades(self):
        store = SessionStore()
        # Simulate a corrupted/locked DB surface.
        store._conn.close()

        class Holder:
            pass

        rt = Holder()
        rt.store = store
        report = deep_health(rt)
        assert report["status"] == "degraded"
        assert report["db"] == {"read": False, "write": False}
        # deep_health's best-effort sweep already hit the closed connection;
        # the store is a lost cause here, so just drop it.
        del store

    def test_deep_health_503_via_api(self):
        store = SessionStore()
        store.close = lambda: None  # keep the connection usable through the test

        class FakeRuntime:
            def __init__(self, st):
                self.store = st

            def describe(self):
                from jarvis.config import settings

                return {
                    "version": "test",
                    "model": settings.ollama_model,
                    "tools": [],
                    "disabled_tools": ["execute_python_code", "computer_control"],
                }

        try:
            set_runtime(FakeRuntime(store))
            with TestClient(app, raise_server_exceptions=False) as c:
                # Healthy store → 200
                assert c.get("/health").status_code == 200
                # Broken store → 503 with structured detail
                store._conn.close()
                resp = c.get("/health")
                assert resp.status_code == 503
                assert resp.json()["detail"]["status"] == "degraded"
        finally:
            set_runtime(None)
            try:
                store._conn.close()
            except sqlite3.ProgrammingError:
                pass


# ── Error sanitization ────────────────────────────────────────────────────────


class TestErrorSanitization:
    def test_chat_503_hides_exception_details(self, client):
        """Internal paths/hostnames from exceptions must never reach clients."""
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=RuntimeError("connect to C:\\secret\\path on internal.corp:11434 failed"),
        ):
            resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 503
        body = resp.text
        assert "secret" not in body and "internal.corp" not in body
        assert "X-Request-ID" in body  # client is pointed at the correlation id

    def test_chat_response_echoes_request_id(self, client):
        from unittest.mock import MagicMock

        class M:
            role = "assistant"
            content = "ok"
            tool_calls = None

        class C:
            message = M()

        class R:
            choices = [C()]

        with patch("jarvis.core.orchestrator.chat_completion", return_value=R()):
            resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"]
        assert body["request_id"] == resp.headers["x-request-id"]


# ── Maintenance doctor ────────────────────────────────────────────────────────


class TestDoctor:
    def test_doctor_reports_and_exit_codes(self, capsys):
        from jarvis import maintenance

        with patch("jarvis.maintenance._open_store") as p:
            store = SessionStore()
            store.close = lambda: None  # type: ignore[method-assign]
            p.return_value = store
            with (
                patch("jarvis.api.health.check_ollama", return_value=True),
                # v0.18 expanded doctor also probes the configured model;
                # mock it so this test stays offline (conftest runs with a
                # fake model name that no real Ollama instance would have).
                patch("jarvis.maintenance._check_ollama_model", return_value=True),
            ):
                code = maintenance.main(["doctor"])
        out = capsys.readouterr().out
        assert code == 0
        assert "[OK]" in out and "database" in out and "all checks passed" in out
        store.close()

    def test_doctor_fails_when_ollama_down(self, capsys):
        from jarvis import maintenance

        with patch("jarvis.maintenance._open_store") as p:
            store = SessionStore()
            store.close = lambda: None  # type: ignore[method-assign]
            p.return_value = store
            with patch("jarvis.api.health.check_ollama", return_value=False):
                code = maintenance.main(["doctor"])
        out = capsys.readouterr().out
        assert code == 1
        assert "[FAIL] ollama" in out
        store.close()

    def test_doctor_flags_enabled_but_unavailable_sandbox(self, capsys):
        from jarvis import maintenance

        with patch("jarvis.maintenance._open_store") as p:
            store = SessionStore()
            store.close = lambda: None  # type: ignore[method-assign]
            p.return_value = store
            with (
                patch("jarvis.api.health.check_ollama", return_value=True),
                patch(
                    "jarvis.config.settings.ENABLE_CODE_EXECUTION", True
                ),
                patch("jarvis.core.sandbox.DockerCodeSandbox.is_available", return_value=False),
            ):
                code = maintenance.main(["doctor"])
        out = capsys.readouterr().out
        assert code == 1
        assert "enabled but unavailable" in out
        store.close()

    def test_check_ollama_url_shapes(self):
        with patch("urllib.request.urlopen") as u:
            assert check_ollama("http://localhost:11434/") is True
            called = u.call_args[0][0]
            assert called == "http://localhost:11434/api/tags"  # no double slash
        with patch("urllib.request.urlopen", side_effect=OSError("nope")):
            assert check_ollama("http://localhost:11434") is False
