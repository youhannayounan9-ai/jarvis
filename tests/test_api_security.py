"""
tests/test_api_security.py
──────────────────────────
Tests for API authentication (opt-in API key) and the SSE streaming endpoint.
All offline: the LLM is mocked, the runtime is isolated per test.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jarvis.api.app import app, set_runtime
from jarvis.runtime import build_runtime


def _mock_llm_response(text: str = "mocked answer"):
    """Shared fake from tests.fakes, re-exported under the historic name."""
    from tests.fakes import fake_text_response

    return fake_text_response(text)


@pytest.fixture()
def client():
    """Isolated runtime, no auth configured (default posture)."""
    with patch("jarvis.runtime.get_vector_store"):
        rt = build_runtime()
    set_runtime(rt)
    with TestClient(app) as c:
        yield c
    set_runtime(None)
    rt.close()


class TestAuthDisabledByDefault:
    """Without JARVIS_API_KEY, everything behaves as before (back-compat)."""

    def test_health_open(self, client):
        assert client.get("/health").status_code == 200

    def test_endpoints_open(self, client):
        assert client.post("/sessions").status_code == 201
        with patch("jarvis.core.orchestrator.chat_completion", return_value=_mock_llm_response()):
            assert client.post("/chat", json={"message": "hi"}).status_code == 200
        assert client.get("/tools").status_code == 200


class TestAuthEnforced:
    """With JARVIS_API_KEY set, protected endpoints require the key."""

    @pytest.fixture()
    def authed_client(self, client):
        with patch("jarvis.api.app.auth_enabled", return_value=True):
            with patch("jarvis.api.auth.settings") as auth_settings:
                # Real string so .strip() and .encode() behave naturally.
                auth_settings.JARVIS_API_KEY = "secret-key-123"
                yield client

    def test_missing_key_rejected(self, authed_client):
        resp = authed_client.post("/sessions")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_wrong_key_rejected(self, authed_client):
        resp = authed_client.post(
            "/sessions", headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401

    def test_bearer_key_accepted(self, authed_client):
        resp = authed_client.post(
            "/sessions", headers={"Authorization": "Bearer secret-key-123"}
        )
        assert resp.status_code == 201

    def test_x_api_key_accepted(self, authed_client):
        resp = authed_client.post("/sessions", headers={"X-API-Key": "secret-key-123"})
        assert resp.status_code == 201

    def test_health_stays_open_for_probes(self, authed_client):
        assert authed_client.get("/health").status_code == 200

    def test_all_protected_endpoints_require_key(self, authed_client):
        sid = "00000000-0000-0000-0000-000000000000"
        paths = [
            ("post", "/chat", {"message": "hi"}),
            ("post", "/chat/stream", {"message": "hi"}),
            ("get", f"/sessions/{sid}/history", None),
            ("get", f"/sessions/{sid}/confirmation", None),
            ("post", f"/sessions/{sid}/confirm", {"confirmed": True}),
            ("get", "/tools", None),
        ]
        for method, path, body in paths:
            call = getattr(authed_client, method)
            resp = call(path, json=body) if body is not None else call(path)
            assert resp.status_code in (401, 404), (path, resp.status_code)
            # 404 only allowed for endpoints that validate the session AFTER auth;
            # without a key nothing may get past 401 for an unknown session.
            if sid in path:
                assert resp.status_code == 401, path

    def test_health_reports_auth_posture(self, client):
        with patch("jarvis.api.app.auth_enabled", return_value=True):
            body = client.get("/health").json()
        assert body["auth_enabled"] is True

    def test_verify_key_constant_time_interface(self):
        from jarvis.api.auth import verify_key

        with patch("jarvis.api.auth.settings") as s:
            s.JARVIS_API_KEY = "abc"
            assert verify_key("abc") is True
            assert verify_key("abd") is False
            assert verify_key(None) is False
            s.JARVIS_API_KEY = ""
            assert verify_key(None) is True  # auth disabled


class TestRequestIdsAndHistoryLimit:
    def test_every_response_carries_request_id(self, client):
        resp = client.get("/health")
        assert resp.headers.get("x-request-id")

        sid = client.post("/sessions").json()["session_id"]
        resp2 = client.get(f"/sessions/{sid}/history")
        assert resp2.headers.get("x-request-id")
        # IDs are unique per request.
        assert resp.headers["x-request-id"] != resp2.headers["x-request-id"]

    def test_history_limit_bounds_response(self, client):
        sid = client.post("/sessions").json()["session_id"]
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=_mock_llm_response(),
        ):
            for i in range(6):
                client.post("/chat", json={"message": f"msg {i}", "session_id": sid})

        full = client.get(f"/sessions/{sid}/history").json()["messages"]
        assert len(full) >= 12  # 6 turns x (user+assistant)

        limited = client.get(f"/sessions/{sid}/history?limit=4").json()["messages"]
        assert len(limited) == 4
        # The window keeps the MOST RECENT messages.
        assert limited[-1]["content"] == full[-1]["content"]

    def test_history_limit_validated(self, client):
        sid = client.post("/sessions").json()["session_id"]
        assert client.get(f"/sessions/{sid}/history?limit=0").status_code == 422
        assert client.get(f"/sessions/{sid}/history?limit=9999").status_code == 422


class TestStreaming:
    def test_stream_happy_path_event_sequence(self, client):
        sid = client.post("/sessions").json()["session_id"]
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=_mock_llm_response("streamed answer"),
        ):
            resp = client.post("/chat/stream", json={"message": "hi", "session_id": sid})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        names = [name for name, _ in events]
        assert names[0] == "begin"
        assert names[-1] == "done"
        assert "intent" in names

        done_data = dict(events)["done"]
        assert done_data["session_id"] == sid
        assert done_data["response"] == "streamed answer"

    def test_stream_auto_creates_session(self, client):
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=_mock_llm_response(),
        ):
            resp = client.post("/chat/stream", json={"message": "hi"})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert dict(events)["done"]["session_id"]

    def test_stream_error_event_on_runtime_failure(self, client):
        sid = client.post("/sessions").json()["session_id"]
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=RuntimeError("ollama down"),
        ):
            resp = client.post("/chat/stream", json={"message": "hi", "session_id": sid})
        assert resp.status_code == 200  # SSE carries errors in-band
        events = _parse_sse(resp.text)
        names = [name for name, _ in events]
        assert names[-1] == "error"
        detail = dict(events)["error"]["detail"]
        # Sanitized: no internal exception text leaks to the client.
        assert "ollama down" not in detail
        assert "X-Request-ID" in detail  # operator joins via request id

    def test_stream_unknown_session_404(self, client):
        resp = client.post(
            "/chat/stream", json={"message": "hi", "session_id": "missing-session"}
        )
        assert resp.status_code == 404

    def test_stream_emits_plan_events_for_complex_path(self, client):
        """Complex route must surface intent + plan + step lifecycle events."""
        sid = client.post("/sessions").json()["session_id"]

        plan = [
            {"step_number": 1, "description": "do the thing", "required_tools": []},
            {"step_number": 2, "description": "wrap up", "required_tools": []},
        ]
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=_mock_llm_response("complex answer"),
        ):
            with patch.object(
                type(_get_orchestrator(client)),
                "route_intent",
                return_value="complex",
            ):
                with patch.object(
                    _get_orchestrator(client)._planner,
                    "generate_plan",
                    return_value=plan,
                ):
                    resp = client.post(
                        "/chat/stream", json={"message": "do something complex", "session_id": sid}
                    )

        events = _parse_sse(resp.text)
        names = [name for name, _ in events]
        assert "plan" in names
        assert names.count("step_start") == 2
        assert names.count("step_done") == 2
        assert "synthesis" in names
        assert names[-1] == "done"


def _get_orchestrator(client: TestClient):
    from jarvis.api.app import get_runtime

    return get_runtime().orchestrator


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data-dict) pairs."""
    events: list[tuple[str, dict]] = []
    current_event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: ") and current_event is not None:
            events.append((current_event, json.loads(line[len("data: "):])))
            current_event = None
    return events
