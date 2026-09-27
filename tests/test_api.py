"""
tests/test_api.py
─────────────────
Tests for the FastAPI service layer. The runtime dependency is overridden
with a locally-built runtime so no global state leaks between tests.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from jarvis.api.app import app, get_runtime, set_runtime
from jarvis.runtime import JarvisRuntime, build_runtime


def _mock_llm_response(text: str = "mocked answer"):
    """Shared fake from tests.fakes, re-exported under the historic name."""
    from tests.fakes import fake_text_response

    return fake_text_response(text)


@pytest.fixture()
def client():
    """TestClient with an isolated runtime (vector store mocked out)."""
    with patch("jarvis.runtime.get_vector_store"):
        rt = build_runtime()
    set_runtime(rt)
    with TestClient(app) as c:
        yield c
    set_runtime(None)
    rt.close()


class TestHealth:
    def test_health_reports_surface(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "web_search" in body["tools"]
        assert "execute_python_code" in body["disabled_tools"]
        assert "computer_control" in body["disabled_tools"]


class TestSessions:
    def test_create_session(self, client):
        resp = client.post("/sessions")
        assert resp.status_code == 201
        sid = resp.json()["session_id"]
        assert sid

    def test_history_of_new_session_is_empty(self, client):
        sid = client.post("/sessions").json()["session_id"]
        resp = client.get(f"/sessions/{sid}/history")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    def test_history_unknown_session_404(self, client):
        assert client.get("/sessions/does-not-exist/history").status_code == 404


class TestChat:
    def test_chat_auto_creates_session(self, client):
        with patch("jarvis.core.orchestrator.chat_completion", return_value=_mock_llm_response()):
            resp = client.post("/chat", json={"message": "hello there"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"]
        assert body["response"] == "mocked answer"

    def test_chat_roundtrip_same_session(self, client):
        sid = client.post("/sessions").json()["session_id"]
        with patch("jarvis.core.orchestrator.chat_completion", return_value=_mock_llm_response("hi back")):
            r1 = client.post("/chat", json={"message": "hi", "session_id": sid})
        assert r1.status_code == 200
        assert r1.json()["session_id"] == sid
        hist = client.get(f"/sessions/{sid}/history").json()["messages"]
        assert any(m["role"] == "user" and m["content"] == "hi" for m in hist)

    def test_chat_unknown_session_404(self, client):
        resp = client.post("/chat", json={"message": "hi", "session_id": "nope"})
        assert resp.status_code == 404

    def test_chat_rejects_empty_message(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_chat_runtime_failure_maps_to_503(self, client):
        """If the LLM is unreachable the API surfaces 503, not a 500 crash."""
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=RuntimeError("ollama down"),
        ):
            resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 503


class TestConfirmations:
    def test_confirmation_flow_end_to_end(self, client):
        """
        v0.15: a high-risk tool call PAUSES the turn (durable confirmation +
        resume context); the API denial endpoint continues the task instead
        of dead-ending with a raw string.
        """
        sid = client.post("/sessions").json()["session_id"]

        # Drive the orchestrator directly into a pending-confirmation state
        class FakeFn:
            name = "write_file"
            arguments = '{"file_path": "x.txt", "content": "hi"}'

        class FakeTC:
            id = "call_api_1"
            function = FakeFn()

        class FakeMsg:
            role = "assistant"
            content = None
            tool_calls = [FakeTC()]

        class FakeChoice:
            message = FakeMsg()

        class FakeResp:
            choices = [FakeChoice()]

        rt = get_runtime()
        with patch("jarvis.core.orchestrator.chat_completion", return_value=FakeResp()):
            with patch.object(rt.orchestrator, "route_intent", return_value="simple"):
                with patch.object(rt.registry, "get_tool_risk_level", return_value="SYSTEM"):
                    with patch.object(rt.guard, "require_confirmation", return_value=True):
                        chat_resp = client.post("/chat", json={"message": "write x.txt", "session_id": sid})
        assert chat_resp.status_code == 200
        body = chat_resp.json()
        assert body["pending_confirmation"] is not None
        # The turn is PAUSED (marker surfaced), not silently finished.
        assert body["response"].startswith("ACTION_REQUIRES_CONFIRMATION")

        # The pending action must be visible via the API
        pending = client.get(f"/sessions/{sid}/confirmation")
        assert pending.status_code == 200
        assert pending.json()["tool_name"] == "write_file"

        # Deny it — never executed, and the task CONTINUES to a final answer
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=_mock_llm_response("I did not write the file as you denied it."),
        ):
            denied = client.post(f"/sessions/{sid}/confirm", json={"confirmed": False})
        assert denied.status_code == 200
        assert "denied" in denied.json()["response"].lower()

        # Confirmation is gone now
        assert client.get(f"/sessions/{sid}/confirmation").status_code == 404


class TestTools:
    def test_tools_lists_active_surface_with_risk(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        tools = {t["name"]: t["risk_level"] for t in resp.json()}
        assert "web_search" in tools
        assert "execute_python_code" not in tools
        assert "computer_control" not in tools
        assert tools["write_file"] == "FILE_WRITE"


class TestRuntimeOverride:
    def test_set_runtime_swaps_dependency(self, client):
        probe = client.get("/health").json()
        assert probe["status"] == "ok"
