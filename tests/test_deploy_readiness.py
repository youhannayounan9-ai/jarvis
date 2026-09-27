"""
tests/test_deploy_readiness.py
──────────────────────────────
Deploy-readiness guarantees: rate limiting, per-session serialization, the
stdlib API client contract, and the sandbox image policy. All offline.
"""

import threading
import time
from unittest.mock import patch

import pytest

from tests.fakes import fake_text_response

from jarvis.api.ratelimit import SlidingWindowRateLimiter, client_key
from jarvis.core.sandbox import DockerCodeSandbox


# ── Rate limiter ──────────────────────────────────────────────────────────────


class TestSlidingWindowLimiter:
    def test_allows_under_limit(self):
        rl = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, retry = rl.check("k1")
            assert allowed

    def test_blocks_over_limit_with_retry_after(self):
        rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("k2")
        allowed, retry = rl.check("k2")
        assert not allowed
        assert retry >= 1

    def test_window_slides(self):
        rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
        t0 = 1000.0
        assert rl.check("k3", now=t0)[0]
        assert rl.check("k3", now=t0 + 1)[0]
        # Third hit inside window blocked...
        assert not rl.check("k3", now=t0 + 2)[0]
        # ...but allowed after the first hit leaves the window.
        allowed, _ = rl.check("k3", now=t0 + 11)
        assert allowed

    def test_keys_are_independent(self):
        rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        assert rl.check("a")[0]
        assert rl.check("b")[0]
        assert not rl.check("a")[0]

    def test_disabled_limiter_allows_everything(self):
        rl = SlidingWindowRateLimiter(max_requests=0, window_seconds=60)
        for _ in range(100):
            assert rl.check("k")[0]

    def test_thread_safety(self):
        rl = SlidingWindowRateLimiter(max_requests=50, window_seconds=60)
        allowed_count = {"n": 0}
        lock = threading.Lock()

        def hammer():
            for _ in range(20):
                ok, _ = rl.check("shared")
                if ok:
                    with lock:
                        allowed_count["n"] += 1

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert allowed_count["n"] == 50  # exactly the window, never over


class TestClientKey:
    def test_prefers_api_key_header(self):
        class R:  # duck-typed request
            headers = {"x-api-key": "secret"}
            client = None

        assert client_key(R()) == "key:secret"

    def test_falls_back_to_bearer(self):
        class R:
            headers = {"authorization": "Bearer tok"}
            client = None

        assert client_key(R()) == "key:tok"

    def test_falls_back_to_ip(self):
        class C:
            host = "10.0.0.9"

        class R:
            headers = {}
            client = C()

        assert client_key(R()) == "ip:10.0.0.9"


class TestApiRateLimitIntegration:
    def test_chat_429_when_limiter_exhausted(self):
        from fastapi.testclient import TestClient

        from jarvis.api.app import app, set_runtime
        from jarvis.api.ratelimit import set_limiter

        rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        set_limiter(rl)
        try:
            with patch("jarvis.runtime.get_vector_store"):
                from jarvis.runtime import build_runtime

                rt = build_runtime()
            set_runtime(rt)
            with TestClient(app) as c:
                first = c.post("/sessions")
                second = c.post("/sessions")
            assert first.status_code == 201
            assert second.status_code == 429
            assert "retry-after" in {k.lower() for k in second.headers}
        finally:
            set_limiter(None)
            set_runtime(None)
            rt.close()

    def test_health_exempt_from_rate_limit(self):
        from fastapi.testclient import TestClient

        from jarvis.api.app import app, set_runtime
        from jarvis.api.ratelimit import set_limiter

        rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        set_limiter(rl)
        try:
            with patch("jarvis.runtime.get_vector_store"):
                from jarvis.runtime import build_runtime

                rt = build_runtime()
            set_runtime(rt)
            with TestClient(app) as c:
                for _ in range(5):
                    assert c.get("/health").status_code == 200
        finally:
            set_limiter(None)
            set_runtime(None)
            rt.close()


# ── Per-session concurrency ──────────────────────────────────────────────────


class TestPerSessionLocking:
    @staticmethod
    def _runtime():
        """A real runtime with a mocked LLM boundary (lock tests only)."""
        from jarvis.runtime import build_runtime

        with patch("jarvis.runtime.get_vector_store"):
            return build_runtime()

    def test_second_concurrent_turn_raises_timeout(self):
        rt = self._runtime()
        gate = threading.Event()
        in_flight = threading.Event()

        def slow_chat(session_id, user_input, **kwargs):
            in_flight.set()
            gate.wait(timeout=5)
            return "done"

        try:
            with patch.object(rt.orchestrator, "chat", side_effect=slow_chat):
                results = {}

                def first_turn():
                    try:
                        results["first"] = rt.chat("s1", "hi")
                    except Exception as e:  # pragma: no cover
                        results["first"] = e

                t = threading.Thread(target=first_turn)
                t.start()
                in_flight.wait(timeout=5)

                with pytest.raises(TimeoutError):
                    rt.chat("s1", "second!")

                gate.set()
                t.join(timeout=5)
                assert results["first"] == "done"
        finally:
            rt.close()

    def test_different_sessions_run_in_parallel(self):
        rt = self._runtime()
        gate = threading.Event()

        def slow_chat(session_id, user_input, **kwargs):
            gate.wait(timeout=5)
            return f"done-{session_id}"

        try:
            with patch.object(rt.orchestrator, "chat", side_effect=slow_chat):
                outs = {}

                def run(sid):
                    outs[sid] = rt.chat(sid, "hi")

                t1 = threading.Thread(target=run, args=("a",))
                t2 = threading.Thread(target=run, args=("b",))
                t1.start()
                t2.start()
                gate.set()
                t1.join(timeout=5)
                t2.join(timeout=5)
            assert outs == {"a": "done-a", "b": "done-b"}
        finally:
            rt.close()

    def test_lock_released_after_exception(self):
        rt = self._runtime()
        try:
            with patch.object(rt.orchestrator, "chat", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError):
                    rt.chat("s-err", "hi")
            # Lock must be free again: no TimeoutError this time.
            with patch.object(rt.orchestrator, "chat", return_value="recovered"):
                assert rt.chat("s-err", "hi") == "recovered"
        finally:
            rt.close()


class TestApiConflictMapping:
    def test_parallel_same_session_chat_returns_409(self):
        from fastapi.testclient import TestClient

        from jarvis.api.app import app, set_runtime

        with patch("jarvis.runtime.get_vector_store"):
            from jarvis.runtime import build_runtime

            rt = build_runtime()
        set_runtime(rt)
        try:
            gate = threading.Event()
            with patch.object(rt.orchestrator, "chat", side_effect=lambda *a, **k: (gate.wait(5), "x")[1]):
                sid = rt.start_session()
                with TestClient(app) as c:
                    t = threading.Thread(
                        target=lambda: c.post("/chat", json={"message": "first", "session_id": sid})
                    )
                    t.start()
                    time.sleep(0.3)  # let the first turn acquire the lock
                    second = c.post("/chat", json={"message": "second", "session_id": sid})
                    gate.set()
                    t.join(timeout=5)
            assert second.status_code == 409
        finally:
            set_runtime(None)
            rt.close()


# ── API client contract ───────────────────────────────────────────────────────


class TestJarvisClientAgainstRealServer:
    """End-to-end: the stdlib client drives a real uvicorn server thread."""

    @pytest.fixture()
    def live_server(self):
        import socket

        import uvicorn

        from jarvis.api.app import app, set_runtime

        with patch("jarvis.runtime.get_vector_store"):
            from jarvis.runtime import build_runtime

            rt = build_runtime()
        set_runtime(rt)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        if not server.started:
            pytest.fail("uvicorn test server did not start")

        from jarvis.api.client import JarvisClient

        client = JarvisClient(f"http://127.0.0.1:{port}")
        yield client
        server.should_exit = True
        thread.join(timeout=10)
        set_runtime(None)
        rt.close()

    def test_health_and_full_chat_loop(self, live_server):
        client = live_server
        health = client.health()
        assert health["status"] == "ok"

        sid = client.create_session()
        assert sid

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=fake_text_response("client answer"),
        ):
            result = client.chat("hello", sid)
        assert result["response"] == "client answer"
        assert result["session_id"] == sid

        history = client.history(sid)
        assert any(m["role"] == "user" and m["content"] == "hello" for m in history)

        assert client.get_confirmation(sid) is None
        tools = client.tools()
        assert any(t["name"] == "web_search" for t in tools)

    def test_stream_events_parse(self, live_server):
        client = live_server
        sid = client.create_session()

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=fake_text_response("streamed"),
        ):
            events = list(client.stream_chat("hi", sid))
        names = [e["event"] for e in events]
        assert names[0] == "begin"
        assert names[-1] == "done"
        assert dict((e["event"], e) for e in events)["done"]["data"]["response"] == "streamed"

    def test_error_shapes_raise_typed(self, live_server):
        from jarvis.api.client import JarvisClientError

        with pytest.raises(JarvisClientError) as ei:
            live_server.chat("hi", "missing-session")
        assert ei.value.status == 404


# ── Sandbox image policy ─────────────────────────────────────────────────────


class TestSandboxImagePolicy:
    @pytest.mark.parametrize(
        "ref",
        ["ubuntu", "ubuntu:latest", "myrepo/jarvis-sbx:latest", "UBUNTU:LATEST"],
    )
    def test_mutable_refs_rejected(self, ref):
        assert not DockerCodeSandbox._validate_image_reference(ref)

    @pytest.mark.parametrize(
        "ref",
        [
            "ubuntu:24.04",
            "myrepo/jarvis-sbx:1.0.0",
            "ghcr.io/org/img:2.3.1",
            "ubuntu:24.04@sha256:" + "a" * 64,
            "myrepo/img@sha256:" + "b" * 64,
        ],
    )
    def test_pinned_refs_accepted(self, ref):
        assert DockerCodeSandbox._validate_image_reference(ref)

    def test_default_image_still_valid(self):
        assert DockerCodeSandbox._validate_image_reference(DockerCodeSandbox.DEFAULT_IMAGE)

    def test_digest_requires_sha256(self):
        assert not DockerCodeSandbox._validate_image_reference("img@md5:abcdef")
