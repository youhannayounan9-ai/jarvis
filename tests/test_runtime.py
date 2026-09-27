"""
tests/test_runtime.py
─────────────────────
Tests for the JarvisRuntime assembly point (agent-runtime / interface seam).
"""

from unittest.mock import patch

from tests.fakes import fake_text_response

from jarvis.runtime import JarvisRuntime, build_runtime


def _build() -> JarvisRuntime:
    with patch("jarvis.runtime.get_vector_store"):
        return build_runtime()


class TestRuntimeAssembly:
    def test_build_runtime_wires_all_components(self):
        rt = _build()
        assert rt.store is not None
        assert rt.registry is not None
        assert rt.guard is not None
        assert rt.orchestrator is not None
        rt.close()

    def test_standard_tool_surface(self):
        rt = _build()
        tools = set(rt.registry.list_tools())
        expected = {
            "get_current_datetime", "web_search", "wikipedia_summary",
            "read_file", "list_directory", "calculator",
            "remember_fact", "recall_facts", "write_file",
            "vision_analyze", "web_scrape",
        }
        assert expected == tools
        rt.close()

    def test_disabled_tools_never_registered(self):
        rt = _build()
        tools = rt.registry.list_tools()
        assert "execute_python_code" not in tools
        assert "computer_control" not in tools
        rt.close()

    def test_describe_reports_surface_and_disabled(self):
        rt = _build()
        info = rt.describe()
        assert info["tools"] == rt.registry.list_tools()
        assert set(info["disabled_tools"]) == {"execute_python_code", "computer_control"}
        assert info["version"]
        assert info["model"]
        rt.close()


class TestRuntimeSessions:
    def test_start_session_binds_and_tracks(self):
        rt = _build()
        sid = rt.start_session()
        assert sid
        assert rt.session_exists(sid)
        assert rt.sessions_created == 1
        rt.close()

    def test_session_exists_for_legacy_rows(self):
        """Sessions created directly in the store (pre-runtime) are recognized."""
        rt = _build()
        sid = rt.store.create_session()
        assert rt.session_exists(sid)
        assert rt.sessions_created == 0
        rt.close()

    def test_session_exists_rejects_unknown(self):
        rt = _build()
        assert not rt.session_exists("no-such-session")
        rt.close()


class TestRuntimeDelegation:
    def test_chat_delegates_to_orchestrator(self):
        rt = _build()
        sid = rt.start_session()

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=fake_text_response("runtime answer"),
        ):
            with patch.object(rt.orchestrator, "route_intent", return_value="simple"):
                answer = rt.chat(sid, "hello")

        assert answer == "runtime answer"
        rt.close()

    def test_confirmation_delegates(self):
        rt = _build()
        sid = rt.start_session()
        assert rt.get_pending_confirmation(sid) is None
        assert "No pending" in rt.handle_confirmation(sid, True)
        rt.close()

    def test_context_manager_support(self):
        with patch("jarvis.runtime.get_vector_store"):
            with _build() as rt:
                assert rt.describe()["version"]
