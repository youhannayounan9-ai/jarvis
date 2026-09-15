"""
tests/test_session_store.py
────────────────────────────
Unit tests for short-term session memory (SQLite).
"""

from jarvis.memory.session_store import SessionStore, _trim_orphan_tool_prefix


class TestSessionStoreHistory:
    def setup_method(self):
        self.store = SessionStore()
        self.session_id = self.store.create_session()

    def teardown_method(self):
        self.store.close()

    def test_loads_most_recent_messages(self):
        for i in range(6):
            self.store.save_message(
                self.session_id,
                {"role": "user", "content": f"msg-{i}"},
            )

        history = self.store.load_history(self.session_id, limit=3)
        assert len(history) == 3
        assert [m["content"] for m in history] == ["msg-3", "msg-4", "msg-5"]

    def test_preserves_chronological_order(self):
        self.store.save_message(self.session_id, {"role": "user", "content": "first"})
        self.store.save_message(
            self.session_id,
            {"role": "assistant", "content": "second"},
        )
        history = self.store.load_history(self.session_id)
        assert history[0]["content"] == "first"
        assert history[1]["content"] == "second"

    def test_trims_long_tool_content_on_load(self):
        self.store.save_message(
            self.session_id,
            {"role": "user", "content": "search please"},
        )
        self.store.save_message(
            self.session_id,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }],
            },
        )
        self.store.save_message(
            self.session_id,
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "web_search",
                "content": "x" * 5000,
            },
        )
        history = self.store.load_history(self.session_id)
        tool_msg = next(m for m in history if m["role"] == "tool")
        assert len(tool_msg["content"]) < 5000
        assert "truncated" in tool_msg["content"]

    def test_message_count(self):
        assert self.store.message_count(self.session_id) == 0
        self.store.save_message(self.session_id, {"role": "user", "content": "hi"})
        assert self.store.message_count(self.session_id) == 1


class TestHistoryHelpers:
    def test_trim_orphan_tool_prefix(self):
        messages = [
            {"role": "tool", "content": "orphan", "tool_call_id": "1"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        trimmed = _trim_orphan_tool_prefix(messages)
        assert trimmed[0]["role"] == "user"
        assert len(trimmed) == 2
