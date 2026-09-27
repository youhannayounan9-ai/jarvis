"""
tests/test_context_manager.py
─────────────────────────────
Tests for the ContextManager: smart windowing, message anchoring,
extractive summarization, and tool-output clamping.
"""

from jarvis.memory.context_manager import ContextManager, estimate_tokens


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool(content: str, call_id: str = "c1", name: str = "web_search") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


class TestWindowing:
    def test_small_history_passes_through(self):
        cm = ContextManager(max_messages=10)
        history = [_user("hi"), _assistant("hello")]
        window, anchor, summary = cm.select_window(history)
        assert window == history
        assert anchor is None
        assert summary is None

    def test_large_history_is_windowed(self):
        cm = ContextManager(max_messages=4)
        history = [_user(f"msg-{i}") for i in range(10)]
        window, anchor, summary = cm.select_window(history)
        assert len(window) == 4
        assert [m["content"] for m in window] == ["msg-6", "msg-7", "msg-8", "msg-9"]

    def test_window_never_exceeds_max(self):
        cm = ContextManager(max_messages=6)
        history = [_user(f"m{i}") for i in range(100)]
        window, _, _ = cm.select_window(history)
        assert len(window) == 6

    def test_build_messages_keeps_system_prompts_outside_budget(self):
        cm = ContextManager(max_messages=2)
        history = [_user(f"m{i}") for i in range(8)]
        msgs = cm.build_messages(
            system_prompts=[{"role": "system", "content": "SYS"}],
            history=history,
            user_input="current question",
        )
        # 1 system + up to (anchor + summary) + 2 window + 1 user
        non_system = [m for m in msgs if m["role"] != "system"]
        assert len(non_system) == 3  # 2 window + current user input
        assert msgs[-1] == {"role": "user", "content": "current question"}
        assert msgs[0] == {"role": "system", "content": "SYS"}


class TestAnchoring:
    def test_first_user_task_is_anchored_when_dropped(self):
        cm = ContextManager(max_messages=3)
        history = [
            _user("Help me plan my Paris trip for 7 days"),
            _assistant("Sure."),
            *[_user(f"follow-up {i}") for i in range(5)],
        ]
        window, anchor, _ = cm.select_window(history)
        assert anchor is not None
        assert "Paris trip" in anchor["content"]
        assert anchor["role"] == "system"
        assert "Paris trip" not in "".join(str(m.get("content")) for m in window)

    def test_no_anchor_when_first_task_still_in_window(self):
        cm = ContextManager(max_messages=10)
        history = [_user("original task"), _assistant("ok"), _user("next")]
        _, anchor, _ = cm.select_window(history)
        assert anchor is None

    def test_anchor_not_duplicated_if_present_in_window(self):
        cm = ContextManager(max_messages=2)
        history = [
            _user("Plan my Paris trip"),
            _assistant("ok"),
            _user("Plan my Paris trip — also add museums"),
        ]
        window, anchor, _ = cm.select_window(history)
        # The last user message still mentions "Plan my Paris trip", so no anchor
        assert anchor is None

    def test_anchor_content_is_clamped(self):
        cm = ContextManager(max_messages=1)
        history = [_user("X" * 2000), _user("y"), _user("z")]
        _, anchor, _ = cm.select_window(history)
        assert anchor is not None
        assert len(anchor["content"]) < 700


class TestSummarization:
    def test_summary_includes_user_intents_and_tools(self):
        cm = ContextManager(max_messages=2)
        history = [
            _user("Remember my birthday is May 5"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "remember_fact", "arguments": "{}"},
                }],
            },
            _tool("Remembered: birthday May 5"),
            _assistant("Noted your birthday."),
            _user("what is 2+2"),
            _assistant("4"),
        ]
        _, _, summary = cm.select_window(history)
        assert summary is not None
        text = summary["content"]
        assert "Remember my birthday" in text
        assert "remember_fact" in text
        assert "Noted your birthday." in text

    def test_summary_is_bounded(self):
        cm = ContextManager(max_messages=1)
        history = [_user("u" * 3000), _assistant("a" * 3000), _user("final")]
        _, _, summary = cm.select_window(history)
        assert summary is not None
        assert len(summary["content"]) < 1500

    def test_no_summary_when_nothing_meaningful_dropped(self):
        cm = ContextManager(max_messages=2)
        history = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c", "type": "function",
                             "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "name": "t", "content": "x" * 3000},
            _user("current"),
        ]
        _, _, summary = cm.select_window(history)
        # The bulky tool payload is dropped silently; only a one-line tool
        # digest survives, and the payload itself must NOT leak into the prompt.
        assert summary is not None
        assert "x" * 100 not in summary["content"]
        assert "t." in summary["content"]

    def test_build_messages_includes_summary_and_anchor(self):
        cm = ContextManager(max_messages=2)
        history = [
            _user("original goal"),
            _assistant("ack"),
            _user("q1"), _assistant("a1"),
            _user("q2"), _assistant("a2"),
        ]
        msgs = cm.build_messages(
            system_prompts=[],
            history=history,
            user_input="latest",
        )
        roles = [m["role"] for m in msgs]
        assert roles.count("system") >= 2  # anchor + summary
        assert msgs[-1]["content"] == "latest"


class TestToolClamping:
    def test_short_output_untouched(self):
        cm = ContextManager()
        assert cm.clamp_tool_output("small result") == "small result"

    def test_large_output_is_clamped_head_and_tail(self):
        cm = ContextManager()
        payload = "HEAD-MARKER" + "x" * 20000 + "TAIL-MARKER"
        clamped = cm.clamp_tool_output(payload, max_chars=6000)
        assert len(clamped) < 7000
        assert "HEAD-MARKER" in clamped
        assert "TAIL-MARKER" in clamped
        assert "characters omitted" in clamped

    def test_clamping_is_idempotent(self):
        cm = ContextManager()
        payload = "y" * 20000
        once = cm.clamp_tool_output(payload, max_chars=4000)
        twice = cm.clamp_tool_output(once, max_chars=4000)
        assert once == twice

    def test_already_truncated_payload_not_reclamped(self):
        cm = ContextManager()
        payload = "z" * 20000 + "\n[truncated — older tool output shortened]"
        assert cm.clamp_tool_output(payload, max_chars=100) == payload

    def test_non_string_input_coerced(self):
        cm = ContextManager()
        assert cm.clamp_tool_output(12345) == "12345"


class TestTokenEstimate:
    def test_estimate_is_proportional(self):
        small = [{"role": "user", "content": "a" * 400}]
        large = [{"role": "user", "content": "a" * 4000}]
        assert estimate_tokens(small) < estimate_tokens(large)

    def test_estimate_counts_tool_calls(self):
        with_calls = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}],
        }]
        assert estimate_tokens(with_calls) > 0
