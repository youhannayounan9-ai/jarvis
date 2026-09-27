"""
tests/test_stability.py
───────────────────────
Stability regressions: context-window edge cases, session-store cache
coherence under writes and threads, and orchestrator event observation.
"""

import threading
from unittest.mock import patch

import pytest

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.context_manager import ContextManager
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.core.permissions import PermissionGuard


# ── ContextManager edge cases ─────────────────────────────────────────────────


class TestContextWindowEdges:
    def test_zero_window_floors_to_one(self):
        """max_messages=0 must not make history[-0:] return EVERYTHING."""
        cm = ContextManager(max_messages=0)
        assert cm.max_messages == 1
        history = [{"role": "user", "content": f"m{i}"} for i in range(50)]
        window, anchor, summary = cm.select_window(history)
        assert len(window) == 1  # bounded, not 50

    def test_negative_window_floors_to_one(self):
        cm = ContextManager(max_messages=-5)
        assert cm.max_messages == 1

    def test_empty_history_returns_empty_window(self):
        cm = ContextManager(max_messages=8)
        window, anchor, summary = cm.select_window([])
        assert window == [] and anchor is None and summary is None

    def test_history_of_only_tool_messages(self):
        """No user anchor exists; must not crash and must not invent one."""
        cm = ContextManager(max_messages=2)
        history = [
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "tool", "tool_call_id": "c2", "content": "result2"},
        ]
        window, anchor, summary = cm.select_window(history)
        assert len(window) == 2
        assert anchor is None

    def test_clamp_is_idempotent(self):
        cm = ContextManager()
        huge = "x" * 50_000
        once = cm.clamp_tool_output(huge)
        twice = cm.clamp_tool_output(once)
        assert once == twice

    def test_clamp_keeps_head_and_tail(self):
        cm = ContextManager()
        content = "HEAD" + "m" * 40_000 + "TAIL"
        clamped = cm.clamp_tool_output(content, max_chars=4000)
        assert clamped.startswith("HEAD")
        assert clamped.endswith("TAIL")
        assert "characters omitted" in clamped


# ── SessionStore cache coherence + thread safety ──────────────────────────────


class TestSessionStoreCoherence:
    def test_load_history_reflects_new_writes(self):
        """The lru_cache must not serve stale history after a write."""
        store = SessionStore()
        sid = store.create_session()
        store.save_message(sid, {"role": "user", "content": "first"})
        assert store.load_history(sid, limit=10)[0]["content"] == "first"
        store.save_message(sid, {"role": "assistant", "content": "second"})
        history = store.load_history(sid, limit=10)
        assert [m["content"] for m in history] == ["first", "second"]
        store.close()

    def test_concurrent_writes_all_persist(self):
        """Parallel writers must not lose rows or corrupt the connection."""
        store = SessionStore()
        sid = store.create_session()
        n_threads, per_thread = 8, 10
        errors: list[Exception] = []

        def writer(tid: int) -> None:
            try:
                for i in range(per_thread):
                    store.save_message(
                        sid, {"role": "user", "content": f"t{tid}-m{i}"}
                    )
            except Exception as e:  # pragma: no cover - surfaced via assert
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.message_count(sid) == n_threads * per_thread
        # History must reflect every message exactly once.
        history = store.load_history(sid, limit=1000)
        contents = [m["content"] for m in history]
        assert len(contents) == n_threads * per_thread
        assert len(set(contents)) == n_threads * per_thread
        store.close()


# ── Orchestrator event observation ────────────────────────────────────────────


class TestOrchestratorEvents:
    def _make(self):
        store = SessionStore()
        orch = Orchestrator(store, ToolRegistry(), PermissionGuard())
        return orch, store

    @staticmethod
    def _fake_llm(text="ok"):
        class M:
            role = "assistant"
            content = text
            tool_calls = None

        class C:
            message = M()

        class R:
            choices = [C()]

        return R()

    def test_simple_path_emits_intent_event(self):
        orch, store = self._make()
        events: list[dict] = []
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=self._fake_llm(),
        ):
            with patch.object(orch, "route_intent", return_value="simple"):
                orch.chat("s_events", "hi", on_event=events.append)
        types = [e["type"] for e in events]
        assert types == ["intent"]
        assert events[0]["intent"] == "simple"
        store.close()

    def test_complex_path_emits_full_lifecycle(self):
        orch, store = self._make()
        events: list[dict] = []
        plan = [
            {"step_number": 1, "description": "step one", "required_tools": []},
            {"step_number": 2, "description": "step two", "required_tools": []},
        ]
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=self._fake_llm(),
        ):
            with patch.object(orch, "route_intent", return_value="complex"):
                with patch.object(orch._planner, "generate_plan", return_value=plan):
                    orch.chat("s_events2", "complex task", on_event=events.append)
        types = [e["type"] for e in events]
        assert types[0] == "intent"
        assert "plan" in types
        assert types.count("step_start") == 2
        assert types.count("step_done") == 2
        assert types[-1] == "synthesis"
        store.close()

    def test_observer_exception_never_breaks_chat(self):
        orch, store = self._make()

        def bomb(evt):
            raise RuntimeError("observer exploded")

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=self._fake_llm(),
        ):
            with patch.object(orch, "route_intent", return_value="simple"):
                answer = orch.chat("s_events3", "hi", on_event=bomb)
        assert answer == "ok"
        store.close()

    def test_no_observer_is_default(self):
        orch, store = self._make()
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            return_value=self._fake_llm(),
        ):
            with patch.object(orch, "route_intent", return_value="simple"):
                assert orch.chat("s_events4", "hi") == "ok"
        store.close()
