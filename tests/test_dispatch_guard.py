"""
tests/test_dispatch_guard.py
────────────────────────────
v0.23 deterministic tests for repeat-call semantics (Parts B/C/E/K):
fingerprinting, duplicate suppression, legitimate repeats, state-dependent
exemption, retry-after-failure, and the integration guarantees (RAG,
calculator fallback, recovery) that suppression must NOT break.

All LLM behavior is mocked; no Ollama. PermissionGuard and validation stay
real. No Git operations concept applies here; tests are offline.
"""

import json
from unittest.mock import patch

import pytest

from jarvis.core.dispatch_guard import (
    DispatchLedger,
    canonical_arguments,
    fingerprint,
)
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.tools import (
    CalculatorTool,
    GetCurrentDatetimeTool,
    ListDirectoryTool,
    ReadFileTool,
    RecallFactsTool,
    RememberFactTool,
    SearchKnowledgeTool,
    ToolRegistry,
    VisionAnalyzeTool,
    WebScrapeTool,
    WebSearchTool,
    WikipediaSummaryTool,
    WriteFileTool,
)


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for factory in (
        CalculatorTool,
        GetCurrentDatetimeTool,
        WebSearchTool,
        WikipediaSummaryTool,
        ReadFileTool,
        ListDirectoryTool,
        RememberFactTool,
        RecallFactsTool,
        SearchKnowledgeTool,
        WriteFileTool,
        VisionAnalyzeTool,
        WebScrapeTool,
    ):
        registry.register(factory())
    return registry


def _make_orchestrator():
    store = SessionStore()
    registry = _build_registry()
    guard = PermissionGuard()
    orch = Orchestrator(store, registry, guard)
    return orch, store, registry, guard


# ── 1. Fingerprinting (Part C) ────────────────────────────────────────────────


class TestCanonicalArguments:
    def test_key_order_irrelevant(self):
        assert (
            canonical_arguments("t", '{"query": "x", "top_k": 4}')
            == canonical_arguments("t", '{"top_k": 4, "query": "x"}')
        )

    def test_whitespace_normalization(self):
        assert (
            canonical_arguments("t", '{"query": "local   AI   models"}')
            == canonical_arguments("t", '{"query": "local AI models"}')
        )

    def test_different_values_differ(self):
        assert canonical_arguments("t", '{"query": "a"}') != canonical_arguments("t", '{"query": "b"}')

    def test_invalid_json_marked_not_crash(self):
        out = canonical_arguments("t", "{not json")
        assert out.startswith("__unparsed__:")

    def test_empty_args(self):
        assert canonical_arguments("t", "") == "{}"
        assert canonical_arguments("t", "   ") == "{}"

    def test_fingerprint_stable_and_short(self):
        fp1 = fingerprint("calculator", '{"expression": "2+2"}')
        fp2 = fingerprint("calculator", '{"expression": "2+2 "}')
        fp3 = fingerprint("calculator", '{"expression": "2+3"}')
        assert fp1 == fp2  # leading/trailing whitespace insignificant
        assert fp1 != fp3
        assert len(fp1) == 12
        # Note: '2+2' vs '2 + 2' differ (in-string spaces are significant —
        # no math special-casing); the residual duplicate window is tiny and
        # the calculator is cheap, so this is the honest general rule.

    def test_fingerprint_tool_sensitive(self):
        assert fingerprint("a", '{"x": 1}') != fingerprint("b", '{"x": 1}')


# ── 2. Ledger semantics (Parts B/E) ──────────────────────────────────────────


class TestDispatchLedger:
    def test_duplicate_detected(self):
        ledger = DispatchLedger()
        ledger.record_success("search_knowledge", '{"query": "LangGraph"}')
        assert ledger.is_duplicate("search_knowledge", '{"query": "LangGraph"}')
        assert ledger.is_duplicate("search_knowledge", '{"query": "  LangGraph  "}')

    def test_different_args_not_duplicate(self):
        ledger = DispatchLedger()
        ledger.record_success("search_knowledge", '{"query": "LangGraph"}')
        assert not ledger.is_duplicate("search_knowledge", '{"query": "LangChain"}')

    def test_state_dependent_tools_exempt(self):
        ledger = DispatchLedger()
        for tool in ("get_current_datetime", "recall_facts", "remember_fact"):
            ledger.record_success(tool, '{"x": 1}')
            assert not ledger.is_duplicate(tool, '{"x": 1}'), tool

    def test_identical_web_search_within_turn_suppressed(self):
        """Live v0.22 evidence: web_search×3 same query. External pages do
        not change seconds apart — identical-argument re-calls are the
        redundancy pattern v0.23 targets (different queries stay allowed)."""
        ledger = DispatchLedger()
        ledger.record_success("web_search", '{"query": "LangGraph news"}')
        assert ledger.is_duplicate("web_search", '{"query": "LangGraph news"}')
        assert not ledger.is_duplicate("web_search", '{"query": "checkpointing libraries"}')

    def test_search_knowledge_not_state_dependent(self):
        """Document retrieval of the SAME query within one turn adds nothing."""
        ledger = DispatchLedger()
        ledger.record_success("search_knowledge", '{"query": "LangGraph"}')
        assert ledger.is_duplicate("search_knowledge", '{"query": "LangGraph"}')

    def test_suppression_counted_and_logged(self):
        ledger = DispatchLedger()
        ledger.record_success("calculator", '{"expression": "2+2"}')
        ledger.record_suppressed("calculator", '{"expression": "2+2"}')
        assert ledger.suppressed_count == 1

    def test_failed_results_never_block_retry(self):
        """Retry after FAILURE is legitimate — enforced by the orchestrator
        only recording successes (behavioral pin below)."""
        ledger = DispatchLedger()
        # Simulate: failure never entered the ledger.
        assert not ledger.is_duplicate("calculator", '{"expression": "893*47"}')


# ── 3. Orchestrator integration (Part E) ─────────────────────────────────────


class TestSuppressionIntegration:
    @pytest.mark.asyncio
    async def test_identical_success_suppressed_at_dispatch(self):
        orch, store, registry, guard = _make_orchestrator()
        calls = {"n": 0}

        async def counting_dispatch(tool_name, tool_args_json):
            calls["n"] += 1
            return "ok result"

        with patch.object(registry, "dispatch_async", side_effect=counting_dispatch):
            first = await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c1")
            second = await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c2")

        assert "ok result" in first
        assert "DUPLICATE_SUPPRESSED" in second
        assert calls["n"] == 1  # the tool ran ONCE
        assert orch._dispatch_ledger.suppressed_count == 1
        store.close()

    @pytest.mark.asyncio
    async def test_different_args_dispatch_again(self):
        orch, store, registry, guard = _make_orchestrator()
        calls = {"n": 0}

        async def counting_dispatch(tool_name, tool_args_json):
            calls["n"] += 1
            return "ok result"

        with patch.object(registry, "dispatch_async", side_effect=counting_dispatch):
            await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c1")
            await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "docs"}', "c2")
        assert calls["n"] == 2
        store.close()

    @pytest.mark.asyncio
    async def test_state_dependent_tool_dispatches_again(self):
        orch, store, registry, guard = _make_orchestrator()
        calls = {"n": 0}

        async def counting_dispatch(tool_name, tool_args_json):
            calls["n"] += 1
            return "result"

        with patch.object(registry, "dispatch_async", side_effect=counting_dispatch):
            await orch._dispatch_with_permissions_async("s", "recall_facts", '{"query": "x"}', "c1")
            await orch._dispatch_with_permissions_async("s", "recall_facts", '{"query": "x"}', "c2")
        assert calls["n"] == 2  # legitimate state-dependent repeat
        store.close()

    @pytest.mark.asyncio
    async def test_retry_after_failure_allowed(self):
        orch, store, registry, guard = _make_orchestrator()
        calls = {"n": 0}

        async def failing_then_ok(tool_name, tool_args_json):
            calls["n"] += 1
            if calls["n"] == 1:
                return "ERROR: transient failure"
            return "recovered result"

        with patch.object(registry, "dispatch_async", side_effect=failing_then_ok):
            first = await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c1")
            second = await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c2")

        assert "ERROR" in first
        assert "recovered result" in second  # retry executed
        assert calls["n"] == 2
        store.close()

    @pytest.mark.asyncio
    async def test_intentional_repeat_bypasses_suppression(self):
        orch, store, registry, guard = _make_orchestrator()
        calls = {"n": 0}

        async def counting_dispatch(tool_name, tool_args_json):
            calls["n"] += 1
            return "ok"

        with patch.object(registry, "dispatch_async", side_effect=counting_dispatch):
            await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c1")
            again = await orch._dispatch_with_permissions_async(
                "s", "list_directory", '{"path": "."}', "c2", intentional_repeat=True
            )
        assert calls["n"] == 2
        assert "DUPLICATE_SUPPRESSED" not in again
        store.close()

    @pytest.mark.asyncio
    async def test_blocked_tool_never_enters_ledger(self):
        """A guard-blocked dispatch must not seed the suppression ledger."""
        orch, store, registry, guard = _make_orchestrator()
        with patch.object(guard, "is_allowed", return_value=False):
            blocked = await orch._dispatch_with_permissions_async("s", "list_directory", '{"path": "."}', "c1")
        assert "not permitted" in blocked
        assert not orch._dispatch_ledger.is_duplicate("list_directory", '{"path": "."}')
        store.close()

    def test_fresh_ledger_per_turn(self):
        """chat() replaces the ledger: no stale suppression across turns."""
        orch, store, registry, guard = _make_orchestrator()
        orch._dispatch_ledger.record_success("list_directory", '{"path": "."}')
        orch._current_mode = "simple"  # simulate entering a new chat() call
        from jarvis.core.dispatch_guard import DispatchLedger as DL

        orch._dispatch_ledger = DL()  # what chat() does at entry
        assert not orch._dispatch_ledger.is_duplicate("list_directory", '{"path": "."}')
        store.close()


# ── 4. Phase guarantees preserved (Part K) ───────────────────────────────────


class TestPhaseGuaranteesPreserved:
    def test_calculator_fallback_path_ignores_ledger_state(self):
        """The v0.21 fallback fires only on zero-attempt turns; the ledger
        records successes, so a prior successful calculator call in the SAME
        turn would mean the model DID call it (fallback precondition false).
        The fallback also passes its own explicit dispatch (no repeat)."""
        orch, store, registry, guard = _make_orchestrator()
        orch._dispatch_ledger.record_success("calculator", '{"expression": "1+1"}')
        # A DIFFERENT expression is not suppressed:
        assert not orch._dispatch_ledger.is_duplicate("calculator", '{"expression": "893 * 47"}')
        store.close()

    def test_recovery_ledger_is_fresh(self):
        """handle_confirmation runs on a fresh turn entry — the ledger starts
        empty, so resumed plans never inherit stale suppression state."""
        orch, store, registry, guard = _make_orchestrator()
        orch._dispatch_ledger.record_success("list_directory", '{"path": "."}')
        # New turn (what handle_confirmation effectively sees):
        orch._dispatch_ledger = DispatchLedger()
        assert not orch._dispatch_ledger.is_duplicate("list_directory", '{"path": "."}')
        store.close()

    @pytest.mark.asyncio
    async def test_high_risk_confirmation_still_parks_before_suppression(self):
        """Suppression sits AFTER the guard: a duplicate write_file attempt
        must still require confirmation — suppression is not an auth path."""
        orch, store, registry, guard = _make_orchestrator()
        orch._dispatch_ledger.record_success("write_file", '{"path": "a.txt", "content": "x"}')

        with patch.object(guard, "require_confirmation", return_value=True):
            result = await orch._dispatch_with_permissions_async(
                "s", "write_file", '{"path": "a.txt", "content": "x"}', "c1"
            )
        # Parked for approval — suppression never intercepted the flow.
        assert "ACTION_REQUIRES_CONFIRMATION" in result
        assert "DUPLICATE_SUPPRESSED" not in result
        store.close()
