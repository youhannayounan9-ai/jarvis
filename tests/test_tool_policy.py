"""
tests/test_tool_policy.py
─────────────────────────
v0.21 deterministic tests for the capability-aware tool-selection policy:
the contract block, the heuristic classifier, the fast-path safety net,
ReAct schema narrowing, unmet-capability honesty, and the kill switch.

All LLM behavior is mocked; no Ollama. PermissionGuard stays real.
"""

from typing import Any
from unittest.mock import patch

import pytest

from jarvis.config import settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.core.tool_policy import (
    build_tool_policy_block,
    classify_intent,
    detect_unmet_capability,
    extract_arithmetic,
    is_single_intent_obligation,
    narrow_schemas_for_react,
)
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

# ── Shared mock helpers (mirror the LiteLLM response shape) ──────────────────


class Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, id, function):
        self.id = id
        self.function = function


class Message:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class Choice:
    def __init__(self, message):
        self.message = message


class Response:
    def __init__(self, choices):
        self.choices = choices


def fake_text_response(text: str) -> Response:
    return Response([Choice(Message("assistant", text))])


def fake_tool_response(calls: list[tuple[str, str]]) -> Response:
    return Response([
        Choice(Message("assistant", None, [ToolCall(f"call_{i}", Function(n, a))]))
        for i, (n, a) in enumerate(calls, start=1)
    ])


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


# ── 1. The tool-selection contract block ─────────────────────────────────────


class TestToolPolicyBlock:
    def test_block_exists_and_is_compact(self):
        block = build_tool_policy_block()
        assert block
        # Compact: no unbounded growth if someone edits it later.
        assert len(block) < 4000

    def test_block_states_the_no_tool_rule_first(self):
        block = build_tool_policy_block()
        assert "NO TOOL" in block

    def test_block_names_obligation_capabilities(self):
        block = build_tool_policy_block()
        assert "calculator" in block
        assert "search_knowledge" in block
        assert "recall_facts" in block

    def test_block_preserves_absent_capability_rule(self):
        """v0.16 honesty rule must survive inside the v0.21 contract."""
        block = build_tool_policy_block()
        assert "ABSENT" in block
        assert "never" in block.lower()


# ── 2. Heuristic capability classifier ───────────────────────────────────────


class TestClassifyIntent:
    @pytest.mark.parametrize(
        "text,expected_tool",
        [
            # Calculator obligations (Part G contract).
            ("What is 893 * 47?", "calculator"),
            ("What is 2+2?", "calculator"),
            ("Calculate 25 * 4", "calculator"),
            ("Compute 137 * 29", "calculator"),
            ("How many days until 2026-12-31?", "calculator"),
            # Knowledge obligations (Part F/O contract) — incl. the exact
            # v0.20 live-eval failure phrasing.
            ("What does my AI roadmap say about LangGraph?", "search_knowledge"),
            ("What does my roadmap say about Phase 5?", "search_knowledge"),
            ("Search my notes for the evaluation plan", "search_knowledge"),
            ("What does my knowledge base contain about evaluations?", "search_knowledge"),
            ("What do the documents I ingested say about RAG?", "search_knowledge"),
        ],
    )
    def test_obligation_tools_forced(self, text, expected_tool):
        registry = _build_registry()
        assert is_single_intent_obligation(text, registry) == expected_tool

    @pytest.mark.parametrize(
        "text",
        [
            # No-tool cases must never be forced (Part B: no-tool policy).
            "Hello there",
            "Explain what gradient descent is",
            "What is the capital of France?",
            "Which do you think is better, SQLite or Postgres?",
            # Datetime anchor, not arithmetic: a date literal would evaluate
            # as numeric subtraction inside the calculator (2026-10-01 → 2015).
            "What day is 2026-10-01?",
            "What time is it?",
            # Memory is recall_facts territory but NOT a forced obligation.
            "What did I say my name was?",
            # Compound asks are planner territory, never fast-path forced.
            "Calculate 12*12 and then write a summary to notes.md",
            "Search for AI news and then remember the headline",
        ],
    )
    def test_no_forcing_for_non_obligations(self, text):
        registry = _build_registry()
        assert is_single_intent_obligation(text, registry) is None

    def test_classifier_never_returns_unregistered_tool(self):
        registry = ToolRegistry()  # empty registry
        result = classify_intent("What is 2+2?", registry)
        assert result["tool"] is None
        assert is_single_intent_obligation("What is 2+2?", registry) is None


# ── 3. ReAct schema narrowing ────────────────────────────────────────────────


class TestNarrowSchemas:
    def test_planned_tools_always_included(self):
        registry = _build_registry()
        schemas = narrow_schemas_for_react(
            registry, "Do the vague thing", ["calculator"]
        )
        names = {s["function"]["name"] for s in schemas}
        # Planned tool present even when the description matches nothing.
        assert "calculator" in names

    def test_knowledge_step_gets_knowledge_family(self):
        registry = _build_registry()
        schemas = narrow_schemas_for_react(
            registry, "Search the knowledge base for LangGraph phases", []
        )
        names = {s["function"]["name"] for s in schemas}
        assert "search_knowledge" in names
        # Narrowing actually narrowed something.
        assert len(names) < 12

    def test_arithmetic_step_gets_compute_family(self):
        registry = _build_registry()
        schemas = narrow_schemas_for_react(
            registry, "Compute 137 * 29 and report the value", ["calculator"]
        )
        names = {s["function"]["name"] for s in schemas}
        assert "calculator" in names
        assert "write_file" not in names

    def test_vague_step_falls_back_to_all_tools(self):
        """Narrowing must never hide a tool the step actually needs."""
        registry = _build_registry()
        schemas = narrow_schemas_for_react(registry, "Summarize the results", [])
        assert len(schemas) == 12

    def test_narrowing_never_returns_empty(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = narrow_schemas_for_react(registry, "mysterious step", [])
        assert schemas  # fallback keeps the full (tiny) surface


# ── 4. Unmet-capability honesty (v0.16 rule reinforcement) ───────────────────


class TestUnmetCapability:
    def test_missing_code_execution_flagged(self):
        registry = _build_registry()
        note = detect_unmet_capability(
            "Run this python snippet for me and show the output: print(2+2)",
            registry,
        )
        assert note is not None
        assert "code execution" in note
        assert "Never simulate" in note

    def test_missing_computer_control_flagged(self):
        registry = _build_registry()
        note = detect_unmet_capability("Restart my computer for me", registry)
        assert note is not None
        assert "computer control" in note

    def test_available_capabilities_not_flagged(self):
        registry = _build_registry()
        assert detect_unmet_capability("What is the capital of France?", registry) is None
        assert detect_unmet_capability("Calculate 25 * 4", registry) is None

    def test_not_flagged_when_capability_actually_registered(self):
        registry = _build_registry()

        class _FakeRegistry:
            def list_tools(self):
                return ["execute_python_code"]

        # If a code tool existed, the honest-refusal note must NOT fire.
        assert detect_unmet_capability("Run my python script", _FakeRegistry()) is None


# ── 5. Orchestrator integration: fast-path safety net ────────────────────────


class TestForcedToolRound:
    def test_net_recovers_no_tool_first_answer(self):
        """Model answers without the tool → one bounded retry produces it."""
        orch, store, registry, guard = _make_orchestrator()
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return "Result: 41971"

        responses = [
            fake_text_response("It is 41971."),            # round 1: no tool
            fake_tool_response([("calculator", '{"expression": "893 * 47"}')]),
            fake_text_response("893 * 47 = 41971."),       # post-tool answer
        ]

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=responses,
        ):
            with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_net1", "What is 893 * 47?")

        assert dispatched == ["calculator"]
        assert answer == "893 * 47 = 41971."
        store.close()

    def test_net_not_needed_when_model_complies(self):
        orch, store, registry, guard = _make_orchestrator()
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return "Result: 41971"

        responses = [
            fake_tool_response([("calculator", '{"expression": "893 * 47"}')]),
            fake_text_response("893 * 47 = 41971."),
        ]

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=responses,
        ) as mock_llm:
            with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_net2", "What is 893 * 47?")

        assert dispatched == ["calculator"]
        assert answer == "893 * 47 = 41971."
        # execute round + post-tool answer; NO extra enforcement call.
        assert mock_llm.call_count == 2
        store.close()

    def test_net_bounded_when_model_refuses_twice(self):
        """A stubborn no-tool model still gets an answer after ONE retry."""
        orch, store, registry, guard = _make_orchestrator()
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return "Result: 1"

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[
                fake_text_response("42."),
                fake_text_response("Still no tools. 42."),
            ],
        ):
            with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat(
                        "s_net3", "What does my AI roadmap say about LangGraph?"
                    )

        assert dispatched == []  # never authorized a silent substitution
        assert answer == "Still no tools. 42."
        store.close()

    def test_no_net_for_greeting(self):
        orch, store, registry, guard = _make_orchestrator()
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return "unused"

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[fake_text_response("Hello!")],
        ) as mock_llm:
            with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_net4", "Hello there")

        assert answer == "Hello!"
        assert dispatched == []
        assert mock_llm.call_count == 1  # no enforcement call
        store.close()

    def test_policy_event_emitted_only_when_net_fires(self):
        orch, store, registry, guard = _make_orchestrator()
        events: list[dict[str, Any]] = []

        async def mock_dispatch_async(tool_name, tool_args):
            return "Result: 41971"

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[
                fake_text_response("41971."),
                fake_tool_response([("calculator", '{"expression": "893 * 47"}')]),
                fake_text_response("893 * 47 = 41971."),
            ],
        ):
            with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
                with patch.object(orch, "route_intent", return_value="simple"):
                    orch.chat("s_net5", "What is 893 * 47?", on_event=events.append)
        assert [e["type"] for e in events] == ["intent", "tool_policy"]

        # No net → no event (SSE event sequences of existing consumers hold).
        events2: list[dict[str, Any]] = []
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[fake_text_response("Hello!")],
        ):
            with patch.object(orch, "route_intent", return_value="simple"):
                orch.chat("s_net6", "Hello there", on_event=events2.append)
        assert [e["type"] for e in events2] == ["intent"]
        store.close()


# ── 6. Policy block injection + kill switch ──────────────────────────────────


class TestPolicyBlockInjection:
    def test_simple_path_messages_include_contract(self):
        orch, store, registry, guard = _make_orchestrator()
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[fake_text_response("Hello!")],
        ) as mock_llm:
            with patch.object(orch, "route_intent", return_value="simple"):
                orch.chat("s_blk1", "Hello there")
        msgs = mock_llm.call_args[1]["messages"]
        system_texts = [m["content"] for m in msgs if m["role"] == "system"]
        assert any("Tool Selection Contract" in t for t in system_texts)
        store.close()

    def test_kill_switch_restores_v020_behavior(self):
        """JARVIS_DISABLE_TOOL_POLICY=true: no contract block, no safety net."""
        orch, store, registry, guard = _make_orchestrator()
        dispatched: list[str] = []

        async def mock_dispatch_async(tool_name, tool_args):
            dispatched.append(tool_name)
            return "unused"

        with patch.object(settings, "JARVIS_DISABLE_TOOL_POLICY", True, create=True):
            with patch(
                "jarvis.core.orchestrator.chat_completion",
                side_effect=[fake_text_response("41971.")],
            ) as mock_llm:
                with patch.object(registry, "dispatch_async", side_effect=mock_dispatch_async):
                    with patch.object(orch, "route_intent", return_value="simple"):
                        answer = orch.chat("s_blk2", "What is 893 * 47?")

        assert answer == "41971."          # direct (v0.20) behavior
        assert dispatched == []            # safety net did NOT fire
        assert mock_llm.call_count == 1
        msgs = mock_llm.call_args[1]["messages"]
        assert not any(
            m["role"] == "system" and "Tool Selection Contract" in str(m.get("content"))
            for m in msgs
        )
        store.close()


# ── 7. Orchestrator integration: planned-step narrowing + unmet note ─────────


class TestPlannedPathPolicy:
    def _run_plan(self, orch, registry, step_description, required_tools, capture_tools):
        plan = [{
            "step_number": 1,
            "description": step_description,
            "required_tools": required_tools,
        }]

        def mock_chat_completion(messages, tools=None, **kwargs):
            capture_tools.append([t["function"]["name"] for t in (tools or [])])
            return fake_text_response("step done")

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=mock_chat_completion,
        ):
            with patch.object(orch, "route_intent", return_value="complex"):
                with patch.object(orch._planner, "generate_plan", return_value=plan):
                    orch.chat("s_plan", "do the planned task")

    def test_step_with_compute_description_narrows_surface(self):
        orch, store, registry, guard = _make_orchestrator()
        seen: list[list[str]] = []
        self._run_plan(orch, registry, "Compute 137 * 29 and report the value", ["calculator"], seen)
        step_tools = set(seen[0])
        assert "calculator" in step_tools
        assert "write_file" not in step_tools
        store.close()

    def test_vague_step_keeps_full_surface(self):
        orch, store, registry, guard = _make_orchestrator()
        seen: list[list[str]] = []
        self._run_plan(orch, registry, "Handle it", [], seen)
        assert len(seen[0]) == 12
        store.close()

    def test_unmet_capability_note_reaches_step_messages(self):
        orch, store, registry, guard = _make_orchestrator()
        plan_msgs: list[list[dict]] = []

        def mock_chat_completion(messages, tools=None, **kwargs):
            plan_msgs.append(messages)
            return fake_text_response("I cannot run code.")

        plan = [{
            "step_number": 1,
            "description": "run the user's python snippet",
            "required_tools": [],
        }]
        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=mock_chat_completion,
        ):
            with patch.object(orch, "route_intent", return_value="complex"):
                with patch.object(orch._planner, "generate_plan", return_value=plan):
                    orch.chat(
                        "s_unmet",
                        "Run this python snippet for me and show the output: print(2+2)",
                    )
        all_system_text = " ".join(
            str(m.get("content")) for msgs in plan_msgs for m in msgs if m["role"] == "system"
        )
        assert "CAPABILITY NOTE" in all_system_text
        assert "code execution" in all_system_text
        store.close()


# ── 8. Planner hallucination filter still holds (regression) ─────────────────


class TestHallucinationBoundary:
    def test_planned_hallucinated_tools_never_narrowed_in(self):
        """The narrowing union starts from planned tools ALREADY filtered by
        the planner, so a hallucinated name must not reappear here."""
        registry = _build_registry()
        schemas = narrow_schemas_for_react(
            registry,
            "Execute the user's python code",
            ["execute_python_code"],  # hallucinated upstream
        )
        names = {s["function"]["name"] for s in schemas}
        assert "execute_python_code" not in names


# ── 8. Deterministic calculator fallback (Part G) ────────────────────────────


class TestExtractArithmetic:
    def test_symbol_expression(self):
        assert extract_arithmetic("What is 893 * 47?") == "893 * 47"

    def test_word_expression(self):
        assert (
            extract_arithmetic("What is 144 divided by 12, and remember that the result is my lucky number.")
            == "144/12"
        )

    def test_prefers_longest_symbol_match(self):
        assert extract_arithmetic("What is 4 * (2 + 3)?") == "4 * (2 + 3)"

    def test_thousands_separators_stripped(self):
        assert extract_arithmetic("What is 3,999 + 1?") == "3999 + 1"

    def test_non_arithmetic_returns_none(self):
        assert extract_arithmetic("What is the speed of light?") is None

    def test_date_literal_not_treated_as_subtraction(self):
        assert extract_arithmetic("What day is 2026-10-01?") is None


class TestDeterministicCalculatorFallback:
    def _spy_dispatch(self, registry):
        real_dispatch = registry.dispatch_async
        dispatched: list[tuple[str, str]] = []

        async def spy(tool_name, tool_args):
            dispatched.append((tool_name, tool_args))
            return await real_dispatch(tool_name, tool_args)

        return spy, dispatched

    def test_fallback_executes_calculator_when_model_refuses(self):
        """Model answers arithmetic mentally (wrongly) with zero tool calls:
        the system runs the real calculator itself and grounds the answer."""
        orch, store, registry, guard = _make_orchestrator()
        spy, dispatched = self._spy_dispatch(registry)

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[
                fake_text_response("I can do that mentally: 42071."),
                fake_text_response("Still refusing tools. 42071."),
                fake_text_response("893 * 47 equals 41971."),
            ],
        ) as mock_llm:
            with patch.object(registry, "dispatch_async", side_effect=spy):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_fb1", "What is 893 * 47?")

        assert dispatched == [("calculator", '{"expression": "893 * 47"}')]
        assert "41971" in answer
        assert "42071" not in answer
        # Exactly 3 LLM calls: refusal + forced recovery + grounded rephrasing
        # (bounded: one recovery round, one deterministic execution, no loops)
        assert mock_llm.call_count == 3
        # The phrasing call received the deterministic tool result
        final_msgs = mock_llm.call_args_list[1][1]["messages"]
        assert any(
            "TOOL RESULT" in str(m.get("content")) and "41971" in str(m.get("content"))
            for m in final_msgs
        )
        store.close()

    def test_fallback_never_fires_for_knowledge_obligations(self):
        """Knowledge stays model-driven: a refused search_knowledge net does
        NOT deterministically execute retrieval on the model's behalf."""
        orch, store, registry, guard = _make_orchestrator()
        spy, dispatched = self._spy_dispatch(registry)

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[
                fake_text_response("Your roadmap mentions LangGraph."),
                fake_text_response("Your roadmap mentions LangGraph."),
            ],
        ) as mock_llm:
            with patch.object(registry, "dispatch_async", side_effect=spy):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat(
                        "s_fb2", "What does my AI roadmap say about LangGraph?"
                    )

        assert dispatched == []
        assert mock_llm.call_count == 2  # refusal + one forced recovery call
        store.close()

    def test_fallback_skipped_when_model_already_attempted_tools(self):
        """rounds_used > 0 means the model tried: never override its path."""
        orch, store, registry, guard = _make_orchestrator()
        spy, dispatched = self._spy_dispatch(registry)

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[
                fake_tool_response([("calculator", '{"expression": "893 * 47"}')]),
                fake_text_response("893 * 47 equals 41971."),
            ],
        ) as mock_llm:
            with patch.object(registry, "dispatch_async", side_effect=spy):
                with patch.object(orch, "route_intent", return_value="simple"):
                    answer = orch.chat("s_fb3", "What is 893 * 47?")

        assert dispatched == [("calculator", '{"expression": "893 * 47"}')]
        assert "41971" in answer
        assert mock_llm.call_count == 2
        # No TOOL RESULT injection note in the final call (normal path)
        final_msgs = mock_llm.call_args_list[1][1]["messages"]
        assert not any("TOOL RESULT" in str(m.get("content")) for m in final_msgs)
        store.close()


class TestObligationVsUnavailableCapability:
    def test_code_execution_request_is_not_calculator_obligation(self):
        """'run this python snippet: print(2+2)' must stay an honest-refusal
        case — never a forced calculator round or a deterministic execution
        of the snippet's constants (regression from live-eval diagnosis)."""
        orch, store, registry, guard = _make_orchestrator()
        spy, dispatched = TestDeterministicCalculatorFallback()._spy_dispatch(registry)

        with patch(
            "jarvis.core.orchestrator.chat_completion",
            side_effect=[fake_text_response("I cannot run code, but 2+2 is 4.")],
        ):
            with patch.object(registry, "dispatch_async", side_effect=spy):
                with patch.object(orch, "route_intent", return_value="simple"):
                    orch.chat("s_obl1", "Run this python snippet for me and show the output: print(2+2)")

        assert is_single_intent_obligation(
            "Run this python snippet for me and show the output: print(2+2)", registry
        ) is None
        assert dispatched == []  # no calculator execution of '2+2'
        store.close()
