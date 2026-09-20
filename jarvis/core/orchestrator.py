"""
jarvis/core/orchestrator.py
────────────────────────────
The Orchestrator — the brain of JARVIS (v0.3 Plan-and-Execute).

Every user message flows through here:
  1. Plan  — decompose the request into discrete steps (Planner).
  2. Execute — run a short ReAct loop per step (tools + PermissionGuard).
  3. Synthesize — one final tool-free LLM call producing the user-facing answer.

Safety bounds:
  - MAX_TOOL_ROUNDS_PER_STEP caps ReAct iterations inside a single step.
  - MAX_TOOL_ROUNDS caps total tool-call rounds across the whole request.
"""

from typing import Any

from jarvis.config import settings
from jarvis.core.permissions import PermissionGuard
from jarvis.core.planner import Planner
from jarvis.llm.client import chat_completion
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Global cap on tool-call rounds for one user message (across all plan steps).
MAX_TOOL_ROUNDS = 5
# Per-step ReAct budget (also clipped by remaining global budget).
MAX_TOOL_ROUNDS_PER_STEP = 2

_SYNTHESIZE_PROMPT = (
    "Synthesize a single, natural, and concise final response based on the completed steps. "
    "CRITICAL: Do NOT repeat raw tool outputs, step descriptions, or previous text verbatim. "
    "Speak naturally and keep it brief. "
    "If the user asks about their memory or personal facts, you MUST rely ONLY on the results from "
    "the `recall_facts` tool. Do not make up or guess any facts."
)


class Orchestrator:
    """
    Central request handler using Plan-and-Execute.

    Args:
        session_store:    Persistence layer for conversation history.
        tool_registry:    Registry of available tools.
        permission_guard: Controls which tools may execute.
    """

    def __init__(
        self,
        session_store: SessionStore,
        tool_registry: ToolRegistry,
        permission_guard: PermissionGuard,
    ) -> None:
        self._store = session_store
        self._registry = tool_registry
        self._guard = permission_guard
        self._planner = Planner(llm_client=chat_completion)
        self._pending_confirmations: dict[str, Any] = {}

    def route_intent(self, user_input: str) -> str:
        text = user_input.lower().strip()
        
        # Complexity triggers that override simple keywords
        complex_triggers = ["and then", "also", "after that", "then", "next", "write a", "create a"]
        if any(c in text for c in complex_triggers):
            return "complex"
            
        simple_keywords = [
            "what time", "time is it", "calculate", "hello", "hi", "hey",
            "who are you", "+", "-", "*", "/", "search", "remember",
            "date", "today", "weather"
        ]
        if any(k in text for k in simple_keywords) and len(text) < 150:
            return "simple"
        return "complex"

    def chat(self, session_id: str, user_input: str) -> str:
        """
        Process one user message via Plan → Execute → Synthesize.

        Args:
            session_id: The ID of the current conversation session.
            user_input: The raw text typed by the user.

        Returns:
            The assistant's final plain-text response string.
        """
        # ── 1. Persist the user turn ───────────────────────────────────────────
        user_message: dict[str, Any] = {"role": "user", "content": user_input}
        self._store.save_message(session_id, user_message)

        history = self._store.load_history(session_id)
        memory_cue = (
            f"Short-term memory: {len(history)} message(s) from this session "
            "are included for reference. Use them to resolve follow-ups."
        )
        context_cue = _build_context_cue(history, memory_cue)

        # ── 1.5. Intent Routing ────────────────────────────────────────────────
        intent = self.route_intent(user_input)
        log.info("intent_routed", intent=intent, bypassed_planner=(intent == "simple"))

        tool_schemas = self._registry.get_schemas()

        if intent == "simple":
            messages = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "system", "content": memory_cue},
                *history,
                {"role": "user", "content": user_input},
            ]
            final_text, _ = self._run_react(
                session_id=session_id,
                messages=messages,
                tool_schemas=tool_schemas,
                max_rounds=2,
            )
            return final_text

        # ── 2. Plan phase ──────────────────────────────────────────────────────
        plan = self._planner.generate_plan(user_input, context_cue)
        log.info(
            "plan_ready",
            session_id=session_id,
            steps=len(plan),
            plan=[{
                "step": s.get("step_number"),
                "description": s.get("description"),
                "tools": s.get("required_tools"),
            } for s in plan],
        )

        # ── 3. Execute phase ───────────────────────────────────────────────────
        completed_steps: list[dict[str, Any]] = []
        remaining_rounds = MAX_TOOL_ROUNDS
        tool_schemas = self._registry.get_schemas()

        for step in plan:
            step_number = int(step.get("step_number") or len(completed_steps) + 1)
            description = str(step.get("description") or "").strip()
            if not description:
                continue

            log.info(
                "step_execute_start",
                session_id=session_id,
                step=step_number,
                description=description,
                remaining_tool_rounds=remaining_rounds,
            )

            step_messages = self._build_step_messages(
                user_input=user_input,
                memory_cue=memory_cue,
                history=history,
                step_number=step_number,
                description=description,
                completed_steps=completed_steps,
            )

            per_step_budget = min(MAX_TOOL_ROUNDS_PER_STEP, max(0, remaining_rounds))
            step_result, rounds_used = self._run_react(
                session_id=session_id,
                messages=step_messages,
                tool_schemas=tool_schemas,
                max_rounds=per_step_budget,
            )
            remaining_rounds -= rounds_used

            completed_steps.append({
                "step_number": step_number,
                "description": description,
                "result": step_result,
            })
            log.info(
                "step_execute_done",
                session_id=session_id,
                step=step_number,
                rounds_used=rounds_used,
                remaining_tool_rounds=remaining_rounds,
            )

            if remaining_rounds <= 0 and step is not plan[-1]:
                log.warning(
                    "global_tool_round_budget_exhausted",
                    session_id=session_id,
                    completed=len(completed_steps),
                    planned=len(plan),
                )
                break

        # ── 4. Synthesize phase ────────────────────────────────────────────────
        final_text = self._synthesize(
            session_id=session_id,
            user_input=user_input,
            memory_cue=memory_cue,
            history=history,
            completed_steps=completed_steps,
        )
        log.info(
            "response_ready",
            session_id=session_id,
            steps_completed=len(completed_steps),
        )
        return final_text

    def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        return self._pending_confirmations.get(session_id)

    def handle_confirmation(self, session_id: str, confirmed: bool) -> str:
        pending = self._pending_confirmations.pop(session_id, None)
        if not pending:
            return "No pending actions to confirm or deny."
        
        tool_name = pending["tool_name"]
        tool_args = pending["tool_args"]
        tool_call_id = pending["tool_call_id"]
        
        if not confirmed:
            result = f"User denied execution of {tool_name}."
        else:
            result = self._registry.dispatch(tool_name, tool_args)
            
        tool_result_message: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
        self._store.save_message(session_id, tool_result_message)
        
        # We need to trigger synthesize to finish the loop, or just tell the user.
        # But this is a simple CLI, we just return the result for now.
        return f"Executed {tool_name}. Result: {result}"

    # ── Step helpers ───────────────────────────────────────────────────────────

    def _build_step_messages(
        self,
        *,
        user_input: str,
        memory_cue: str,
        history: list[dict[str, Any]],
        step_number: int,
        description: str,
        completed_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Focused message list for one plan step."""
        prior = _format_completed_steps(completed_steps)
        step_brief = (
            f"You are executing step {step_number} of a multi-step plan.\n"
            f"Step description: {description}\n\n"
            "Complete ONLY this step. Use tools if needed. "
            "When done, reply with a concise result for this step."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": settings.system_prompt},
            {"role": "system", "content": memory_cue},
            *history,
            {"role": "user", "content": f"Original request:\n{user_input}"},
            {"role": "system", "content": step_brief},
        ]
        if prior:
            messages.append({
                "role": "system",
                "content": f"Results from previously completed steps:\n{prior}",
            })
        return messages

    def _run_react(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        max_rounds: int,
    ) -> tuple[str, int]:
        """
        Mini ReAct loop for a single plan step.

        Returns:
            (final_text, rounds_used) where rounds_used counts LLM calls that
            requested tools (or forced fallback rounds).
        """
        if max_rounds <= 0:
            # No tool budget left — one tool-free call for a best-effort answer.
            response = chat_completion(messages=messages, tools=None)
            assistant_message = response.choices[0].message
            assistant_dict = _message_to_dict(assistant_message)
            self._store.save_message(session_id, assistant_dict)
            return (assistant_message.content or ""), 0

        rounds_used = 0
        for round_num in range(max_rounds):
            log.info(
                "llm_call",
                session_id=session_id,
                round=round_num + 1,
                phase="execute_step",
            )
            response = chat_completion(messages=messages, tools=tool_schemas)
            assistant_message = response.choices[0].message
            assistant_dict = _message_to_dict(assistant_message)
            tool_calls = getattr(assistant_message, "tool_calls", None)

            if not tool_calls:
                self._store.save_message(session_id, assistant_dict)
                return (assistant_message.content or ""), rounds_used

            rounds_used += 1
            log.info(
                "tool_calls_requested",
                count=len(tool_calls),
                round=round_num + 1,
            )
            self._store.save_message(session_id, assistant_dict)
            messages.append(assistant_dict)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                tool_call_id = tool_call.id
                result = self._dispatch_with_permissions(session_id, tool_name, tool_args, tool_call_id)

                tool_result_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
                self._store.save_message(session_id, tool_result_message)
                messages.append(tool_result_message)

        # Step budget exhausted — force a text wrap-up without tools.
        log.warning("max_tool_rounds_exceeded", session_id=session_id, phase="step")
        final_response = chat_completion(messages=messages, tools=None)
        final_message = final_response.choices[0].message
        final_dict = _message_to_dict(final_message)
        self._store.save_message(session_id, final_dict)
        return (
            final_message.content or "ERROR: Step could not be completed within tool budget.",
            rounds_used,
        )

    def _dispatch_with_permissions(self, session_id: str, tool_name: str, tool_args: str, tool_call_id: str) -> str:
        """Run PermissionGuard checks, then registry dispatch."""
        risk_level = self._registry.get_tool_risk_level(tool_name)

        if getattr(settings, "REQUIRE_CONFIRMATION_FOR_HIGH_RISK", False) and self._guard.require_confirmation(tool_name, risk_level):
            log.warning(
                "tool_requires_confirmation",
                tool=tool_name,
                risk_level=risk_level,
            )
            self._pending_confirmations[session_id] = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_call_id": tool_call_id,
                "risk_level": risk_level
            }
            return f"ACTION_REQUIRES_CONFIRMATION: This action ({tool_name}) requires explicit user approval. Please confirm to proceed."

        if not self._guard.is_allowed(tool_name, risk_level):
            log.warning(
                "tool_blocked",
                tool=tool_name,
                risk_level=risk_level,
            )
            return (
                f"ERROR: Tool '{tool_name}' is not permitted "
                f"(Risk Level: {risk_level})."
            )

        return self._registry.dispatch(tool_name, tool_args)

    def _synthesize(
        self,
        *,
        session_id: str,
        user_input: str,
        memory_cue: str,
        history: list[dict[str, Any]],
        completed_steps: list[dict[str, Any]],
    ) -> str:
        """Final tool-free call that turns step results into the user answer."""
        steps_blob = _format_completed_steps(completed_steps) or "(no steps completed)"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": settings.system_prompt},
            {"role": "system", "content": memory_cue},
            *history,
            {"role": "user", "content": f"Original request:\n{user_input}"},
            {
                "role": "system",
                "content": (
                    f"{_SYNTHESIZE_PROMPT}\n\n"
                    f"Executed steps and results:\n{steps_blob}"
                ),
            },
        ]

        log.info("llm_call", session_id=session_id, phase="synthesize")
        response = chat_completion(messages=messages, tools=None)
        assistant_message = response.choices[0].message
        assistant_dict = _message_to_dict(assistant_message)
        self._store.save_message(session_id, assistant_dict)
        return (
            assistant_message.content
            or "I ran into an issue completing that request."
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_context_cue(history: list[dict[str, Any]], memory_cue: str) -> str:
    """Compact context string for the planner (not full message objects)."""
    snippets: list[str] = []
    for msg in history[-6:]:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if msg.get("tool_calls"):
            content = "[tool call]"
        if not content:
            continue
        snippets.append(f"{role}: {content[:240]}")
    recent = "\n".join(snippets) if snippets else "(no prior messages)"
    return f"{memory_cue}\nRecent conversation:\n{recent}"


def _format_completed_steps(completed_steps: list[dict[str, Any]]) -> str:
    if not completed_steps:
        return ""
    lines: list[str] = []
    for step in completed_steps:
        lines.append(
            f"Step {step.get('step_number')}: {step.get('description')}\n"
            f"Result: {step.get('result')}"
        )
    return "\n\n".join(lines)


def _message_to_dict(message: Any) -> dict[str, Any]:
    """
    Convert a LiteLLM/OpenAI Message object to a plain dict.
    We need plain dicts for SQLite storage and for appending to the messages list.
    """
    d: dict[str, Any] = {"role": message.role}

    if message.content is not None:
        d["content"] = message.content

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    return d
