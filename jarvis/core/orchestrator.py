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

import json
import re
import time
from collections.abc import Callable
from typing import Any

from jarvis.config import settings
from jarvis.core.permissions import PermissionGuard
from jarvis.core.planner import Planner
from jarvis.llm.client import chat_completion
from jarvis.memory.context_manager import ContextManager, estimate_tokens
from jarvis.memory.session_store import (
    ACTION_STATE_FAILED,
    ACTION_STATE_PENDING,
    ACTION_STATE_SUCCEEDED,
    ACTION_STATE_UNKNOWN,
    SessionStore,
    new_owner_token,
)
from jarvis.tools.registry import ToolRegistry
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Global cap on tool-call rounds for one user message (across all plan steps).
MAX_TOOL_ROUNDS = 5
# Per-step ReAct budget (also clipped by remaining global budget).
MAX_TOOL_ROUNDS_PER_STEP = 2
# Max additional recovery rounds the LLM gets after a tool returns an error.
MAX_SELF_CORRECTION_ATTEMPTS = 2
# Marker returned when a step paused for a durable confirmation. Interfaces
# surface this to the user instead of a final answer; the turn resumes via
# ``handle_confirmation`` after approve/deny.
PAUSED_FOR_CONFIRMATION = (
    "ACTION_REQUIRES_CONFIRMATION: This action requires your approval. "
    "Approve or deny it and the task will continue from where it paused."
)
# Keys of the durable resume context stored alongside a pending confirmation.
_RESUME_CONTEXT_VERSION = 1

_SYNTHESIZE_PROMPT = (
    "Synthesize a single, natural, and concise final response based on the completed steps. "
    "CRITICAL: Do NOT repeat raw tool outputs, step descriptions, or previous text verbatim. "
    "Speak naturally and keep it brief. "
    "If the user asks about their memory or personal facts, you MUST rely ONLY on the results from "
    "the `recall_facts` tool. Do not make up or guess any facts.\n\n"
    "Answer-quality rules:\n"
    "- Lead with the answer to the user's actual question, not with process narration.\n"
    "- Quote numbers, names, and dates exactly as they appear in the step results.\n"
    "- If some steps failed or produced errors, say briefly what could not be done; never paper over failures.\n"
    "- If the steps did not produce enough information, say what is missing instead of inventing content."
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
        store: SessionStore,
        tool_registry: ToolRegistry,
        permission_guard: PermissionGuard,
    ) -> None:
        self._store = store
        self._registry = tool_registry
        self._guard = permission_guard
        self._planner = Planner(
            llm_client=chat_completion,
            tool_names=tool_registry.list_tools(),
        )
        self._context = ContextManager()
        # "simple" | "complex" for the turn currently executing on this
        # orchestrator; captured into the durable pause context so resume
        # knows whether the paused step came from the fast path or a plan.
        self._current_mode = "complex"

    def route_intent(self, user_input: str) -> str:
        """
        Heuristic zero-cost intent router.

        Known trade-off (see AGENTS.md): keyword matching has false positives
        (complex queries containing simple keywords) and false negatives
        (simple queries missing keywords). The rules below prioritize:
          - AVOID routing genuinely multi-step requests to the fast path.
          - Allow short greetings and single-fact questions through cheaply.
        """
        text = user_input.lower().strip()

        # Multi-step signals always go through the planner.
        complex_triggers = [
            " and then ", " and also ", "after that", "first... then",
            "step 1", "step one", "write a", "write an ", "create a",
            "create an ", "make me", "build me",
        ]
        if any(c in text for c in complex_triggers):
            return "complex"

        # A bare "then" / "next" / "also" as a connecting word usually means
        # the user is chaining tasks — planner territory.
        if re.search(r"\b(then|next|afterwards)\b", text) and len(text.split()) > 6:
            return "complex"

        # Multiple sentences frequently mean multiple asks.
        sentence_count = len([s for s in re.split(r"[.!?]+\s", text) if s.strip()])
        if sentence_count >= 3:
            return "complex"

        # Two substantive sentences usually carry two asks (e.g. a math question
        # plus an unrelated instruction). A minimum length keeps short chatty
        # pairs like "hey there. what time is it?" on the cheap fast path.
        if sentence_count >= 2 and len(text.split()) >= 12:
            return "complex"

        # Single-intent fast path: short inputs containing one simple intent.
        simple_keywords = [
            "what time", "time is it", "what date", "calculate", "compute",
            "hello", "hi", "hey", "who are you", "what can you do",
            "+", "-", "*", "/", "search", "remember", "date", "today",
            "weather",
        ]
        if (
            any(k in text for k in simple_keywords)
            and len(text) < 150
            and sentence_count <= 2
        ):
            return "simple"

        return "complex"

    def chat(
        self,
        session_id: str,
        user_input: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """
        Process one user message via Plan → Execute → Synthesize.

        Args:
            session_id: The ID of the current conversation session.
            user_input: The raw text typed by the user.
            on_event:   Optional observer callback receiving dict events
                ({"type": "intent"|"plan"|"step_start"|"step_done"|"synthesis",
                ...}) as the request progresses. Used by the SSE endpoint and
                for structured tracing. Callback exceptions are swallowed —
                observation must never break execution.

        Returns:
            The assistant's final plain-text response string.
        """
        # ── 1. Persist the user turn ───────────────────────────────────────────
        user_message: dict[str, Any] = {"role": "user", "content": user_input}
        self._store.save_message(session_id, user_message)

        # Fetch a WIDER window than the LLM will see: the ContextManager needs
        # dropped turns to build the anchor + rolling summary. The prompt itself
        # stays bounded by max_context_messages regardless of session length.
        history = self._store.load_history(
            session_id, limit=settings.max_context_messages * 3
        )
        memory_cue = (
            "Short-term memory: recent messages from this session are included "
            "for reference. Use them to resolve follow-ups."
        )
        context_cue = _build_context_cue(history, memory_cue)

        # ── 1.5. Intent Routing ────────────────────────────────────────────────
        intent = self.route_intent(user_input)
        log.info("intent_routed", intent=intent, bypassed_planner=(intent == "simple"))
        _emit(on_event, type="intent", intent=intent)
        self._current_mode = "simple" if intent == "simple" else "complex"

        tool_schemas = self._registry.get_schemas()
        request_started = time.perf_counter()

        if intent == "simple":
            messages = self._context.build_messages(
                system_prompts=[
                    {"role": "system", "content": settings.system_prompt},
                    {"role": "system", "content": memory_cue},
                ],
                history=history,
                user_input=user_input,
            )
            final_text, _ = self._run_react(
                session_id=session_id,
                messages=messages,
                tool_schemas=tool_schemas,
                max_rounds=2,
                on_event=on_event,
                pause_context={
                    # Fast-path turns park too: the approved action IS the
                    # whole task, so resume goes straight to synthesis.
                    "original_request": user_input,
                    "memory_cue": memory_cue,
                    "step_number": 1,
                    "pending_plan": [],
                    "completed_steps": [],
                    "remaining_rounds": 2,
                },
            )
            log.info(
                "response_ready",
                session_id=session_id,
                steps_completed=1,
                duration_ms=round((time.perf_counter() - request_started) * 1000, 1),
            )
            return final_text

        # ── 2. Plan phase ──────────────────────────────────────────────────────
        plan = self._planner.generate_plan(user_input, context_cue)
        log.info(
            "plan_ready",
            session_id=session_id,
            steps=len(plan),
            est_context_tokens=estimate_tokens(history),
            plan=[{
                "step": s.get("step_number"),
                "description": s.get("description"),
                "tools": s.get("required_tools"),
            } for s in plan],
        )
        _emit(
            on_event,
            type="plan",
            steps=[{
                "step_number": s.get("step_number"),
                "description": s.get("description"),
                "tools": s.get("required_tools"),
            } for s in plan],
        )

        # ── 3. Execute phase ───────────────────────────────────────────────────
        completed_steps: list[dict[str, Any]] = []
        remaining_rounds = MAX_TOOL_ROUNDS

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

            _emit(on_event, type="step_start", step_number=step_number, description=description)

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
                on_event=on_event,
                pause_context={
                    "original_request": user_input,
                    "memory_cue": memory_cue,
                    "step_number": step_number,
                    "pending_plan": plan,
                    "completed_steps": completed_steps,
                    "remaining_rounds": remaining_rounds,
                },
            )
            # The step parked a high-risk action: persist resume state is
            # already done (inside _dispatch...), so stop and hand control to
            # the user WITHOUT treating this as a completed step.
            if step_result == PAUSED_FOR_CONFIRMATION:
                log.info(
                    "turn_paused_for_confirmation",
                    session_id=session_id,
                    step=step_number,
                )
                _emit(on_event, type="paused", step_number=step_number)
                return PAUSED_FOR_CONFIRMATION
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
            _emit(on_event, type="step_done", step_number=step_number, rounds_used=rounds_used)

            if remaining_rounds <= 0 and step is not plan[-1]:
                log.warning(
                    "global_tool_round_budget_exhausted",
                    session_id=session_id,
                    completed=len(completed_steps),
                    planned=len(plan),
                )
                break

        # ── 4. Synthesize phase ────────────────────────────────────────────────
        _emit(on_event, type="synthesis")
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
            duration_ms=round((time.perf_counter() - request_started) * 1000, 1),
        )
        return final_text

    def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        return self._store.load_pending_confirmation(session_id)

    # ── Confirmation continuation (resume path) ─────────────────────────────

    def handle_confirmation(self, session_id: str, confirmed: bool) -> str:
        """
        Resolve the session's pending high-risk action and RESUME the paused
        turn when a durable resume context exists.

        Flow (approve): pop the confirmation → claim the action's execution-
        ledger row (at-most-once) → dispatch the tool → persist the result
        as a `tool` message → restore the pause context → execute remaining
        plan steps → synthesize a final answer.

        Flow (deny): record the denial in the ledger (no execution attempt)
        → persist a denial tool message → restore context and continue, so
        the agent can explain what was NOT done and still deliver the rest
        of the original request.

        Duplicate safety (v0.17): the confirmation pop and the ledger claim
        are both atomic, so two concurrent approvals cannot both dispatch;
        a repeat approval after the action reached a terminal state reports
        the recorded outcome instead of re-executing; an UNKNOWN action
        (possible execution whose result was never durably recorded) is
        never automatically re-executed.

        States handled: approval, denial, tool failure (the error flows into
        synthesis like any other tool error), expired confirmation (None →
        plain message), duplicate resolution (last recorded outcome is
        reported), missing/corrupt context (legacy rows → behave like the
        pre-v0.15 raw-result reply), and restart between park and resolve
        (everything needed is in SQLite).
        """
        pending = self._store.complete_pending_confirmation(session_id)
        if not pending:
            return self._duplicate_resolution_report(session_id)

        tool_name = pending["tool_name"]
        tool_args = pending["tool_args"]
        tool_call_id = pending["tool_call_id"]
        context = pending.get("context") or {}
        confirmation_id = str(pending.get("confirmation_id") or "")

        # ── 1. Resolve the action itself (never raises) ────────────────────
        if not confirmed:
            result = f"User denied execution of {tool_name}."
            self._record_denial(confirmation_id, tool_name, result)
        else:
            result, action_id, ambiguous = self._execute_protected_action(
                session_id, confirmation_id, tool_name, tool_args
            )
            if ambiguous:
                # Crash ambiguity: the side effect may or may not have
                # happened. Never auto-retry; never resume the plan on a
                # guessed result. State is visible in the ledger.
                return self._unknown_action_report(session_id, action_id, tool_name)

        tool_result_message: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
        self._store.save_message(session_id, tool_result_message)

        # ── 2. Resume only with a usable durable context ───────────────────
        original_request = str(context.get("original_request") or "").strip()
        if not original_request:
            # Legacy row (parked before v0.15) or corrupt context: keep the
            # historical raw-result behavior rather than fail.
            verb = "Executed" if confirmed else "Denied"
            return f"{verb} {tool_name}. Result: {result}"

        return self._resume_paused_turn(session_id, context, tool_name, result, confirmed)

    # ── Action ledger integration (v0.17) ──────────────────────────────────

    def _execute_protected_action(
        self,
        session_id: str,
        confirmation_id: str,
        tool_name: str,
        tool_args: str,
    ) -> tuple[str, str | None, bool]:
        """
        Dispatch a protected action at-most-once via the execution ledger.

        Returns:
            (result, action_id, ambiguous)

            - ``claimed``: the caller owns the single execution attempt; the
              tool was dispatched and the outcome (SUCCEEDED/FAILED) durably
              recorded before returning.
            - ``already_running``: another resolver owns the attempt; a clear
              message is returned and nothing is dispatched here.
            - terminal states: the recorded outcome is returned instead of
              re-executing. ``ambiguous=True`` (UNKNOWN state) means the side
              effect may already have happened — the caller must not resume
              the turn automatically.
        """
        action = (
            self._store.get_action_execution_by_confirmation(confirmation_id)
            if confirmation_id
            else None
        )
        if action is None:
            # Legacy row parked before v0.17 (no ledger pair): historical
            # direct-dispatch behavior. New parks always have a ledger row.
            log.warning(
                "action_ledger_missing_legacy_dispatch",
                session_id=session_id,
                tool_name=tool_name,
            )
            return self._registry.dispatch(tool_name, tool_args), None, False

        owner = new_owner_token("action")
        claim = self._store.claim_action_execution(action.action_id, owner)
        log.info(
            "action_claim_result",
            session_id=session_id,
            action_id=action.action_id,
            tool=tool_name,
            claim=claim,
        )

        if claim == "claimed":
            try:
                result = self._registry.dispatch(tool_name, tool_args)  # ERROR:... on failure
            except Exception as e:  # a crash here must never strand RUNNING
                result = f"ERROR: dispatch raised {type(e).__name__}: {e}"
            state = (
                ACTION_STATE_FAILED
                if _is_tool_error(result)
                else ACTION_STATE_SUCCEEDED
            )
            self._store.finish_action_execution(action.action_id, state, result)
            return result, action.action_id, False

        if claim == "already_running":
            # Defense in depth: the atomic confirmation pop should normally
            # make this unreachable, but a second resolver (or a crashed
            # claim that recovery has not yet swept) must not double-run.
            return (
                f"ERROR: Action '{tool_name}' is already being executed by "
                "another request; no duplicate dispatch was performed.",
                action.action_id,
                False,
            )

        state = claim.split(":", 1)[1]
        current = self._store.get_action_execution(action.action_id) or action
        if state == ACTION_STATE_UNKNOWN:
            log.warning(
                "action_blocked_unknown_state",
                session_id=session_id,
                action_id=action.action_id,
                tool=tool_name,
            )
            return "", action.action_id, True

        # SUCCEEDED / FAILED: report the durably recorded outcome, no rerun.
        word = "succeeded" if state == ACTION_STATE_SUCCEEDED else "failed"
        recorded = str(current.result or "(no result recorded)")
        return (
            f"[Action already {word}; recorded result] {recorded}",
            action.action_id,
            False,
        )

    def _record_denial(self, confirmation_id: str, tool_name: str, result: str) -> None:
        """Close the ledger row for a denied action (no execution attempt)."""
        if not confirmation_id:
            return
        action = self._store.get_action_execution_by_confirmation(confirmation_id)
        if action is not None and action.state == ACTION_STATE_PENDING:
            self._store.finish_action_execution(action.action_id, ACTION_STATE_FAILED, result)
            log.info(
                "action_denied",
                action_id=action.action_id,
                tool=tool_name,
            )

    def _duplicate_resolution_report(self, session_id: str) -> str:
        """Deterministic response when there is nothing left to resolve."""
        last = self._store.get_last_action_execution(session_id)
        if last is None:
            return "No pending actions to confirm or deny."
        if last.state == ACTION_STATE_UNKNOWN:
            return self._unknown_action_report(session_id, last.action_id, last.tool_name)
        if last.state == ACTION_STATE_SUCCEEDED:
            return (
                f"No pending actions. The most recent protected action "
                f"({last.tool_name}) already succeeded and was not executed "
                f"again. Recorded result: {str(last.result or '')[:500]}"
            )
        if last.state == ACTION_STATE_FAILED:
            return (
                f"No pending actions. The most recent protected action "
                f"({last.tool_name}) did not execute successfully "
                f"(state: FAILED); it was not re-executed. "
                f"Recorded outcome: {str(last.result or '')[:500]}"
            )
        return "No pending actions to confirm or deny."

    def _unknown_action_report(self, session_id: str, action_id: str | None, tool_name: str) -> str:
        """User-facing status for crash-ambiguous (UNKNOWN) actions."""
        aid = f" (action {action_id})" if action_id else ""
        return (
            f"ACTION_EXECUTION_STATE_UNKNOWN{aid}: '{tool_name}' may or may "
            "not have executed before an interruption, and its result was "
            "never durably recorded. To prevent a duplicate side effect, "
            "automatic retry was prevented and the paused task was not "
            "resumed. Inspect the ledger row and resolve explicitly "
            "(re-issue the original request when you have confirmed the "
            "external state)."
        )

    def _resume_paused_turn(
        self,
        session_id: str,
        context: dict[str, Any],
        tool_name: str,
        tool_result: str,
        confirmed: bool,
    ) -> str:
        """
        Continue a turn that was paused by a confirmation.

        Restores the plan state from the durable context, injects the paused
        step's outcome, executes the REMAINING steps, and synthesizes the
        final user-facing answer. Reuses _run_react/_synthesize — no
        duplicated orchestrator.
        """
        memory_cue = str(context.get("memory_cue") or "")
        original_request = str(context["original_request"])
        plan: list[dict[str, Any]] = list(context.get("pending_plan") or [])
        completed_steps: list[dict[str, Any]] = list(context.get("completed_steps") or [])
        remaining_rounds = int(context.get("remaining_rounds") or MAX_TOOL_ROUNDS)
        paused_step = int(context.get("step_number") or 0)
        mode = str(context.get("mode") or "complex")

        history = self._store.load_history(
            session_id, limit=settings.max_context_messages * 3
        )
        tool_schemas = self._registry.get_schemas()

        # The paused step's outcome (approval result, denial, or tool error)
        # becomes a completed step so synthesis and later steps can use it.
        outcome_note = (
            f"User approved execution of {tool_name}."
            if confirmed and not _is_tool_error(tool_result)
            else f"User DENIED execution of {tool_name}."
            if not confirmed
            else f"User approved {tool_name}, but execution failed."
        )
        if paused_step:
            completed_steps.append({
                "step_number": paused_step,
                "description": f"(paused for confirmation) {outcome_note}",
                "result": tool_result,
            })

        # Find the steps that come AFTER the paused one.
        remaining = [
            step for step in plan
            if int(step.get("step_number") or 0) > paused_step
            and str(step.get("description") or "").strip()
        ]

        log.info(
            "turn_resuming_after_confirmation",
            session_id=session_id,
            paused_step=paused_step,
            remaining_steps=len(remaining),
            approved=confirmed,
            tool_ok=not _is_tool_error(tool_result),
        )

        for step in remaining:
            step_number = int(step.get("step_number") or (len(completed_steps) + 1))
            description = str(step.get("description")).strip()
            _emit(None, type="step_start", step_number=step_number, description=description)

            step_messages = self._build_step_messages(
                user_input=original_request,
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
                # Nested pauses during resumed steps re-park with fresh state.
                pause_context={
                    "original_request": original_request,
                    "memory_cue": memory_cue,
                    "step_number": step_number,
                    "pending_plan": plan,
                    "completed_steps": completed_steps,
                    "remaining_rounds": remaining_rounds,
                    "mode": mode,
                },
            )
            if step_result == PAUSED_FOR_CONFIRMATION:
                log.info("turn_paused_again_during_resume", session_id=session_id, step=step_number)
                return PAUSED_FOR_CONFIRMATION
            remaining_rounds -= rounds_used
            completed_steps.append({
                "step_number": step_number,
                "description": description,
                "result": step_result,
            })
            if remaining_rounds <= 0:
                log.warning("resume_tool_budget_exhausted", session_id=session_id)
                break

        # ── Final synthesis over the whole (original + resumed) task ────────
        if mode == "simple" and not remaining:
            # Fast-path turn: the approved/denied action WAS the whole task.
            # The tool/denial message is already in history; one tool-free
            # call turns it into a natural answer.
            return self._synthesize(
                session_id=session_id,
                user_input=original_request,
                memory_cue=memory_cue,
                history=history,
                completed_steps=completed_steps,
            )

        return self._synthesize(
            session_id=session_id,
            user_input=original_request,
            memory_cue=memory_cue,
            history=history,
            completed_steps=completed_steps,
        )

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
            "When done, reply with a concise result for this step. "
            "Do not start the next step; do not answer the overall request yet."
        )
        messages: list[dict[str, Any]] = self._context.build_messages(
            system_prompts=[
                {"role": "system", "content": settings.system_prompt},
                {"role": "system", "content": memory_cue},
            ],
            history=history,
            # The current user turn is the step brief itself, so the window
            # budget is spent on real conversation history instead.
            user_input=f"Original request:\n{user_input}\n\n{step_brief}",
        )
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
        on_event: Callable[[dict[str, Any]], None] | None = None,
        pause_context: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """
        Mini ReAct loop for a single plan step.

        Includes a self-correction sub-loop: when a tool returns an error, the
        loop feeds the observation back to the LLM and grants up to
        MAX_SELF_CORRECTION_ATTEMPTS additional rounds so the agent can recover
        (e.g., fix arguments, call a different tool, or answer without the tool).

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
        consecutive_errors = 0  # track back-to-back tool failures for correction

        for round_num in range(max_rounds + MAX_SELF_CORRECTION_ATTEMPTS):
            # ── Hard stop: global round budget is the absolute ceiling ───────────
            if rounds_used >= max_rounds + MAX_SELF_CORRECTION_ATTEMPTS:
                break

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
                # LLM chose to answer directly (possibly after recovering from an error)
                self._store.save_message(session_id, assistant_dict)
                return (assistant_message.content or ""), rounds_used

            rounds_used += 1
            log.info(
                "tool_calls_requested",
                count=len(tool_calls),
                round=round_num + 1,
            )
            _emit(
                on_event,
                type="tool_calls",
                round=round_num + 1,
                tools=[tc.function.name for tc in tool_calls],
            )
            self._store.save_message(session_id, assistant_dict)
            messages.append(assistant_dict)

            # ── Dispatch each tool call and collect results ──────────────────────
            round_had_error = False
            paused = False
            
            import asyncio
            async def _dispatch_all():
                tasks = []
                for tc in tool_calls:
                    tasks.append(
                        self._dispatch_with_permissions_async(
                            session_id,
                            tc.function.name,
                            tc.function.arguments,
                            tc.id,
                            pause_context=pause_context,
                        )
                    )
                return await asyncio.gather(*tasks)

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    results = pool.submit(asyncio.run, _dispatch_all()).result()
            except RuntimeError:
                results = asyncio.run(_dispatch_all())

            for tc, result in zip(tool_calls, results):
                if result == PAUSED_FOR_CONFIRMATION:
                    paused = True
                    break  # durable state saved; stop this round entirely
                if paused:
                    break

                if _is_tool_error(result):
                    round_had_error = True

                # Save full result to DB
                full_tool_result_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result,
                }
                self._store.save_message(session_id, full_tool_result_message)
                
                # Clamp for the active LLM context: head + tail preserved with an
                # explicit omission marker, so scrapes/tracebacks cannot blow up
                # the prompt. The full payload is already saved to SQLite above.
                active_content = self._context.clamp_tool_output(result)
                active_tool_result_message = {
                    **full_tool_result_message,
                    "content": active_content,
                }
                messages.append(active_tool_result_message)

            # ── Pause propagation ────────────────────────────────────────────────
            if paused:
                # Hand PAUSED_FOR_CONFIRMATION up to chat() as this step's
                # result. rounds_used stays as-is: rounds already spent are
                # restored from the durable context on resume.
                return PAUSED_FOR_CONFIRMATION, rounds_used

            # ── Self-correction logic ────────────────────────────────────────────
            if round_had_error:
                consecutive_errors += 1
                if consecutive_errors > MAX_SELF_CORRECTION_ATTEMPTS:
                    log.warning(
                        "self_correction_limit_reached",
                        session_id=session_id,
                        attempts=consecutive_errors,
                    )
                    break  # stop looping; fall through to forced wrap-up

                log.info(
                    "self_correction_attempt",
                    session_id=session_id,
                    attempt=consecutive_errors,
                    max_attempts=MAX_SELF_CORRECTION_ATTEMPTS,
                )
                # Inject a recovery hint so the LLM understands it should try
                # a different approach rather than repeat the same failing call.
                hint: dict[str, Any] = {
                    "role": "system",
                    "content": (
                        f"The previous tool call returned an error "
                        f"(attempt {consecutive_errors} of {MAX_SELF_CORRECTION_ATTEMPTS}). "
                        "Please observe the error above and try to recover: "
                        "fix the arguments, call a different tool, "
                        "or answer the user's question directly without the tool."
                    ),
                }
                messages.append(hint)
                # Do NOT break — continue the loop to give the LLM another go.
            else:
                consecutive_errors = 0  # successful round; reset error counter

        # Step budget (including correction rounds) exhausted — force a wrap-up.
        log.warning("max_tool_rounds_exceeded", session_id=session_id, phase="step")
        final_response = chat_completion(messages=messages, tools=None)
        final_message = final_response.choices[0].message
        final_dict = _message_to_dict(final_message)
        self._store.save_message(session_id, final_dict)
        return (
            final_message.content or "ERROR: Step could not be completed within tool budget.",
            rounds_used,
        )

    def _permission_decision(
        self,
        session_id: str,
        tool_name: str,
        tool_args: str,
        tool_call_id: str,
    ) -> str | None:
        """
        Shared PermissionGuard logic for sync/async dispatch.

        Returns:
            A refusal/confirmation string when the call must NOT execute
            (durable confirmation parked, or guard blocked), or None when the
            dispatch may proceed.
        """
        risk_level = self._registry.get_tool_risk_level(tool_name)

        if getattr(settings, "REQUIRE_CONFIRMATION_FOR_HIGH_RISK", False) and self._guard.require_confirmation(tool_name, risk_level):
            log.warning(
                "tool_requires_confirmation",
                tool=tool_name,
                risk_level=risk_level,
            )
            self._store.save_pending_confirmation(
                session_id=session_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                risk_level=risk_level,
            )
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

        return None

    def _dispatch_with_permissions(self, session_id: str, tool_name: str, tool_args: str, tool_call_id: str) -> str:
        """Run PermissionGuard checks, then registry dispatch."""
        decision = self._permission_decision(session_id, tool_name, tool_args, tool_call_id)
        if decision is not None:
            return decision
        return self._registry.dispatch(tool_name, tool_args)

    async def _dispatch_with_permissions_async(
        self,
        session_id: str,
        tool_name: str,
        tool_args: str,
        tool_call_id: str,
        pause_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Run PermissionGuard checks, then registry dispatch asynchronously.

        When the guard parks a confirmation, ``pause_context`` (the durable
        resume state) is persisted alongside it so the turn can CONTINUE
        after the user resolves the action — even across a restart.
        """
        risk_level = self._registry.get_tool_risk_level(tool_name)

        if getattr(settings, "REQUIRE_CONFIRMATION_FOR_HIGH_RISK", False) and self._guard.require_confirmation(tool_name, risk_level):
            log.warning(
                "tool_requires_confirmation",
                tool=tool_name,
                risk_level=risk_level,
            )
            pause_ctx = {**({} if pause_context is None else pause_context),
                         "mode": self._current_mode}
            confirmation_id = self._store.save_pending_confirmation(
                session_id=session_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                risk_level=risk_level,
                context=pause_ctx,
            )
            log.info(
                "action_parked",
                session_id=session_id,
                tool=tool_name,
                confirmation_id=confirmation_id,
            )
            return PAUSED_FOR_CONFIRMATION

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

        return await self._registry.dispatch_async(tool_name, tool_args)

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
        # Step results can embed huge tool outputs; clamp them so the final
        # synthesis call cannot blow up the context window either.
        clamped_steps = [
            {**step, "result": self._context.clamp_tool_output(str(step.get("result") or ""))}
            for step in completed_steps
        ]
        steps_blob = _format_completed_steps(clamped_steps) or "(no steps completed)"
        messages: list[dict[str, Any]] = self._context.build_messages(
            system_prompts=[
                {"role": "system", "content": settings.system_prompt},
                {"role": "system", "content": memory_cue},
            ],
            history=history,
            user_input=f"Original request:\n{user_input}",
        )
        messages.append({
            "role": "system",
            "content": (
                f"{_SYNTHESIZE_PROMPT}\n\n"
                f"Executed steps and results:\n{steps_blob}"
            ),
        })

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

def _emit(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    """
    Deliver an observation event to the callback, swallowing any callback
    exception — observation must never break agent execution.
    """
    if callback is None:
        return
    try:
        callback(event)
    except Exception as e:  # noqa: BLE001 - observer isolation is the point
        log.warning("event_observer_failed", error=str(e))


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


def _is_tool_error(result: str) -> bool:
    """
    Return True if a tool result string represents a failure.

    Convention: all tool errors in JARVIS start with "ERROR:" or
    "ACTION_REQUIRES_CONFIRMATION:". A confirmation request is not a
    failure for self-correction purposes — the loop should simply wait.
    Only "ERROR:" prefixed results trigger the recovery logic.
    """
    return isinstance(result, str) and result.startswith("ERROR:")

