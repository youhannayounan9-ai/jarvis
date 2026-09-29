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
import uuid
import time
from collections.abc import Callable
from typing import Any

from jarvis.config import settings
from jarvis.core.permissions import PermissionGuard
from jarvis.core.planner import Planner
from jarvis.core.dispatch_guard import DispatchLedger
from jarvis.core.plan_quality import report_plan_quality
from jarvis.core.plan_validator import validate_plan
from jarvis.core.tool_policy import (
    build_tool_policy_block,
    detect_unmet_capability,
    extract_arithmetic,
    is_single_intent_obligation,
    narrow_schemas_for_react,
)
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
            # v0.22: the planner plans by CAPABILITY, so it sees what each
            # tool does (compact one-liners), not just bare names.
            tool_descriptions={
                t.name: t.description for t in tool_registry._tools.values()
            },
        )
        self._context = ContextManager()
        # "simple" | "complex" for the turn currently executing on this
        # orchestrator; captured into the durable pause context so resume
        # knows whether the paused step came from the fast path or a plan.
        self._current_mode = "complex"
        # v0.23 (Parts C/E): per-turn ledger of successful dispatch
        # fingerprints. One user request = one ledger; a resumed turn
        # rebuilds from persisted state instead of carrying stale entries.
        self._dispatch_ledger = DispatchLedger()

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

        # v0.22 (Part L): a compute/memory conjunction is a true TWO-capability
        # task ('Calculate X and remember the result') — the fast path cannot
        # carry it (the single-intent net is disabled by the conjunction, and
        # the live v0.21 eval failed exactly this shape). Send to planner.
        if re.search(
            r"\b(calculate|compute)\b.*\band\s+(remember|save|store|note)\b"
            r"|\b(remember|note|save)\b.*\band\s+(calculate|compute)\b",
            text,
        ):
            return "complex"

        # v0.22 (Part L): two RETRIEVAL capabilities in one request
        # ('search my documents and compare with the web') need ordered
        # multi-tool execution — planner territory.
        if re.search(
            r"\b(search|find|look\s*up|check)\b[^.!?]*\band\b[^.!?]*"
            r"\b(compare|contrast|cross-?check)\b"
            r"|\bcompare\b[^.!?]*\bwith\b[^.!?]*\b(knowledge|document|note|web|search)\b",
            text,
        ):
            return "complex"

        # v0.23: retrieval + memory conjunction ('find what my roadmap says
        # about X and remember it') is a two-capability task — planner.
        if re.search(
            r"\b(search|find|look\s*up|check)\b[^.!?]*\band\s+(remember|save|store|note)\b"
            r"|\b(remember|save|store|note)\b[^.!?]*\bwhat\b[^.!?]*\b(say|says|said|found|find)\b",
            text,
        ):
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
            # v0.22: operator keywords ('+', '-', ...) must not substring-match
            # inside words — 'plan-and-execute' contains '-' and previously
            # routed ANY hyphenated text to the fast path. Arithmetic without
            # operator keywords is still caught by the digit-operator pattern.
            any(k in text for k in simple_keywords if k not in "+-*/")
            or re.search(r"\d\s*[-+*/^]\s*\d", text)
        ) and (
            len(text) < 150
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
        # v0.23: fresh repeat-semantics ledger for this turn. (Thread-level
        # isolation is provided by the session lease; the ledger is per-turn
        # by construction because chat() replaces it at entry.)
        self._dispatch_ledger = DispatchLedger()

        # v0.21 capability-aware tool policy (kill switch restores v0.20):
        # the contract block teaches WHEN tools apply; the fast-path safety
        # net forces one tool round when a single-intent request clearly
        # needs an obligation capability (calculator / search_knowledge).
        # Telemetry (tool_policy_applied) records WHY tool rounds were or
        # were not available, so policy misses are measurable from logs.
        policy_on = not bool(getattr(settings, "JARVIS_DISABLE_TOOL_POLICY", False))
        force_tool_round = False
        forced_tool_name: str | None = None
        tool_policy_block: str | None = None
        if policy_on:
            tool_policy_block = build_tool_policy_block()
            forced_tool = is_single_intent_obligation(user_input, self._registry)
            if forced_tool:
                force_tool_round = True
                forced_tool_name = forced_tool
            log.info(
                "tool_policy_applied",
                intent=intent,
                forced_tool=forced_tool,
                force_tool_round=force_tool_round,
            )
            # SSE consumers see this event ONLY when the safety net changes
            # routing; the decision itself is always in the structured log.
            if force_tool_round:
                _emit(
                    on_event,
                    type="tool_policy",
                    forced_tool=forced_tool,
                    force_tool_round=True,
                )
        unmet_note = detect_unmet_capability(user_input, self._registry)
        # v0.21: which tool the single-intent safety net demands (consumed by
        # the deterministic calculator fallback in _enforce_min_tool_round).
        self._forced_tool_name = forced_tool_name

        tool_schemas = self._registry.get_schemas()
        request_started = time.perf_counter()

        if intent == "simple":
            simple_system_prompts = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "system", "content": memory_cue},
            ]
            if tool_policy_block:
                simple_system_prompts.append(
                    {"role": "system", "content": tool_policy_block}
                )
            fast_path_messages = self._context.build_messages(
                system_prompts=simple_system_prompts,
                history=history,
                user_input=user_input,
            )
            if unmet_note:
                fast_path_messages.append({"role": "system", "content": unmet_note})
            fast_path_rounds = 2 if not force_tool_round else 3
            final_text, _ = self._run_react(
                session_id=session_id,
                messages=fast_path_messages,
                tool_schemas=tool_schemas,
                max_rounds=fast_path_rounds,
                min_rounds=1 if force_tool_round else 0,
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
        raw_plan = self._planner.generate_plan(user_input, context_cue)
        # v0.22 deterministic plan validation (structural only; a plan is a
        # request, never authorization): shape, bounds, registry truth,
        # duplicates, dependency ordering, orphan reasoning steps.
        validation = validate_plan(raw_plan, self._registry.list_tools())
        plan = validation.plan
        log.info(
            "plan_validated",
            session_id=session_id,
            raw_steps=len(raw_plan),
            steps=len(plan),
            issues=validation.issues,
        )
        if not plan:
            log.warning("plan_rejected_empty", session_id=session_id)
            return self._synthesize(
                session_id=session_id,
                user_input=user_input,
                memory_cue=memory_cue,
                history=history,
                completed_steps=[],
            )
        log.info(
            "plan_ready",
            session_id=session_id,
            steps=len(plan),
            est_context_tokens=estimate_tokens(history),
            quality=report_plan_quality(plan, set(self._registry.list_tools())),
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
        # v0.22 (Part E): exact clamped tool evidence carried across steps so
        # a later step can quote real values, not an earlier step's prose.
        step_observations: list[str] = []

        completed_descriptions: set[str] = set()
        failed_step_count = 0
        for step in plan:
            step_number = int(step.get("step_number") or len(completed_steps) + 1)
            description = str(step.get("description") or "").strip()
            if not description:
                continue

            # v0.23 (Part F): a plan step whose task was already completed by
            # an earlier step (non-consecutive duplicates survive the
            # validator's consecutive-dedup) is redundant — record it and
            # skip execution instead of re-running the same work.
            desc_key = " ".join(description.lower().split())
            if desc_key in completed_descriptions:
                log.info(
                    "redundant_plan_step",
                    session_id=session_id,
                    step=step_number,
                    duplicate_of=description[:80],
                )
                _emit(on_event, type="redundant_step", step_number=step_number)
                completed_steps.append({
                    "step_number": step_number,
                    "description": description,
                    "result": "(skipped: identical to an earlier completed step)",
                })
                continue
            completed_descriptions.add(desc_key)

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
                completed_steps=_clamp_step_results(completed_steps, self._context.clamp_tool_output),
                unmet_note=unmet_note,
                observations=list(step_observations),
            )

            per_step_budget = min(MAX_TOOL_ROUNDS_PER_STEP, max(0, remaining_rounds))
            # v0.22 (Parts F/H): a validated step that names required tools
            # must attempt one tool round — the executor's bounded correction
            # when the executor model would otherwise answer the step from
            # memory. (The step's OWN single-intent fallback does not fire
            # here; the forced recovery round does, and tool errors still
            # feed the normal self-correction sub-loop.)
            step_result, rounds_used = self._run_react(
                session_id=session_id,
                messages=step_messages,
                tool_schemas=narrow_schemas_for_react(
                    self._registry,
                    description,
                    step.get("required_tools") or [],
                ),
                max_rounds=per_step_budget,
                min_rounds=1 if step.get("required_tools") else 0,
                on_event=on_event,
                tool_observations=step_observations,
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
            # Part P telemetry: a failed step is visible by state, not guesswork.
            if _is_tool_error(step_result):
                failed_step_count += 1
                log.warning(
                    "plan_step_failed",
                    session_id=session_id,
                    step=step_number,
                    required_tools=step.get("required_tools") or [],
                )
                _emit(on_event, type="step_failed", step_number=step_number)
            else:
                # Part F/P telemetry: the step satisfied its requirement.
                log.info(
                    "plan_step_satisfied",
                    session_id=session_id,
                    step=step_number,
                    required_tools=step.get("required_tools") or [],
                )
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

        # v0.19 full-context recovery: this confirmation may be the reissue
        # of a previously UNKNOWN action (its context was copied verbatim
        # from the original ledger row). The resolution itself needs no
        # special casing — the same at-most-once claim and the same
        # permission flow apply — but the recovery lineage is logged so the
        # audit trail shows the chain original → reissue → resolution.
        recovered_from = str(context.get("recovered_from_action") or "")
        if recovered_from:
            log.info(
                "recovered_action_resolved",
                session_id=session_id,
                recovered_from=recovered_from,
                approved=confirmed,
                confirmation_id=confirmation_id,
            )

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
        # v0.22 (Part E): rebuild the observation ledger from completed steps'
        # tool messages (persisted in the store) so resumed steps keep the
        # same exact-evidence flow as a non-paused turn.
        step_observations: list[str] = [
            self._context.clamp_tool_output(str(m.get("content") or ""))
            for m in history
            if m.get("role") == "tool" and not _is_tool_error(str(m.get("content") or ""))
        ]

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
            # v0.19: label a recovered step truthfully so synthesis can tell
            # the user the task resumed after an explicit reissue of an
            # UNKNOWN action (never silently claim the original succeeded).
            recovered_from = str(context.get("recovered_from_action") or "")
            recovered_label = ", recovered action" if recovered_from else ""
            completed_steps.append({
                "step_number": paused_step,
                "description": f"(paused for confirmation{recovered_label}) {outcome_note}",
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
                completed_steps=_clamp_step_results(completed_steps, self._context.clamp_tool_output),
                observations=list(step_observations),
            )
            per_step_budget = min(MAX_TOOL_ROUNDS_PER_STEP, max(0, remaining_rounds))
            step_result, rounds_used = self._run_react(
                session_id=session_id,
                messages=step_messages,
                tool_schemas=tool_schemas,
                max_rounds=per_step_budget,
                min_rounds=1 if step.get("required_tools") else 0,
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
                tool_observations=step_observations,
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
        unmet_note: str | None = None,
        observations: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Focused message list for one plan step."""
        prior = _format_completed_steps(completed_steps)
        step_brief = (
            f"You are executing step {step_number} of a multi-step plan.\n"
            f"Step description: {description}\n\n"
            "Complete ONLY this step. "
            "If the step needs a capability JARVIS has a tool for (search, "
            "calculation, memory, files, web), the tool call is REQUIRED — "
            "do the step by tool, not from memory. "
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
        if observations:
            # v0.22 (Part E): exact, clamped tool evidence from earlier steps —
            # a later step must not depend on the verbosity of an earlier
            # step's prose summary. Identity preserved: each line names its
            # source step. ContextManager remains the size boundary (the
            # observation lines are already clamp_tool_output-bounded).
            messages.append({
                "role": "system",
                "content": "Exact tool results from earlier steps (quote values from here):\n"
                + "\n".join(observations),
            })
        if unmet_note:
            messages.append({"role": "system", "content": unmet_note})
        return messages

    def _run_react(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        max_rounds: int,
        min_rounds: int = 0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        pause_context: dict[str, Any] | None = None,
        tool_observations: list[str] | None = None,
    ) -> tuple[str, int]:
        """
        Mini ReAct loop for a single plan step.

        v0.22: when ``tool_observations`` is a list, every clamped raw tool
        result is appended to it so the CALLER can propagate exact evidence
        to later steps (a later step must not depend on how verbosely the
        model summarized an earlier step).

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
                if rounds_used < max(0, min_rounds):
                    # v0.21 fast-path safety net: the caller guaranteed a tool
                    # round but the model answered without any tool. One
                    # bounded recovery attempt before accepting the answer.
                    log.warning(
                        "forced_tool_round_unfulfilled",
                        session_id=session_id,
                        min_rounds=min_rounds,
                    )
                    return self._enforce_min_tool_round_sync(
                        session_id=session_id,
                        messages=messages,
                        tool_schemas=tool_schemas,
                        rounds_used=rounds_used,
                        pause_context=pause_context,
                    )
                self._store.save_message(session_id, assistant_dict)
                # Part P telemetry: record WHY no tool ran this turn (the
                # model chose a direct answer) so fast-path policy misses
                # are measurable without reading raw model output.
                log.info(
                    "no_tool_direct_answer",
                    session_id=session_id,
                    round=round_num + 1,
                    forced_round_pending=bool(min_rounds),
                )
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
                # v0.22: raw evidence for later steps (bounded by the clamp).
                if tool_observations is not None:
                    tool_observations.append(str(active_content))

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
        if rounds_used < max(0, min_rounds):
            log.warning(
                "forced_tool_round_unfulfilled",
                session_id=session_id,
                min_rounds=min_rounds,
            )
            return self._enforce_min_tool_round_sync(
                session_id=session_id,
                messages=messages,
                tool_schemas=tool_schemas,
                rounds_used=rounds_used,
                pause_context=pause_context,
            )
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

    def _enforce_min_tool_round_sync(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        rounds_used: int,
        pause_context: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Sync bridge for _enforce_min_tool_round from the sync _run_react.

        Uses the same running-loop fallback as the tool-dispatch block:
        a fresh event loop when none is running, a worker thread otherwise.
        """
        import asyncio
        import concurrent.futures

        enforcement = self._enforce_min_tool_round(
            session_id=session_id,
            messages=messages,
            tool_schemas=tool_schemas,
            rounds_used=rounds_used,
            pause_context=pause_context,
        )
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, enforcement).result()
        except RuntimeError:
            return asyncio.run(enforcement)

    async def _enforce_min_tool_round(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        rounds_used: int,
        pause_context: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """
        v0.21 fast-path safety net (bounded): give the model ONE tool-enabled
        recovery call after it tried to answer without the guaranteed tool
        round. If it now emits tool calls, dispatch them (PermissionGuard +
        registry validation, same as the normal loop) and produce a grounded
        answer; otherwise its direct answer stands. Never retries twice.
        """
        forced_observation = (
            "FORCED-TOOL-ROUND NOTE: this request needs a tool attempt. If a "
            "suitable tool exists for it, call it now; otherwise answer and "
            "state plainly why no tool applied."
        )
        messages.append({"role": "system", "content": forced_observation})
        response = chat_completion(messages=messages, tools=tool_schemas)
        message = response.choices[0].message
        message_dict = _message_to_dict(message)
        self._store.save_message(session_id, message_dict)
        calls = getattr(message, "tool_calls", None)
        if not calls:
            # v0.21 deterministic fallback (calculator only, bounded): when a
            # single-intent arithmetic request went through the whole prompt
            # surface without a single calculator attempt, the system executes
            # the extraction-validated expression ITSELF instead of accepting
            # an ungrounded (and often wrong) mental-arithmetic answer.
            # Authorization unchanged: PermissionGuard still guards the tool,
            # schema validation still applies, and this NEVER fires when the
            # model attempted the tool (rounds_used > 0) or for non-forced
            # requests. Knowledge stays model-driven (no deterministic path).
            if (
                rounds_used == 0
                and self._forced_tool_name == "calculator"
                and "calculator" in self._registry.list_tools()
            ):
                expression = extract_arithmetic(_user_request_from_messages(messages))
                if expression:
                    log.info(
                        "deterministic_tool_fallback",
                        tool="calculator",
                        reason="model_emitted_no_tool_call",
                    )
                    result = await self._dispatch_with_permissions_async(
                        session_id,
                        "calculator",
                        json.dumps({"expression": expression}),
                        f"fallback_{uuid.uuid4().hex[:8]}",
                        pause_context=pause_context,
                    )
                    if result != PAUSED_FOR_CONFIRMATION and not _is_tool_error(result):
                        fallback_note = {
                            "role": "system",
                            "content": (
                                f"TOOL RESULT (executed by JARVIS after the model "
                                f"failed to call the calculator): calculator({expression!r}) "
                                f"-> {result}. Use this exact value in your answer; "
                                "do not recompute or contradict it."
                            ),
                        }
                        messages.append(fallback_note)
                        answer = chat_completion(messages=messages, tools=None)
                        answer_message = answer.choices[0].message
                        self._store.save_message(session_id, _message_to_dict(answer_message))
                        return (answer_message.content or ""), 1
            return (message.content or ""), rounds_used

        messages.append(message_dict)
        for tc in calls:
            result = await self._dispatch_with_permissions_async(
                session_id,
                tc.function.name,
                tc.function.arguments,
                tc.id,
                pause_context=pause_context,
            )
            if result == PAUSED_FOR_CONFIRMATION:
                # The action needs approval; surface it like any other pause
                # so the durable resume machinery takes over.
                return PAUSED_FOR_CONFIRMATION, rounds_used + 1
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            }
            self._store.save_message(session_id, tool_msg)
            messages.append(
                {**tool_msg, "content": self._context.clamp_tool_output(result)}
            )
        answer = chat_completion(messages=messages, tools=None)
        answer_message = answer.choices[0].message
        self._store.save_message(session_id, _message_to_dict(answer_message))
        return (answer_message.content or ""), rounds_used + 1

    async def _dispatch_with_permissions_async(
        self,
        session_id: str,
        tool_name: str,
        tool_args: str,
        tool_call_id: str,
        pause_context: dict[str, Any] | None = None,
        intentional_repeat: bool = False,
    ) -> str:
        """
        Run PermissionGuard checks, then registry dispatch asynchronously.

        When the guard parks a confirmation, ``pause_context`` (the durable
        resume state) is persisted alongside it so the turn can CONTINUE
        after the user resolves the action — even across a restart.

        v0.23 (Part E): an identical (tool, canonical-args) dispatch that
        ALREADY succeeded this turn is suppressed — the step receives the
        previous result via the caller's message history, so no observation
        is lost (Part N) and no external side effect runs twice. This is
        NOT an authorization change: the guard runs first, and suppression
        never marks anything approved. Failed results are never recorded,
        so retry-after-failure remains legitimate (Part K); state-dependent
        tools are exempt by class.
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

        # v0.23 repeat semantics (AFTER the guard: suppression is never an
        # authorization path). deliberate repeats (recovery rounds) and
        # state-dependent tools pass through untouched.
        if not intentional_repeat and self._dispatch_ledger.is_duplicate(tool_name, tool_args):
            self._dispatch_ledger.record_suppressed(tool_name, tool_args)
            return (
                "DUPLICATE_SUPPRESSED: this exact tool call already succeeded "
                "earlier in this task and its full result is present above. "
                "Do not repeat it; use the existing result to continue."
            )

        result = await self._registry.dispatch_async(tool_name, tool_args)
        if not _is_tool_error(result):
            self._dispatch_ledger.record_success(tool_name, tool_args)
        return result

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
        # synthesis call cannot blow up the context window either (v0.22:
        # shared with the per-step injection via _clamp_step_results).
        clamped_steps = _clamp_step_results(completed_steps, self._context.clamp_tool_output)
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


def _clamp_step_results(
    completed_steps: list[dict[str, Any]], clamp: Any
) -> list[dict[str, Any]]:
    """v0.22 (Part E): clamp each step result for prompt injection.

    The DB keeps the full result; the PROMPT copy is bounded so one huge
    tool output (scrape, dump) cannot consume the context for later steps
    or for synthesis. Identity (step number/description) is preserved.
    """
    return [
        {
            **step,
            "result": clamp(str(step.get("result") or "")),
        }
        for step in completed_steps
    ]


def _user_request_from_messages(messages: list[dict[str, Any]]) -> str:
    """Return the latest user message text (the current request)."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _is_tool_error(result: str) -> bool:
    """
    Return True if a tool result string represents a failure.

    Convention: all tool errors in JARVIS start with "ERROR:" or
    "ACTION_REQUIRES_CONFIRMATION:". A confirmation request is not a
    failure for self-correction purposes — the loop should simply wait.
    Only "ERROR:" prefixed results trigger the recovery logic.
    """
    return isinstance(result, str) and result.startswith("ERROR:")

