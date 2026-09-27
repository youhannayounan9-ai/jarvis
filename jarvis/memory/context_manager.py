"""
jarvis/memory/context_manager.py
────────────────────────────────
Context Manager — bounded context assembly for the Orchestrator.

Strategy (simple and reliable, no new dependencies):

  1. Smart windowing   — keep only the most recent N messages.
  2. Anchor pins       — the FIRST user message of the session (the original
                         task) is always re-injected, so "important recent
                         messages" and the conversation's goal survive
                         windowing even in very long sessions.
  3. Rolling summary   — dropped assistant/tool turns are condensed into a
                         compact deterministic digest (extractive, not LLM
                         generated) that is injected as a system message.
  4. Tool-output clamp — tool results are hard-clamped per message so a
                         web scrape or stack trace can never dominate the
                         prompt.

The SQLite session store remains the source of truth (full fidelity on
disk); this module only shapes what the LLM *sees*. Full payloads are kept
in SQLite for audit/debug; only the in-prompt view is bounded.
"""

from __future__ import annotations

from typing import Any

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Characters kept from the START of a clamped tool output before the tail
# marker. Tail keeps line-ends where stack traces / conclusions often live.
_TOOL_HEAD_KEEP = 2800
_TOOL_TAIL_KEEP = 1200

_TRUNCATION_MARKERS = ("[truncated", "[output truncated", "[TRUNCATED]")


class ContextManager:
    """
    Builds a bounded message list for LLM calls.

    Usage:
        cm = ContextManager(store)
        messages = cm.build_messages(
            system_prompts=[settings.system_prompt, memory_cue],
            history=store.load_history(session_id),
            user_input=user_input,
        )
    """

    def __init__(self, max_messages: int | None = None) -> None:
        # Window size: generous, but bounded so token cost stays predictable.
        # Floor at 1: 0 would make history[-0:] return the ENTIRE history
        # (unbounded context) instead of an empty window.
        self._max_messages = max(
            1,
            int(
                max_messages if max_messages is not None else settings.max_context_messages
            ),
        )

    @property
    def max_messages(self) -> int:
        return self._max_messages

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_messages(
        self,
        *,
        system_prompts: list[dict[str, Any]],
        history: list[dict[str, Any]],
        user_input: str,
    ) -> list[dict[str, Any]]:
        """
        Compose the final message list for one LLM call.

        Order: system prompts → anchor → rolling summary → windowed history
        → current user input. System prompts and the current user message are
        never counted against the window budget.
        """
        window, anchor, summary = self.select_window(history)

        messages: list[dict[str, Any]] = list(system_prompts)
        if anchor:
            messages.append(anchor)
        if summary:
            messages.append(summary)
        messages.extend(window)
        messages.append({"role": "user", "content": user_input})
        return messages

    def select_window(
        self,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        """
        Pick the message window and produce anchor / summary system messages.

        Returns:
            (window, anchor, summary) — ``window`` is the chronological list
            of messages to include verbatim; ``anchor`` re-injects the session's
            original task (first user message) when it fell out of the window;
            ``summary`` is a compact digest of dropped turns (or None).
        """
        if len(history) <= self._max_messages:
            return list(history), None, None

        window = list(history[-self._max_messages :])

        dropped = history[: len(history) - self._max_messages]
        anchor = self._build_anchor(dropped, window)
        summary = self._build_summary(dropped) if dropped else None
        return window, anchor, summary

    # ── Anchoring ──────────────────────────────────────────────────────────────

    def _build_anchor(
        self,
        dropped: list[dict[str, Any]],
        window: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Re-inject the session's original task when it slid out of the window.

        The first user message usually carries the standing goal; losing it
        makes follow-ups in long sessions drift. It is injected as a system
        message so it never appears as a duplicate turn.
        """
        first_user = next(
            (m for m in dropped if m.get("role") == "user" and (m.get("content") or "").strip()),
            None,
        )
        if first_user is None:
            return None

        content = " ".join(str(first_user["content"]).split())
        if len(content) > 500:
            content = content[:499].rstrip() + "…"

        # Skip if the same content is already visible in the window.
        for msg in window:
            if msg.get("role") == "user":
                window_text = " ".join(str(msg.get("content") or "").split())
                if content and content in window_text:
                    return None

        return {
            "role": "system",
            "content": (
                "[Conversation anchor] The user's original task in this "
                f"session was: {content}"
            ),
        }

    # ── Summarization ──────────────────────────────────────────────────────────

    def _build_summary(self, dropped: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Deterministic extractive digest of dropped turns.

        Deliberately NOT an LLM call: it is free, fast, and cannot hallucinate.
        Key user intents (first user turns) and tool outcomes (names only) are
        preserved so the model keeps a sense of what already happened.
        """
        parts: list[str] = []

        user_intents = [
            " ".join(str(m.get("content") or "").split())
            for m in dropped
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        if user_intents:
            joined = " | ".join(user_intents)
            if len(joined) > 600:
                joined = joined[:599].rstrip() + "…"
            parts.append(f"Earlier user requests: {joined}")

        tool_names: list[str] = []
        for m in dropped:
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name")
                if name and name not in tool_names:
                    tool_names.append(name)
        if tool_names:
            parts.append(f"Tools used earlier: {', '.join(tool_names)}.")

        assistant_notes = [
            " ".join(str(m.get("content") or "").split())[:160]
            for m in dropped
            if m.get("role") == "assistant" and (m.get("content") or "").strip()
        ]
        if assistant_notes:
            parts.append(
                "Earlier assistant answers (abridged): "
                + " | ".join(assistant_notes[-3:])
            )

        if not parts:
            return None

        summary_text = "\n".join(parts)
        if len(summary_text) > 1200:
            summary_text = summary_text[:1199].rstrip() + "…"

        return {
            "role": "system",
            "content": (
                "[Context summary] Older messages were compacted to keep the "
                "conversation focused. Digest of the earlier part of this "
                f"session:\n{summary_text}"
            ),
        }

    # ── Tool-output clamping ───────────────────────────────────────────────────

    def clamp_tool_output(self, content: str, max_chars: int | None = None) -> str:
        """
        Bound a tool result for the LLM while keeping head AND tail.

        A stack trace's conclusion and a web page's ending both live at the
        end, so we keep the tail too. Idempotent: already-truncated payloads
        (from the store or the tool itself) pass through untouched.
        """
        if not isinstance(content, str):
            content = str(content)

        max_chars = max_chars if max_chars is not None else settings.max_tool_output_chars

        if len(content) <= max_chars:
            return content

        if any(marker in content for marker in _TRUNCATION_MARKERS):
            return content

        head_keep = max(0, max_chars - _TOOL_TAIL_KEEP - 80)
        head_keep = min(head_keep, _TOOL_HEAD_KEEP)
        tail_keep = _TOOL_TAIL_KEEP

        omitted = len(content) - head_keep - tail_keep
        clipped = (
            content[:head_keep].rstrip()
            + f"\n…[{omitted} characters omitted]…\n"
            + content[-tail_keep:].lstrip()
        )
        log.info(
            "tool_output_clamped",
            original_chars=len(content),
            kept_chars=len(clipped),
        )
        return clipped


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Rough token estimate (~4 chars/token) for budget logging and tests.

    Intentionally simple: we only need a comparative signal, not an exact
    tokenizer count, and this avoids new dependencies.
    """
    total_chars = 0
    for m in messages:
        total_chars += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total_chars += len(str(tc))
    return total_chars // 4
