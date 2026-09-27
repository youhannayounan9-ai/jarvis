"""
jarvis/core/planner.py
──────────────────────
Lightweight Plan-and-Execute planner (v0.3).

Produces a structured JSON list of discrete steps for complex tasks.
No external agent frameworks — uses the existing LiteLLM chat_completion wrapper.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = (
    "You are a strategic planner for a tool-using assistant.\n"
    "Break the user's request into a numbered list of actionable, discrete steps.\n\n"
    "Rules:\n"
    "1. Prefer the FEWEST steps that fully cover the request (1-3 is typical; "
    "   never invent busywork steps like 'understand' or 'synthesize').\n"
    "2. Each step must be independently executable and build toward the final answer.\n"
    "3. 'required_tools' must contain ONLY tool names that exist in the provided "
    "   tool list; use [] for a pure-reasoning step.\n"
    "4. Steps that do not depend on earlier results are fine to list in any order; "
    "   the executor runs them sequentially and the executor, not you, decides "
    "   the concrete tool arguments.\n"
    "5. If the request mentions memory of the user ('my', 'remember', personal "
    "   facts), include the recall_facts / remember_fact tools as required.\n"
    "6. Resolve references to earlier results ('it', 'that page', 'the result') "
    "   into each step's own description: every step must be self-contained, "
    "   because the executor sees prior results but re-reads each description cold.\n"
    "7. Return ONLY a valid JSON array of objects, where each object has "
    "   'step_number' (int), 'description' (str), and 'required_tools' (list of str). "
    "   No markdown fences, no prose outside the JSON array."
)

# Cap on plan size to bound cost of the execute phase.
MAX_PLAN_STEPS = 5

# Type alias: matches jarvis.llm.client.chat_completion
LLMClient = Callable[..., Any]


class Planner:
    """
    Generates a structured multi-step plan for a user request.

    Args:
        llm_client: Callable with the same signature as ``chat_completion``.
    """

    def __init__(self, llm_client: LLMClient, tool_names: list[str] | None = None) -> None:
        self._llm = llm_client
        self._tool_names = list(tool_names) if tool_names else []

    def generate_plan(self, user_input: str, context: str) -> list[dict[str, Any]]:
        """
        Ask the planner model to decompose ``user_input`` into discrete steps.

        Args:
            user_input: The raw user request.
            context:    Extra session / memory context for better planning.

        Returns:
            A list of step dicts with keys:
            ``step_number``, ``description``, ``required_tools``.
            On parse failure, returns a single-step fallback plan.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_planner_user_prompt(
                    user_input, context, self._tool_names
                ),
            },
        ]

        try:
            response = self._llm(
                messages=messages,
                tools=None,
                model=settings.litellm_planner_model,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as e:
            log.error("planner_llm_failed", error=str(e))
            return self._fallback_plan(user_input)

        plan = self._parse_plan(raw, user_input=user_input)
        if len(plan) > MAX_PLAN_STEPS:
            log.warning(
                "plan_truncated_to_max_steps",
                requested=len(plan),
                max_steps=MAX_PLAN_STEPS,
            )
            plan = plan[:MAX_PLAN_STEPS]
            # Renumber so step numbers stay contiguous.
            for i, step in enumerate(plan, start=1):
                step["step_number"] = i
        log.info("plan_generated", steps=len(plan))
        return plan

    def _parse_plan(self, raw: str, user_input: str) -> list[dict[str, Any]]:
        """Strip fences, parse JSON, and normalise step objects."""
        cleaned = _strip_code_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            log.warning("planner_json_decode_error", error=str(e), raw_preview=raw[:200])
            return self._fallback_plan(user_input)

        if not isinstance(data, list) or not data:
            log.warning("planner_invalid_shape", got=type(data).__name__)
            return self._fallback_plan(user_input)

        normalised: list[dict[str, Any]] = []
        for i, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            tools = item.get("required_tools") or []
            if not isinstance(tools, list):
                tools = []
            normalised.append({
                "step_number": int(item.get("step_number") or i),
                "description": description,
                "required_tools": [str(t) for t in tools],
            })

        if not normalised:
            return self._fallback_plan(user_input)

        # LLMs hallucinate tool names (execute_python_code, list_files, ...).
        # A step whose required_tools do not exist would send the executor
        # into guaranteed tool errors, so filter them against the real
        # registry surface when it is known. (Empty tool_names = planner used
        # standalone without registry info; keep plans untouched there.)
        known = set(self._tool_names)
        if known:
            for step in normalised:
                step["required_tools"] = [
                    t for t in step["required_tools"] if t in known
                ]

        return normalised

    @staticmethod
    def _fallback_plan(user_input: str) -> list[dict[str, Any]]:
        """Single-step plan used when the planner output is unusable."""
        log.info("planner_fallback_single_step")
        return [{
            "step_number": 1,
            "description": user_input,
            "required_tools": [],
        }]


def build_planner_user_prompt(
    user_input: str,
    context: str,
    tool_names: list[str] | None = None,
) -> str:
    """
    Compose the planner's user message, optionally grounding it in the
    actual registered tool names so 'required_tools' stays truthful.
    """
    tool_block = ""
    if tool_names:
        tool_block = "Available tools:\n" + "\n".join(f"- {t}" for t in tool_names) + "\n\n"
    return (
        f"Context:\n{context or '(none)'}\n\n"
        f"{tool_block}"
        f"User request:\n{user_input}\n\n"
        "Return a JSON array of plan steps now."
    )


def _strip_code_fences(text: str) -> str:
    """Remove optional ``` / ```json wrappers around a JSON payload."""
    text = text.strip()
    fence = re.match(
        r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        return fence.group(1).strip()
    # Sometimes models add prose before/after a fenced block — try to find one.
    inner = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if inner:
        return inner.group(1).strip()
    # Or locate the outermost JSON array.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text
