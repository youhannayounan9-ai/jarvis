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
    "You are a strategic planner. Break down the user's request into a numbered "
    "list of actionable, discrete steps. Return ONLY a valid JSON array of objects, "
    "where each object has 'step_number' (int), 'description' (str), and "
    "'required_tools' (list of str). Do not include markdown formatting or any "
    "text outside the JSON array."
)

# Type alias: matches jarvis.llm.client.chat_completion
LLMClient = Callable[..., Any]


class Planner:
    """
    Generates a structured multi-step plan for a user request.

    Args:
        llm_client: Callable with the same signature as ``chat_completion``.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

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
                "content": (
                    f"Context:\n{context or '(none)'}\n\n"
                    f"User request:\n{user_input}"
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
