"""
jarvis/llm/client.py
────────────────────
LiteLLM wrapper for Ollama.

Why LiteLLM?
  LiteLLM provides a single unified completion() / acompletion() interface
  across 100+ LLM providers. Switching from Ollama to OpenAI, Anthropic, or
  Gemini in the future requires changing exactly ONE value (the model string)
  — no other code changes.

Why the wrapper?
  We don't call LiteLLM directly from the orchestrator. This module:
    1. Injects the Ollama base URL from config.
    2. Provides a single call site to add retries, cost tracking, or
       fallback models later — without touching orchestrator code.
    3. Silences LiteLLM's default verbose logging (it's noisy).
"""

from typing import Any

import litellm

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# LiteLLM prints a lot of debug info by default. Suppress it so our own
# structured logs stay readable.
litellm.suppress_debug_info = True
litellm.set_verbose = False


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> Any:
    """
    Send a synchronous chat completion request to Ollama via LiteLLM.

    Args:
        messages: Full conversation history in OpenAI message format.
                  e.g. [{"role": "user", "content": "Hello"}]
        tools:    Optional list of tool schemas in OpenAI function-calling
                  format. When provided, the model may respond with
                  tool_calls instead of (or in addition to) text.
        model:    Optional LiteLLM model string override (e.g. planner model).
                  Defaults to ``settings.litellm_model``.

    Returns:
        A LiteLLM ModelResponse object. Callers inspect:
          - response.choices[0].message.content   → text response
          - response.choices[0].message.tool_calls → tool call requests

    Raises:
        litellm.exceptions.APIConnectionError: Ollama not running.
        litellm.exceptions.BadRequestError:    Malformed request.
        Any other litellm exception for API-level errors.
    """
    litellm_model = model or settings.litellm_model

    log.debug(
        "llm_request",
        model=litellm_model,
        message_count=len(messages),
        tools_count=len(tools) if tools else 0,
    )

    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "max_tokens": settings.max_tokens,
        # Tell LiteLLM where the Ollama server lives.
        "api_base": settings.ollama_base_url,
        # Do not raise exceptions for tool_call responses that have no text.
        "drop_params": True,
        # Prevent OOM crashes by restricting Ollama's memory allocation
        "num_ctx": settings.ollama_num_ctx,
    }

    if tools:
        kwargs["tools"] = tools
        # "auto" lets the model decide whether to call a tool or reply directly.
        kwargs["tool_choice"] = "auto"

    response = litellm.completion(**kwargs)

    log.debug(
        "llm_response",
        finish_reason=response.choices[0].finish_reason,
        has_tool_calls=bool(
            getattr(response.choices[0].message, "tool_calls", None)
        ),
    )

    return response
