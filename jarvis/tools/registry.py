"""
jarvis/tools/registry.py
────────────────────────
Central registry that manages all available tools.

The registry is the single source of truth for:
  1. Which tools exist (register / lookup by name).
  2. The OpenAI-format tool schemas sent to the LLM.
  3. Routing an LLM tool-call request to the right implementation.

Design decision — manual registration over auto-discovery:
  Auto-discovery (importing all .py files in the tools/ dir) is clever but
  makes it hard to trace what's active and easy to accidentally expose a
  half-built tool. We explicitly register tools in main.py, which is one
  clear, readable place that lists exactly what JARVIS can do.
"""

import concurrent.futures
import json
import time
from typing import Any

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class ToolRegistry:
    """Holds all registered tools and dispatches tool-call requests."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """
        Add a tool to the registry.

        Args:
            tool: An instance of a BaseTool subclass.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Each tool must have a unique name."
            )
        self._tools[tool.name] = tool
        log.info(
            "tool_registered",
            tool=tool.name,
            risk_level=tool.risk_level,
            timeout_seconds=tool.timeout_seconds,
        )

    # ── Schema export ──────────────────────────────────────────────────────────

    def get_schemas(self) -> list[dict[str, Any]]:
        """
        Return all tool schemas in OpenAI function-calling format.
        This list is passed to LiteLLM with every chat_completion call.
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def get(self, tool_name: str) -> BaseTool | None:
        """Return a registered tool by name, or None if missing."""
        return self._tools.get(tool_name)

    def get_tool_risk_level(self, tool_name: str) -> str:
        """
        Return the declared risk_level for a registered tool.

        Returns:
            The tool's risk_level string, or ``"UNKNOWN"`` if the tool
            is not in the registry.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return "UNKNOWN"
        return str(tool.risk_level)

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def dispatch(self, tool_name: str, tool_args_json: str) -> str:
        """
        Execute a tool by name with JSON-encoded arguments.

        This is called by the orchestrator when the LLM issues a tool_call.
        Execution is bounded by the tool's ``timeout_seconds`` using a
        single-worker thread pool (synchronous API; no asyncio).

        Args:
            tool_name:      The function name from the LLM's tool_call.
            tool_args_json: A JSON string of argument key-value pairs.

        Returns:
            The tool's string result (or an "ERROR: ..." string on failure).
        """
        if tool_name not in self._tools:
            error = f"ERROR: Unknown tool '{tool_name}'. Available: {list(self._tools)}"
            log.warning("tool_not_found", tool=tool_name)
            return error

        try:
            args: dict[str, Any] = json.loads(tool_args_json) if tool_args_json else {}
        except json.JSONDecodeError as e:
            error = f"ERROR: Could not parse tool arguments as JSON: {e}"
            log.error("tool_args_parse_error", tool=tool_name, error=str(e))
            return error

        tool = self._tools[tool_name]
        timeout = float(tool.timeout_seconds)

        log.info(
            "tool_dispatching",
            tool=tool_name,
            args=args,
            risk_level=tool.risk_level,
            timeout_seconds=timeout,
        )

        started = time.monotonic()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.run, **args)
                result = future.result(timeout=timeout)

            elapsed = time.monotonic() - started
            log.info("tool_success", tool=tool_name, elapsed_seconds=round(elapsed, 3))
            return result

        except concurrent.futures.TimeoutError:
            elapsed = time.monotonic() - started
            error = (
                f"ERROR: Tool '{tool_name}' execution timed out "
                f"after {timeout} seconds."
            )
            log.error(
                "tool_timeout",
                tool=tool_name,
                timeout_seconds=timeout,
                elapsed_seconds=round(elapsed, 3),
            )
            return error

        except Exception as e:
            elapsed = time.monotonic() - started
            error = f"ERROR: Tool '{tool_name}' raised an unexpected exception: {e}"
            log.error(
                "tool_exception",
                tool=tool_name,
                error=str(e),
                elapsed_seconds=round(elapsed, 3),
            )
            return error

    # ── Introspection ──────────────────────────────────────────────────────────

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
