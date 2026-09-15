"""
jarvis/tools/base.py
────────────────────
Abstract base class that every JARVIS tool must implement.

Why a base class?
  - Enforces a consistent interface: every tool has a name, description,
    a JSON-Schema parameters spec, and a run() method.
  - The tool registry can treat all tools uniformly (no isinstance checks).
  - Adding a new tool = creating a new file + subclassing BaseTool.
    Nothing else in the system needs to change.

Tool result contract:
  run() always returns a plain string. The string may contain:
    - A successful result (text, JSON, etc.)
    - An error message prefixed with "ERROR: "
  The orchestrator passes this string back to the LLM as a tool result.
  Returning strings (not exceptions) keeps the tool-calling loop simple and
  lets the LLM handle tool errors gracefully in its response.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

# Risk tiers used by PermissionGuard to decide allow / block / confirm.
RiskLevel = Literal[
    "SAFE",
    "NETWORK",
    "FILE_READ",
    "FILE_WRITE",
    "SYSTEM",
    "DESTRUCTIVE",
]

_VALID_RISK_LEVELS: frozenset[str] = frozenset(
    {"SAFE", "NETWORK", "FILE_READ", "FILE_WRITE", "SYSTEM", "DESTRUCTIVE"}
)


class BaseTool(ABC):
    """Abstract base for all JARVIS tools."""

    # ── Subclasses must define these as class attributes ───────────────────────
    name: str  # Unique tool identifier (used in function-calling schema)
    description: str  # What this tool does — shown to the LLM
    parameters: dict[str, Any]  # JSON Schema for the tool's input arguments

    # Security / reliability — subclasses SHOULD override risk_level to match
    # their side effects. timeout_seconds may be raised for slow network tools.
    risk_level: RiskLevel = "SAFE"
    timeout_seconds: float = 15.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate risk_level / timeout_seconds when a concrete tool is defined."""
        super().__init_subclass__(**kwargs)

        risk = getattr(cls, "risk_level", "SAFE")
        if risk not in _VALID_RISK_LEVELS:
            raise TypeError(
                f"{cls.__name__}.risk_level must be one of "
                f"{sorted(_VALID_RISK_LEVELS)}, got {risk!r}"
            )

        timeout = getattr(cls, "timeout_seconds", 15.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise TypeError(
                f"{cls.__name__}.timeout_seconds must be a positive number, "
                f"got {timeout!r}"
            )

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """
        Execute the tool with the given keyword arguments.

        Args:
            **kwargs: Arguments matching the JSON Schema in `parameters`.

        Returns:
            A string result to be returned to the LLM.
            On error, return a descriptive "ERROR: ..." string — do not raise.
        """

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Render this tool as an OpenAI-compatible function-calling schema.

        This is the format LiteLLM (and Ollama) expect in the `tools` list.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return (
            f"<Tool name={self.name!r} "
            f"risk_level={self.risk_level!r} "
            f"timeout_seconds={self.timeout_seconds}>"
        )
