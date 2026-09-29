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
from dataclasses import dataclass
from typing import Any, Literal, Type

from pydantic import BaseModel, create_model, Field, ValidationError

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


@dataclass(frozen=True)
class CachePolicy:
    """
    v0.24 explicit cross-turn cacheability declaration (Parts B/C2).

    A tool OPTS IN to the cross-turn result cache by setting ``cache_policy``
    on the class; tools without a policy are NEVER cached (the safe default —
    side-effect and state-coupled tools simply don't declare one).

    Fields:
        cacheable:       True → the result may be stored and reused across turns.
        ttl_seconds:     Time-based expiration. None → freshness comes from
                         ``freshness`` (source state), not the clock.
        scope:           "global" (any session may reuse) or "session"
                         (only the session that produced the result).
                         Global is reserved for public/deterministic content.
        freshness:       "ttl" (clock only) | "source_stat" (re-stat the
                         argument path; size/mtime change ⇒ stale) |
                         "knowledge_generation" (stale when the knowledge
                         base registry changes).
        normalizer:      Cache-KEY normalization for arguments:
                         "generic" (v0.23 canonical JSON; collapses value
                         whitespace) | "verbatim" (key-sort only — paths,
                         URLs, code) | "calculator_expression" (existing AST
                         parser proves equivalence: 2+2 ≡ 2 + 2 ≡ (2+2)).
    """

    cacheable: bool = True
    ttl_seconds: float | None = None
    scope: Literal["global", "session"] = "global"
    freshness: Literal["ttl", "source_stat", "knowledge_generation"] = "ttl"
    normalizer: Literal["generic", "verbatim", "calculator_expression"] = "generic"


class BaseTool(ABC):
    """Abstract base for all JARVIS tools."""

    # ── Subclasses must define these as class attributes ───────────────────────
    name: str  # Unique tool identifier (used in function-calling schema)
    description: str  # What this tool does — shown to the LLM
    parameters: dict[str, Any]  # JSON Schema for the tool's input arguments

    # Automatically generated during __init_subclass__
    _args_model: Type[BaseModel]

    # Security / reliability — subclasses SHOULD override risk_level to match
    # their side effects. timeout_seconds may be raised for slow network tools.
    risk_level: RiskLevel = "SAFE"
    timeout_seconds: float = 15.0

    # v0.24: cross-turn result-cache policy. None (the default) = never cached.
    # Declare ONLY for read-only retrieval whose semantics survive reuse;
    # see jarvis/core/result_cache.py for the enforcement boundary.
    cache_policy: "CachePolicy | None" = None

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

        # Generate Pydantic validation model from JSON schema
        if hasattr(cls, "parameters") and isinstance(cls.parameters, dict):
            cls._args_model = _build_pydantic_model(cls.name, cls.parameters)
        else:
            cls._args_model = create_model(f"{cls.__name__}Args")

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

    async def run_async(self, **kwargs: Any) -> str:
        """
        Asynchronous execution of the tool.

        By default, this offloads the synchronous `run` method to a thread
        so it doesn't block the event loop. Subclasses can override this
        for native async IO.
        """
        import asyncio
        return await asyncio.to_thread(self.run, **kwargs)

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


def _build_pydantic_model(tool_name: str, schema: dict[str, Any]) -> Type[BaseModel]:
    """Dynamically build a Pydantic model from a simple JSON schema."""
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    
    fields: dict[str, Any] = {}
    for field_name, field_info in properties.items():
        type_str = field_info.get("type", "string")
        
        # Map JSON schema types to Python types
        if type_str == "string":
            py_type = str
        elif type_str == "integer":
            py_type = int
        elif type_str == "number":
            py_type = float
        elif type_str == "boolean":
            py_type = bool
        elif type_str == "array":
            py_type = list
        elif type_str == "object":
            py_type = dict
        else:
            py_type = Any
            
        # Handle enums
        if "enum" in field_info:
            enum_values = field_info["enum"]
            if enum_values:
                # Use Literal for typing
                py_type = Literal[tuple(enum_values)]  # type: ignore

        description = field_info.get("description", "")
        
        if field_name in required_fields:
            fields[field_name] = (py_type, Field(..., description=description))
        else:
            fields[field_name] = (py_type, Field(default=None, description=description))
            
    from pydantic import ConfigDict
    config = ConfigDict(extra="forbid")

    model_name = f"{"".join(word.capitalize() for word in tool_name.split('_'))}Args"
    return create_model(model_name, __config__=config, **fields)
