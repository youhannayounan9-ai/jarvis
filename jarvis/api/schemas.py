"""
jarvis/api/schemas.py
─────────────────────
Pydantic request/response models for the JARVIS REST API.

Keeping schemas separate from the app module makes them importable by
clients (or a future generated SDK) without pulling in FastAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """One user turn for an agent session."""

    message: str = Field(..., min_length=1, max_length=32_000, description="The user's message.")
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Existing session ID. Omit to start a new session.",
    )


class ConfirmationRequest(BaseModel):
    """Decision on a pending high-risk action."""

    confirmed: bool = Field(..., description="True to approve, False to deny.")


class PendingAction(BaseModel):
    tool_name: str
    tool_args: str
    risk_level: str


class ChatResponse(BaseModel):
    """The assistant's reply plus light request metadata."""

    session_id: str
    response: str
    pending_confirmation: PendingAction | None = None
    # Echoed to clients so an error report can cite a correlation id; the
    # middleware also sets it as an X-Request-ID header.
    request_id: str | None = None


class SessionCreated(BaseModel):
    session_id: str


class ToolInfo(BaseModel):
    name: str
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    tools: list[str]
    disabled_tools: list[str]
    # v0.10: auth + code-execution posture surfaced for operators.
    auth_enabled: bool = False
    code_execution: str = "disabled"  # disabled | docker_isolated
