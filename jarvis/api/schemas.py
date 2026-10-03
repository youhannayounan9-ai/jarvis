"""
jarvis/api/schemas.py
─────────────────────
Pydantic request/response models for the JARVIS REST API.

Keeping schemas separate from the app module makes them importable by
clients (or a future generated SDK) without pulling in FastAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BrowserStatus(BaseModel):
    """
    v0.28 safe browser status (read-only; safe metadata only — never page
    content, URLs visited, or download payloads).
    """

    enabled: bool = Field(..., description="Whether the browser tool surface is registered.")
    driver: str = Field(..., description="Configured driver: simulated | playwright.")
    emergency_stop_active: bool
    emergency_stop_reason: str = ""
    emergency_stop_token: int = 0
    open_sessions: int = Field(0, description="Currently open browser controllers.")
    max_sessions: int
    downloads_captured: int = Field(0, description="Bounded download metadata count for open sessions.")


class BrowserStopRequest(BaseModel):
    """Body for POST /browser/emergency-stop (optional reason)."""

    reason: str = Field(
        default="user requested emergency stop",
        max_length=200,
        description="Short human reason recorded with the stop token.",
    )


class BrowserStopResponse(BaseModel):
    triggered: bool
    token: int
    reason: str


class BrowserResetResponse(BaseModel):
    was_active: bool
    message: str = ""


class ChatRequest(BaseModel):
    """One user turn for an agent session."""

    message: str = Field(..., min_length=1, max_length=32_000, description="The user's message.")
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Existing session ID. Omit to start a new session.",
    )
    refresh: bool = Field(
        default=False,
        description=(
            "v0.25 explicit programmatic freshness control. When true, "
            "cache-eligible read-only retrieval tools (web/search/file/knowledge) "
            "bypass the cross-turn result cache for THIS turn and execute for "
            "real; fresh successful results are stored again as usual. Never "
            "bypasses permissions, schema validation, or confirmation — cache "
            "eligibility only. Natural-language freshness wording continues to "
            "work independently."
        ),
    )


class MultimodalResponse(BaseModel):
    """Reply for a multimodal turn (v0.27): text plus light metadata."""

    session_id: str
    response: str
    modality: str
    pending_confirmation: PendingAction | None = None
    request_id: str | None = None


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


# ── v0.29: integration management (metadata only — never credentials) ────────


class IntegrationAccountInfo(BaseModel):
    """One connected account's PUBLIC metadata (credential-free by type)."""

    account_id: str
    provider: str
    display_label: str
    scopes: list[str]
    authenticated: bool
    auth_state: str
    granted_at: str
    last_verified_at: str | None = None
    provider_account_ref: str = ""
    # v0.30 (safe metadata only): normalized OAuth lifecycle + timestamps.
    # Token material is structurally absent — these are strings/None. The
    # authorization_* names keep the v0.29 "no token-ish public key" guard.
    authorization_status: str | None = None
    authorization_expires_at: str | None = None
    authorization_updated_at: str | None = None


class IntegrationConnectRequest(BaseModel):
    """Body for POST /integrations/{provider}/connect (user-controlled)."""

    display_label: str = Field(..., min_length=1, max_length=64)
    scopes: list[str] = Field(
        ..., min_length=1, max_length=8,
        description="EXACT scopes to grant (validated against the provider menu).",
    )
    credential: str | None = Field(
        default=None, max_length=512,
        description=(
            "Optional credential for development providers. Never sent by "
            "the chat model; omitted = generated local-dev token. Never "
            "returned by any endpoint after this request."
        ),
    )
    provider_account_ref: str = Field(default="", max_length=120)


class IntegrationAuthorizeRequest(BaseModel):
    """Body for POST /integrations/{provider}/authorize (OPERATOR-controlled)."""

    session_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Local session the authorization is bound to (state binding).",
    )
    display_label: str = Field(..., min_length=1, max_length=64)
    scopes: list[str] = Field(..., min_length=1, max_length=8)


class IntegrationAuthorizationStart(BaseModel):
    """
    A started OAuth flow. The OPERATOR opens ``authorization_url`` in a
    browser; the provider redirects back to ``redirect_uri`` (our fixed
    callback + our own session parameter). The raw state lives ONLY inside
    the URL — it is never returned separately, logged, or shown to the model.
    """

    provider: str
    display_label: str
    scopes: list[str]
    authorization_url: str
    redirect_uri: str
    expires_in: int


class IntegrationStatusResponse(BaseModel):
    """Bounded authorization-flow polling view (no state values, ever)."""

    status: str
    provider: str
    display_label: str | None = None


class IntegrationDisconnectResponse(BaseModel):
    disconnected: bool
    account_id: str
    message: str


class IntegrationAuditEntry(BaseModel):
    """One EXTERNAL side-effect audit row (bounded, sanitized summary)."""

    ts: str
    provider: str
    account_id: str
    operation: str
    resource_kind: str
    resource_id: str | None = None
    risk_category: str
    request_id: str
    idempotency_key: str | None = None
    state: str
    verification: str = "NOT_APPLICABLE"
    result_summary: str | None = None


# ── v0.18: operator introspection + explicit UNKNOWN recovery ─────────────────


class ActionInfo(BaseModel):
    """
    Safe metadata for one execution-ledger row.

    Deliberately EXCLUDES tool_args and result bodies (they may contain user
    data); tool_name/risk/ids/timestamps are the operator-relevant facts.
    """

    action_id: str
    session_id: str
    confirmation_id: str
    tool_name: str
    risk_level: str
    state: str
    attempt: int
    created_at: str
    claimed_at: str | None = None
    finished_at: str | None = None
    reissue_depth: int = 0
    reissued_from: str | None = None


class ReissueRequest(BaseModel):
    """Explicit operator/user decision to re-issue an UNKNOWN action."""

    request_id: str | None = Field(
        None,
        min_length=8,
        max_length=128,
        description=(
            "Caller-generated idempotency key: the same key twice returns the "
            "same new action instead of creating two. Optional — when omitted, "
            "the server generates one and returns it; clients that want replay "
            "protection across retries should send their own."
        ),
    )


class ReissueResponse(BaseModel):
    original_action_id: str
    new_action_id: str
    state: str
    reissue_depth: int
    max_reissues: int
    reused_existing: bool = Field(
        False,
        description="True when this exact request had already been processed (idempotent replay).",
    )
    request_id: str = Field(
        "",
        description=(
            "The idempotency key actually used (echoed back when the server "
            "generated one, so the client can replay the same request safely)."
        ),
    )


class KnowledgeDocumentInfo(BaseModel):
    """Safe knowledge-document metadata (never file contents)."""

    document_id: str
    source_path: str
    filename: str
    media_type: str
    size_bytes: int
    content_hash: str
    chunk_count: int
    live_chunk_count: int = 0
    parser_version: str
    ingested_at: str
    modified_at: str | None = None


class KnowledgeIngestRequest(BaseModel):
    """Explicit single-document ingestion (absolute or allowed-relative path)."""

    path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Path to ONE document inside the allowed directory (no crawling).",
    )
    target_chars: int = Field(1200, ge=200, le=8000)
    overlap_chars: int = Field(150, ge=0, le=1000)


class KnowledgeIngestResponse(BaseModel):
    status: str
    document_id: str = ""
    chunk_count: int = 0
    content_hash: str = ""
    filename: str = ""
    source_path: str = ""
    reason: str | None = None
    same_content_as: str | None = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=20)
    source: str | None = Field(None, max_length=512)
    document_id: str | None = Field(None, max_length=64)


class KnowledgeSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int = 0
    section: str | None = None
    distance: float | None = None
    citation: str
    snippet: str


class KnowledgeSearchResponse(BaseModel):
    results: list[KnowledgeSearchResult]
    total_chunks: int


class KnowledgeRemoveResponse(BaseModel):
    document_id: str
    chunks_removed: int
    registry_row_removed: bool


class TimelineEvent(BaseModel):
    """
    One bounded, causal timeline event for a session (v0.19).

    Safe metadata only: never tool_args, result bodies, message content, or
    raw owner tokens. ``kind`` ∈ message / confirmation_parked /
    action_state / reissue / lease; the remaining fields are populated
    per kind.
    """

    ts: str
    seq: int = 0
    kind: str
    model_config = {"extra": "allow"}


class LeaseInfo(BaseModel):
    """Safe lease metadata — owner token is redacted, never exposed raw."""

    session_id: str
    owner: str
    acquired_at: str
    expires_at: str
    fencing: int
    active: bool
