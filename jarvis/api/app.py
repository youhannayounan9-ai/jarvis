"""
jarvis/api/app.py
─────────────────
FastAPI application — the *service* interface for JARVIS.

Design:
  - One JarvisRuntime per app instance (built lazily at startup; swappable in
    tests via set_runtime()).
  - Sessions are just IDs: state lives in SQLite (the runtime), so the API
    layer stays stateless and horizontally scalable later.
  - High-risk tools never execute here: the orchestrator parks them as
    durable pending confirmations, surfaced via `pending_confirmation` in
    the chat response, resolved via POST /sessions/{id}/confirm.
  - Optional API-key auth (JARVIS_API_KEY): Bearer/X-API-Key on every
    endpoint except /health and OpenAPI metadata. Disabled by default for
    localhost deployments.
  - Streaming: POST /chat/stream is an SSE endpoint emitting the same
    lifecycle events the orchestrator produces internally, plus the final
    answer.

Run:
    uv run uvicorn jarvis.api.app:app --port 8000

Endpoints:
    GET  /health                        → version, model, tool surface, posture
    POST /sessions                      → create a session
    POST /chat                          → send a user turn (auto-creates session)
    POST /chat/stream                   → SSE: agent step events + final answer
    GET  /sessions/{id}/history         → recent conversation window
    GET  /sessions/{id}/confirmation    → pending high-risk action, if any
    POST /sessions/{id}/confirm         → approve/deny pending action
    GET  /tools                         → active capability surface
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from jarvis.api.auth import auth_enabled, require_api_key
from jarvis.api.health import deep_health
from jarvis.api import ratelimit as _ratelimit
from jarvis.api.ratelimit import client_key
from jarvis.api.schemas import (
    ActionInfo,
    BrowserResetResponse,
    BrowserStatus,
    BrowserStopRequest,
    BrowserStopResponse,
    ChatRequest,
    ChatResponse,
    ConfirmationRequest,
    HealthResponse,
    IntegrationAccountInfo,
    IntegrationAuditEntry,
    IntegrationAuthorizationStart,
    IntegrationAuthorizeRequest,
    IntegrationConnectRequest,
    IntegrationDisconnectResponse,
    IntegrationStatusResponse,
    KnowledgeDocumentInfo,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeRemoveResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
    LeaseInfo,
    PendingAction,
    ReissueRequest,
    ReissueResponse,
    SessionCreated,
    TimelineEvent,
    ToolInfo,
)
from jarvis.integrations.manager import IntegrationManagerError
from jarvis.memory.session_store import (
    MAX_REISSUES_PER_ACTION,
    SessionStore,
    redact_owner,
)
from jarvis.memory.session_store import ActionExecution as _ActionExecution

# Details are never leaked to clients: exception strings can embed internal
# paths or hostnames. Clients get a generic message + the request id; the
# full error is in the log under that id.
_GENERIC_RUNTIME_ERROR = (
    "Agent runtime error. Quote the X-Request-ID header of this response "
    "when contacting the operator."
)
from jarvis.api.schemas import MultimodalResponse
from jarvis.browser.emergency import get_emergency_stop
from jarvis.browser.registry import get_browser_registry
from jarvis.config import settings
from jarvis.runtime import JarvisRuntime, build_runtime
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="JARVIS API",
    description="Local-first AI assistant — agent runtime over HTTP.",
    version="0.30.0",
)

# ── Runtime dependency (overridable in tests) ─────────────────────────────────

_runtime: JarvisRuntime | None = None


def get_runtime() -> JarvisRuntime:
    global _runtime
    if _runtime is None:
        _runtime = build_runtime()
    return _runtime


def set_runtime(runtime: JarvisRuntime | None) -> None:
    """Swap the process runtime (tests / embedding the API in another app).

    Clearing the runtime (``None``) also restores the default in-memory
    rate limiter, so a store-backed limiter never outlives the runtime whose
    database it writes to (v0.17 hygiene).
    """
    global _runtime
    _runtime = runtime
    if runtime is None:
        _ratelimit.set_limiter(None)


# Every mutating/reading endpoint requires the key when configured.
_AUTH = Depends(require_api_key)


def _pending_or_none(runtime: JarvisRuntime, session_id: str) -> PendingAction | None:
    pending = runtime.get_pending_confirmation(session_id)
    if not pending:
        return None
    return PendingAction(
        tool_name=pending["tool_name"],
        tool_args=pending["tool_args"],
        risk_level=pending["risk_level"],
    )


def _resolve_session(runtime: JarvisRuntime, session_id: str | None) -> str:
    """Validate or create the session; 404 on unknown IDs."""
    if session_id:
        if not runtime.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Unknown session_id.")
        return session_id
    new_id = runtime.start_session()
    log.info("api_session_created", session_id=new_id, via="chat")
    return new_id


# ── Observability middleware ──────────────────────────────────────────────────


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """
    One structured log line per API request: request id, method, path, status,
    duration. The generated request id is echoed as X-Request-ID so a client
    error report can be joined with the server log line.
    """
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    # Endpoints read this to echo the id in their response bodies.
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        log.error(
            "api_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=500,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        raise
    response.headers["X-Request-ID"] = request_id
    log.info(
        "api_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(runtime: JarvisRuntime = Depends(get_runtime)) -> HealthResponse:
    """
    Liveness + deep health: performs a real DB read/write round-trip and
    degrades to 503 when persistence is broken (fail-closed).
    """
    info = runtime.describe()
    probe = deep_health(runtime)
    status = probe["status"]
    if status != "ok":
        raise HTTPException(
            status_code=503,
            detail={
                "status": status,
                "db": probe["db"],
                "version": info["version"],
            },
        )
    return HealthResponse(
        status=status,
        version=info["version"],
        model=info["model"],
        tools=info["tools"],
        disabled_tools=info["disabled_tools"],
        auth_enabled=auth_enabled(),
        code_execution=(
            "docker_isolated" if "execute_python_code" in info["tools"] else "disabled"
        ),
    )


@app.post("/sessions", response_model=SessionCreated, status_code=201)
def create_session(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> SessionCreated:
    _enforce_rate_limit(request)
    session_id = runtime.start_session()
    log.info("api_session_created", session_id=session_id)
    return SessionCreated(session_id=session_id)


def _enforce_rate_limit(request: Request) -> None:
    """Raise 429 with Retry-After when the client exceeds the window."""
    # Module-attribute access (not a from-import) so set_limiter() rebinding
    # is honored at call time.
    allowed, retry_after = _ratelimit.limiter.check(client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Slow down.",
            headers={"Retry-After": str(retry_after)},
        )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> ChatResponse:
    """
    Process one user turn.

    If `session_id` is omitted a new session is created and returned.
    If the turn triggered a high-risk action, the response carries
    `pending_confirmation` instead of executing it.
    Concurrent turns on the SAME session return 409 (per-session lock).
    """
    _enforce_rate_limit(request)
    session_id = _resolve_session(runtime, payload.session_id)

    try:
        response_text = runtime.chat(session_id, payload.message, refresh=payload.refresh)
    except TimeoutError as e:
        log.warning("api_chat_busy", session_id=session_id)
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        # The request id is already logged by the middleware; include the
        # exception under the same correlation for the operator, but never
        # leak exception internals to the client.
        log.error("api_chat_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=503, detail=_GENERIC_RUNTIME_ERROR) from e

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        pending_confirmation=_pending_or_none(runtime, session_id),
        request_id=getattr(request.state, "request_id", None),
    )


# ── v0.27: multimodal (image + text / audio) ────────────────────────────────

# Same security model as /chat: auth (_AUTH), rate limiting, per-session
# lease — no separate multimodal security model (Part 13).


@app.post("/chat/multimodal", response_model=MultimodalResponse)
async def chat_multimodal(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> MultimodalResponse:
    """
    One multimodal user turn (v0.27).

    Accepts multipart/form-data:
      - ``text``      required prompt (used verbatim when no image);
      - ``image``     optional upload (JPEG/PNG/WebP/GIF, content-sniffed,
                      size + pixel bounds enforced);
      - ``audio``     optional upload (WAV/MP3/WebM, content-sniffed,
                      transcribed by the LOCAL Whisper provider);
      - ``session_id`` optional; ``refresh`` optional (same semantics as
                      /chat).

    The request is normalized into MultimodalRequest and executed by the
    SAME runtime (planning/tools/permissions/evidence/grounding unchanged).
    Image contents are untrusted; vision output is an observation, never a
    trusted tool result. Client file names are never used for storage.
    """
    from jarvis.multimodal.models import (
        Attachment,
        MultimodalRequest,
        MultimodalValidationError,
        validate_audio_bytes,
    )
    from jarvis.multimodal.service import MultimodalService

    _enforce_rate_limit(request)

    form = await request.form()
    text = str(form.get("text") or "").strip()
    session_id = str(form.get("session_id") or "").strip() or None
    refresh = str(form.get("refresh") or "").lower() in ("1", "true", "yes")

    image: Attachment | None = None
    audio_text: str | None = None
    modality = "text"
    upload = form.get("image")
    if upload is not None and hasattr(upload, "read"):
        try:
            data = await upload.read()
            image = Attachment.from_upload(
                data,
                declared_mime=getattr(upload, "content_type", None),
                client_name=getattr(upload, "filename", None),
            )
        except MultimodalValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        modality = "image"

    audio_upload = form.get("audio")
    if audio_upload is not None and hasattr(audio_upload, "read"):
        try:
            audio_data = await audio_upload.read()
            audio_mime = validate_audio_bytes(
                audio_data,
                getattr(audio_upload, "content_type", None),
            )
        except MultimodalValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        # LOCAL Whisper STT (v0.27 provider interface).
        from jarvis.voice.stt import SpeechToText

        stt = SpeechToText()
        audio_text = stt.transcribe_bytes(audio_data, audio_mime)
        if audio_text.startswith("ERROR:"):
            raise HTTPException(status_code=502, detail=audio_text)
        if not audio_text:
            raise HTTPException(status_code=422, detail="No speech detected in the audio upload.")
        text = f"{text} {audio_text}".strip() if text else audio_text
        modality = "audio"

    if image is not None and modality == "audio":
        modality = "image+audio"
    elif image is not None:
        modality = "image+text" if text != "Describe this image in detail." else "image"

    if not text and image is None:
        raise HTTPException(status_code=422, detail="Provide 'text', 'image', or 'audio'.")

    session_id = _resolve_session(runtime, session_id)
    request_id = getattr(request.state, "request_id", None)
    normalized = MultimodalRequest(
        text=text,
        session_id=session_id,
        request_id=request_id,
        modality=modality,
        image=image,
        refresh=refresh,
    )
    try:
        response_text = MultimodalService(runtime.orchestrator).run(normalized)
    except MultimodalValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except TimeoutError as e:
        log.warning("api_chat_busy", session_id=session_id)
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        log.error(
            "api_multimodal_failed",
            session_id=session_id,
            modality=modality,
            error_category="runtime_error",
        )
        raise HTTPException(status_code=503, detail=_GENERIC_RUNTIME_ERROR) from e
    finally:
        # Temp hygiene (Part 19/20): the persisted upload lives only for the
        # request (the session history keeps the derived TEXT, not the bytes).
        if image is not None:
            image.cleanup()

    return MultimodalResponse(
        session_id=session_id,
        response=response_text,
        modality=modality,
        pending_confirmation=_pending_or_none(runtime, session_id),
        request_id=request_id,
    )


# ── Streaming (SSE) ───────────────────────────────────────────────────────────


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event (JSON payload for typed clients)."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# Heartbeat cadence for the SSE stream: proxies (nginx default 60s idle
# timeout) close quiet connections; a keepalive comment every 15s keeps the
# path open during long LLM turns.
_SSE_HEARTBEAT_SECONDS = 15.0


def _event_stream(runtime: JarvisRuntime, session_id: str, message: str, refresh: bool = False):
    """
    Generator bridging orchestrator lifecycle events to SSE.

    chat() runs in a worker thread and pushes events into a queue; this
    generator drains it, so clients receive events *as they happen* (true
    interleaving, not a post-hoc replay), plus heartbeat comments during long
    quiet stretches so proxies do not close the connection.

    Event sequence:
      event: begin      {session_id}
      event: intent     {intent}
      event: plan       {steps: [...]}                      (complex path only)
      event: step_start {step_number, description}
      event: tool_calls {round, tools}
      event: step_done  {step_number, rounds_used}
      event: synthesis  {}
      event: done       {session_id, response, pending_confirmation}
      event: error      {detail}                            (runtime failure)
    """
    import queue as _queue
    import threading as _threading

    yield _sse("begin", {"session_id": session_id})

    events: _queue.Queue = _queue.Queue()
    _SENTINEL = object()

    def collect(evt: dict) -> None:
        events.put(evt)

    def worker() -> None:
        try:
            response_text = runtime.chat(session_id, message, on_event=collect, refresh=refresh)
            events.put({"__result": response_text})
        except Exception as e:
            # Sanitized like POST /chat: details stay in the log (keyed by
            # the X-Request-ID header), clients get a generic in-band error.
            log.error("api_stream_failed", session_id=session_id, error=str(e))
            events.put({"__error": _GENERIC_RUNTIME_ERROR})
        finally:
            events.put(_SENTINEL)

    thread = _threading.Thread(target=worker, daemon=True, name=f"sse-{session_id[:8]}")
    thread.start()

    try:
        while True:
            try:
                evt = events.get(timeout=_SSE_HEARTBEAT_SECONDS)
            except _queue.Empty:
                # Keepalive: SSE comment lines are ignored by event parsers.
                yield ": keepalive\n\n"
                continue

            if evt is _SENTINEL:
                break
            if "__result" in evt:
                pending = _pending_or_none(runtime, session_id)
                yield _sse(
                    "done",
                    {
                        "session_id": session_id,
                        "response": evt["__result"],
                        "pending_confirmation": pending.model_dump() if pending else None,
                    },
                )
                break
            if "__error" in evt:
                yield _sse("error", {"detail": evt["__error"]})
                break
            yield _sse(evt.get("type", "event"), {k: v for k, v in evt.items() if k != "type"})
    finally:
        # The per-session lock in runtime.chat releases when the worker's
        # chat() call returns/raises; a brief join keeps the generator from
        # outliving a worker that is still unwinding.
        thread.join(timeout=5)


@app.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> StreamingResponse:
    """
    SSE variant of /chat.

    Media type: text/event-stream. Clients should read until the `done` or
    `error` terminal event. The session is created/validated exactly like
    POST /chat before streaming starts. Concurrent turns on the SAME session
    surface as an in-band `error` event (the response has already begun).
    """
    _enforce_rate_limit(request)
    session_id = _resolve_session(runtime, payload.session_id)
    return StreamingResponse(
        _event_stream(runtime, session_id, payload.message, refresh=payload.refresh),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{session_id}/history")
def session_history(
    session_id: str,
    request: Request,
    limit: int | None = Query(default=None, ge=1, le=1000),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> dict:
    """Persisted history. `limit` bounds the response (default: store default)."""
    _enforce_rate_limit(request)
    if not runtime.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    history = runtime.store.load_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": history}


@app.get("/sessions/{session_id}/confirmation")
def get_confirmation(
    session_id: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> PendingAction:
    _enforce_rate_limit(request)
    pending = runtime.get_pending_confirmation(session_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending confirmation for this session.")
    return PendingAction(
        tool_name=pending["tool_name"],
        tool_args=pending["tool_args"],
        risk_level=pending["risk_level"],
    )


@app.post("/sessions/{session_id}/confirm")
def resolve_confirmation(
    session_id: str,
    payload: ConfirmationRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> dict:
    _enforce_rate_limit(request)
    if not runtime.get_pending_confirmation(session_id):
        raise HTTPException(status_code=404, detail="No pending confirmation for this session.")
    try:
        response_text = runtime.handle_confirmation(session_id, payload.confirmed)
    except TimeoutError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"session_id": session_id, "resolved": True, "response": response_text}


@app.get("/tools", response_model=list[ToolInfo])
def list_tools(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[ToolInfo]:
    _enforce_rate_limit(request)
    return [
        ToolInfo(name=name, risk_level=runtime.registry.get_tool_risk_level(name))
        for name in runtime.registry.list_tools()
    ]


# ── Operator introspection + explicit UNKNOWN recovery (v0.18) ────────────────
# Reading is protected by the same auth as every other endpoint; the one
# MUTATING operation (reissue) goes through the same _AUTH dependency and can
# never be triggered by a GET.


def _safe_action_info(store: SessionStore, row: "ActionExecution") -> ActionInfo:
    """Project a ledger row to its safe, disclosable metadata."""
    origin = store.get_reissue_origin(row.action_id)
    return ActionInfo(
        action_id=row.action_id,
        session_id=row.session_id,
        confirmation_id=row.confirmation_id,
        tool_name=row.tool_name,
        risk_level=row.risk_level,
        state=row.state,
        attempt=row.attempt,
        created_at=row.created_at,
        claimed_at=row.claimed_at,
        finished_at=row.finished_at,
        reissue_depth=store.count_reissues_for_action(row.action_id),
        reissued_from=origin,
    )


@app.get("/actions", response_model=list[ActionInfo])
def list_actions(
    request: Request,
    state: str | None = Query(
        default=None,
        description="Filter by ledger state (PENDING/RUNNING/SUCCEEDED/FAILED/UNKNOWN).",
    ),
    session_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=500),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[ActionInfo]:
    """Read-only execution-ledger inspection (bounded, newest first)."""
    _enforce_rate_limit(request)
    try:
        rows = runtime.store.list_action_executions(
            state=state, session_id=session_id, limit=limit
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return [_safe_action_info(runtime.store, r) for r in rows]


@app.get("/actions/{action_id}", response_model=ActionInfo)
def get_action(
    action_id: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> ActionInfo:
    """Read-only detail for one ledger row (404 when unknown)."""
    _enforce_rate_limit(request)
    row = runtime.store.get_action_execution(action_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown action_id.")
    return _safe_action_info(runtime.store, row)


@app.get("/sessions/leases", response_model=list[LeaseInfo])
def list_leases(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[LeaseInfo]:
    """
    Read-only session-lease inspection: who (redacted) holds which session,
    until when, at which fencing token. Stale leases show ``active: false``.
    """
    _enforce_rate_limit(request)
    return [
        LeaseInfo(
            session_id=lease["session_id"],
            owner=redact_owner(lease["owner_token"]),
            acquired_at=lease["acquired_at"],
            expires_at=lease["expires_at"],
            fencing=lease["fencing"],
            active=lease["active"],
        )
        for lease in runtime.store.list_session_leases(limit=limit)
    ]


@app.post("/actions/{action_id}/reissue", response_model=ReissueResponse)
def reissue_action(
    action_id: str,
    payload: ReissueRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> ReissueResponse:
    """
    Explicitly re-issue an UNKNOWN action as a NEW action identity.

    This is NOT a retry and not automatic: it mints a fresh PENDING ledger
    row (new action_id) that must pass the normal permission/confirmation
    flow before anything executes. Idempotent per request_id; bounded by
    MAX_REISSUES_PER_ACTION; the original row stays UNKNOWN for audit.
    """
    _enforce_rate_limit(request)
    store = runtime.store
    # Idempotency key: client-supplied when given; otherwise server-generated
    # (returned in the response so the client can replay it). A server key is
    # NOT an authorization credential — auth is enforced by the dependency.
    request_id = payload.request_id or f"srv-{uuid.uuid4().hex}"
    log.info(
        "action_reissue_requested",
        action_id=action_id,
        request_id=request_id,
    )
    # Was this exact request already satisfied before this call? (The store
    # enforces true idempotency via a UNIQUE index; this flag is informational.)
    already_served = any(
        r["request_id"] == request_id
        for r in store.get_reissue_chain(action_id)
    )
    try:
        new_id = store.request_action_reissue(action_id, request_id)
    except ValueError as e:
        msg = str(e)
        status = 404 if "unknown action_id" in msg else 409
        raise HTTPException(status_code=status, detail=msg) from e
    return ReissueResponse(
        original_action_id=action_id,
        new_action_id=new_id,
        state=store.get_action_execution(new_id).state,
        reissue_depth=store.count_reissues_for_action(action_id),
        max_reissues=MAX_REISSUES_PER_ACTION,
        reused_existing=already_served,
        request_id=request_id,
    )


@app.get("/sessions/{session_id}/timeline", response_model=list[TimelineEvent])
def get_session_timeline(
    session_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[TimelineEvent]:
    """
    Read-only causal timeline for one session (v0.19): messages,
    confirmation parks/resolutions, action states, reissues, and the
    lease — merged in timestamp order. Safe metadata only: never tool
    arguments, result bodies, message content, or raw owner tokens.
    """
    _enforce_rate_limit(request)
    if not runtime.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    events = runtime.store.get_session_timeline(session_id, limit=limit)
    return [TimelineEvent(**e) for e in events]


# ── v0.24: cross-turn result cache (safe metadata only) ───────────────────────


@app.get("/ops/cache/stats")
def get_cache_stats(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> dict:
    """
    Bounded cache telemetry (Part R): entries / hits / expired / per-tool
    counts. NEVER exposes cached payloads or argument fingerprints.
    """
    _enforce_rate_limit(request)
    return runtime.store.cache_stats()


@app.get("/ops/cache/stats/history")
def get_cache_stats_history(
    request: Request,
    days: int = Query(default=14, ge=1, le=365),
    limit: int = Query(default=30, ge=1, le=365),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[dict]:
    """
    v0.25 historical cache-operations aggregates (Part E3): one row per day
    {day, hits, misses, stale, bypass, stores, per_tool}. Aggregates only —
    no queries, fingerprints, or payloads. Bounded by days/limit.
    """
    _enforce_rate_limit(request)
    return runtime.store.cache_metrics_history(days=days, limit=limit)


# ── v0.26: grounding-guard telemetry (aggregate counts only) ─────────────────


@app.get("/ops/grounding/stats")
def get_grounding_stats(
    request: Request,
    days: int = Query(default=14, ge=1, le=365),
    limit: int = Query(default=30, ge=1, le=365),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[dict]:
    """
    v0.26 daily grounding-guard aggregates: one row per day {day, checks,
    contradictions, corrections, corrections_ok, corrections_failed,
    fallbacks, per_tool}. Counts only — never answers, evidence text, or
    arguments. Bounded by days/limit.
    """
    _enforce_rate_limit(request)
    return runtime.store.grounding_metrics_history(days=days, limit=limit)


# ── v0.28: safe browser control — status + emergency stop (human-only) ───────


def _browser_status_payload(runtime: JarvisRuntime) -> BrowserStatus:
    """Assemble safe browser status (no page content, no visited URLs)."""
    from jarvis.browser.emergency import get_emergency_stop
    from jarvis.browser.registry import get_browser_registry

    stop = get_emergency_stop()
    registry = get_browser_registry()
    downloads = 0
    for controller in registry.controllers():
        try:
            downloads += int(controller.status().get("downloads", 0))
        except Exception:  # pragma: no cover - best-effort status
            continue
    return BrowserStatus(
        enabled=bool(getattr(settings, "ENABLE_BROWSER_CONTROL", False)) and any(
            name in set(runtime.registry.list_tools())
            for name in ("open_url", "click_element")
        ),
        driver=str(settings.BROWSER_DRIVER),
        emergency_stop_active=bool(stop.status()["active"]),
        emergency_stop_reason=str(stop.status()["reason"]),
        emergency_stop_token=int(stop.status()["token"]),
        open_sessions=registry.open_count(),
        max_sessions=int(settings.BROWSER_MAX_SESSIONS),
        downloads_captured=downloads,
    )


@app.get("/browser/status", response_model=BrowserStatus)
def browser_status(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> BrowserStatus:
    """
    v0.28: safe-browser posture snapshot. Bounded, safe metadata only —
    whether the surface is enabled, the configured driver, the emergency
    stop state, and open-session/download counts. Never page content,
    URLs, or arguments.
    """
    _enforce_rate_limit(request)
    return _browser_status_payload(runtime)


@app.post("/browser/emergency-stop", response_model=BrowserStopResponse)
def browser_emergency_stop(
    payload: BrowserStopRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> BrowserStopResponse:
    """
    v0.28: HUMAN-ONLY emergency stop. Interrupts any in-flight browser
    action at the next runtime gate check; blocked actions report
    ACTION_INTERRUPTED. The model has no tool that can trigger or reset
    this. Also closes all open browser controllers (bounded resources).
    """
    _enforce_rate_limit(request)
    stop = get_emergency_stop()
    token = stop.trigger(reason=payload.reason)
    try:
        get_browser_registry().close_all()
    except Exception as e:  # pragma: no cover - close is best-effort
        log.warning("browser_registry_close_failed_on_stop", error=str(e))
    log.info("browser_api_emergency_stop", token=token, reason=payload.reason[:120])
    return BrowserStopResponse(triggered=True, token=token, reason=payload.reason)


@app.post("/browser/emergency-reset", response_model=BrowserResetResponse)
def browser_emergency_reset(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> BrowserResetResponse:
    """v0.28: explicit operator reset of the emergency stop (idempotent)."""
    _enforce_rate_limit(request)
    was_active = get_emergency_stop().reset()
    return BrowserResetResponse(
        was_active=was_active,
        message="emergency stop cleared" if was_active else "no stop was active",
    )


# ── v0.20: personal knowledge base (documents are untrusted data) ────────────


def _knowledge_service(runtime: JarvisRuntime):
    """KnowledgeService over the runtime's own stores (isolated in tests)."""
    from jarvis.memory.knowledge import KnowledgeService

    return KnowledgeService(store=runtime.store)


@app.get("/knowledge/documents", response_model=list[KnowledgeDocumentInfo])
def list_knowledge_documents(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[KnowledgeDocumentInfo]:
    """Read-only knowledge index listing (safe metadata, bounded)."""
    _enforce_rate_limit(request)
    return [
        KnowledgeDocumentInfo(**d)
        for d in _knowledge_service(runtime).list_documents(limit=limit)
    ]


@app.get("/knowledge/documents/{document_id}", response_model=KnowledgeDocumentInfo)
def get_knowledge_document(
    document_id: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> KnowledgeDocumentInfo:
    """One document's safe metadata (404 unknown)."""
    _enforce_rate_limit(request)
    doc = _knowledge_service(runtime).inspect_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Unknown document_id.")
    return KnowledgeDocumentInfo(**doc)


@app.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
def ingest_knowledge_document(
    payload: KnowledgeIngestRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> KnowledgeIngestResponse:
    """
    Explicitly ingest ONE document (mutating; authenticated).

    Path safety is enforced server-side: the resolved path must be inside
    the allowed directory, credential-like files are refused, and no
    crawling ever happens. Unknown/unsupported/oversized files → 400 with
    the reason; a duplicate unchanged file returns ``unchanged`` without
    re-embedding.
    """
    _enforce_rate_limit(request)
    report = _knowledge_service(runtime).ingest(
        payload.path,
        target_chars=payload.target_chars,
        overlap_chars=payload.overlap_chars,
    )
    if report.get("status") == "error":
        raise HTTPException(status_code=400, detail=str(report.get("reason")))
    log.info(
        "knowledge_api_ingest",
        status=report.get("status"),
        document_id=report.get("document_id"),
    )
    return KnowledgeIngestResponse(**report)


@app.post("/knowledge/search", response_model=KnowledgeSearchResponse)
def search_knowledge_endpoint(
    payload: KnowledgeSearchRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> KnowledgeSearchResponse:
    """Read-only bounded retrieval with citation-ready metadata."""
    _enforce_rate_limit(request)
    report = _knowledge_service(runtime).search(
        payload.query,
        top_k=payload.top_k,
        source=payload.source,
        document_id=payload.document_id,
    )
    return KnowledgeSearchResponse(
        results=[KnowledgeSearchResult(**r) for r in report["results"]],
        total_chunks=report["total_chunks"],
    )


@app.delete("/knowledge/documents/{document_id}", response_model=KnowledgeRemoveResponse)
def delete_knowledge_document(
    document_id: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> KnowledgeRemoveResponse:
    """Explicitly remove a document (mutating; authenticated; 404 unknown)."""
    _enforce_rate_limit(request)
    svc = _knowledge_service(runtime)
    if svc.inspect_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Unknown document_id.")
    report = svc.remove_document(document_id)
    log.info(
        "knowledge_api_remove",
        document_id=document_id,
        chunks=report["chunks_removed"],
    )
    return KnowledgeRemoveResponse(**report)


# ── v0.29: integration management (metadata + connect/disconnect ONLY) ──────
#
# These endpoints manage CONNECTIONS, never external resources. Creating/
# updating/deleting calendar events or tasks happens ONLY through the chat
# runtime (normal tool path: dynamic risk → confirmation → action ledger).
# There is deliberately NO endpoint that dispatches a provider operation:
# no second execution API that bypasses permissions (Part 20).
# Account metadata is public projection only — credential material is
# structurally absent from every response (Part 4).


def _integration_manager_or_404(runtime: JarvisRuntime):
    manager = getattr(runtime, "integration_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=404,
            detail="Integrations are not enabled on this runtime "
            "(set ENABLE_INTEGRATIONS=true).",
        )
    return manager


@app.get("/integrations")
def list_integrations(
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[dict]:
    """
    Provider catalog + connected accounts (safe metadata: scopes, auth
    state, verification timestamps; NEVER credentials/tokens).
    """
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    out: list[dict] = []
    for name, provider in manager.providers().items():
        cap = provider.capabilities
        out.append(
            {
                "provider": name,
                "display_name": cap.display_name,
                "production_like": cap.production_like,
                "grantable_scopes": sorted(cap.grantable_scopes),
                "resource_kind": cap.resource.kind,
                "operations": sorted(op.value for op in cap.resource.operations),
                "supports_idempotency_key": cap.supports_idempotency_key,
                "supports_oauth": bool(getattr(provider, "supports_oauth", False)),
                "accounts": [
                    a.public_metadata() for a in manager.list_accounts(name)
                ],
            }
        )
    return out


@app.get("/integrations/{provider}/scopes")
def integration_scopes(
    provider: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> dict:
    """The exact scope menu for one provider (no wildcards exist)."""
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    impl = manager.providers().get(provider)
    if impl is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
    cap = impl.capabilities
    return {
        "provider": provider,
        "grantable_scopes": sorted(cap.grantable_scopes),
        "per_operation": {
            op.value: cap.resource.scope_for(op)
            for op in sorted(cap.resource.operations, key=lambda o: o.value)
        },
        "per_operation_risk": {
            op.value: cap.resource.operation_risk[op].value
            for op in sorted(cap.resource.operations, key=lambda o: o.value)
        },
    }


@app.post("/integrations/{provider}/connect", response_model=IntegrationAccountInfo)
def integration_connect(
    provider: str,
    payload: IntegrationConnectRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> IntegrationAccountInfo:
    """
    Explicitly connect one account (user-controlled; Part 19). The
    credential is supplied by the OPERATOR here — never by the chat model.
    Omitting the credential generates a random local-dev token (development
    providers). Invalid providers/scopes are 400; duplicates are 409.
    """
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    try:
        account = manager.connect(
            provider=provider,
            display_label=payload.display_label,
            credential=payload.credential,
            scopes=frozenset(payload.scopes),
            provider_account_ref=payload.provider_account_ref or "",
        )
    except IntegrationManagerError as e:
        msg = str(e)
        status = 409 if "already connected" in msg else 400
        raise HTTPException(status_code=status, detail=msg) from e
    return IntegrationAccountInfo(**account.public_metadata())


def _oauth_page(ok: bool, detail: str) -> str:
    """
    v0.30: the fixed LOCAL result page for the OAuth callback. No redirects
    are ever issued from here (no open redirect), no token material is ever
    rendered, and the detail string is HTML-escaped and bounded.
    """
    import html as _html

    title = "Authorization complete" if ok else "Authorization failed"
    color = "#1a7f37" if ok else "#b3261e"
    safe = _html.escape(str(detail)[:220])
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{title}</title></head>"
        "<body style=\"font-family:system-ui;padding:2rem;max-width:40rem\">"
        f"<h2 style=\"color:{color}\">{title}</h2><p>{safe}</p>"
        "<p style=\"color:#666\">You can close this window and return to JARVIS.</p>"
        "</body></html>"
    )


@app.post("/integrations/{provider}/authorize", response_model=IntegrationAuthorizationStart)
def integration_authorize(
    provider: str,
    payload: IntegrationAuthorizeRequest,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> IntegrationAuthorizationStart:
    """
    Start ONE user-controlled OAuth authorization (v0.30). The returned
    authorization_url is opened by the OPERATOR in a browser — it is never
    given to the chat model, and the one-time state it carries is persisted
    only as a hash, bound to this session/provider/label/scopes/redirect.
    """
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    try:
        start = manager.begin_authorization(
            provider=provider,
            session_id=payload.session_id,
            display_label=payload.display_label,
            scopes=frozenset(payload.scopes),
        )
    except IntegrationManagerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return IntegrationAuthorizationStart(**start)


@app.get("/integrations/{provider}/authorize/status", response_model=IntegrationStatusResponse)
def integration_authorize_status(
    provider: str,
    request: Request,
    session_id: str = Query(..., min_length=1, max_length=64),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> IntegrationStatusResponse:
    """Bounded polling view for a started authorization (no state values)."""
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    return IntegrationStatusResponse(
        **manager.authorization_flow_outcome(provider, session_id)
    )


@app.get("/integrations/oauth/callback/{provider}")
def integration_oauth_callback(
    provider: str,
    request: Request,
    code: str | None = Query(default=None, max_length=512),
    state: str | None = Query(default=None, max_length=256),
    session_id: str | None = Query(default=None, max_length=64),
    error: str | None = Query(default=None, max_length=64),
    runtime: JarvisRuntime = Depends(get_runtime),
):
    """
    The OAuth redirect target (v0.30 Parts 4/5/6).

    DELIBERATELY NOT behind ``_AUTH``: this URL is visited by the user's
    BROWSER after the provider redirect, so an API key cannot be attached.
    The ONE-TIME, hash-stored, session/provider/redirect-bound STATE is the
    authenticator here (standard OAuth semantics): a missing, expired,
    replayed, or mismatched state fails closed and no account is created.
    The response is a fixed local HTML page — never a redirect, never a
    token, never a raw provider payload.
    """
    from fastapi.responses import HTMLResponse

    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    if error:
        return HTMLResponse(
            _oauth_page(False, "the provider reported the request was not approved"),
            status_code=400,
        )
    if not code or not state or not session_id:
        return HTMLResponse(
            _oauth_page(False, "the callback is missing code/state"),
            status_code=400,
        )
    try:
        account = manager.handle_callback(
            provider=provider, code=code, state=state, session_id=session_id
        )
    except IntegrationManagerError as e:
        return HTMLResponse(_oauth_page(False, str(e)), status_code=400)
    return HTMLResponse(
        _oauth_page(
            True,
            f"{account.provider} · {account.display_label} "
            f"({account.account_id}) — status {account.authorization_status}",
        )
    )


@app.post("/integrations/accounts/{account_id}/refresh", response_model=IntegrationAccountInfo)
def integration_refresh(
    account_id: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> IntegrationAccountInfo:
    """
    Re-verify one account NOW (opportunistic OAuth token refresh included).
    Returns SAFE metadata only; token material never appears in a response.
    """
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    account = manager.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Unknown account_id.")
    updated = manager.refresh_auth_state(account)
    return IntegrationAccountInfo(**updated.public_metadata())


@app.post("/integrations/accounts/{account_id}/disconnect", response_model=IntegrationDisconnectResponse)
def integration_disconnect(
    account_id: str,
    request: Request,
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> IntegrationDisconnectResponse:
    """Disconnect (and locally revoke) one connected account. 404 unknown."""
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    if manager.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail="Unknown account_id.")
    removed = manager.disconnect(account_id)
    return IntegrationDisconnectResponse(
        disconnected=removed,
        account_id=account_id,
        message="Account disconnected. Stored credentials were removed from "
        "the local database; revoke the grant at the provider if it is a "
        "real account.",
    )


@app.get("/integrations/accounts/{account_id}/audit", response_model=list[IntegrationAuditEntry])
def integration_account_audit(
    account_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    runtime: JarvisRuntime = Depends(get_runtime),
    _auth: None = _AUTH,
) -> list[IntegrationAuditEntry]:
    """Recent EXTERNAL side effects for one account (safe fields only)."""
    _enforce_rate_limit(request)
    manager = _integration_manager_or_404(runtime)
    if manager.get_account(account_id) is None:
        raise HTTPException(status_code=404, detail="Unknown account_id.")
    rows = manager.recent_audit(account_id=account_id, limit=limit)
    return [IntegrationAuditEntry(**r) for r in rows]
