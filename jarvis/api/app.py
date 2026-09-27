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
    ChatRequest,
    ChatResponse,
    ConfirmationRequest,
    HealthResponse,
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
from jarvis.runtime import JarvisRuntime, build_runtime
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="JARVIS API",
    description="Local-first AI assistant — agent runtime over HTTP.",
    version="0.20.0",
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
        response_text = runtime.chat(session_id, payload.message)
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


# ── Streaming (SSE) ───────────────────────────────────────────────────────────


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event (JSON payload for typed clients)."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# Heartbeat cadence for the SSE stream: proxies (nginx default 60s idle
# timeout) close quiet connections; a keepalive comment every 15s keeps the
# path open during long LLM turns.
_SSE_HEARTBEAT_SECONDS = 15.0


def _event_stream(runtime: JarvisRuntime, session_id: str, message: str):
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
            response_text = runtime.chat(session_id, message, on_event=collect)
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
        _event_stream(runtime, session_id, payload.message),
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
