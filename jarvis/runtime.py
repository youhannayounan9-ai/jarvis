"""
jarvis/runtime.py
─────────────────
JarvisRuntime — the single assembly point for a JARVIS "agent runtime".

This is the seam between the *agent runtime* (orchestrator, store, tools,
guard) and the *interfaces* (CLI, REST API, Streamlit dashboard, voice).
Interfaces build a runtime via ``build_runtime()`` and talk to it; they never
wire individual components themselves.

Why one assembly point?
  Previously main.py and ui/dashboard.py each had their own ~15-line wiring
  that had drifted apart (different tool sets). A single runtime keeps the
  capability surface identical across CLI, API, and UI, and gives the service
  layer a clean object to own.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from jarvis import __version__
from jarvis.config import settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.core.sandbox import DockerCodeSandbox
from jarvis.memory.session_store import SessionStore, new_owner_token
from jarvis.memory.vector_store import get_vector_store
from jarvis.tools.search_knowledge import SearchKnowledgeTool
from jarvis.tools import (
    CalculatorTool,
    CodeExecutionTool,
    GetCurrentDatetimeTool,
    ListDirectoryTool,
    ReadFileTool,
    RecallFactsTool,
    RememberFactTool,
    ToolRegistry,
    VisionAnalyzeTool,
    WebScrapeTool,
    WebSearchTool,
    WikipediaSummaryTool,
    WriteFileTool,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# The single source of truth for the ACTIVE capability surface.
# computer_control is NEVER registered (fail-closed placeholder only).
# execute_python_code joins the surface ONLY when explicitly enabled in
# config AND verified Docker isolation is actually available (checked via
# _build_code_execution_tool). See docs/architecture.md § Safety Model.
_TOOL_FACTORIES: tuple = (
    GetCurrentDatetimeTool,
    WebSearchTool,
    WikipediaSummaryTool,
    ReadFileTool,
    ListDirectoryTool,
    CalculatorTool,
    RememberFactTool,
    RecallFactsTool,
    SearchKnowledgeTool,
    WriteFileTool,
    VisionAnalyzeTool,
    WebScrapeTool,
)


def _build_code_execution_tool() -> CodeExecutionTool | None:
    """
    Return a CodeExecutionTool only when code execution is BOTH explicitly
    enabled in config AND backed by verified container isolation. Otherwise
    return None (tool absent from the surface — the LLM never sees it).

    Fail-closed ladder:
      - not enabled in config                            → None
      - bad SANDBOX_IMAGE reference (ValueError)         → None
      - docker missing / daemon down / image not pulled
        / unsupported host (is_available() False)        → None
    """
    if not getattr(settings, "ENABLE_CODE_EXECUTION", False):
        return None
    try:
        sandbox = DockerCodeSandbox(image=settings.SANDBOX_IMAGE)
    except ValueError as e:
        log.error("code_execution_bad_sandbox_image", error=str(e))
        return None
    if not sandbox.is_available():
        log.error(
            "code_execution_enabled_but_docker_unavailable",
            image=settings.SANDBOX_IMAGE,
        )
        return None
    log.info("code_execution_enabled", image=settings.SANDBOX_IMAGE)
    return CodeExecutionTool(sandbox=sandbox)


@dataclass
class JarvisRuntime:
    """Owns a fully-wired agent runtime: store, registry, guard, orchestrator."""

    store: SessionStore
    registry: ToolRegistry
    guard: PermissionGuard
    orchestrator: Orchestrator
    sessions_created: int = 0
    _created_sessions: set[str] = field(default_factory=set)
    # One chat turn per session at a time: parallel turns on the same session
    # would interleave persisted history and double-resolve confirmations.
    # Locks are created lazily per session; the map itself is guarded.
    _session_locks: dict[str, threading.Lock] = field(default_factory=dict)
    _session_locks_guard: threading.Lock = field(default_factory=threading.Lock)
    # v0.17: durable identity of THIS runtime instance for session leases.
    owner_token: str = field(default_factory=lambda: new_owner_token("runtime"))

    @contextmanager
    def _exclusive_session(self, session_id: str) -> Iterator[None]:
        """
        Hold a per-session mutex for the duration of one chat turn.

        Non-blocking: if another turn is already running for this session,
        raise TimeoutError immediately so the caller can return 409 Conflict
        instead of interleaving history or queueing an unbounded wait.
        Different sessions never contend.

        v0.17: this in-process mutex is now the FAST layer; cross-process
        exclusion is provided by the database-backed session lease acquired
        inside it (see chat / handle_confirmation).
        """
        with self._session_locks_guard:
            lock = self._session_locks.setdefault(session_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise TimeoutError(f"session {session_id} is busy processing another turn")
        try:
            yield
        finally:
            lock.release()

    def _acquire_session_lease_or_busy(self, session_id: str) -> int:
        """
        Acquire the database-backed session lease or raise TimeoutError.

        Returns the fencing token. Raises TimeoutError when another live
        owner holds the lease (mapped to HTTP 409 by the API layer). The
        lease expires after SESSION_LEASE_TTL_SECONDS, so a crashed owner
        cannot lock a session forever.
        """
        acquired, fencing = self.store.acquire_session_lease(
            session_id, self.owner_token
        )
        if not acquired:
            raise TimeoutError(
                f"session {session_id} is busy processing another turn"
            )
        return fencing

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start_session(self) -> str:
        """Create a new conversation session and bind long-term memory to it."""
        session_id = self.store.create_session()
        get_vector_store().set_session(session_id)
        self._created_sessions.add(session_id)
        self.sessions_created += 1
        return session_id

    def chat(self, session_id: str, user_input: str, **kwargs) -> str:
        """
        Process one user turn (thin passthrough so callers need only the runtime).

        Keyword args (e.g. ``on_event`` for step observation) forward to the
        orchestrator unchanged.

        Concurrency: guarded by BOTH the in-process per-session mutex (fast
        path) and a database-backed session lease (multi-process safety).
        The lease is released when the turn finishes; a crashed process's
        lease expires after SESSION_LEASE_TTL_SECONDS so the session can
        recover. Residual race (documented): a single turn that outlives the
        lease TTL may lose cross-process exclusivity near the TTL boundary.

        Raises:
            TimeoutError: if another turn is already in flight for this
                session, in this process (mutex) or another process sharing
                the database (lease). Callers map this to HTTP 409.
        """
        with self._exclusive_session(session_id):
            self._acquire_session_lease_or_busy(session_id)
            try:
                return self.orchestrator.chat(session_id, user_input, **kwargs)
            finally:
                # Release even on failure so one error never wedges the
                # session until the TTL. (Best-effort; expiry backstops.)
                try:
                    self.store.release_session_lease(session_id, self.owner_token)
                except Exception:  # pragma: no cover - release is best-effort
                    log.warning("session_lease_release_failed", session_id=session_id)

    def handle_confirmation(self, session_id: str, confirmed: bool) -> str:
        """
        Confirm or deny the session's pending high-risk action.

        Takes the same session lease as chat(): resolution resumes the paused
        turn (plan steps + synthesis), which must not interleave with a
        concurrent chat on the same session. Duplicate resolutions are still
        safe even without the lease (the confirmation pop and the action
        claim are atomic) — the lease prevents interleaved HISTORY writes.
        """
        with self._exclusive_session(session_id):
            self._acquire_session_lease_or_busy(session_id)
            try:
                return self.orchestrator.handle_confirmation(session_id, confirmed)
            finally:
                try:
                    self.store.release_session_lease(session_id, self.owner_token)
                except Exception:  # pragma: no cover - release is best-effort
                    log.warning("session_lease_release_failed", session_id=session_id)

    def get_pending_confirmation(self, session_id: str) -> dict | None:
        return self.orchestrator.get_pending_confirmation(session_id)

    def session_exists(self, session_id: str) -> bool:
        """True if the session was created by this runtime or exists in the store."""
        if session_id in self._created_sessions:
            return True
        return self.store.message_count(session_id) >= 0 and self._session_row_exists(session_id)

    def _session_row_exists(self, session_id: str) -> bool:
        row = self.store._conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        """Release resources (DB connection)."""
        self.store.close()

    def __enter__(self) -> "JarvisRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Introspection ──────────────────────────────────────────────────────────

    def describe(self) -> dict:
        """Machine-readable summary for /health and startup banners."""
        registered = self.registry.list_tools()
        registered_set = set(registered)
        disabled = [
            name
            for name in ("execute_python_code", "computer_control")
            if name not in registered_set
        ]
        return {
            "version": __version__,
            "model": settings.ollama_model,
            "planner_model": settings.planner_model,
            "tools": registered,
            "disabled_tools": disabled,
            "db_path": settings.db_path,
        }


def build_runtime() -> JarvisRuntime:
    """
    Assemble a complete runtime with the standard tool surface.

    Interfaces (CLI, API, dashboard) call this instead of wiring components
    themselves — one place to add a tool or change assembly.
    """
    store = SessionStore()
    # Warm the long-term memory singleton (creates local Chroma path if needed).
    get_vector_store()

    registry = ToolRegistry()
    for factory in _TOOL_FACTORIES:
        registry.register(factory())

    # Conditional capability: only a verified-isolation sandbox earns a
    # registry slot. Absent otherwise — the LLM never sees the tool.
    code_tool = _build_code_execution_tool()
    if code_tool is not None:
        registry.register(code_tool)

    guard = PermissionGuard()
    orchestrator = Orchestrator(store=store, tool_registry=registry, permission_guard=guard)

    log.info(
        "runtime_built",
        tools=len(registry),
        model=settings.ollama_model,
        version=__version__,
    )
    return JarvisRuntime(store=store, registry=registry, guard=guard, orchestrator=orchestrator)
