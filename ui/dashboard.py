"""
ui/dashboard.py
───────────────
JARVIS Web Dashboard (Streamlit).

v0.11: the dashboard is a *client* of the JARVIS REST API — it no longer
wires an in-process agent runtime. Set JARVIS_API_URL to point at a running
API server (see deploy/README.md); the dashboard can then run on a different
machine than the agent runtime.

Backward compatibility: if JARVIS_API_URL is empty, the dashboard falls back
to the legacy in-process runtime (same process builds build_runtime()) so
existing single-machine setups keep working.

Run with: streamlit run ui/dashboard.py
"""

import os
import uuid
from typing import Any

import streamlit as st

from jarvis.api.client import JarvisClientError
from jarvis.config import settings

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="JARVIS Dashboard", page_icon="🧠", layout="wide")


# ── Backend abstraction (API client first, legacy runtime fallback) ──────────


class ApiBackend:
    """Talks to a running JARVIS API server (the service surface)."""

    mode = "api"

    def __init__(self) -> None:
        from jarvis.api.client import JarvisClient

        self._client = JarvisClient(
            base_url=settings.JARVIS_API_URL or "http://127.0.0.1:8000",
            api_key=settings.JARVIS_CLIENT_API_KEY or None,
        )

    def health(self) -> dict[str, Any]:
        return self._client.health()

    def start_session(self) -> str:
        return self._client.create_session()

    def chat(self, session_id: str, message: str, refresh: bool = False) -> str:
        result = self._client.chat(message, session_id, refresh=refresh)
        return result.get("response", "")

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self._client.history(session_id)

    def message_count(self, session_id: str) -> int:
        return len(self.history(session_id))

    def pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        return self._client.get_confirmation(session_id)

    def resolve_confirmation(self, session_id: str, confirmed: bool) -> str:
        body = self._client.resolve_confirmation(session_id, confirmed)
        return body.get("response", "")

    # ── v0.19 operator surface (all read-only except reissue) ────────────────

    def list_actions(
        self, state: str | None = None, session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._client.list_actions(state=state, session_id=session_id,
                                         limit=limit)

    def get_action(self, action_id: str) -> dict[str, Any]:
        return self._client.get_action(action_id)

    def list_leases(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._client.list_leases(limit=limit)

    def reissue_action(self, action_id: str,
                       request_id: str | None = None) -> dict[str, Any]:
        return self._client.reissue_action(action_id, request_id)

    def session_timeline(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self._client.session_timeline(session_id, limit=limit)

    # ── v0.20 knowledge base ─────────────────────────────────────────────────

    def list_knowledge_documents(self, limit: int = 100):
        return self._client.list_knowledge_documents(limit=limit)

    def ingest_knowledge_document(self, path: str):
        return self._client.ingest_knowledge_document(path)

    def search_knowledge(self, query: str, top_k: int = 4, source: str | None = None):
        return self._client.search_knowledge(query, top_k=top_k, source=source)

    def remove_knowledge_document(self, document_id: str):
        return self._client.remove_knowledge_document(document_id)

    # ── v0.24 result cache ─────────────────────────────────────────────────

    def cache_stats(self) -> dict[str, Any]:
        return self._client._request("GET", "/ops/cache/stats")

    def cache_history(self, days: int = 14, limit: int = 30) -> list[dict[str, Any]]:
        """Daily cache-operations aggregates (v0.25 Part E) — counts only."""
        return self._client._request(
            "GET",
            "/ops/cache/stats/history",
            params={"days": days, "limit": limit},
        )

    def cache_entries(self, limit: int = 25):
        raise JarvisClientError(501, "Cache payload inspection is not exposed over the API.")


class LegacyBackend:
    """In-process runtime (single-machine fallback; pre-v0.11 behavior)."""

    mode = "legacy"

    def __init__(self) -> None:
        from jarvis.runtime import build_runtime

        self._runtime = build_runtime()

    def start_session(self) -> str:
        return self._runtime.start_session()

    def chat(self, session_id: str, message: str, refresh: bool = False) -> str:
        return self._runtime.chat(session_id, message, refresh=refresh)

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self._runtime.store.load_history(session_id)

    def message_count(self, session_id: str) -> int:
        return self._runtime.store.message_count(session_id)

    def pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        return self._runtime.get_pending_confirmation(session_id)

    def resolve_confirmation(self, session_id: str, confirmed: bool) -> str:
        return self._runtime.handle_confirmation(session_id, confirmed)

    # ── v0.19 operator surface (direct store reads; same safe projections) ───

    def list_actions(
        self, state: str | None = None, session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = self._runtime.store.list_action_executions(
            state=state, session_id=session_id, limit=limit
        )
        return [
            {
                "action_id": r.action_id,
                "session_id": r.session_id,
                "confirmation_id": r.confirmation_id,
                "tool_name": r.tool_name,
                "risk_level": r.risk_level,
                "state": r.state,
                "attempt": r.attempt,
                "created_at": r.created_at,
                "claimed_at": r.claimed_at,
                "finished_at": r.finished_at,
            }
            for r in rows
        ]

    def get_action(self, action_id: str) -> dict[str, Any]:
        r = self._runtime.store.get_action_execution(action_id)
        if r is None:
            raise JarvisClientError(404, "Unknown action_id.")
        return {
            "action_id": r.action_id,
            "session_id": r.session_id,
            "confirmation_id": r.confirmation_id,
            "tool_name": r.tool_name,
            "risk_level": r.risk_level,
            "state": r.state,
            "attempt": r.attempt,
            "created_at": r.created_at,
            "claimed_at": r.claimed_at,
            "finished_at": r.finished_at,
        }

    def list_leases(self, limit: int = 100) -> list[dict[str, Any]]:
        from jarvis.memory.session_store import redact_owner

        return [
            {
                "session_id": l["session_id"],
                "owner": redact_owner(l["owner_token"]),
                "acquired_at": l["acquired_at"],
                "expires_at": l["expires_at"],
                "fencing": l["fencing"],
                "active": l["active"],
            }
            for l in self._runtime.store.list_session_leases(limit=limit)
        ]

    def reissue_action(self, action_id: str,
                       request_id: str | None = None) -> dict[str, Any]:
        from jarvis.memory.session_store import MAX_REISSUES_PER_ACTION

        store = self._runtime.store
        rid = request_id or f"dash-{uuid.uuid4().hex}"
        already_served = any(
            r["request_id"] == rid
            for r in store.get_reissue_chain(action_id)
        )
        try:
            new_id = store.request_action_reissue(action_id, rid)
        except ValueError as e:
            msg = str(e)
            status = 404 if "unknown action_id" in msg else 409
            raise JarvisClientError(status, msg) from e
        return {
            "original_action_id": action_id,
            "new_action_id": new_id,
            "state": store.get_action_execution(new_id).state,
            "reissue_depth": store.count_reissues_for_action(action_id),
            "max_reissues": MAX_REISSUES_PER_ACTION,
            "reused_existing": already_served,
            "request_id": rid,
        }

    def session_timeline(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self._runtime.store.get_session_timeline(session_id, limit=limit)

    # ── v0.20 knowledge base (in-process service; same safe operations) ──────

    def _kb(self):
        from jarvis.memory.knowledge import get_knowledge_service

        return get_knowledge_service(store=self._runtime.store)

    def list_knowledge_documents(self, limit: int = 100):
        return self._kb().list_documents(limit=limit)

    def ingest_knowledge_document(self, path: str):
        return self._kb().ingest(path)

    def search_knowledge(self, query: str, top_k: int = 4, source: str | None = None):
        report = self._kb().search(query, top_k=top_k, source=source)
        return {
            "results": [
                {k: r.get(k) for k in (
                    "chunk_id", "document_id", "filename", "page",
                    "section", "distance", "citation", "snippet",
                )}
                for r in report["results"]
            ],
            "total_chunks": report["total_chunks"],
        }

    def remove_knowledge_document(self, document_id: str):
        return self._kb().remove_document(document_id)

    # ── v0.24 result cache ─────────────────────────────────────────────────

    def cache_stats(self) -> dict[str, Any]:
        return self._runtime.store.cache_stats()

    def cache_history(self, days: int = 14, limit: int = 30) -> list[dict[str, Any]]:
        """Daily cache-operations aggregates (v0.25 Part E) — counts only."""
        return self._runtime.store.cache_metrics_history(days=days, limit=limit)

    def cache_entries(self, limit: int = 25) -> list[dict[str, Any]]:
        from jarvis.memory.session_store import _parse_ts

        rows = self._runtime.store.list_cache_entries(limit=limit)
        import time as _time

        for r in rows:
            try:
                age = max(0.0, _time.time() - _parse_ts(str(r["created_at"])).timestamp())
            except Exception:
                age = None
            r["age_human"] = (
                f"{int(age // 60)}m" if age is not None and age < 86400
                else (f"{int(age // 86400)}d" if age is not None else "?")
            )
        return rows


@st.cache_resource
def get_backend() -> Any:
    """API mode when JARVIS_API_URL is set; legacy in-process runtime otherwise."""
    if settings.JARVIS_API_URL:
        return ApiBackend()
    return LegacyBackend()


backend = get_backend()

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = backend.start_session()


def start_new_session() -> None:
    st.session_state.session_id = backend.start_session()


# ── Operations view (v0.19): read-only introspection + explicit recovery ──────


def _ops_error(e: Exception, context: str) -> None:
    """Operator-level error message; never crash the page."""
    if isinstance(e, JarvisClientError):
        if e.status == 401:
            st.error("🔒 Authentication required: set JARVIS_CLIENT_API_KEY to the server's JARVIS_API_KEY.")
            return
        if e.status == 404:
            st.warning(f"Not found: {e.detail}")
            return
        if e.status == 409:
            st.warning(f"Conflict: {e.detail}")
            return
        if e.status == 429:
            st.warning(f"Rate limited by the API: {e.detail} — wait a moment and retry.")
            return
        if e.status == 0:
            st.error(f"🔌 API unreachable while {context}: {e.detail}")
            return
        st.error(f"API error while {context} (HTTP {e.status}): {e.detail}")
        return
    st.error(f"Unexpected error while {context}: {type(e).__name__}: {e}")


def _safe(caller, context: str, default=None):
    """Run a backend call; map failures to operator messages."""
    try:
        return caller(), None
    except Exception as e:  # noqa: BLE001 — the page must never crash
        _ops_error(e, context)
        return (default if default is not None else (None,)), e


def _render_ops(backend: Any, current_session: str) -> None:
    st.title("🛠️ Operations")
    st.caption(
        "Read-only reliability state. Reissue is the ONLY mutation here and "
        "always requires two explicit confirmations. Owner tokens are "
        "redacted; tool arguments and result bodies are never shown."
    )

    section = st.radio(
        "Section",
        ["Recent actions", "UNKNOWN actions", "Session leases", "Session timeline", "Plan status", "Result cache"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Recent actions ───────────────────────────────────────────────────
    if section == "Recent actions":
        c1, c2, c3 = st.columns([2, 2, 1])
        state = c1.selectbox("State", ["(any)", "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"])
        session_filter = c2.text_input("Session ID (optional)")
        limit = c3.number_input("Limit", 1, 500, 50)
        actions, err = _safe(
            lambda: backend.list_actions(
                state=None if state == "(any)" else state,
                session_id=session_filter.strip() or None,
                limit=int(limit),
            ),
            "loading actions",
        )
        if err is None and actions is not None:
            if not actions:
                st.info("No actions match.")
            else:
                st.dataframe(
                    [
                        {
                            "action": a["action_id"][:16] + "…",
                            "state": a["state"],
                            "tool": a["tool_name"],
                            "risk": a["risk_level"],
                            "session": str(a["session_id"])[:12] + "…",
                            "confirmation": str(a.get("confirmation_id") or "")[:12] + "…",
                            "attempt": a.get("attempt"),
                            "created": a["created_at"][:19],
                            "finished": (a.get("finished_at") or "-")[:19],
                        }
                        for a in actions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Full IDs: expand a row's action in UNKNOWN actions, or use the CLI `inspect --action`.")

    # ── UNKNOWN actions + explicit reissue ────────────────────────────────
    elif section == "UNKNOWN actions":
        actions, err = _safe(
            lambda: backend.list_actions(state="UNKNOWN", limit=100),
            "loading UNKNOWN actions",
        )
        if err is None and actions is not None:
            if not actions:
                st.success("No UNKNOWN actions — no ambiguous executions await resolution.")
            for a in actions:
                aid = a["action_id"]
                with st.expander(f"❓ `{aid[:16]}…` — tool `{a['tool_name']}` (attempt {a.get('attempt')})"):
                    st.markdown(
                        f"**Session:** `{a['session_id']}`  \n"
                        f"**Created:** {a['created_at']}  \n"
                        "**Reason:** outcome not durably recorded (crash/restart "
                        "between dispatch and result) — the side effect may or "
                        "may not have happened."
                    )
                    # Detail + recovery preview (read-only).
                    detail, derr = _safe(
                        lambda aid=aid: backend.get_action(aid), "loading action detail"
                    )
                    if derr is None and detail:
                        st.caption(
                            f"attempt {detail.get('attempt')} · created "
                            f"{detail.get('created_at', '')[:19]} · finished "
                            f"{(detail.get('finished_at') or '-')[:19]}"
                        )
                    st.info(
                        "Reissue creates a **NEW action id** (this row stays "
                        "UNKNOWN for audit) and parks a fresh confirmation for "
                        "the session. Nothing executes until that confirmation "
                        "is approved through the normal permission flow. "
                        "Max 3 reissues per original."
                    )
                    if st.checkbox(
                        "I understand this action may have partially executed",
                        key=f"ack-{aid}",
                    ):
                        if st.button("♻️ Reissue this action…", key=f"reissue-{aid}"):
                            st.session_state[f"confirm-reissue-{aid}"] = True
                    if st.session_state.get(f"confirm-reissue-{aid}"):
                        st.warning(
                            f"**Final confirmation** — re-issue `{aid[:16]}…` "
                            f"(tool `{a['tool_name']}`)? A NEW action will be "
                            "created; the original stays UNKNOWN; nothing runs "
                            "until the new confirmation is approved."
                        )
                        cc1, cc2 = st.columns(2)
                        if cc1.button("Confirm reissue", key=f"go-{aid}", type="primary"):
                            rid = f"dash-{uuid.uuid4().hex}"
                            result, rerr = _safe(
                                lambda aid=aid, rid=rid: backend.reissue_action(aid, rid),
                                "reissuing action",
                            )
                            st.session_state[f"confirm-reissue-{aid}"] = False
                            if rerr is None and result:
                                if result.get("reused_existing"):
                                    st.info(
                                        f"Idempotent replay: this request had already "
                                        f"been processed — new action is "
                                        f"`{result['new_action_id'][:16]}…`"
                                    )
                                else:
                                    st.success(
                                        f"Reissued `{aid[:16]}…` → NEW action "
                                        f"`{result['new_action_id'][:16]}…` "
                                        f"(state {result.get('state')}, depth "
                                        f"{result.get('reissue_depth')}/"
                                        f"{result.get('max_reissues')})"
                                    )
                                st.caption(
                                    "The new action is a pending confirmation for "
                                    "its session — approve it in Chat (or via API/CLI). "
                                    f"Idempotency key: `{result.get('request_id')}` "
                                    "(reusing it returns the same new action)."
                                )
                        if cc2.button("Cancel", key=f"cancel-{aid}"):
                            st.session_state[f"confirm-reissue-{aid}"] = False
                            st.rerun()

    # ── Session leases ───────────────────────────────────────────────────
    elif section == "Session leases":
        leases, err = _safe(lambda: backend.list_leases(100), "loading leases")
        if err is None and leases is not None:
            if not leases:
                st.info("No session leases recorded.")
            else:
                st.dataframe(
                    [
                        {
                            "session": l["session_id"][:16] + "…",
                            "status": "ACTIVE" if l["active"] else "STALE",
                            "owner (redacted)": l["owner"],
                            "fencing": l["fencing"],
                            "acquired": l["acquired_at"][:19],
                            "expires": l["expires_at"][:19],
                        }
                        for l in leases
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                stale = [l for l in leases if not l["active"]]
                if stale:
                    st.warning(
                        f"{len(stale)} stale lease(s): held by a process that died "
                        "(or a turn past its TTL). They self-expire and are safe "
                        "to take over; inspect via the CLI `sessions --expired`."
                    )

    # ── Session timeline ──────────────────────────────────────────────────
    elif section == "Session timeline":
        sid = st.text_input(
            "Session ID",
            value=current_session,
            help="Defaults to the active chat session.",
        )
        if sid.strip():
            events, err = _safe(
                lambda: backend.session_timeline(sid.strip(), 200),
                "loading timeline",
            )
            if err is None and events is not None:
                if not events:
                    st.info("No events recorded for this session.")
                else:
                    icons = {
                        "message": "💬", "confirmation_parked": "⏸️",
                        "action_state": "⚡", "reissue": "♻️", "lease": "🔑",
                    }
                    for e in events:
                        ts = str(e.get("ts", ""))[:19]
                        kind = e.get("kind", "?")
                        if kind == "message":
                            detail = f"role={e.get('role')}"
                        elif kind == "confirmation_parked":
                            detail = f"tool={e.get('tool')} resolved={e.get('resolved')}"
                        elif kind == "action_state":
                            detail = (
                                f"{str(e.get('action_id'))[:12]}… tool={e.get('tool')} "
                                f"state={e.get('state')}"
                            )
                        elif kind == "reissue":
                            detail = (
                                f"{str(e.get('original_action_id'))[:12]}… → "
                                f"{str(e.get('new_action_id'))[:12]}…"
                            )
                        else:
                            detail = f"owner={e.get('owner')} active={e.get('active')}"
                        st.markdown(f"{icons.get(kind, '•')} `{ts}` **{kind}** — {detail}")
                    st.caption(f"{len(events)} events (bounded). Safe metadata only.")

    # ── Plan status (v0.23) ──────────────────────────────────────────────
    elif section == "Plan status":
        """Minimal plan/repeat indicators for the current session.

        Derived read-only from the session timeline: completed plan steps,
        failed steps, suppressed duplicate dispatches, and tool usage
        counts (repeat visibility). Safe metadata only — no tool arguments,
        no result bodies.
        """
        sid = st.text_input(
            "Session ID",
            value=current_session,
            key="plan_status_session",
            help="Defaults to the active chat session.",
        )
        if sid.strip():
            events, err = _safe(
                lambda: backend.session_timeline(sid.strip(), 200),
                "loading plan status",
            )
            if err is None and events is not None:
                tool_counts: dict[str, int] = {}
                failed_actions = 0
                succeeded_actions = 0
                for e in events:
                    if e.get("kind") == "action_state":
                        tool = str(e.get("tool") or "?")
                        tool_counts[tool] = tool_counts.get(tool, 0) + 1
                        if e.get("state") == "FAILED":
                            failed_actions += 1
                        elif e.get("state") == "SUCCEEDED":
                            succeeded_actions += 1
                c1, c2, c3 = st.columns(3)
                c1.metric("Tool executions", succeeded_actions + failed_actions)
                c2.metric("Succeeded", succeeded_actions)
                c3.metric("Failed", failed_actions)
                repeats = {t: n for t, n in sorted(tool_counts.items()) if n > 1}
                if repeats:
                    st.warning(
                        "Tools used more than once in this session: "
                        + ", ".join(f"{t} ×{n}" for t, n in repeats.items())
                        + " — repeats may be legitimate (different arguments) "
                        "or suppressed duplicates; check the logs for "
                        "duplicate_tool_call_suppressed."
                    )
                else:
                    st.success("No repeated tool usage in this session.")
                st.caption(
                    "Plan step state is derived from persisted actions; live "
                    "per-step telemetry is in the structured logs "
                    "(plan_validated / plan_step_satisfied / plan_step_failed / "
                    "replan_triggered / plan_completed / replan_diff — v0.25 "
                    "replan_diff logs the structural shape of each replan: "
                    "steps added/removed and capabilities changed, never "
                    "arguments or results)."
                )

    # ── Result cache (v0.24) ─────────────────────────────────────────────
    elif section == "Result cache":
        """Cross-turn reuse indicators: hits, expired entries, per-tool
        counts, and a metadata-only listing. Payloads are never shown —
        cached content is reachable only through the runtime's own
        permission-guarded dispatch path (Part R/Q)."""
        stats, err = _safe(backend.cache_stats, "loading cache stats")
        if err is None and stats is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Cached entries", stats["entries"])
            c2.metric("Total hits", stats["hits"])
            c3.metric("Expired", stats["expired"])
            if stats["per_tool"]:
                st.dataframe(
                    [
                        {
                            "tool": row["tool_name"],
                            "entries": row["entries"],
                            "hits": row["hits"],
                        }
                        for row in stats["per_tool"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("The result cache is empty — run a retrieval-backed request first.")

            # v0.25 (Part E4): bounded daily aggregates — counts only, no
            # payloads. Newest first from the store; shown oldest→newest so
            # the trend reads naturally.
            history_rows, herr = _safe(
                lambda: backend.cache_history(days=14, limit=30),
                "loading cache metrics history",
            )
            if herr is None and history_rows is not None:
                if history_rows:
                    st.markdown("**Daily cache activity (last 14 days)**")
                    st.dataframe(
                        [
                            {
                                "day": row["day"],
                                "hits": row["hits"],
                                "misses": row["misses"],
                                "stale": row["stale"],
                                "bypass": row["bypass"],
                                "stores": row["stores"],
                                "top tools": ", ".join(
                                    sorted(row.get("per_tool", {}).keys())
                                ) or "—",
                            }
                            for row in reversed(history_rows)
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No cache activity recorded yet.")

            st.caption(
                "Cross-turn reuse for read-only retrieval only (never side "
                "effects). Cached answers carry a provenance header; a request "
                "saying 'latest' — or Refresh mode in the sidebar — bypasses "
                "time-sensitive cache entries. Retention is bounded (default "
                "30 days). Inspect/cleanup via CLI: "
                "`maintenance cache stats|inspect|cleanup`."
            )
            if backend.mode == "legacy":
                entries, err2 = _safe(
                    lambda: backend.cache_entries(25), "loading cache entries"
                )
                if err2 is None and entries:
                    st.dataframe(
                        [
                            {
                                "fingerprint": r["cache_key"][:12] + "…",
                                "tool": r["tool_name"],
                                "scope": r["scope"],
                                "hits": r["hit_count"],
                                "age": r.get("age_human", "?"),
                            }
                            for r in entries
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )


def _render_knowledge(backend: Any) -> None:
    """v0.20 personal knowledge base: list/ingest/search (bounded, explicit)."""
    st.title("📚 Knowledge Base")
    st.caption(
        "Your ingested documents (PDF / Markdown / text / code / JSON) as "
        "citable evidence for JARVIS. Personal memory (remember_fact) is a "
        "separate store. Ingestion requires an explicit path inside the "
        "allowed directory — credential-like files are refused."
    )

    section = st.radio(
        "Section",
        ["Indexed documents", "Ingest a document", "Search knowledge"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if section == "Indexed documents":
        docs, err = _safe(
            lambda: backend.list_knowledge_documents(100), "loading documents"
        )
        if err is None and docs is not None:
            if not docs:
                st.info("No documents indexed yet.")
            else:
                st.dataframe(
                    [
                        {
                            "document": d["document_id"][:12] + "…",
                            "filename": d["filename"],
                            "type": d["media_type"],
                            "chunks": f"{d.get('live_chunk_count', '?')}/{d['chunk_count']}",
                            "size": f"{d['size_bytes']:,}B",
                            "ingested": str(d["ingested_at"])[:19],
                        }
                        for d in docs
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Full metadata via the CLI `knowledge inspect --id`.")

    elif section == "Ingest a document":
        st.info(
            "Provide an explicit path to ONE file (PDF, Markdown, text, code, "
            "JSON) inside the allowed directory. No crawling; unchanged files "
            "are skipped; changed files are re-indexed."
        )
        path = st.text_input("Document path", placeholder="docs/ai_roadmap.md")
        if st.button("⬆️ Ingest", type="primary", disabled=not path.strip()):
            result, err = _safe(
                lambda: backend.ingest_knowledge_document(path.strip()),
                "ingesting document",
            )
            if err is None and result:
                if result.get("status") == "error":
                    st.error(f"Ingestion refused: {result.get('reason')}")
                elif result.get("status") == "unchanged":
                    st.info(
                        f"Unchanged — already indexed as "
                        f"`{result.get('document_id', '')[:12]}…` "
                        f"({result.get('chunk_count')} chunks); no re-embedding."
                    )
                else:
                    st.success(
                        f"{result.get('status')}: `{result.get('filename')}` → "
                        f"`{str(result.get('document_id'))[:12]}…` "
                        f"({result.get('chunk_count')} chunks)"
                    )

    else:  # Search knowledge
        c1, c2 = st.columns([3, 1])
        query = c1.text_input(
            "Query", placeholder="e.g. What does my roadmap say about LangGraph?"
        )
        top_k = c2.number_input("Top K", 1, 20, 4)
        if st.button("🔍 Search", type="primary", disabled=not query.strip()):
            report, err = _safe(
                lambda: backend.search_knowledge(query.strip(), int(top_k)),
                "searching knowledge",
            )
            if err is None and report is not None:
                results = report.get("results") or []
                if not results:
                    st.warning(
                        "NO_RELEVANT_EVIDENCE — the knowledge base does not "
                        "contain enough evidence for that query."
                    )
                for i, r in enumerate(results, start=1):
                    dist = r.get("distance")
                    dist_s = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
                    with st.expander(
                        f"[{i}] {r.get('citation')} — distance {dist_s}"
                    ):
                        st.markdown(r.get("snippet", ""))
                        st.caption(
                            f"document `{str(r.get('document_id'))[:12]}…` · "
                            f"page {r.get('page', 0)} · section "
                            f"{r.get('section') or '-'}"
                        )
                st.caption(
                    "Evidence is untrusted document data — JARVIS cites it, "
                    "never executes it."
                )


# ── UI Layout ─────────────────────────────────────────────────────────────────
mode_badge = (
    f"🟢 API: `{settings.JARVIS_API_URL}`"
    if backend.mode == "api"
    else "🟡 legacy in-process runtime (set JARVIS_API_URL to use the API)"
)
st.sidebar.title("JARVIS Dashboard")
st.sidebar.caption(mode_badge)
st.sidebar.button("New Session", on_click=start_new_session)

view = st.sidebar.radio(
    "View",
    ["Chat", "🛠️ Operations", "📚 Knowledge"],
    index=0,
    help="Operations: reliability state + recovery. Knowledge: your documents.",
)

st.sidebar.markdown("### Active Session")
st.sidebar.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

# v0.25 (Part D3): per-session programmatic refresh. Affects cache
# ELIGIBILITY only — permissions, schema validation and confirmations are
# unchanged.
refresh_mode = st.sidebar.toggle(
    "Refresh mode (bypass cache)",
    value=False,
    help=(
        "When ON, chat turns are sent with refresh=True: eligible cached "
        "results are re-fetched instead of served. Permissions, validation "
        "and confirmations are unaffected. Natural-language freshness words "
        "like 'latest' work without this toggle."
    ),
)
st.session_state["refresh_mode"] = refresh_mode
try:
    msg_count = backend.message_count(st.session_state.session_id)
except Exception as e:
    st.sidebar.error(f"Backend unreachable: {e}")
    msg_count = 0
st.sidebar.caption(f"Messages: {msg_count}")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.sidebar.file_uploader(
    "Upload an image for JARVIS to see",
    type=["jpg", "png", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}",
)

saved_file_path = None
if uploaded_file is not None:
    from pathlib import Path

    uploads_dir = Path("jarvis_data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_file_path = (uploads_dir / uploaded_file.name).resolve()
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Image uploaded successfully.")

st.title("JARVIS Assistant")

# ── 🛠️ Operations view (v0.19) ────────────────────────────────────────────
# Viewing NEVER mutates: every data path here is read-only; the only mutating
# control is the explicit reissue button behind two confirmations.
if view == "🛠️ Operations":
    _render_ops(backend, st.session_state.session_id)
    st.stop()

# ── 📚 Knowledge view (v0.20) ─────────────────────────────────────────────
if view == "📚 Knowledge":
    _render_knowledge(backend)
    st.stop()

try:
    history = backend.history(st.session_state.session_id)
except Exception:
    history = []

# Filter history for chat display
chat_messages = []
for msg in history:
    if msg.get("role") in ("user", "assistant"):
        # Skip assistant messages that were just tool calls (no content)
        if msg.get("role") == "assistant" and not msg.get("content"):
            continue
        chat_messages.append(msg)

for msg in chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))

# Chat Input
if user_input := st.chat_input("How can I help you?"):
    if saved_file_path is not None:
        user_input += f"\n[Attached Image: {saved_file_path}]"
        st.session_state.uploader_key += 1

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("JARVIS is planning and executing..."):
            try:
                response = backend.chat(
                    st.session_state.session_id,
                    user_input,
                    refresh=bool(st.session_state.get("refresh_mode")),
                )
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
                response = None

try:
    pending = backend.pending_confirmation(st.session_state.session_id)
except Exception:
    pending = None
if pending:
    st.error("⚠️ HIGH-RISK ACTION REQUIRES APPROVAL")
    st.markdown(f"**Tool:** `{pending['tool_name']}`\n\n**Risk Level:** `{pending['risk_level']}`")
    st.markdown("**Arguments:**")
    st.code(pending["tool_args"], language="json")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", type="primary"):
            response = backend.resolve_confirmation(st.session_state.session_id, True)
            st.success(response)
            st.rerun()
    with col2:
        if st.button("❌ Deny Action"):
            response = backend.resolve_confirmation(st.session_state.session_id, False)
            st.error(response)
            st.rerun()

    # Load updated history to find the thought process for this turn
    try:
        updated_history = backend.history(st.session_state.session_id)
    except Exception:
        updated_history = []

    # Extract only the tools/thoughts from this specific interaction (after the user input)
    last_user_idx = -1
    for i in range(len(updated_history) - 1, -1, -1):
        if updated_history[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx != -1:
        turn_history = updated_history[last_user_idx + 1 :]

        with st.expander("🧠 Agent Thought Process", expanded=False):
            for msg in turn_history:
                if msg.get("tool_calls"):
                    for call in msg["tool_calls"]:
                        fn = call.get("function", {})
                        st.info(
                            f"**🛠️ Calling Tool:** `{fn.get('name')}`\n```json\n{fn.get('arguments')}\n```"
                        )
                elif msg.get("role") == "tool":
                    st.success(f"**📄 Tool Result ({msg.get('name')}):**\n```\n{msg.get('content')}\n```")
