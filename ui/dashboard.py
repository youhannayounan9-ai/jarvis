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

    def chat_multimodal(
        self,
        session_id: str,
        message: str,
        *,
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """v0.27 multimodal turn over the API (same auth/rate limits)."""
        return self._client.chat_multimodal(
            message,
            session_id,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            refresh=refresh,
        )

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

    # ── v0.26 grounding guard ──────────────────────────────────────────────

    def grounding_history(self, days: int = 14, limit: int = 30) -> list[dict[str, Any]]:
        """Daily grounding-guard aggregates (v0.26) — counts only."""
        return self._client._request(
            "GET",
            "/ops/grounding/stats",
            params={"days": days, "limit": limit},
        )

    # ── v0.28 safe browser control ─────────────────────────────────────────

    def browser_status(self) -> dict[str, Any]:
        """Safe browser posture snapshot (bounded metadata only)."""
        return self._client.browser_status()

    def emergency_stop(self, reason: str = "dashboard operator stop") -> dict[str, Any]:
        return self._client.emergency_stop(reason)

    def emergency_reset(self) -> dict[str, Any]:
        return self._client.emergency_reset()


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

    def chat_multimodal(
        self,
        session_id: str,
        message: str,
        *,
        image_bytes: bytes | None = None,
        audio_bytes: bytes | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """v0.27 multimodal turn in-process (legacy fallback backend)."""
        import io as _io

        from jarvis.multimodal.models import Attachment, MultimodalRequest
        from jarvis.multimodal.service import MultimodalService
        from jarvis.voice.stt import SpeechToText

        text = message
        image = None
        modality = "text"
        if audio_bytes is not None:
            from jarvis.multimodal.models import validate_audio_bytes

            mime = validate_audio_bytes(audio_bytes)
            transcribed = SpeechToText().transcribe_bytes(audio_bytes, mime)
            if transcribed.startswith("ERROR:") or not transcribed:
                raise ValueError(transcribed or "No speech detected in the audio.")
            text = f"{message} {transcribed}".strip()
            modality = "audio"
        if image_bytes is not None:
            image = Attachment.from_upload(image_bytes)
            modality = "image+text" if modality == "audio" or text.strip() else "image"
        request = MultimodalRequest(
            text=text,
            session_id=session_id,
            modality=modality,
            image=image,
            refresh=refresh,
        )
        try:
            response = MultimodalService(self._runtime.orchestrator).run(request)
        finally:
            if image is not None:
                image.cleanup()
        return {
            "session_id": session_id,
            "response": response,
            "modality": request.modality,
        }

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

    # ── v0.26 grounding guard ──────────────────────────────────────────────

    def grounding_history(self, days: int = 14, limit: int = 30) -> list[dict[str, Any]]:
        """Daily grounding-guard aggregates (v0.26) — counts only."""
        return self._runtime.store.grounding_metrics_history(days=days, limit=limit)

    # ── v0.28 safe browser control ─────────────────────────────────────────

    def browser_status(self) -> dict[str, Any]:
        from jarvis.browser.emergency import get_emergency_stop
        from jarvis.browser.registry import get_browser_registry

        stop = get_emergency_stop()
        registry = get_browser_registry()
        downloads = sum(
            int(c.status().get("downloads", 0)) for c in registry.controllers()
        )
        return {
            "enabled": bool(
                getattr(settings, "ENABLE_BROWSER_CONTROL", False)
            ),
            "driver": str(settings.BROWSER_DRIVER),
            "emergency_stop_active": bool(stop.status()["active"]),
            "emergency_stop_reason": str(stop.status()["reason"]),
            "emergency_stop_token": int(stop.status()["token"]),
            "open_sessions": registry.open_count(),
            "max_sessions": int(settings.BROWSER_MAX_SESSIONS),
            "downloads_captured": downloads,
        }

    def emergency_stop(self, reason: str = "dashboard operator stop") -> dict[str, Any]:
        from jarvis.browser.emergency import get_emergency_stop
        from jarvis.browser.registry import get_browser_registry

        token = get_emergency_stop().trigger(reason=reason)
        get_browser_registry().close_all()
        return {"triggered": True, "token": token, "reason": reason}

    def emergency_reset(self) -> dict[str, Any]:
        from jarvis.browser.emergency import get_emergency_stop

        was_active = get_emergency_stop().reset()
        return {"was_active": was_active}

    # ── v0.29 integration management (legacy in-process backend) ───────────

    def list_integrations(self) -> list[dict[str, Any]]:
        """Provider catalog + accounts (public metadata; never credentials)."""
        manager = getattr(self._runtime, "integration_manager", None)
        if manager is None:
            return []
        out: list[dict[str, Any]] = []
        for name, provider in manager.providers().items():
            cap = provider.capabilities
            out.append({
                "provider": name,
                "display_name": cap.display_name,
                "production_like": cap.production_like,
                "grantable_scopes": sorted(cap.grantable_scopes),
                "resource_kind": cap.resource.kind,
                "operations": sorted(op.value for op in cap.resource.operations),
                "supports_idempotency_key": cap.supports_idempotency_key,
                "accounts": [a.public_metadata() for a in manager.list_accounts(name)],
            })
        return out

    def integration_disconnect(self, account_id: str) -> dict[str, Any]:
        manager = getattr(self._runtime, "integration_manager", None)
        if manager is None:
            return {"disconnected": False, "account_id": account_id, "message": "Integrations disabled."}
        removed = manager.disconnect(account_id)
        return {
            "disconnected": removed,
            "account_id": account_id,
            "message": "Account disconnected; stored credentials removed locally.",
        }


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
        ["Recent actions", "UNKNOWN actions", "Session leases", "Session timeline", "Plan status", "Result cache", "Browser control", "Integrations"],
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

        # ── v0.26 (Part 11): grounding-guard daily aggregates ─────────────
        grounding_rows, gerr = _safe(
            lambda: backend.grounding_history(days=14, limit=30),
            "loading grounding metrics history",
        )
        if gerr is None and grounding_rows is not None:
            st.markdown("**Answer grounding (last 14 days)**")
            if grounding_rows:
                st.dataframe(
                    [
                        {
                            "day": row["day"],
                            "checks": row["checks"],
                            "contradictions": row["contradictions"],
                            "corrections": row["corrections"],
                            "corrections ok": row["corrections_ok"],
                            "corrections failed": row["corrections_failed"],
                            "fallbacks": row["fallbacks"],
                            "top tools": ", ".join(
                                sorted(row.get("per_tool", {}).keys())
                            ) or "—",
                        }
                        for row in reversed(grounding_rows)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No grounding activity recorded yet.")
            st.caption(
                "Every final answer is deterministically checked against the "
                "turn's trusted tool evidence. A high-confidence contradiction "
                "triggers exactly ONE transcription-only correction round; if "
                "that still contradicts, the generated answer is withheld and "
                "the authoritative value is reported instead. Counts only — "
                "never answers or evidence text."
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

    # ── Browser control (v0.28): posture + human-only emergency stop ──────
    elif section == "Browser control":
        """Safe-browser surface state and the emergency stop. Status is
        bounded safe metadata (enabled/driver/stop state/session counts) —
        never page content, URLs, or arguments. The stop button and reset
        are HUMAN-ONLY controls: the model has no tool that can trigger or
        clear them."""
        status, err = _safe(backend.browser_status, "loading browser status")
        if err is None and status is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Surface", "enabled" if status["enabled"] else "disabled")
            c2.metric("Driver", status["driver"])
            c3.metric(
                "Emergency stop",
                "ACTIVE" if status["emergency_stop_active"] else "clear",
            )
            c4, c5, c6 = st.columns(3)
            c4.metric("Open sessions", status["open_sessions"], f"max {status['max_sessions']}")
            c5.metric("Downloads captured", status["downloads_captured"])
            c6.metric("Stop token", status["emergency_stop_token"])
            if status["emergency_stop_active"]:
                st.warning(f"Emergency stop active: {status['emergency_stop_reason']}")
            st.caption(
                "The browser surface is opt-in (ENABLE_BROWSER_CONTROL). Every "
                "action passes URL policy, observation freshness, pacing limits, "
                "dynamic risk → confirmation, and deterministic verification; "
                "page text is untrusted data. HIGH-risk actions (submit/delete/"
                "password-worded) always require your explicit confirmation."
            )
            b1, b2 = st.columns(2)
            if b1.button("🛑 Emergency stop", type="primary", help="Interrupt any in-flight browser action now. Human-only."):
                _safe(
                    lambda: backend.emergency_stop("dashboard operator stop"),
                    "triggering emergency stop",
                )
                st.rerun()
            if b2.button("Reset stop", disabled=not status["emergency_stop_active"], help="Clear the emergency stop (operator action)."):
                _safe(backend.emergency_reset, "resetting emergency stop")
                st.rerun()

    # ── Integrations (v0.29): connected accounts + external actions ───────
    elif section == "Integrations":
        """Connected personal services. Shows account identity, scopes,
        auth state and recent EXTERNAL actions. Never displays credentials
        or tokens (they are structurally absent from the API projection).
        Connect/disconnect are the only mutations here; external resource
        actions always flow through the chat runtime with confirmation."""
        providers, err = _safe(backend.list_integrations, "loading integrations")
        if err is None and providers is not None:
            if not providers:
                st.info("No integration providers registered (ENABLE_INTEGRATIONS=false or none built).")
            for p in providers:
                st.subheader(f"🔗 {p['display_name']}")
                st.caption(
                    ("Production-like" if p.get("production_like") else "Development-only provider")
                    + f" · resource: {p.get('resource_kind', '?')} · operations: "
                    + ", ".join(p.get("operations", []))
                )
                st.markdown("Scopes: `" + "` `".join(p.get("grantable_scopes", [])) + "`")
                _render_oauth_connect(backend, p)
                accounts = p.get("accounts", [])
                if not accounts:
                    st.info("No accounts connected for this provider.")
                for a in accounts:
                    state = a.get("auth_state", "ERROR")
                    icon = {"AUTHENTICATED": "✅", "EXPIRED": "⚠️", "REVOKED": "⛔", "DISCONNECTED": "➖"}.get(state, "❓")
                    oauth_status = a.get("authorization_status") or "—"
                    oauth_icon = {
                        "AUTHORIZED": "🔐", "TOKEN_EXPIRING": "🕒", "REFRESHING": "♻️",
                        "AUTHENTICATION_REQUIRED": "🔑", "REVOKED": "⛔", "ERROR": "❓",
                        "AUTHORIZING": "⏳", "DISCONNECTED": "➖",
                    }.get(oauth_status, "•")
                    col_a, col_b, col_c = st.columns([3, 2, 1])
                    col_a.markdown(
                        f"{icon} **{a.get('display_label', '?')}** · `{a.get('account_id', '?')}`"
                    )
                    col_b.markdown(
                        f"{state} · {oauth_icon} {oauth_status} · "
                        f"verified: {(a.get('last_verified_at') or 'never')[:19]}"
                    )
                    if col_c.button("Disconnect", key=f"dc-{a.get('account_id')}", help="Revoke at the provider (when supported) and remove stored local credentials."):
                        _safe(
                            lambda aid=a.get("account_id"): backend.integration_disconnect(aid),
                            "disconnecting account",
                        )
                        st.rerun()
                    st.markdown("Granted: `" + "` `".join(a.get("scopes", [])) + "`")
                    if a.get("authorization_expires_at"):
                        st.caption(f"Authorization expires: {a.get('authorization_expires_at')}")
                    b_re, b_rf, _sp = st.columns([1, 1, 3])
                    if b_re.button("↻ Re-authenticate", key=f"re-{a.get('account_id')}", help="Start a fresh OAuth authorization for this label (rotates tokens)."):
                        start, err = _safe(
                            lambda: backend.integration_authorize(
                                a.get("provider", ""),
                                _dashboard_oauth_session(),
                                a.get("display_label", ""),
                                a.get("scopes", []),
                            ),
                            "starting re-authorization",
                        )
                        if err is None and start:
                            st.markdown(f"[Open the authorization page]({start.get('authorization_url', '')})")
                    if b_rf.button("⟳ Refresh now", key=f"rf-{a.get('account_id')}", help="Re-verify this account now (opportunistic token refresh)."):
                        _safe(
                            lambda aid=a.get("account_id"): backend.integration_refresh(aid),
                            "refreshing account",
                        )
                        st.rerun()
            st.divider()
            st.caption(
                "External writes (create/update/delete/complete) are NOT "
                "performed here: the agent performs them only through the "
                "chat runtime with your explicit confirmation, the action "
                "ledger, and read-back verification. Credentials are never "
                "displayed in this dashboard."
            )


def _dashboard_oauth_session() -> str:
    """Stable per-browser dashboard session id for OAuth state binding."""
    key = "jarvis_oauth_session"
    if key not in st.session_state:
        import uuid as _uuid

        st.session_state[key] = f"dashboard-{_uuid.uuid4().hex[:8]}"
    return str(st.session_state[key])


def _render_oauth_connect(backend: Any, provider_info: dict) -> None:
    """
    v0.30: start a user-controlled OAuth authorization from the dashboard.

    The authorization URL is displayed for the OPERATOR to open; the callback
    lands on the API server's fixed endpoint. The dashboard never displays
    tokens, authorization codes, or state values beyond that URL.
    """
    provider = str(provider_info.get("provider") or "")
    if not provider:
        return
    if not provider_info.get("supports_oauth", False):
        return
    with st.expander("🔐 Connect with OAuth (recommended)"):
        label = st.text_input(
            "Account label", key=f"oauth-label-{provider}", max_chars=64
        )
        scopes = st.multiselect(
            "Scopes to grant (exact — never expandable by the model)",
            provider_info.get("grantable_scopes", []),
            key=f"oauth-scopes-{provider}",
        )
        if st.button("Start authorization", key=f"oauth-start-{provider}"):
            if not label.strip() or not scopes:
                st.warning("A label and at least one scope are required.")
            else:
                start, err = _safe(
                    lambda: backend.integration_authorize(
                        provider, _dashboard_oauth_session(), label.strip(), scopes
                    ),
                    "starting authorization",
                )
                if err is None and start:
                    st.session_state[f"oauth-url-{provider}"] = start.get("authorization_url", "")
        url = st.session_state.get(f"oauth-url-{provider}")
        if url:
            st.markdown(f"**[Open the authorization page]({url})** and approve.")
            st.caption(
                "The provider redirects to JARVIS's fixed callback endpoint. "
                "Local simulated providers may require an out-of-band consent step."
            )
            if st.button("Check authorization status", key=f"oauth-check-{provider}"):
                status, err = _safe(
                    lambda: backend.integration_authorize_status(
                        provider, _dashboard_oauth_session()
                    ),
                    "checking authorization status",
                )
                if err is None and status:
                    outcome = str(status.get("status", ""))
                    st.info(f"Authorization status: {outcome}")
                    if outcome == "AUTHORIZED":
                        st.rerun()


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

# ── v0.27 multimodal inputs ──────────────────────────────────────────────
# The image NO LONGER touches disk under the client's name: bytes are sent
# to the API, which content-sniffs, bounds, and stores them under a random
# name inside the sandbox (client filenames/paths are never trusted).
uploaded_file = st.sidebar.file_uploader(
    "Attach an image (JPEG/PNG/WebP/GIF, ≤10 MB)",
    type=["jpg", "jpeg", "png", "webp", "gif"],
    key=f"uploader_{st.session_state.uploader_key}",
)
image_bytes = None
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.sidebar.caption(
        f"Image attached: {len(image_bytes) / 1024:.0f} KB — validated by "
        "the server (content-sniffed; stored under a random name)."
    )

uploaded_audio = st.sidebar.file_uploader(
    "Or attach audio (WAV/MP3, ≤25 MB) — transcribed with local Whisper",
    type=["wav", "mp3"],
    key=f"audio_{st.session_state.uploader_key}",
)
audio_bytes = None
if uploaded_audio is not None:
    audio_bytes = uploaded_audio.getvalue()
    st.sidebar.caption(f"Audio attached: {len(audio_bytes) / 1024:.0f} KB.")

saved_file_path = None  # v0.27: images no longer pre-written to disk client-side

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
        if image_bytes is not None:
            st.caption("🖼️ image attached")
        if audio_bytes is not None:
            st.caption("🎙️ audio attached")

    with st.chat_message("assistant"):
        # v0.27: visible processing state (Part 14).
        status = st.status("JARVIS is working…", expanded=True)
        status.update(label="Routing request…", state="running")
        try:
            if image_bytes is not None or audio_bytes is not None:
                status.update(label="Sending multimodal request…")
                result = backend.chat_multimodal(
                    st.session_state.session_id,
                    user_input,
                    image_bytes=image_bytes,
                    audio_bytes=audio_bytes,
                    refresh=bool(st.session_state.get("refresh_mode")),
                )
                response = result.get("response", "")
                status.update(
                    label=f"Done ({result.get('modality', 'multimodal')} turn)",
                    state="complete",
                )
            else:
                response = backend.chat(
                    st.session_state.session_id,
                    user_input,
                    refresh=bool(st.session_state.get("refresh_mode")),
                )
                status.update(label="Done", state="complete")
            st.markdown(response)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Error: {e}")
            response = None
        finally:
            # One-shot attachments: the bytes are consumed by this turn and
            # are NOT persisted beyond it (privacy/retention, Part 16).
            image_bytes = None
            audio_bytes = None
            st.session_state.uploader_key += 1

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
