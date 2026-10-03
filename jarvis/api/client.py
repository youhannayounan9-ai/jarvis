"""
jarvis/api/client.py
────────────────────
Dependency-free HTTP client over the JARVIS REST API.

The interface/runtime split: UIs (Streamlit dashboard, scripts, tests) use
this client instead of wiring an in-process runtime, so the agent runtime
runs in exactly one place — the API server. Only the Python stdlib is used
(urllib), keeping JARVIS installable without extra packages.

Usage:
    from jarvis.api.client import JarvisClient
    client = JarvisClient("http://127.0.0.1:8000", api_key="...")
    sid = client.create_session()
    result = client.chat(sid, "What time is it?")
    print(result["response"])

    for event in client.stream_chat(sid, "Plan my trip"):
        print(event["event"], event["data"])

Errors:
    JarvisClientError       — any non-2xx (has .status and .detail)
    SessionConflict         — 409: another turn is running on that session
    RateLimitedError        — 429: honor .retry_after seconds
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid as _uuid
from collections.abc import Iterator
from typing import Any

DEFAULT_TIMEOUT_CONNECT = 10.0
# One chat turn can legitimately take minutes on local LLMs.
DEFAULT_TIMEOUT_CHAT = 600.0


class JarvisClientError(Exception):
    """Non-2xx response from the JARVIS API."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class SessionConflict(JarvisClientError):
    """409 — another turn is already running for that session."""


class RateLimitedError(JarvisClientError):
    """429 — client exceeded the rate limit."""

    def __init__(self, detail: str, retry_after: int) -> None:
        super().__init__(429, detail)
        self.retry_after = retry_after


class JarvisClient:
    """Thin, typed wrapper over the JARVIS REST surface."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_CHAT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout

    # ── Transport ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """One JSON request; returns parsed JSON (dict/list) or raises."""
        url = f"{self.base_url}{path}"
        if params:
            query = urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                url = f"{url}?{query}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except (json.JSONDecodeError, AttributeError):
                detail = raw
            if e.code == 409:
                raise SessionConflict(409, str(detail)) from e
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "1") or 1)
                raise RateLimitedError(str(detail), retry_after) from e
            raise JarvisClientError(e.code, str(detail)) from e
        except urllib.error.URLError as e:
            raise JarvisClientError(0, f"connection failed: {e.reason}") from e

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def chat_multimodal(
        self,
        message: str,
        session_id: str | None = None,
        *,
        image_bytes: bytes | None = None,
        image_mime: str | None = None,
        audio_bytes: bytes | None = None,
        audio_mime: str | None = None,
        refresh: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        v0.27 multimodal turn (multipart/form-data). The server validates
        and normalizes; the SAME runtime (tools, permissions, grounding)
        executes. Returns {session_id, response, modality, ...}.
        """
        boundary = f"----jarvis{_uuid.uuid4().hex[:16]}"
        parts: list[bytes] = []

        def _field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n"
            ).encode("utf-8")

        def _file(name: str, filename: str, mime: str, data: bytes) -> bytes:
            return (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n"
            ).encode("utf-8") + data + b"\r\n"

        parts.append(_field("text", message))
        if session_id:
            parts.append(_field("session_id", session_id))
        if refresh:
            parts.append(_field("refresh", "true"))
        if image_bytes is not None:
            parts.append(_file("image", "upload.png", image_mime or "image/png", image_bytes))
        if audio_bytes is not None:
            parts.append(_file("audio", "upload.wav", audio_mime or "audio/wav", audio_bytes))
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        url = f"{self.base_url}/chat/multimodal"
        headers = self._headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except (json.JSONDecodeError, AttributeError):
                detail = raw
            if e.code == 409:
                raise SessionConflict(409, str(detail)) from e
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "1") or 1)
                raise RateLimitedError(str(detail), retry_after) from e
            raise JarvisClientError(e.code, str(detail)) from e
        except urllib.error.URLError as e:
            raise JarvisClientError(0, f"connection failed: {e.reason}") from e

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=DEFAULT_TIMEOUT_CONNECT)

    def create_session(self) -> str:
        body = self._request("POST", "/sessions", timeout=DEFAULT_TIMEOUT_CONNECT)
        return body["session_id"]

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """One turn; auto-creates a session when session_id is omitted.

        v0.25 ``refresh=True``: programmatic freshness control — cache-eligible
        read-only retrieval tools skip the cross-turn result cache for THIS
        turn and execute for real (fresh successful results are re-stored).
        Affects cache eligibility ONLY: permissions, schema validation and
        confirmation apply unchanged. Natural-language freshness wording
        ("latest", "today", …) keeps working independently.
        """
        payload: dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        if refresh:
            payload["refresh"] = True
        return self._request("POST", "/chat", payload)

    def stream_chat(
        self,
        message: str,
        session_id: str | None = None,
        *,
        refresh: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """
        One turn over SSE.

        Yields dicts: {"event": <name>, "data": <parsed json>}.
        The terminal events are `done` (normal) and `error` (in-band failure);
        a 4xx before streaming (unknown session, rate limit) raises as usual.

        ``refresh=True``: same v0.25 semantics as :meth:`chat` — bypass the
        cross-turn result cache for this turn (cache eligibility only).
        """
        payload: dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        if refresh:
            payload["refresh"] = True
        req = urllib.request.Request(
            f"{self.base_url}/chat/stream",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                event_name: str | None = None
                for raw_line in resp:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if line.startswith("event: "):
                        event_name = line[len("event: "):]
                    elif line.startswith("data: ") and event_name is not None:
                        data = json.loads(line[len("data: "):])
                        yield {"event": event_name, "data": data}
                        event_name = None
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except (json.JSONDecodeError, AttributeError):
                detail = raw
            if e.code == 409:
                raise SessionConflict(409, str(detail)) from e
            if e.code == 429:
                raise RateLimitedError(str(detail), int(e.headers.get("Retry-After", "1") or 1)) from e
            raise JarvisClientError(e.code, str(detail)) from e
        except urllib.error.URLError as e:
            raise JarvisClientError(0, f"connection failed: {e.reason}") from e

    def history(self, session_id: str) -> list[dict[str, Any]]:
        body = self._request(
            "GET", f"/sessions/{session_id}/history", timeout=DEFAULT_TIMEOUT_CONNECT
        )
        return body["messages"]

    def get_confirmation(self, session_id: str) -> dict[str, Any] | None:
        """Pending action for the session, or None when nothing is pending."""
        try:
            return self._request(
                "GET",
                f"/sessions/{session_id}/confirmation",
                timeout=DEFAULT_TIMEOUT_CONNECT,
            )
        except JarvisClientError as e:
            if e.status == 404:
                return None
            raise

    def resolve_confirmation(self, session_id: str, confirmed: bool) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/sessions/{session_id}/confirm",
            {"confirmed": confirmed},
        )

    def tools(self) -> list[dict[str, Any]]:
        return self._request("GET", "/tools", timeout=DEFAULT_TIMEOUT_CONNECT)

    # ── v0.19 operator endpoints ─────────────────────────────────────────────

    @staticmethod
    def _qs(params: dict[str, Any]) -> str:
        """URL-encode non-empty params (urllib-based; no extra deps)."""
        clean = {
            k: v for k, v in params.items()
            if v is not None and v != ""
        }
        return f"?{urllib.parse.urlencode(clean)}" if clean else ""

    def list_actions(
        self,
        state: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Read-only execution-ledger report (safe metadata only)."""
        qs = self._qs({"limit": limit, "state": state, "session_id": session_id})
        return self._request(
            "GET", f"/actions{qs}", timeout=DEFAULT_TIMEOUT_CONNECT
        )

    def get_action(self, action_id: str) -> dict[str, Any]:
        """One ledger row's safe metadata. Raises JarvisClientError 404."""
        return self._request(
            "GET",
            f"/actions/{action_id}",
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def list_leases(self, limit: int = 100) -> list[dict[str, Any]]:
        """Read-only session-lease report (owner redacted server-side)."""
        return self._request(
            "GET",
            f"/sessions/leases{self._qs({'limit': limit})}",
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def reissue_action(
        self, action_id: str, request_id: str | None = None
    ) -> dict[str, Any]:
        """Explicit UNKNOWN-action reissue (mutating; server auth applies).

        Without a client request_id the server generates one and echoes it
        in the response — retry the SAME request_id to stay idempotent.
        Always sends a JSON body (``{}`` when no key), since the endpoint's
        body model is required even though its field is optional.
        """
        body: dict[str, Any] = {"request_id": request_id} if request_id else {}
        return self._request(
            "POST", f"/actions/{action_id}/reissue", body
        )

    def session_timeline(
        self, session_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Read-only causal timeline (safe metadata only)."""
        return self._request(
            "GET",
            f"/sessions/{session_id}/timeline{self._qs({'limit': limit})}",
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    # ── v0.20 knowledge base ───────────────────────────────────────────────

    def list_knowledge_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/knowledge/documents{self._qs({'limit': limit})}",
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def get_knowledge_document(self, document_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/knowledge/documents/{document_id}",
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def ingest_knowledge_document(
        self, path: str, target_chars: int = 1200, overlap_chars: int = 150
    ) -> dict[str, Any]:
        """Explicit single-document ingestion (server enforces path safety)."""
        return self._request(
            "POST",
            "/knowledge/ingest",
            {
                "path": path,
                "target_chars": target_chars,
                "overlap_chars": overlap_chars,
            },
        )

    def search_knowledge(
        self, query: str, top_k: int = 4, source: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if source:
            body["source"] = source
        return self._request("POST", "/knowledge/search", body)

    def remove_knowledge_document(self, document_id: str) -> dict[str, Any]:
        return self._request(
            "DELETE", f"/knowledge/documents/{document_id}"
        )

    # ── v0.28 safe browser control ─────────────────────────────────────────

    def browser_status(self) -> dict[str, Any]:
        """Safe browser posture snapshot (no page content, no URLs)."""
        return self._request(
            "GET", "/browser/status", timeout=DEFAULT_TIMEOUT_CONNECT
        )

    def emergency_stop(self, reason: str = "user requested emergency stop") -> dict[str, Any]:
        """Trigger the HUMAN-ONLY browser emergency stop."""
        return self._request("POST", "/browser/emergency-stop", {"reason": reason})

    def emergency_reset(self) -> dict[str, Any]:
        """Operator reset of the emergency stop (idempotent)."""
        return self._request(
            "POST", "/browser/emergency-reset", {},
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    # ── v0.29 integration management (metadata only; never credentials) ────

    def list_integrations(self) -> list[dict[str, Any]]:
        """Provider catalog + connected accounts (safe metadata)."""
        return self._request(
            "GET", "/integrations", timeout=DEFAULT_TIMEOUT_CONNECT
        )

    def integration_scopes(self, provider: str) -> dict[str, Any]:
        """Exact scope menu for one provider."""
        return self._request(
            "GET", f"/integrations/{provider}/scopes",
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def integration_connect(
        self,
        provider: str,
        display_label: str,
        scopes: list[str],
        credential: str | None = None,
        provider_account_ref: str = "",
    ) -> dict[str, Any]:
        """Explicitly connect one account (user-controlled)."""
        body: dict[str, Any] = {
            "display_label": display_label,
            "scopes": scopes,
        }
        if credential:
            body["credential"] = credential
        if provider_account_ref:
            body["provider_account_ref"] = provider_account_ref
        return self._request(
            "POST", f"/integrations/{provider}/connect", body,
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def integration_disconnect(self, account_id: str) -> dict[str, Any]:
        """Disconnect one account (removes stored credentials locally)."""
        return self._request(
            "POST", f"/integrations/accounts/{account_id}/disconnect", {},
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def integration_audit(self, account_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Recent external side effects for one account (safe fields)."""
        return self._request(
            "GET", f"/integrations/accounts/{account_id}/audit",
            params={"limit": limit},
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    # ── v0.30 OAuth (real authorization; never returns token material) ─────

    def integration_authorize(
        self, provider: str, session_id: str, display_label: str, scopes: list[str]
    ) -> dict[str, Any]:
        """Start an OAuth authorization; returns the URL the OPERATOR opens."""
        return self._request(
            "POST", f"/integrations/{provider}/authorize",
            {"session_id": session_id, "display_label": display_label, "scopes": scopes},
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def integration_authorize_status(self, provider: str, session_id: str) -> dict[str, Any]:
        """Bounded polling view for a started authorization (no state values)."""
        return self._request(
            "GET", f"/integrations/{provider}/authorize/status",
            params={"session_id": session_id},
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )

    def integration_refresh(self, account_id: str) -> dict[str, Any]:
        """Re-verify one account (opportunistic token refresh included)."""
        return self._request(
            "POST", f"/integrations/accounts/{account_id}/refresh", {},
            timeout=DEFAULT_TIMEOUT_CONNECT,
        )


__all__ = [
    "JarvisClient",
    "JarvisClientError",
    "RateLimitedError",
    "SessionConflict",
]
