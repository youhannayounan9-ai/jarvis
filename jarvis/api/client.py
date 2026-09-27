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
    ) -> Any:
        """One JSON request; returns parsed JSON (dict/list) or raises."""
        url = f"{self.base_url}{path}"
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

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=DEFAULT_TIMEOUT_CONNECT)

    def create_session(self) -> str:
        body = self._request("POST", "/sessions", timeout=DEFAULT_TIMEOUT_CONNECT)
        return body["session_id"]

    def chat(self, message: str, session_id: str | None = None) -> dict[str, Any]:
        """One turn; auto-creates a session when session_id is omitted."""
        payload: dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/chat", payload)

    def stream_chat(
        self, message: str, session_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """
        One turn over SSE.

        Yields dicts: {"event": <name>, "data": <parsed json>}.
        The terminal events are `done` (normal) and `error` (in-band failure);
        a 4xx before streaming (unknown session, rate limit) raises as usual.
        """
        payload: dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
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


__all__ = [
    "JarvisClient",
    "JarvisClientError",
    "RateLimitedError",
    "SessionConflict",
]
