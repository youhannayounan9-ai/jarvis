"""
jarvis/api/auth.py
──────────────────
Optional API-key authentication for the JARVIS REST API.

Model:
  - JARVIS_API_KEY unset/empty → authentication disabled (local-only trust).
  - JARVIS_API_KEY set         → every non-exempt endpoint requires
    `Authorization: Bearer <key>` (or `X-API-Key: <key>` for clients that
    cannot set headers).

Exempt paths (no auth): /health, /docs, /openapi.json, /redoc — probes and
OpenAPI metadata stay reachable; they expose no user data.

Comparison uses hmac.compare_digest (constant-time) — never `==`.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Paths reachable without a key (liveness probes + OpenAPI metadata).
AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


def auth_enabled() -> bool:
    """True when an API key is configured."""
    return bool(settings.JARVIS_API_KEY)


def verify_key(presented: str | None) -> bool:
    """Constant-time comparison of the presented key against the configured one."""
    expected = settings.JARVIS_API_KEY
    if not expected:
        return True  # auth disabled
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def extract_api_key(request: Request) -> str | None:
    """
    Extract the presented key from a request.

    Accepts (in order):
      - Authorization: Bearer <key>
      - X-API-Key: <key>
    """
    auth_header = request.headers.get("authorization") or ""
    scheme, param = get_authorization_scheme_param(auth_header)
    if scheme.lower() == "bearer" and param:
        return param
    return request.headers.get("x-api-key") or None


async def require_api_key(request: Request) -> None:
    """
    FastAPI dependency enforcing the configured API key.

    Raises:
        401: no/invalid credentials (WWW-Authenticate hints at the scheme).
        503: auth misconfigured in a way that must fail closed.
    """
    if not auth_enabled():
        return
    # Defense in depth: an empty configured key would disable auth silently,
    # so treat a whitespace-only key as misconfiguration and refuse everything.
    if not settings.JARVIS_API_KEY.strip():
        log.error("api_auth_misconfigured_empty_key")
        raise HTTPException(status_code=503, detail="API auth misconfigured.")

    presented = extract_api_key(request)
    if not verify_key(presented):
        client = request.client.host if request.client else "unknown"
        log.warning("api_auth_rejected", client=client, path=request.url.path)
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = [
    "AUTH_EXEMPT_PATHS",
    "auth_enabled",
    "extract_api_key",
    "require_api_key",
    "verify_key",
]
