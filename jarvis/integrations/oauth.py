"""
jarvis/integrations/oauth.py
────────────────────────────
v0.30 Parts 2/3/4/6/9/16 — REAL OAuth authorization + token lifecycle for
the v0.29 integration framework, as a PROVIDER-NEUTRAL extension of the
existing IntegrationProvider / CredentialStore architecture (no redesign,
no second security model).

What this module owns (and nothing else does):

  - The normalized AUTHORIZATION LIFECYCLE (Part 16): one state machine,
    DISCONNECTED → AUTHORIZING → AUTHORIZED → (TOKEN_EXPIRING →
    REFRESHING) → AUTHORIZED | AUTHENTICATION_REQUIRED | REVOKED | ERROR,
    mapped onto the EXISTING operational AuthState (manager enforcement is
    unchanged — OAuth only feeds it honest states).
  - DURABLE ONE-TIME authorization state (Part 4): 256-bit
    cryptographically random state values; SQLite rows keyed by SHA-256
    HASH of the state (the raw value is never persisted, never logged,
    never shown to the model); hard binding to (provider, session,
    display_label, scopes, redirect_uri); expiry; atomic one-time
    consumption with replay rejection.
  - The authorization-code EXCHANGE + token REFRESH/REVOCATION seam
    (Parts 6/9): one abstract OAuthTokenClient with a stdlib-HTTP
    implementation (RFC 6749 form posts, explicit timeout, NO blind
    retries — a timed-out exchange is never retried because one-time
    codes make retries meaningless) and a deterministic
    LocalAuthorizationServer (simulated provider, PKCE S256, refresh-token
    rotation with replay detection) used for development, tests, and the
    controlled live validation.
  - PKCE (S256) for every flow — code verifier lives only server-side in
    the flow row, code challenge goes into the authorization URL.
  - The OAuthIntegrationProvider mixin: begin_authorization /
    handle_callback / revoke_remote / introspection-based verification.
    Provider-specific details (endpoints, scope strings, identity
    discovery) stay INSIDE adapters — the orchestrator never sees them
    (Part 3).

Honest boundaries (documented, not hidden):
  - Token material is stored through the SAME XOR+base64 local-development
    obfuscation as v0.29 credentials (credentials.py) — a privacy guard,
    NOT encryption. Production-grade secret storage (OS keychain) is still
    the documented next step behind the same CredentialStore interface.
  - Tokens never appear in prompts, tool output, logs, audit rows, API
    responses, or error messages (redaction is pinned by tests, Part 8).
  - The callback endpoint authenticates with the ONE-TIME STATE itself
    (standard OAuth semantics) — the state is the proof the redirect is
    ours; nothing else is accepted.

All failures are fail-closed: missing/invalid/expired/mismatched state,
replayed state, wrong provider/session/redirect, exchange failure, or a
revoked grant refuse the flow with a bounded, safe message.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from jarvis.config import settings
from jarvis.integrations.base import AuthState
from jarvis.integrations.scopes import validate_scope_set
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── Part 16: the single normalized authorization lifecycle ────────────────────


class AuthorizationStatus(str, Enum):
    """One lifecycle model; provider-specific strings never leak (Part 16)."""

    DISCONNECTED = "DISCONNECTED"
    AUTHORIZING = "AUTHORIZING"
    AUTHORIZED = "AUTHORIZED"
    TOKEN_EXPIRING = "TOKEN_EXPIRING"
    REFRESHING = "REFRESHING"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    REVOKED = "REVOKED"
    ERROR = "ERROR"


def operational_auth_state(status: AuthorizationStatus | str | None) -> AuthState:
    """
    Map the OAuth lifecycle onto the EXISTING operational AuthState the
    manager already enforces. This mapping is the ONLY bridge between the
    two models — permission decisions never read provider strings.
    """
    value = (status.value if isinstance(status, AuthorizationStatus) else str(status or "")).upper()
    return {
        "AUTHORIZED": AuthState.AUTHENTICATED,
        "TOKEN_EXPIRING": AuthState.AUTHENTICATED,   # still valid; refresh opportunistically
        "REFRESHING": AuthState.AUTHENTICATED,       # transient; ops still allowed
        "AUTHENTICATION_REQUIRED": AuthState.EXPIRED,
        "REVOKED": AuthState.REVOKED,
        "AUTHORIZING": AuthState.DISCONNECTED,       # no usable credential yet
        "DISCONNECTED": AuthState.DISCONNECTED,
        "ERROR": AuthState.ERROR,
    }.get(value, AuthState.ERROR)


# ── Part 2: flow errors — fail-closed, safe, categorized ──────────────────────


class OAuthErrorCategory(str, Enum):
    INVALID_STATE = "INVALID_STATE"                # unknown state hash
    EXPIRED_STATE = "EXPIRED_STATE"                # TTL passed
    REPLAYED_STATE = "REPLAYED_STATE"              # already consumed (one-time)
    SESSION_MISMATCH = "SESSION_MISMATCH"          # bound to another session
    PROVIDER_MISMATCH = "PROVIDER_MISMATCH"        # state minted for another provider
    REDIRECT_DENIED = "REDIRECT_DENIED"            # redirect does not match the exact allowlist
    MALFORMED_CALLBACK = "MALFORMED_CALLBACK"      # missing/garbled code or state
    EXCHANGE_FAILED = "EXCHANGE_FAILED"            # token endpoint rejected the exchange
    SCOPE_DOWNGRADED = "SCOPE_DOWNGRADED"          # provider granted less than requested
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"  # user refused consent
    REVOKED_GRANT = "REVOKED_GRANT"                # refresh token invalid/rotated-replay
    UNAVAILABLE = "UNAVAILABLE"                    # provider unreachable/timeout


class OAuthFlowError(Exception):
    """A normalized authorization failure. Safe bounded message; NEVER a token/code."""

    def __init__(self, category: OAuthErrorCategory, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message

    def to_public_message(self) -> str:
        """Bounded, secret-free rendering for API/CLI/dashboard surfaces."""
        return f"OAUTH_{self.category.value}: {self.message[:200]}"


# ── Part 7: the token record (secrets structurally excluded from repr) ────────


@dataclass(frozen=True)
class TokenSet:
    """One grant's tokens. repr/str NEVER contain token material (Part 8)."""

    access_token: str = field(repr=False, compare=False)
    refresh_token: str | None = field(default=None, repr=False, compare=False)
    expires_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    scopes: frozenset[str] = frozenset()          # PROVIDER scope strings
    account_ref: str = ""                          # provider-side stable identity (sub/email)
    token_type: str = "Bearer"

    def expires_in(self, now: datetime | None = None) -> float:
        now = now or datetime.now(tz=timezone.utc)
        return (self.expires_at - now).total_seconds()

    def is_expired(self, *, margin_seconds: float = 0.0, now: datetime | None = None) -> bool:
        return self.expires_in(now) <= margin_seconds

    def expiry_iso(self) -> str:
        return self.expires_at.isoformat()


# ── Part 3: provider-neutral client config (owned by the adapter) ─────────────


@dataclass(frozen=True)
class OAuthClientConfig:
    """
    Everything ONE provider needs for its OAuth flow. Built inside the
    adapter; the orchestrator/model never sees an instance (Part 3).
    ``client_secret`` is configuration material — it is never logged and
    never leaves the token client (Part 8).
    """

    provider: str
    client_id: str
    client_secret: str = field(repr=False, compare=False)
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str | None = None
    redirect_uri: str = ""                         # exact; from settings base + fixed path
    timeout_seconds: float = 10.0
    # JARVIS scope → provider scope string. Only these are ever requested;
    # the model cannot add one (scopes are validated at begin_authorization).
    scope_map: dict[str, str] = field(default_factory=dict)
    scope_separator: str = " "

    def provider_scope_string(self, jarvis_scopes: frozenset[str] | set[str]) -> str:
        mapped = [self.scope_map[s] for s in sorted(jarvis_scopes) if s in self.scope_map]
        return self.scope_separator.join(mapped)

    def jarvis_scopes_from_provider(self, provider_scope_str: str) -> set[str]:
        """Inverse map (used to verify the grant was not downgraded)."""
        got = {s for s in (provider_scope_str or "").split(self.scope_separator) if s}
        return {j for j, p in self.scope_map.items() if p in got}


# ── PKCE (S256) — mandatory on every flow ─────────────────────────────────────


def new_code_verifier() -> str:
    """RFC 7636 verifier: 48 random bytes, URL-safe (≈64 chars)."""
    return secrets.token_urlsafe(48)


def code_challenge_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


# ── Part 4: durable one-time state (SQLite rows keyed by state HASH) ──────────

STATE_TTL_SECONDS_DEFAULT = 600


@dataclass(frozen=True)
class StoredFlow:
    """The persisted binding behind one authorization attempt."""

    state_hash: str
    provider: str
    session_id: str
    display_label: str
    scopes: frozenset[str]
    redirect_uri: str
    code_challenge: str
    code_verifier: str = field(repr=False, compare=False)
    created_at: str = ""
    expires_at: str = ""
    consumed_at: str | None = None
    outcome: str | None = None


def _flow_from_row(row: dict[str, Any]) -> StoredFlow:
    """Project a store row (dict) onto the typed flow record."""
    import json as _json

    try:
        scopes = frozenset(_json.loads(row.get("scopes_json") or "[]"))
    except Exception:  # noqa: BLE001 — corrupt row → empty grant (fail closed)
        scopes = frozenset()
    return StoredFlow(
        state_hash=str(row.get("state_hash") or ""),
        provider=str(row.get("provider") or ""),
        session_id=str(row.get("session_id") or ""),
        display_label=str(row.get("display_label") or ""),
        scopes=scopes,
        redirect_uri=str(row.get("redirect_uri") or ""),
        code_challenge=str(row.get("code_challenge") or ""),
        code_verifier=str(row.get("code_verifier") or ""),
        created_at=str(row.get("created_at") or ""),
        expires_at=str(row.get("expires_at") or ""),
        consumed_at=row.get("consumed_at"),
        outcome=row.get("outcome"),
    )


class OAuthFlowManager:
    """
    Create / consume one-time authorization state over the store's
    ``oauth_states`` table. SECURITY PROPERTIES (all fail-closed):

      - the raw state is 256 bits of ``secrets`` entropy and is stored ONLY
        as a SHA-256 hash (a leaked DB row cannot be replayed as a state);
      - every consume validates the FULL binding (provider + session +
        redirect) and atomically flips ``consumed_at`` — a second consume
        of the same state is a REPLAYED_STATE refusal;
      - expired flows refuse and are purged opportunistically;
      - the raw state exists only inside the authorization URL handed to
        the OPERATOR (API/CLI/dashboard) — never to the model, never in
        logs, never in chat history.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def begin_flow(
        self,
        *,
        provider: str,
        session_id: str,
        display_label: str,
        scopes: frozenset[str],
        redirect_uri: str,
        ttl_seconds: int | None = None,
    ) -> tuple[str, StoredFlow]:
        """Mint a fresh one-time state bound to every context dimension."""
        session = (session_id or "").strip()
        if not session or len(session) > 64:
            raise OAuthFlowError(OAuthErrorCategory.SESSION_MISMATCH, "a valid session_id is required")
        if not (display_label or "").strip():
            raise OAuthFlowError(OAuthErrorCategory.MALFORMED_CALLBACK, "display_label is required")
        ttl = int(ttl_seconds or getattr(settings, "OAUTH_STATE_TTL_SECONDS", STATE_TTL_SECONDS_DEFAULT))
        raw_state = secrets.token_urlsafe(32)
        state_hash = hashlib.sha256(raw_state.encode("ascii")).hexdigest()
        verifier = new_code_verifier()
        challenge = code_challenge_s256(verifier)
        flow = self._store.insert_oauth_flow(
            state_hash=state_hash,
            provider=provider,
            session_id=session,
            display_label=display_label.strip()[:64],
            scopes=frozenset(scopes),
            redirect_uri=redirect_uri,
            code_challenge=challenge,
            code_verifier=verifier,
            ttl_seconds=ttl,
        )
        log.info("oauth_flow_started", provider=provider, ttl=ttl)  # no state value
        return raw_state, _flow_from_row(flow)

    def consume_flow(
        self,
        *,
        raw_state: str,
        provider: str,
        session_id: str,
        redirect_uri: str,
    ) -> StoredFlow:
        """
        Atomic one-time consumption with FULL binding validation. Any
        failure raises a precise OAuthFlowError and leaves the row for
        diagnosis (a mismatched binding does NOT consume the legitimate
        flow — only a full match flips it).
        """
        state = (raw_state or "").strip()
        if not state or len(state) > 256 or (session_id or "").strip() == "":
            raise OAuthFlowError(OAuthErrorCategory.MALFORMED_CALLBACK, "callback is missing state/session")
        state_hash = hashlib.sha256(state.encode("ascii")).hexdigest()
        raw_row = self._store.get_oauth_flow(state_hash)
        if raw_row is None:
            raise OAuthFlowError(OAuthErrorCategory.INVALID_STATE, "unknown authorization state")
        row = _flow_from_row(raw_row)
        if row.consumed_at is not None:
            raise OAuthFlowError(OAuthErrorCategory.REPLAYED_STATE, "authorization state was already used")
        now = datetime.now(tz=timezone.utc)
        if row.expires_at and _parse_iso(row.expires_at) is not None and _parse_iso(row.expires_at) < now:
            raise OAuthFlowError(OAuthErrorCategory.EXPIRED_STATE, "authorization state expired; start again")
        if row.provider != provider:
            raise OAuthFlowError(OAuthErrorCategory.PROVIDER_MISMATCH, "state does not match this provider")
        if row.session_id != (session_id or "").strip():
            raise OAuthFlowError(OAuthErrorCategory.SESSION_MISMATCH, "state is bound to a different session")
        if row.redirect_uri != (redirect_uri or ""):
            raise OAuthFlowError(OAuthErrorCategory.REDIRECT_DENIED, "redirect does not match the authorization request")
        consumed = self._store.try_consume_oauth_flow(state_hash, consumed_at=now.isoformat())
        if not consumed:  # lost a race between two callbacks → the other one won
            raise OAuthFlowError(OAuthErrorCategory.REPLAYED_STATE, "authorization state was already used")
        log.info("oauth_flow_consumed", provider=provider)
        return row

    def record_outcome(self, state_hash: str, outcome: str) -> None:
        """Persist the terminal outcome ('AUTHORIZED' or 'FAILED:<category>')."""
        try:
            self._store.record_oauth_flow_outcome(state_hash, outcome[:40])
        except Exception as e:  # noqa: BLE001 — telemetry never breaks the flow
            log.warning("oauth_flow_outcome_record_failed", error=str(e))

    def purge_expired(self) -> int:
        try:
            return int(self._store.purge_expired_oauth_states())
        except Exception as e:  # noqa: BLE001
            log.warning("oauth_state_purge_failed", error=str(e))
            return 0


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except Exception:  # noqa: BLE001
        return None


# ── Part 6: the token-exchange seam (adapter-owned; never model-visible) ─────


class OAuthTokenClient(ABC):
    """
    ONE seam for code exchange / refresh / revocation. Implementations:

      - ``HttpOAuthTokenClient`` — real RFC 6749 form posts over stdlib
        urllib (for production adapters; no third-party dependency).
      - ``LocalOAuthTokenClient`` — talks to the in-process
        LocalAuthorizationServer (deterministic development provider).

    RETRY DISCIPLINE (Part 6, spec): NO blind retries. A one-time code or a
    rotating refresh token makes a repeated exchange/refresh WRONG, not
    just wasteful: exactly one attempt per call, explicit timeout, and
    normalized failures. Transport trouble maps to UNAVAILABLE.
    """

    @abstractmethod
    def exchange(
        self,
        config: OAuthClientConfig,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> TokenSet: ...

    @abstractmethod
    def refresh(self, config: OAuthClientConfig, *, refresh_token: str) -> TokenSet: ...

    @abstractmethod
    def revoke(self, config: OAuthClientConfig, *, token: str) -> bool: ...


class HttpOAuthTokenClient(OAuthTokenClient):
    """Real-provider client: RFC 6749/7636 over stdlib urllib. No retries."""

    _TRANSIENT_TOKENS = frozenset({"timeout", "temporarily_unavailable"})

    def _post_form(self, url: str, form: dict[str, str], timeout: float) -> dict[str, Any]:
        from urllib.error import HTTPError, URLError
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen

        data = urlencode(form).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        try:
            with urlopen(request, timeout=timeout) as resp:
                import json as _json

                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            # 4xx = deterministic refusal (no retry); 5xx = outage (no retry
            # here either — the caller may surface AUTHENTICATION_REQUIRED).
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                body = ""
            if "invalid_grant" in body:
                raise OAuthFlowError(
                    OAuthErrorCategory.REVOKED_GRANT,
                    "the provider rejected the grant (invalid_grant)",
                ) from e
            category = OAuthErrorCategory.UNAVAILABLE if e.code >= 500 else OAuthErrorCategory.EXCHANGE_FAILED
            raise OAuthFlowError(category, f"token endpoint returned HTTP {e.code}") from e
        except (URLError, TimeoutError) as e:
            raise OAuthFlowError(OAuthErrorCategory.UNAVAILABLE, "token endpoint unreachable") from e

    def _token_set(self, config: OAuthClientConfig, payload: dict[str, Any]) -> TokenSet:
        access = str(payload.get("access_token") or "")
        if not access:
            raise OAuthFlowError(OAuthErrorCategory.EXCHANGE_FAILED, "token response missing access_token")
        expires_in = float(payload.get("expires_in") or 3600)
        granted = str(payload.get("scope") or "")
        refresh = payload.get("refresh_token")
        return TokenSet(
            access_token=access,
            refresh_token=(str(refresh) if refresh else None),
            expires_at=datetime.now(tz=timezone.utc) + timedelta(seconds=max(1.0, expires_in)),
            scopes=frozenset(g for g in granted.split(config.scope_separator) if g),
        )

    def exchange(
        self, config: OAuthClientConfig, *, code: str, redirect_uri: str, code_verifier: str
    ) -> TokenSet:
        payload = self._post_form(
            config.token_endpoint,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code_verifier": code_verifier,
            },
            config.timeout_seconds,
        )
        return self._token_set(config, payload)

    def refresh(self, config: OAuthClientConfig, *, refresh_token: str) -> TokenSet:
        payload = self._post_form(
            config.token_endpoint,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": config.client_id,
                "client_secret": config.client_secret,
            },
            config.timeout_seconds,
        )
        token_set = self._token_set(config, payload)
        if token_set.refresh_token is None:
            # RFC 6749: the provider MAY omit a new refresh token — keep the
            # existing one valid rather than dropping refresh capability.
            token_set = TokenSet(
                access_token=token_set.access_token,
                refresh_token=refresh_token,
                expires_at=token_set.expires_at,
                scopes=token_set.scopes,
                account_ref=token_set.account_ref,
                token_type=token_set.token_type,
            )
        return token_set

    def revoke(self, config: OAuthClientConfig, *, token: str) -> bool:
        if not config.revocation_endpoint:
            return False
        try:
            self._post_form(
                config.revocation_endpoint,
                {"token": token, "client_id": config.client_id, "client_secret": config.client_secret},
                config.timeout_seconds,
            )
            return True
        except OAuthFlowError:
            return False  # best-effort revocation (disconnect proceeds anyway)


class LocalOAuthTokenClient(OAuthTokenClient):
    """Development client bound to the in-process LocalAuthorizationServer."""

    def __init__(self, server: "LocalAuthorizationServer") -> None:
        self._server = server

    def exchange(
        self, config: OAuthClientConfig, *, code: str, redirect_uri: str, code_verifier: str
    ) -> TokenSet:
        payload = self._server.exchange(
            client_id=config.client_id,
            client_secret=config.client_secret,
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )
        return TokenSet(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            expires_at=datetime.now(tz=timezone.utc) + timedelta(seconds=float(payload["expires_in"])),
            scopes=frozenset(g for g in str(payload.get("scope") or "").split(" ") if g),
            account_ref=str(payload.get("account_ref") or ""),
        )

    def refresh(self, config: OAuthClientConfig, *, refresh_token: str) -> TokenSet:
        payload = self._server.refresh(
            client_id=config.client_id,
            client_secret=config.client_secret,
            refresh_token=refresh_token,
        )
        return TokenSet(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            expires_at=datetime.now(tz=timezone.utc) + timedelta(seconds=float(payload["expires_in"])),
            scopes=frozenset(g for g in str(payload.get("scope") or "").split(" ") if g),
            account_ref=str(payload.get("account_ref") or ""),
        )

    def revoke(self, config: OAuthClientConfig, *, token: str) -> bool:
        return bool(self._server.revoke(token))


# ── Part 26: deterministic LOCAL AUTHORIZATION SERVER (simulated provider) ────


class LocalAuthorizationServer:
    """
    A deterministic, in-process OAuth authorization server — the "controlled
    development/test provider" for v0.30. Implements REAL authorization-code
    semantics so the machinery is exercised for what it is:

      - registered clients (client_id + secret) and consented subjects;
      - one-time authorization CODES (short TTL) bound to
        (client, redirect, scope, PKCE challenge, subject);
      - PKCE S256 verification at the token endpoint;
      - access tokens with expiry + refresh-token ROTATION where replaying a
        rotated refresh token REVOKES the whole grant (industry behavior);
      - introspection for provider-side verification.

    NO network I/O; all clocks are simulated-offset-controllable for tests.
    It is provider-SIDE simulation only: it never holds JARVIS state.
    """

    CODE_TTL_SECONDS = 120
    ACCESS_TTL_SECONDS_DEFAULT = 3600

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._clients: dict[str, str] = {}            # client_id → client_secret
        self._subjects: dict[str, dict[str, str]] = {}  # subject → {email, name}
        self._pending: dict[str, dict[str, Any]] = {}  # state → pending authorization
        self._codes: dict[str, dict[str, Any]] = {}    # code → authorization grant request
        self._grants: dict[str, dict[str, Any]] = {}   # access_token → grant view
        self._refresh: dict[str, str] = {}             # refresh_token → grant_id
        self._grant_counter = 0
        self._clock_offset = 0.0
        self._fail_exchange: list[str] = []
        self._fail_refresh: list[str] = []

    # ── test/ops controls ────────────────────────────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._pending.clear(); self._codes.clear(); self._grants.clear(); self._refresh.clear()
            self._grant_counter = 0; self._clock_offset = 0.0
            self._fail_exchange.clear(); self._fail_refresh.clear()

    def advance_clock(self, seconds: float) -> None:
        with self._lock:
            self._clock_offset += float(seconds)

    def fail_next_exchange(self, kind: str = "server") -> None:
        with self._lock:
            self._fail_exchange.append(kind)

    def fail_next_refresh(self, kind: str = "server") -> None:
        with self._lock:
            self._fail_refresh.append(kind)

    def register_client(self, client_id: str, client_secret: str) -> None:
        with self._lock:
            self._clients[client_id] = client_secret

    def register_subject(self, subject: str, *, email: str = "", name: str = "") -> None:
        with self._lock:
            self._subjects[subject] = {"email": email, "name": name}

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc) + timedelta(seconds=self._clock_offset)

    # ── authorization endpoint ───────────────────────────────────────────────

    def begin_authorization(
        self, *, client_id: str, redirect_uri: str, scope: str, state: str, code_challenge: str
    ) -> None:
        with self._lock:
            self._pending[state] = {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "code_challenge": code_challenge,
                "created_at": self._now(),
            }

    def authorize(self, state: str, *, granted: bool = True, subject: str = "user-local-1") -> str:
        """
        The (simulated) user consents — mints the one-time code and returns
        it (the value a real provider would put in the redirect). Returns ""
        when consent is denied (the authorization is simply dropped).
        """
        with self._lock:
            pending = self._pending.get(state)
            if pending is None:
                raise OAuthFlowError(OAuthErrorCategory.INVALID_STATE, "no pending authorization")
            self._pending.pop(state, None)
            if not granted:
                return ""
            code = "ac_" + secrets.token_urlsafe(32)
            self._codes[code] = {**pending, "subject": subject, "created_at": self._now()}
            return code

    # ── token endpoint ───────────────────────────────────────────────────────

    def exchange(
        self,
        *,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        with self._lock:
            if self._fail_exchange:
                kind = self._fail_exchange.pop(0)
                if kind == "timeout":
                    raise OAuthFlowError(OAuthErrorCategory.UNAVAILABLE, "simulated timeout")
                if kind == "invalid_grant":
                    raise OAuthFlowError(OAuthErrorCategory.REVOKED_GRANT, "simulated invalid_grant")
                raise OAuthFlowError(OAuthErrorCategory.EXCHANGE_FAILED, "simulated exchange failure")
            if self._clients.get(client_id) != client_secret:
                raise OAuthFlowError(OAuthErrorCategory.EXCHANGE_FAILED, "client authentication failed")
            grant_req = self._codes.pop(code, None)  # one-time: replay is structurally impossible
            if grant_req is None:
                raise OAuthFlowError(OAuthErrorCategory.REVOKED_GRANT, "unknown or already-used authorization code")
            if grant_req["client_id"] != client_id or grant_req["redirect_uri"] != redirect_uri:
                raise OAuthFlowError(OAuthErrorCategory.REDIRECT_DENIED, "code does not match this client/redirect")
            if self._now() - grant_req["created_at"] > timedelta(seconds=self.CODE_TTL_SECONDS):
                raise OAuthFlowError(OAuthErrorCategory.EXPIRED_STATE, "authorization code expired")
            expected = grant_req.get("code_challenge") or ""
            if not expected or code_challenge_s256(code_verifier) != expected:
                raise OAuthFlowError(OAuthErrorCategory.EXCHANGE_FAILED, "PKCE verification failed")
            return self._mint_grant(grant_req["scope"], grant_req["subject"], client_id)

    def refresh(self, *, client_id: str, client_secret: str, refresh_token: str) -> dict[str, Any]:
        with self._lock:
            if self._fail_refresh:
                kind = self._fail_refresh.pop(0)
                if kind == "invalid_grant":
                    self._revoke_by_refresh(refresh_token)
                    raise OAuthFlowError(OAuthErrorCategory.REVOKED_GRANT, "refresh token rejected")
                raise OAuthFlowError(OAuthErrorCategory.UNAVAILABLE, "simulated refresh failure")
            if self._clients.get(client_id) != client_secret:
                raise OAuthFlowError(OAuthErrorCategory.EXCHANGE_FAILED, "client authentication failed")
            grant_id = self._refresh.get(refresh_token)
            if grant_id is None:
                raise OAuthFlowError(OAuthErrorCategory.REVOKED_GRANT, "unknown refresh token")
            grant = self._grants.get(grant_id)
            if grant is None or grant["revoked"]:
                raise OAuthFlowError(OAuthErrorCategory.REVOKED_GRANT, "grant is revoked")
            if grant["refresh_token"] != refresh_token:
                # ROTATION REPLAY: a rotated-out refresh token was reused —
                # real providers revoke the whole grant here; so do we.
                grant["revoked"] = True
                raise OAuthFlowError(OAuthErrorCategory.REVOKED_GRANT, "refresh token replay detected; grant revoked")
            return self._mint_grant(grant["scope"], grant["subject"], client_id, grant_id=grant_id)

    def _mint_grant(self, scope: str, subject: str, client_id: str, *, grant_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if grant_id is None:
                self._grant_counter += 1
                grant_id = f"grant-{self._grant_counter}"
            access = "at_" + secrets.token_urlsafe(32)
            refresh = "rt_" + secrets.token_urlsafe(32)
            ttl = float(getattr(settings, "OAUTH_ACCESS_TOKEN_TTL_SECONDS", self.ACCESS_TTL_SECONDS_DEFAULT))
            exp = self._now() + timedelta(seconds=ttl)
            self._grants[grant_id] = {
                "grant_id": grant_id,
                "scope": scope,
                "subject": subject,
                "client_id": client_id,
                "access_token": access,
                "access_exp": exp,
                "refresh_token": refresh,
                "revoked": False,
            }
            self._refresh[refresh] = grant_id
            subject_info = self._subjects.get(subject, {})
            return {
                "access_token": access,
                "refresh_token": refresh,
                "expires_in": max(1, int(ttl)),
                "scope": scope,
                "token_type": "Bearer",
                "account_ref": subject_info.get("email") or subject,
            }

    def revoke(self, token: str) -> bool:
        with self._lock:
            grant = self._grants.get(token) or self._grants.get(self._refresh.get(token, ""))
            if grant is None:
                return False
            grant["revoked"] = True
            return True

    def _revoke_by_refresh(self, refresh_token: str) -> None:
        grant_id = self._refresh.get(refresh_token)
        if grant_id and grant_id in self._grants:
            self._grants[grant_id]["revoked"] = True

    # ── introspection (provider-side verification seam) ──────────────────────

    def introspect(self, access_token: str) -> dict[str, Any]:
        with self._lock:
            for grant in self._grants.values():
                if grant["access_token"] == access_token:
                    if grant["revoked"]:
                        return {"active": False, "reason": "revoked"}
                    if grant["access_exp"] < self._now():
                        return {"active": False, "reason": "expired"}
                    return {
                        "active": True,
                        "sub": grant["subject"],
                        "scope": grant["scope"],
                        "exp": grant["access_exp"].isoformat(),
                    }
            return {"active": False, "reason": "unknown"}

    def subject_info(self, subject: str) -> dict[str, str]:
        with self._lock:
            return dict(self._subjects.get(subject, {}))


_server = LocalAuthorizationServer()


def local_authorization_server() -> LocalAuthorizationServer:
    return _server


def local_introspection_auth_state(account: Any) -> AuthState:
    """
    Provider-side verification for OAuth accounts on the LOCAL simulated
    server (used by the local calendar/tasks adapters). Fail-closed: an
    unknown/inactive token is never reported as authenticated.
    """
    access = getattr(account, "_access_token", "") or ""
    if not access:
        return AuthState.REVOKED
    info = local_authorization_server().introspect(access)
    if info.get("active"):
        return AuthState.AUTHENTICATED
    reason = str(info.get("reason") or "")
    if reason == "expired":
        return AuthState.EXPIRED
    return AuthState.REVOKED  # revoked or unknown → fail closed


@dataclass(frozen=True)
class LocalConsentResult:
    """Callback parameters a real provider would redirect with."""

    redirect_uri: str
    code: str = field(repr=False)
    state: str = field(repr=False)
    granted: bool = True


def local_simulate_consent(
    authorization_url: str,
    *,
    granted: bool = True,
    subject: str = "user-local-1",
) -> LocalConsentResult:
    """
    DEVELOPMENT-ONLY bridge for the simulated provider: parse the authorization
    URL our manager just produced, register the pending request on the local
    authorization server, record the (simulated) user's decision, and return
    the redirect parameters.

    This plays the role of (a) the provider's authorization endpoint and
    (b) the human clicking Grant in the consent screen. A REAL provider needs
    none of this: the operator opens ``authorization_url`` in a browser and the
    provider redirects the browser to our callback endpoint. It exists so
    dev/tests/live-eval can exercise the REAL state+PKCE+exchange machinery
    without network I/O.
    """
    parsed = urllib.parse.urlparse(authorization_url)
    query = urllib.parse.parse_qs(parsed.query)

    def _one(name: str) -> str:
        values = query.get(name) or []
        return str(values[0]) if values else ""

    state = _one("state")
    redirect_uri = _one("redirect_uri")
    if not state or not redirect_uri:
        raise OAuthFlowError(
            OAuthErrorCategory.MALFORMED_CALLBACK,
            "authorization URL is missing state/redirect_uri",
        )
    server = local_authorization_server()
    server.begin_authorization(
        client_id=_one("client_id"),
        redirect_uri=redirect_uri,
        scope=_one("scope"),
        state=state,
        code_challenge=_one("code_challenge"),
    )
    code = server.authorize(state, granted=granted, subject=subject)
    return LocalConsentResult(
        redirect_uri=redirect_uri, code=code, state=state, granted=bool(code)
    )


# ── Part 3/5: the OAuth-aware provider mixin + redirect discipline ───────────


def default_redirect_uri(provider: str) -> str:
    """
    The ONLY redirect construction in the system: settings base + a fixed
    per-provider path. Operator-supplied redirect URLs are never accepted
    (open-redirect structurally impossible, Part 5); the base URL is
    deployment configuration, not model or provider input.
    """
    base = str(getattr(settings, "OAUTH_REDIRECT_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
    return f"{base}/integrations/oauth/callback/{provider}"


def redirect_uri_for_session(provider: str, session_id: str) -> str:
    """
    The redirect we register for one flow: the fixed path plus OUR OWN
    ``session_id`` query parameter. The parameter travels through the
    provider verbatim (a redirect URI is a fixed string, not a template), so
    the callback endpoint can complete the flow without any ambient session
    state — and the state row still validates the FULL string exactly, so a
    tampered ``session_id`` is a SESSION_MISMATCH refusal, never a
    cross-session completion.
    """
    base = default_redirect_uri(provider)
    return f"{base}?session_id={urllib.parse.quote(str(session_id or ''), safe='')}"


@dataclass(frozen=True)
class AuthorizationStart:
    """What begin_authorization hands back (operator-facing; state in URL)."""

    authorization_url: str
    redirect_uri: str
    expires_in_seconds: int


class OAuthIntegrationProvider:
    """
    Mixin for IntegrationProvider adapters that support real OAuth. An
    adapter declares its OAuthClientConfig + token client and inherits the
    full flow; provider-specific identity discovery is one method.

    The mixin NEVER trusts callback-carried context: display label, scopes,
    session, and redirect come from the STORED FLOW (minted at
    begin_authorization), never from the callback query.
    """

    supports_oauth: bool = True

    # ── adapter hooks (implement per provider) ───────────────────────────────

    def oauth_config(self) -> OAuthClientConfig:
        raise NotImplementedError("adapter must provide its OAuthClientConfig")

    def token_client(self) -> OAuthTokenClient:
        raise NotImplementedError("adapter must provide its OAuthTokenClient")

    # ── flow (used by the manager only) ──────────────────────────────────────

    def begin_authorization(
        self,
        store: Any,
        *,
        session_id: str,
        display_label: str,
        scopes: frozenset[str],
        redirect_uri: str | None = None,
    ) -> AuthorizationStart:
        config = self.oauth_config()
        flows = OAuthFlowManager(store)
        flows.purge_expired()
        effective_redirect = (
            redirect_uri
            or config.redirect_uri
            or redirect_uri_for_session(config.provider, session_id)
        )
        raw_state, flow = flows.begin_flow(
            provider=config.provider,
            session_id=session_id,
            display_label=display_label,
            scopes=validate_scope_set(config.provider, scopes),
            redirect_uri=effective_redirect,
        )
        challenge = flow.code_challenge
        params = urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": config.client_id,
                "redirect_uri": effective_redirect,
                "scope": config.provider_scope_string(flow.scopes),
                "state": raw_state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        separator = "&" if "?" in config.authorization_endpoint else "?"
        return AuthorizationStart(
            authorization_url=f"{config.authorization_endpoint}{separator}{params}",
            redirect_uri=effective_redirect,
            expires_in_seconds=int(getattr(settings, "OAUTH_STATE_TTL_SECONDS", STATE_TTL_SECONDS_DEFAULT)),
        )

    def handle_callback(
        self,
        store: Any,
        *,
        code: str,
        raw_state: str,
        session_id: str,
        redirect_uri: str | None = None,
    ) -> tuple[StoredFlow, TokenSet]:
        """
        Consume the one-time state (full binding), exchange the code (once,
        PKCE), and verify the grant was not scope-downgraded. Returns the
        consumed flow (carries label/scopes/session for persistence) and
        the TokenSet. Every failure is a fail-closed OAuthFlowError.
        """
        if not (code or "").strip() or len(code) > 512:
            raise OAuthFlowError(OAuthErrorCategory.MALFORMED_CALLBACK, "callback is missing a usable code")
        config = self.oauth_config()
        flows = OAuthFlowManager(store)
        effective_redirect = (
            redirect_uri
            or config.redirect_uri
            or redirect_uri_for_session(config.provider, session_id)
        )
        flow = flows.consume_flow(
            raw_state=raw_state,
            provider=config.provider,
            session_id=session_id,
            redirect_uri=effective_redirect,
        )
        try:
            token_set = self.token_client().exchange(
                config,
                code=code.strip(),
                redirect_uri=effective_redirect,
                code_verifier=flow.code_verifier,
            )
        except OAuthFlowError as e:
            flows.record_outcome(flow.state_hash, f"FAILED:{e.category.value}")
            raise
        # Scope verification (Part 10): the grant must cover the request.
        granted_jarvis = config.jarvis_scopes_from_provider(_scope_string(token_set, config))
        missing = sorted(set(flow.scopes) - granted_jarvis)
        if missing:
            flows.record_outcome(flow.state_hash, f"FAILED:{OAuthErrorCategory.SCOPE_DOWNGRADED.value}")
            raise OAuthFlowError(
                OAuthErrorCategory.SCOPE_DOWNGRADED,
                f"the provider granted less than requested ({', '.join(missing[:3])})",
            )
        flows.record_outcome(flow.state_hash, "AUTHORIZED")
        return flow, token_set

    def discover_account_ref(self, token_set: TokenSet) -> str:
        """Provider-specific identity discovery; default = token's account_ref."""
        return token_set.account_ref

    def revoke_remote(self, account: "Any") -> bool:
        """Best-effort provider-side revocation (Part 13 disconnect)."""
        access = getattr(account, "_access_token", "") or ""
        refresh = getattr(account, "_refresh_token", "") or ""
        config = self.oauth_config()
        client = self.token_client()
        revoked = False
        if refresh:
            revoked = client.revoke(config, token=refresh) or revoked
        if access and not revoked:
            revoked = client.revoke(config, token=access)
        return revoked

    def oauth_introspect(self, account: "Any") -> dict[str, Any] | None:
        """Cheap provider-side verification seam (adapters may override)."""
        access = getattr(account, "_access_token", "") or ""
        if not access:
            return None
        client = self.token_client()
        introspect = getattr(client, "introspect", None)
        if callable(introspect):
            return introspect(access)
        return None


def _scope_string(token_set: TokenSet, config: OAuthClientConfig) -> str:
    return config.scope_separator.join(sorted(token_set.scopes))
