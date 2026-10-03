"""
jarvis/integrations/manager.py
──────────────────────────────
v0.29 Parts 3/5/13/19 — the IntegrationManager: the ONLY component that
connects, inspects, authorizes, and drives external providers.

Enforcement order for EVERY operation (all fail-closed):

  1. Account exists + provider adapter registered (else honest refusal).
  2. Provider availability check.
  3. REAL authentication state from the adapter (never model text, never
     stale optimism). EXPIRED/REVOKED/DISCONNECTED refuse the operation
     with the honest state (no blind retry of side effects).
  4. EXACT scope check for the requested operation (Part 5).
  5. Operation → side-effect risk → the caller (tool layer) maps it onto
     the EXISTING permission tiers before dispatch (no second system).

Writes record an integration_audit row (Part 13) with request id +
idempotency key; UNKNOWN outcomes stay UNKNOWN (AMBIGUOUS_CATEGORIES).
Reads are auto-retried at most twice, only for RATE_LIMITED/PROVIDER_OUTAGE
(Part 15: never a side effect).

Connect flow (Part 19, local-dev shape): the USER supplies the credential
through the CLI/API/dashboard — never through the chat model. The manager
validates it OUTSIDE any LLM context, records identity + scopes explicitly,
and verifies immediately. No raw codes or tokens are ever exposed to the
model: the model-visible surface is the account's public metadata only.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from jarvis.integrations.base import (
    AuthState,
    ConnectedAccount,
    IntegrationProvider,
    Operation,
    ProviderResource,
    SideEffectRisk,
)
from jarvis.config import settings
from jarvis.integrations.credentials import (
    credential_fingerprint,
    deobfuscate,
    new_credential_material,
    obfuscate,
)
from jarvis.integrations.errors import (
    AMBIGUOUS_CATEGORIES,
    READ_RETRYABLE_CATEGORIES,
    ProviderError,
    ProviderErrorCategory,
)
from jarvis.integrations.oauth import (
    AuthorizationStatus,
    OAuthErrorCategory,
    OAuthFlowError,
    OAuthFlowManager,
)
from jarvis.integrations.scopes import (
    PROVIDER_SCOPE_MENUS,
    validate_scope_set,
)
from jarvis.memory.session_store import SessionStore
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Bounded auto-retry for deterministic reads only (Part 15).
_MAX_READ_RETRIES = 2
# Bounded result_summary in the audit trail (chars).
_MAX_AUDIT_SUMMARY_CHARS = 300


class IntegrationManagerError(Exception):
    """Operator-facing refusal (safe message; never a credential)."""


class IntegrationManager:
    """Central, stateful authority for connected personal services."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._providers: dict[str, IntegrationProvider] = {}

    # ── provider registration (runtime assembly; not model-reachable) ────────

    def register_provider(self, provider: IntegrationProvider) -> None:
        name = provider.capabilities.provider
        if name in self._providers:
            raise ValueError(f"provider {name!r} already registered")
        self._providers[name] = provider
        log.info(
            "integration_provider_registered",
            provider=name,
            production_like=provider.capabilities.production_like,
        )

    def providers(self) -> dict[str, IntegrationProvider]:
        return dict(self._providers)

    # ── Part 3: explicit account identity ─────────────────────────────────────

    def connect(
        self,
        *,
        provider: str,
        display_label: str,
        credential: str | None = None,
        scopes: frozenset[str] | set[str] | None = None,
        provider_account_ref: str = "",
    ) -> ConnectedAccount:
        """
        Explicitly connect one account (user-controlled; Part 19).

        The credential NEVER passes through the model: callers are the CLI,
        API, or dashboard. Scopes are validated against the provider menu
        (exact vocabulary; the model cannot expand them). When ``credential``
        is omitted a random local-dev token is generated (mock/local
        providers; real OAuth providers would run their flow here and store
        the resulting refresh token the same way).
        """
        cap = self._provider_capabilities(provider)
        granted = validate_scope_set(provider, frozenset(scopes or set()))
        label = display_label.strip()
        if not label or len(label) > 64:
            raise IntegrationManagerError(
                "display_label must be 1-64 characters"
            )
        existing = self._store.get_integration_account_by_label(provider, label)
        if existing is not None:
            raise IntegrationManagerError(
                f"account '{label}' already connected for {provider}; "
                "disconnect it first"
            )
        secret = credential or new_credential_material()
        account_id = f"{provider}:{uuid.uuid4().hex[:12]}"
        self._store.upsert_integration_account(
            account_id=account_id,
            provider=provider,
            display_label=label,
            scopes=granted,
            credential_obfuscated=obfuscate(secret),
            credential_fingerprint=credential_fingerprint(secret),
            provider_account_ref=provider_account_ref.strip()[:120],
            auth_state=AuthState.AUTHENTICATED.value,
            granted_at=_utcnow(),
        )
        account = self._load_account(account_id)
        if account is None:  # pragma: no cover - defensive
            raise IntegrationManagerError("connect failed to persist the account")
        # Immediate honest verification; persist the result.
        account = self.refresh_auth_state(account)
        log.info(
            "integration_account_connected",
            provider=provider,
            account_id=account_id,
            scopes=sorted(granted),
            auth_state=account.auth_state.value,
        )
        return account

    def disconnect(self, account_id: str) -> bool:
        """
        Remove a connected account entirely (Part 19; extended v0.30 Part 13):
        best-effort PROVIDER-SIDE revocation first (where the adapter
        supports it — disconnect proceeds even when revocation fails, since
        the local fail-closed state is what gates every operation), then the
        row deletion, then invalidation of that account's cached private
        reads (Part 22).
        """
        account = self._load_account(account_id)
        if account is not None and account.is_oauth_account:
            provider = self._providers.get(account.provider)
            if provider is not None and getattr(provider, "supports_oauth", False):
                try:
                    revoked = provider.revoke_remote(account)
                    log.info(
                        "integration_provider_revocation_attempted",
                        account_id=account_id,
                        revoked=revoked,
                    )
                except Exception as e:  # noqa: BLE001 — revocation is best-effort
                    log.warning("integration_provider_revocation_failed", error=str(e))
        existed = self._store.delete_integration_account(account_id)
        if existed:
            invalidated = self._store.delete_result_cache_for_account(account_id)
            log.info(
                "integration_account_disconnected",
                account_id=account_id,
                invalidated_cache=invalidated,
            )
        return existed

    def list_accounts(self, provider: str | None = None) -> list[ConnectedAccount]:
        """All accounts as SAFE public projections (no credentials)."""
        out: list[ConnectedAccount] = []
        for row in self._store.list_integration_accounts(provider):
            account = self._account_from_row(row)
            if account is not None:
                out.append(account)
        return out

    def get_account(self, account_id: str) -> ConnectedAccount | None:
        return self._load_account(account_id)

    def refresh_auth_state(self, account: ConnectedAccount) -> ConnectedAccount:
        """
        Ask the provider for the REAL current auth state and persist it
        (Part 3: last-verified + expired/revoked are explicit, never guessed).
        v0.30: OAuth accounts first run the token-lifecycle step
        (opportunistic refresh when expired/near expiry; a REVOKED grant is
        NEVER retried), which feeds the operational state persisted below.
        Falls back to ERROR (honest unknowable) when the provider is down —
        the previous state is not silently kept as if fresh.
        """
        if account.is_oauth_account:
            account = self._refresh_oauth_tokens_if_needed(account)
        provider = self._providers.get(account.provider)
        if provider is None:
            state = AuthState.ERROR
        else:
            try:
                state = provider.verify_authentication(account)
            except ProviderError:
                state = AuthState.ERROR
            except Exception:  # noqa: BLE001 - verification never crashes a turn
                log.warning("integration_verify_failed", provider=account.provider)
                state = AuthState.ERROR
        verified = (
            datetime.now(tz=timezone.utc).isoformat() if state == AuthState.AUTHENTICATED else None
        )
        self._store.update_integration_account_auth(
            account.account_id, state.value, verified
        )
        refreshed = self._load_account(account.account_id)
        if refreshed is not None:
            return refreshed
        # The row is gone (disconnect / crash window): return the state we just
        # learned from the provider rather than a stale in-memory projection.
        return replace(
            account, auth_state=state, authenticated=(state == AuthState.AUTHENTICATED)
        )

    # ── v0.30: real OAuth authorization + token lifecycle (Parts 3/6/9) ─────

    def begin_authorization(
        self,
        *,
        provider: str,
        session_id: str,
        display_label: str,
        scopes: frozenset[str] | set[str] | None = None,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        """
        Start ONE user-controlled OAuth authorization (Parts 4/17). The
        authorization URL is handed to the OPERATOR (API/CLI/dashboard) —
        never to the chat model; the one-time state it carries is persisted
        only as a hash and is hard-bound to (provider, session, label,
        scopes, exact redirect). Re-authorization of an EXISTING OAuth
        account (same label) rotates its tokens; colliding with a legacy
        raw-credential account is refused.
        """
        adapter = self._providers.get(provider)
        if adapter is None:
            raise IntegrationManagerError(
                f"provider {provider!r} is not enabled in this runtime"
            )
        if not getattr(adapter, "supports_oauth", False):
            raise IntegrationManagerError(
                f"provider {provider!r} does not support OAuth authorization"
            )
        granted = validate_scope_set(provider, frozenset(scopes or set()))
        label = (display_label or "").strip()
        if not label or len(label) > 64:
            raise IntegrationManagerError("display_label must be 1-64 characters")
        existing = self._store.get_integration_account_by_label(provider, label)
        if existing is not None and not existing.get("authorization_status"):
            raise IntegrationManagerError(
                f"account '{label}' already connected for {provider} via a "
                "local credential; disconnect it before authorizing with OAuth"
            )
        try:
            start = adapter.begin_authorization(
                self._store,
                session_id=session_id,
                display_label=label,
                scopes=granted,
                redirect_uri=redirect_uri,
            )
        except OAuthFlowError as e:
            raise IntegrationManagerError(e.to_public_message()) from e
        log.info("integration_authorization_started", provider=provider)
        return {
            "authorization_url": start.authorization_url,
            "redirect_uri": start.redirect_uri,
            "expires_in": start.expires_in_seconds,
            "provider": provider,
            "display_label": label,
            "scopes": sorted(granted),
        }

    def handle_callback(
        self,
        *,
        provider: str,
        code: str,
        state: str,
        session_id: str,
        redirect_uri: str | None = None,
    ) -> ConnectedAccount:
        """
        Complete ONE authorization (Parts 4/5/6/7): atomic one-time state
        consumption with full binding → single PKCE code exchange → identity
        discovery → one-transaction account persistence → cache invalidation
        (Part 22) → immediate honest verification. Fail-closed on every
        OAuthFlowError; no code/state values ever reach logs or messages.
        """
        adapter = self._providers.get(provider)
        if adapter is None or not getattr(adapter, "supports_oauth", False):
            raise IntegrationManagerError(
                f"provider {provider!r} is not OAuth-enabled"
            )
        try:
            flow, token_set = adapter.handle_callback(
                self._store,
                code=code,
                raw_state=state,
                session_id=session_id,
                redirect_uri=redirect_uri,
            )
        except OAuthFlowError as e:
            log.info(
                "integration_oauth_callback_failed",
                provider=provider,
                category=e.category.value,
            )
            raise IntegrationManagerError(e.to_public_message()) from e
        account_ref = token_set.account_ref
        try:
            account_ref = adapter.discover_account_ref(token_set) or account_ref
        except Exception:  # noqa: BLE001 — identity discovery is best-effort
            pass
        now = _utcnow()
        existing = self._store.get_integration_account_by_label(provider, flow.display_label)
        rotate = existing is not None and bool(existing.get("authorization_status"))
        account_id = (
            str(existing["account_id"]) if rotate else f"{provider}:{uuid.uuid4().hex[:12]}"
        )
        granted_at = str(existing["granted_at"]) if existing else now
        self._store.upsert_oauth_account(
            account_id=account_id,
            provider=provider,
            display_label=flow.display_label,
            scopes=frozenset(flow.scopes),
            credential_obfuscated=obfuscate(token_set.access_token),
            credential_fingerprint=credential_fingerprint(token_set.access_token),
            provider_account_ref=str(account_ref)[:120],
            granted_at=granted_at,
            authorization_status=AuthorizationStatus.AUTHORIZED.value,
            token_access_obfuscated=obfuscate(token_set.access_token),
            token_refresh_obfuscated=(
                obfuscate(token_set.refresh_token) if token_set.refresh_token else ""
            ),
            token_expires_at=token_set.expiry_iso(),
            token_updated_at=now,
            auth_state=AuthState.AUTHENTICATED.value,
        )
        # Part 22: tokens/scopes changed → this account's cached private
        # reads are invalid by definition; drop them before anything reads.
        invalidated = self._store.delete_result_cache_for_account(account_id)
        account = self._load_account(account_id)
        if account is None:  # pragma: no cover - defensive
            raise IntegrationManagerError(
                "callback succeeded but the account row is missing"
            )
        account = self.refresh_auth_state(account)
        log.info(
            "integration_oauth_authorized",
            provider=provider,
            account_id=account_id,
            reauthorized=rotate,
            invalidated_cache=invalidated,
        )
        return account

    def authorization_flow_outcome(self, provider: str, session_id: str) -> dict[str, Any]:
        """Bounded polling view for CLI/dashboard (never a state value)."""
        row = self._store.latest_oauth_flow(provider, session_id)
        if row is None:
            return {"status": "NO_FLOW", "provider": provider}
        if not row.get("consumed_at"):
            return {
                "status": "AUTHORIZING",
                "provider": provider,
                "display_label": row.get("display_label"),
            }
        return {
            "status": row.get("outcome") or "CONSUMED",
            "provider": provider,
            "display_label": row.get("display_label"),
        }

    def _refresh_oauth_tokens_if_needed(self, account: ConnectedAccount) -> ConnectedAccount:
        """
        Part 9: refresh when the access token is expired or inside the
        refresh margin. EXACTLY ONE attempt; a rejected/revoked refresh
        token flips the account to REVOKED and is NEVER retried (a replayed
        rotation would revoke real grants); any other failure produces the
        explicit AUTHENTICATION_REQUIRED state (never a silent retry, never
        a fabricated success). Token updates are atomic store operations.
        """
        status = account.authorization_status or AuthorizationStatus.DISCONNECTED.value
        if status in (
            AuthorizationStatus.DISCONNECTED.value,
            AuthorizationStatus.AUTHORIZING.value,
            AuthorizationStatus.REVOKED.value,
        ):
            return account  # a revoked grant's refresh token is never retried
        margin = float(getattr(settings, "OAUTH_REFRESH_MARGIN_SECONDS", 300))
        from datetime import datetime as _dt

        expires: _dt | None = None
        if account.token_expires_at:
            try:
                expires = _dt.fromisoformat(str(account.token_expires_at))
            except Exception:  # noqa: BLE001 — unparseable expiry → refresh
                expires = None
        if expires is not None and (
            expires - _dt.now(tz=expires.tzinfo)
        ).total_seconds() > margin:
            return account  # token still comfortably valid
        if not account._refresh_token:
            self._store.update_integration_account_tokens(
                account.account_id,
                authorization_status=AuthorizationStatus.AUTHENTICATION_REQUIRED.value,
                token_access_obfuscated=obfuscate(account._access_token),
                token_refresh_obfuscated=None,
                token_expires_at=account.token_expires_at or "",
                token_updated_at=_utcnow(),
                auth_state=AuthState.EXPIRED.value,
            )
            log.info("integration_token_refresh_unavailable", account_id=account.account_id)
            return self._load_account(account.account_id) or account
        provider = self._providers.get(account.provider)
        if provider is None or not getattr(provider, "supports_oauth", False):
            return account
        try:
            token_set = provider.token_client().refresh(
                provider.oauth_config(), refresh_token=account._refresh_token
            )
        except OAuthFlowError as e:
            if e.category == OAuthErrorCategory.REVOKED_GRANT:
                new_status = AuthorizationStatus.REVOKED.value
                auth_state = AuthState.REVOKED.value
            else:
                new_status = AuthorizationStatus.AUTHENTICATION_REQUIRED.value
                auth_state = AuthState.EXPIRED.value
            self._store.update_integration_account_tokens(
                account.account_id,
                authorization_status=new_status,
                token_access_obfuscated=obfuscate(account._access_token),
                token_refresh_obfuscated=None,
                token_expires_at=account.token_expires_at or "",
                token_updated_at=_utcnow(),
                auth_state=auth_state,
            )
            log.info(
                "integration_token_refresh_failed",
                provider=account.provider,
                category=e.category.value,
            )
            return self._load_account(account.account_id) or account
        except Exception as e:  # noqa: BLE001 — never leak, never retry blindly
            self._store.update_integration_account_tokens(
                account.account_id,
                authorization_status=AuthorizationStatus.AUTHENTICATION_REQUIRED.value,
                token_access_obfuscated=obfuscate(account._access_token),
                token_refresh_obfuscated=None,
                token_expires_at=account.token_expires_at or "",
                token_updated_at=_utcnow(),
                auth_state=AuthState.EXPIRED.value,
            )
            log.warning("integration_token_refresh_error", error=str(e))
            return self._load_account(account.account_id) or account
        # Success: atomic rotation. When the provider omitted a new refresh
        # token the previous one REMAINS the live rotation (RFC 6749) — the
        # client already preserves it, so we persist what we received.
        new_refresh = token_set.refresh_token or account._refresh_token
        self._store.update_integration_account_tokens(
            account.account_id,
            authorization_status=AuthorizationStatus.AUTHORIZED.value,
            token_access_obfuscated=obfuscate(token_set.access_token),
            token_refresh_obfuscated=obfuscate(new_refresh),
            token_expires_at=token_set.expiry_iso(),
            token_updated_at=_utcnow(),
            auth_state=AuthState.AUTHENTICATED.value,
        )
        log.info("integration_token_refreshed", provider=account.provider)
        return self._load_account(account.account_id) or account

    # ── Part 12: risk surface (read by the tool layer) ────────────────────────

    def capabilities(self, provider: str) -> Any:
        return self._provider_capabilities(provider)

    def operation_risk(self, provider: str, operation: Operation) -> SideEffectRisk:
        cap = self._provider_capabilities(provider)
        return cap.resource.operation_risk[operation]

    def required_scope(self, provider: str, operation: Operation) -> str:
        cap = self._provider_capabilities(provider)
        return cap.resource.scope_for(operation)

    # ── Parts 5/15/13: scope-checked, retry-disciplined operations ────────────

    def execute_read(
        self,
        account: ConnectedAccount,
        operation: Operation,
        *,
        resource_id: str | None = None,
        limit: int = 10,
    ) -> list[ProviderResource] | ProviderResource:
        """
        LIST or GET: scope check → auth check → bounded auto-retry on
        transient categories only. Deterministic and idempotent.
        """
        if operation not in (Operation.LIST, Operation.GET):
            raise IntegrationManagerError(
                f"execute_read only accepts list/get, got {operation.value}"
            )
        provider = self._available_provider(account)
        self._ensure_usable_auth(account)
        self._ensure_scope(account, operation)

        def _once() -> list[ProviderResource] | ProviderResource:
            if operation == Operation.LIST:
                return provider.list_resources(account, max(1, min(int(limit), 50)))
            if not resource_id:
                raise IntegrationManagerError("get requires resource_id")
            return provider.get_resource(account, resource_id)

        last_error: ProviderError | None = None
        for attempt in range(_MAX_READ_RETRIES + 1):
            try:
                return _once()
            except ProviderError as e:
                last_error = e
                if e.category not in READ_RETRYABLE_CATEGORIES or attempt >= _MAX_READ_RETRIES:
                    raise
                log.info(
                    "integration_read_retried",
                    provider=account.provider,
                    category=e.category.value,
                    attempt=attempt + 1,
                )
        raise last_error or ProviderError(  # pragma: no cover
            ProviderErrorCategory.PROVIDER_ERROR, "read failed"
        )

    def execute_write(
        self,
        account: ConnectedAccount,
        operation: Operation,
        *,
        resource_id: str | None = None,
        fields: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[ProviderResource | None, str]:
        """
        CREATE / UPDATE / DELETE — the side-effect path. The CALLER has
        already obtained user confirmation (the existing action ledger);
        this method enforces scope + auth, executes ONCE (never blind
        retries), records the audit row, and classifies the outcome:

          success          → SUCCEEDED  (+ resource)
          ambiguous error  → UNKNOWN    (Part 13: stays UNKNOWN)
          clean error      → FAILED

        Returns (resource_or_None, state_string).
        """
        if operation not in (Operation.CREATE, Operation.UPDATE, Operation.DELETE):
            raise IntegrationManagerError(
                f"execute_write only accepts create/update/delete, got {operation.value}"
            )
        provider = self._available_provider(account)
        self._ensure_usable_auth(account)
        self._ensure_scope(account, operation)
        cap = provider.capabilities
        request_id = f"ext-{uuid.uuid4().hex}"
        idem = idempotency_key if cap.supports_idempotency_key else None
        risk = cap.resource.operation_risk[operation]

        try:
            if operation == Operation.CREATE:
                resource = provider.create_resource(
                    account, dict(fields or {}), idempotency_key=idem
                )
            elif operation == Operation.UPDATE:
                if not resource_id:
                    raise IntegrationManagerError("update requires resource_id")
                resource = provider.update_resource(
                    account, resource_id, dict(fields or {})
                )
            else:
                if not resource_id:
                    raise IntegrationManagerError("delete requires resource_id")
                ok = provider.delete_resource(account, resource_id)
                resource = None
                self._record_audit(
                    provider=cap.provider, account=account, operation=operation,
                    kind=cap.resource.kind, risk=risk, request_id=request_id,
                    idem=idem, state="SUCCEEDED",
                    resource_id=resource_id,
                    summary="delete confirmed by provider" if ok else "delete reported already absent",
                )
                return None, "SUCCEEDED"
        except ProviderError as e:
            state = "UNKNOWN" if e.category in AMBIGUOUS_CATEGORIES else "FAILED"
            self._record_audit(
                provider=cap.provider, account=account, operation=operation,
                kind=cap.resource.kind, risk=risk, request_id=request_id,
                idem=idem, state=state, resource_id=resource_id,
                summary=f"{e.category.value}: {e.message[:_MAX_AUDIT_SUMMARY_CHARS]}",
            )
            raise
        except IntegrationManagerError:
            raise
        except Exception as e:  # noqa: BLE001 - never leak raw exceptions upward
            from jarvis.integrations.errors import provider_error_from_exception

            pe = provider_error_from_exception(e, operation.value)
            self._record_audit(
                provider=cap.provider, account=account, operation=operation,
                kind=cap.resource.kind, risk=risk, request_id=request_id,
                idem=idem, state="FAILED", resource_id=resource_id,
                summary=f"{pe.category.value}: {pe.message[:_MAX_AUDIT_SUMMARY_CHARS]}",
            )
            raise pe from e

        self._record_audit(
            provider=cap.provider, account=account, operation=operation,
            kind=cap.resource.kind, risk=risk, request_id=request_id,
            idem=idem, state="SUCCEEDED", resource_id=resource.resource_id,
            summary=_resource_summary(resource),
        )
        return resource, "SUCCEEDED"

    def recent_audit(self, account_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self._store.list_integration_audit(account_id=account_id, limit=limit)

    # ── internals ─────────────────────────────────────────────────────────────

    def _provider_capabilities(self, provider: str) -> Any:
        if provider not in PROVIDER_SCOPE_MENUS:
            raise IntegrationManagerError(f"unknown provider: {provider!r}")
        impl = self._providers.get(provider)
        if impl is None:
            raise IntegrationManagerError(
                f"provider {provider!r} is not enabled in this runtime "
                "(enable integrations in configuration)"
            )
        return impl.capabilities

    def _available_provider(self, account: ConnectedAccount) -> IntegrationProvider:
        impl = self._providers.get(account.provider)
        if impl is None:
            raise IntegrationManagerError(
                f"provider {account.provider!r} is not enabled in this runtime"
            )
        if not impl.is_available():
            raise ProviderError(
                ProviderErrorCategory.PROVIDER_OUTAGE,
                f"{account.provider} provider is currently unavailable",
            )
        return impl

    def _ensure_usable_auth(self, account: ConnectedAccount) -> None:
        # Fail closed when the handle no longer exists (e.g. disconnected while
        # a turn held it): a stale in-memory projection never authorizes a call.
        if self._load_account(account.account_id) is None:
            raise ProviderError(
                ProviderErrorCategory.AUTHORIZATION_DENIED,
                f"the connected {account.provider} account is no longer "
                "connected; reconnect it before using this integration",
            )
        account = self.refresh_auth_state(account)
        if account.auth_state in (AuthState.EXPIRED, AuthState.REVOKED, AuthState.DISCONNECTED):
            raise ProviderError(
                ProviderErrorCategory.AUTH_EXPIRED
                if account.auth_state == AuthState.EXPIRED
                else ProviderErrorCategory.AUTHORIZATION_DENIED,
                f"the connected {account.provider} account is "
                f"{account.auth_state.value.lower()}; reconnect it before "
                "using this integration",
            )

    def _ensure_scope(self, account: ConnectedAccount, operation: Operation) -> None:
        needed = self.required_scope(account.provider, operation)
        from jarvis.integrations.scopes import require_scope

        try:
            require_scope(account.scopes, needed)
        except Exception as e:
            raise ProviderError(
                ProviderErrorCategory.AUTHORIZATION_DENIED, str(e)
            ) from e

    def _record_audit(
        self,
        *,
        provider: str,
        account: ConnectedAccount,
        operation: Operation,
        kind: str,
        risk: SideEffectRisk,
        request_id: str,
        idem: str | None,
        state: str,
        resource_id: str | None,
        summary: str,
    ) -> None:
        from jarvis.integrations.sanitize import sanitize

        try:
            self._store.record_integration_audit(
                provider=provider,
                account_id=account.account_id,
                operation=operation.value,
                resource_kind=kind,
                risk_category=risk.value,
                request_id=request_id,
                idempotency_key=idem,
                state=state,
                resource_id=resource_id,
                verification="NOT_APPLICABLE",
                result_summary=sanitize(str(summary or ""))[:_MAX_AUDIT_SUMMARY_CHARS],
            )
        except Exception as e:  # noqa: BLE001 - audit failure never breaks the op
            log.warning("integration_audit_record_failed", error=str(e))

    def _account_from_row(self, row: dict[str, Any]) -> ConnectedAccount | None:
        try:
            import json as _json

            scopes = frozenset(_json.loads(row.get("scopes_json") or "[]"))
            return ConnectedAccount(
                account_id=str(row["account_id"]),
                provider=str(row["provider"]),
                display_label=str(row["display_label"]),
                scopes=scopes,
                authenticated=str(row.get("auth_state")) == AuthState.AUTHENTICATED.value,
                auth_state=AuthState(str(row.get("auth_state") or AuthState.ERROR.value)),
                granted_at=str(row.get("granted_at") or ""),
                last_verified_at=row.get("last_verified_at"),
                provider_account_ref=str(row.get("provider_account_ref") or ""),
                _credential_secret=deobfuscate(str(row.get("credential_obfuscated") or "")),
                authorization_status=row.get("authorization_status") or None,
                token_expires_at=row.get("token_expires_at"),
                token_updated_at=row.get("token_updated_at"),
                _access_token=deobfuscate(str(row.get("token_access_obfuscated") or "")),
                _refresh_token=deobfuscate(str(row.get("token_refresh_obfuscated") or "")),
            )
        except Exception as e:  # noqa: BLE001 - corrupt row degrades honestly
            log.warning("integration_account_row_corrupt", error=str(e))
            return None

    def _load_account(self, account_id: str) -> ConnectedAccount | None:
        row = self._store.get_integration_account(account_id)
        return self._account_from_row(row) if row else None


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _resource_summary(resource: ProviderResource) -> str:
    """Bounded, sanitized one-line summary for the audit row (no payloads)."""
    from jarvis.integrations.sanitize import sanitize

    kind = sanitize(resource.kind)[:40]
    title = sanitize(resource.fields.get("title") or resource.fields.get("name") or "")[:80]
    return f"{kind} {resource.resource_id}: {title}".strip()
