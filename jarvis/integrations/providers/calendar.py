"""
jarvis/integrations/providers/calendar.py
─────────────────────────────────────────
v0.29 Part 7 — the CALENDAR reference integration (local dev provider).

This is the reference implementation of the IntegrationProvider seam. The
bundled backend is a DETERMINISTIC LOCAL provider (in-memory calendar,
explicitly labeled development-only — ProviderCapabilities.production_like
=False). Its interface is deliberately identical to what a production
Google Calendar adapter would implement:

  - verify_authentication: shaped like a real token introspection
    (deterministic prefix rules stand in for token expiry/revocation);
  - create_resource: shaped like a real event insert, honoring the
    idempotency key (Part 7 duplicate protection);
  - errors: normalized ProviderError categories from the start.

A production adapter (Google Calendar via OAuth refresh-token flow) plugs
in by implementing the same five operations + verification — no runtime,
tool, or permission change. The safety model for that adapter is specified
in docs/JARVIS_V029_SECURITY_MODEL.md §2 and is intentionally NOT faked
here (Part 19: document rather than fake automation).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone as _tz

from jarvis.config import settings
from jarvis.integrations.base import (
    AuthState,
    ConnectedAccount,
    IntegrationProvider,
    Operation,
    ProviderCapabilities,
    ProviderResource,
    ResourceSpec,
    SideEffectRisk,
)
from jarvis.integrations.errors import ProviderError, ProviderErrorCategory
from jarvis.integrations.oauth import (
    LocalOAuthTokenClient,
    OAuthClientConfig,
    OAuthIntegrationProvider,
    local_authorization_server,
    local_introspection_auth_state,
)
from jarvis.integrations.scopes import (
    CALENDAR_DELETE,
    CALENDAR_READ,
    CALENDAR_WRITE,
)

# Deterministic token-prefix rules that stand in for real introspection
# (documented; the local provider is the only consumer).
PREFIX_AUTHENTICATED = "loc-dev_"
PREFIX_EXPIRED = "expired_"
PREFIX_REVOKED = "revoked_"


@dataclass
class _StoredEvent:
    event_id: str
    account_id: str
    title: str
    start_utc: str          # ISO instant
    end_utc: str
    tz: str
    notes: str
    attendees: tuple[str, ...] = ()
    cancelled: bool = False
    idempotency_key: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(tz=_tz.utc).isoformat())


class LocalCalendarBackend:
    """
    Deterministic in-memory calendar. Per-account isolation; thread-safe;
    bounded (creates capped) so tests/demos can grow without leaks.
    """

    MAX_EVENTS = 500

    def __init__(self) -> None:
        self._events: dict[str, _StoredEvent] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def _account_events(self, account_id: str) -> list[_StoredEvent]:
        return [e for e in self._events.values() if e.account_id == account_id]

    def create(
        self,
        account_id: str,
        fields: dict[str, str],
        idempotency_key: str | None,
    ) -> _StoredEvent:
        with self._lock:
            if idempotency_key:
                for e in self._account_events(account_id):
                    if e.idempotency_key == idempotency_key and not e.cancelled:
                        return e  # duplicate-create protection: return the ORIGINAL
            if len(self._events) >= self.MAX_EVENTS:
                raise ProviderError(
                    ProviderErrorCategory.PROVIDER_OUTAGE,
                    "calendar store is full (bounded development provider)",
                )
            event = _StoredEvent(
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
                account_id=account_id,
                title=fields.get("title", ""),
                start_utc=fields.get("start_utc", ""),
                end_utc=fields.get("end_utc", ""),
                tz=fields.get("tz", "UTC"),
                notes=fields.get("notes", ""),
                attendees=tuple(
                    a for a in fields.get("attendees", "").split(",") if a
                ),
                idempotency_key=idempotency_key,
            )
            self._events[event.event_id] = event
            return event

    def list(self, account_id: str, limit: int) -> list[_StoredEvent]:
        with self._lock:
            events = [e for e in self._account_events(account_id) if not e.cancelled]
            events.sort(key=lambda e: (e.start_utc, e.created_at))
            return events[: max(1, min(int(limit), 50))]

    def get(self, account_id: str, event_id: str) -> _StoredEvent:
        with self._lock:
            event = self._events.get(event_id)
            if event is None or event.account_id != account_id or event.cancelled:
                raise ProviderError(
                    ProviderErrorCategory.NOT_FOUND,
                    f"event {event_id[:24]} not found",
                )
            return event

    def update(self, account_id: str, event_id: str, fields: dict[str, str]) -> _StoredEvent:
        with self._lock:
            event = self.get(account_id, event_id)
            if "title" in fields:
                event.title = fields["title"]
            if "start_utc" in fields:
                event.start_utc = fields["start_utc"]
            if "end_utc" in fields:
                event.end_utc = fields["end_utc"]
            if "tz" in fields:
                event.tz = fields["tz"]
            if "notes" in fields:
                event.notes = fields["notes"]
            return event

    def delete(self, account_id: str, event_id: str) -> bool:
        with self._lock:
            event = self.get(account_id, event_id)  # NOT_FOUND on missing
            event.cancelled = True
            return True


# Singleton backend (per process; tests reset it).
_backend = LocalCalendarBackend()


def calendar_backend() -> LocalCalendarBackend:
    return _backend


def reset_calendar_backend() -> None:
    _backend.reset()


class LocalCalendarProvider(OAuthIntegrationProvider, IntegrationProvider):
    """
    The reference adapter over LocalCalendarBackend (development-only).

    v0.30: OAuth-capable — the SAME adapter seam now also drives the real
    authorization flow against the simulated local authorization server
    (which plays the provider's role) through the provider-neutral mixin.
    A production Google Calendar adapter implements the same hooks against
    real endpoints and keeps everything else identical: no runtime, tool,
    permission, ledger, or grounding change.
    """

    def __init__(self) -> None:
        super().__init__(
            ProviderCapabilities(
                provider="calendar",
                display_name="Calendar (local development provider)",
                resource=ResourceSpec(
                    kind="event",
                    operations=frozenset(
                        {Operation.LIST, Operation.GET, Operation.CREATE,
                         Operation.UPDATE, Operation.DELETE}
                    ),
                    required_scopes={
                        Operation.LIST: CALENDAR_READ,
                        Operation.GET: CALENDAR_READ,
                        Operation.CREATE: CALENDAR_WRITE,
                        Operation.UPDATE: CALENDAR_WRITE,
                        Operation.DELETE: CALENDAR_DELETE,
                    },
                    operation_risk={
                        Operation.LIST: SideEffectRisk.READ_ONLY,
                        Operation.GET: SideEffectRisk.READ_ONLY,
                        Operation.CREATE: SideEffectRisk.MEDIUM_SIDE_EFFECT,
                        Operation.UPDATE: SideEffectRisk.MEDIUM_SIDE_EFFECT,
                        Operation.DELETE: SideEffectRisk.HIGH_SIDE_EFFECT,
                    },
                ),
                grantable_scopes=frozenset(
                    {CALENDAR_READ, CALENDAR_WRITE, CALENDAR_DELETE}
                ),
                production_like=False,
                supports_idempotency_key=True,
                description=(
                    "Deterministic local calendar used for development, "
                    "tests, and the v0.29 live-provider validation. NOT a "
                    "production calendar service."
                ),
            )
        )

    # ── v0.30: OAuth capability (provider-neutral mixin hooks) ───────────────

    def oauth_config(self) -> OAuthClientConfig:
        """Provider-owned OAuth details (Part 3) — never visible to the model."""
        provider = self.capabilities.provider
        client_id = str(getattr(settings, "OAUTH_CALENDAR_CLIENT_ID", "local-dev-calendar") or "")
        client_secret = str(
            getattr(settings, "OAUTH_CALENDAR_CLIENT_SECRET", "local-dev-calendar-secret") or ""
        )
        local_authorization_server().register_client(client_id, client_secret)
        return OAuthClientConfig(
            provider=provider,
            client_id=client_id,
            client_secret=client_secret,
            authorization_endpoint=f"local-oauth://{provider}/authorize",
            token_endpoint=f"local-oauth://{provider}/token",
            revocation_endpoint=f"local-oauth://{provider}/revoke",
            # Left empty: the mixin derives the redirect (settings base + the
            # fixed callback path + our own session_id parameter).
            redirect_uri="",
            scope_map={s: s for s in (CALENDAR_READ, CALENDAR_WRITE, CALENDAR_DELETE)},
        )

    def token_client(self) -> LocalOAuthTokenClient:
        return LocalOAuthTokenClient(local_authorization_server())

    # ── auth (Part 3; deterministic stand-in for token introspection) ─────────

    def verify_authentication(self, account: ConnectedAccount) -> AuthState:
        # v0.30: OAuth accounts are verified by REAL introspection against
        # the (simulated) authorization server; legacy local-dev accounts
        # keep the deterministic prefix rules unchanged.
        if account.is_oauth_account:
            return local_introspection_auth_state(account)
        secret = account._credential_secret
        if not secret:
            return AuthState.REVOKED
        if secret.startswith(PREFIX_REVOKED):
            return AuthState.REVOKED
        if secret.startswith(PREFIX_EXPIRED):
            return AuthState.EXPIRED
        if secret.startswith(PREFIX_AUTHENTICATED):
            return AuthState.AUTHENTICATED
        return AuthState.ERROR

    def is_available(self) -> bool:
        return True  # local, in-process

    # ── operations (Part 6) ───────────────────────────────────────────────────

    def list_resources(self, account: ConnectedAccount, limit: int) -> list[ProviderResource]:
        events = _backend.list(account.account_id, limit)
        return [_event_to_resource(e) for e in events]

    def get_resource(self, account: ConnectedAccount, resource_id: str) -> ProviderResource:
        return _event_to_resource(_backend.get(account.account_id, resource_id))

    def create_resource(
        self,
        account: ConnectedAccount,
        fields: dict[str, str],
        *,
        idempotency_key: str | None = None,
    ) -> ProviderResource:
        if not fields.get("start_utc"):
            raise ProviderError(
                ProviderErrorCategory.VALIDATION_ERROR,
                "start_utc is required by the provider",
            )
        event = _backend.create(account.account_id, fields, idempotency_key)
        return _event_to_resource(event)

    def update_resource(
        self, account: ConnectedAccount, resource_id: str, fields: dict[str, str]
    ) -> ProviderResource:
        return _event_to_resource(_backend.update(account.account_id, resource_id, fields))

    def delete_resource(self, account: ConnectedAccount, resource_id: str) -> bool:
        return _backend.delete(account.account_id, resource_id)


def _event_to_resource(e: _StoredEvent) -> ProviderResource:
    return ProviderResource(
        resource_id=e.event_id,
        kind="event",
        fields={
            "id": e.event_id,
            "title": e.title,
            "start_utc": e.start_utc,
            "end_utc": e.end_utc,
            "tz": e.tz,
            "attendees": ", ".join(e.attendees),
        },
        raw_external={"title": e.title, "notes": e.notes},
    )
