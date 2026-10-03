"""
jarvis/integrations/base.py
───────────────────────────
v0.29 Parts 2/6/12 — the integration abstraction and provider boundary.

The runtime talks ONLY to the normalized concepts here — never to a
provider SDK, never to raw HTTP. A provider adapter:

  - declares its identity, capability menu, required scope vocabulary,
    and per-operation risk classification (Part 12 categories mapped onto
    the EXISTING permission tiers — no second risk system);
  - reports its authentication state honestly (Part 3 identity comes from
    the persisted account record + provider verification, NEVER from
    model text);
  - implements EXACTLY the narrow operations it supports (Part 6):
    list_resources / get_resource / create_resource / update_resource /
    delete_resource. There is NO generic request/HTTP operation and the
    model can never supply endpoints, headers, auth values, or methods.

Risk mapping (Part 12):
    READ_ONLY           → static tier SAFE   (auto-allowed, cache-eligible)
    LOW_SIDE_EFFECT     → static tier NETWORK, escalates to SYSTEM on
                          validated write args (existing confirmation flow)
    MEDIUM_SIDE_EFFECT  → same as LOW (create/update-shaped writes)
    HIGH_SIDE_EFFECT    → static tier SYSTEM (always confirmed)
    IRREVERSIBLE        → static tier DESTRUCTIVE (always confirmed;
                          delete-shaped operations at minimum)

Escalation is args-based and conservative (any error → static ceiling),
exactly like jarvis/browser/risk.py in v0.28.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jarvis.integrations.errors import ProviderError, ProviderErrorCategory
from jarvis.integrations.scopes import validate_scope_set
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── Part 12: external side-effect risk categories ─────────────────────────────


class SideEffectRisk(str, Enum):
    READ_ONLY = "READ_ONLY"
    LOW_SIDE_EFFECT = "LOW_SIDE_EFFECT"
    MEDIUM_SIDE_EFFECT = "MEDIUM_SIDE_EFFECT"
    HIGH_SIDE_EFFECT = "HIGH_SIDE_EFFECT"
    IRREVERSIBLE = "IRREVERSIBLE"


# Mapped onto the EXISTING PermissionGuard tiers. Static tier is the
# ceiling for read-shaped dispatch; write tools escalate via risk_for_args.
_RISK_TO_STATIC_TIER: dict[SideEffectRisk, str] = {
    SideEffectRisk.READ_ONLY: "SAFE",
    SideEffectRisk.LOW_SIDE_EFFECT: "NETWORK",
    SideEffectRisk.MEDIUM_SIDE_EFFECT: "NETWORK",
    SideEffectRisk.HIGH_SIDE_EFFECT: "SYSTEM",
    SideEffectRisk.IRREVERSIBLE: "DESTRUCTIVE",
}


def static_tier_for(risk: SideEffectRisk) -> str:
    """The existing permission tier a side-effect risk maps onto."""
    return _RISK_TO_STATIC_TIER[risk]


# ── Part 2: normalized capabilities & operations ──────────────────────────────


class AuthState(str, Enum):
    DISCONNECTED = "DISCONNECTED"      # no account connected
    AUTHENTICATED = "AUTHENTICATED"    # connected + verified working
    EXPIRED = "EXPIRED"                # credentials present but expired
    REVOKED = "REVOKED"                # authorization revoked / denied by provider
    ERROR = "ERROR"                    # account exists; state unknowable right now


class Operation(str, Enum):
    LIST = "list"
    GET = "get"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True)
class ResourceSpec:
    """One resource type a provider exposes (e.g. calendar events)."""

    kind: str                                   # "event", "task"
    operations: frozenset[Operation]
    # Exact scope each operation requires (deny-by-default).
    required_scopes: dict[Operation, str]
    # Side-effect risk per operation (Part 12).
    operation_risk: dict[Operation, SideEffectRisk]

    def tier_for(self, operation: Operation) -> str:
        return static_tier_for(self.operation_risk[operation])

    def scope_for(self, operation: Operation) -> str:
        return self.required_scopes[operation]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Everything the runtime knows about a provider WITHOUT touching it."""

    provider: str
    display_name: str
    resource: ResourceSpec
    # Scopes the provider is CAPABLE of granting (its menu; validated at
    # connect time — the model cannot expand it).
    grantable_scopes: frozenset[str]
    production_like: bool = False   # False = development-only provider (docs must say so)
    supports_idempotency_key: bool = False
    description: str = ""


# ── Part 3: connected-account identity (persisted, explicit) ──────────────────


@dataclass(frozen=True)
class ConnectedAccount:
    """
    The explicit identity of one connected account (persisted in SQLite by
    IntegrationManager; this is the in-memory projection).

    Identity is NEVER inferred from model text: every external action
    carries the account_id it was authorized against, and the manager
    re-checks authentication state from THIS record plus the provider
    verification — not from anything the model said.
    """

    account_id: str
    provider: str
    display_label: str            # operator-chosen label; NOT a credential
    scopes: frozenset[str]
    authenticated: bool
    auth_state: AuthState
    granted_at: str               # ISO timestamp of the original grant
    last_verified_at: str | None  # last successful provider verification
    # Provider-side account metadata WITHOUT secrets (e.g. a calendar id
    # or "primary"). Sanitized before display; never a token.
    provider_account_ref: str = ""
    # Encrypted-at-rest credential material (v0.29 local-dev boundary:
    # see credentials.py). NEVER rendered, logged, or returned.
    _credential_secret: str = field(default="", repr=False, compare=False)
    # ── v0.30: OAuth lifecycle (real authorization + token records) ────────
    # authorization_status is the NORMALIZED v0.30 lifecycle value
    # (jarvis/integrations/oauth.py::AuthorizationStatus); None marks a
    # legacy v0.29 account (raw credential + prefix rules). Token material
    # is repr/compare-excluded exactly like the credential — it is
    # structurally absent from every projection below.
    authorization_status: str | None = None
    token_expires_at: str | None = None
    token_updated_at: str | None = None
    _access_token: str = field(default="", repr=False, compare=False)
    _refresh_token: str = field(default="", repr=False, compare=False)

    @property
    def is_oauth_account(self) -> bool:
        return bool(self.authorization_status)

    def public_metadata(self) -> dict[str, Any]:
        """Safe projection for API/dashboard/CLI — secrets structurally absent."""
        return {
            "account_id": self.account_id,
            "provider": self.provider,
            "display_label": self.display_label,
            "scopes": sorted(self.scopes),
            "authenticated": self.authenticated,
            "auth_state": self.auth_state.value,
            "granted_at": self.granted_at,
            "last_verified_at": self.last_verified_at,
            "provider_account_ref": self.provider_account_ref,
            "authorization_status": self.authorization_status,
            # Named authorization_* deliberately: the v0.29 metadata tripwire
            # asserts NO public key even mentions credentials/secrets/tokens
            # (structural, not semantic) — these are lifecycle timestamps.
            "authorization_expires_at": self.token_expires_at,
            "authorization_updated_at": self.token_updated_at,
        }


# ── Part 6: the provider adapter interface ────────────────────────────────────


@dataclass(frozen=True)
class ProviderResource:
    """One normalized resource returned by a provider (pre-sanitization)."""

    resource_id: str
    kind: str
    # Ordered display fields — provider adapters emit exact label: value
    # lines (StructuredFieldGroundingPolicy-compatible), never raw JSON.
    fields: dict[str, str]
    raw_external: dict[str, str] = field(default_factory=dict)  # free text (titles/notes)


class IntegrationProvider(ABC):
    """
    The ONLY seam between JARVIS and an external personal service.

    Implementations are synchronous and bounded; they raise ProviderError
    (normalized categories) instead of leaking native exceptions. The
    manager checks scopes BEFORE any call reaches the adapter — adapters
    may defensively re-check but never grant.

    ``supports_oauth`` (v0.30): True when the adapter also mixes in
    OAuthIntegrationProvider (real authorization + token lifecycle).
    """

    supports_oauth: bool = False

    def __init__(self, capabilities: ProviderCapabilities) -> None:
        self.capabilities = capabilities

    # ── identity / availability (Part 2) ──────────────────────────────────────

    @abstractmethod
    def verify_authentication(self, account: ConnectedAccount) -> AuthState:
        """
        Determine the REAL auth state for this account by consulting the
        provider (never by trusting cached text). Cheap; no side effects.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the provider backend is reachable/configured right now."""

    # ── operations (Part 6 — exactly these, nothing generic) ─────────────────

    @abstractmethod
    def list_resources(
        self, account: ConnectedAccount, limit: int
    ) -> list[ProviderResource]:
        """Bounded newest/soonest-first listing."""

    @abstractmethod
    def get_resource(self, account: ConnectedAccount, resource_id: str) -> ProviderResource:
        """Fetch one resource; ProviderError(NOT_FOUND) when absent."""

    @abstractmethod
    def create_resource(
        self,
        account: ConnectedAccount,
        fields: dict[str, str],
        *,
        idempotency_key: str | None = None,
    ) -> ProviderResource:
        """Create one resource. Idempotency key honored when supported."""

    @abstractmethod
    def update_resource(
        self, account: ConnectedAccount, resource_id: str, fields: dict[str, str]
    ) -> ProviderResource:
        """Update given fields of one resource; returns the updated view."""

    def delete_resource(self, account: ConnectedAccount, resource_id: str) -> bool:
        """
        Delete/cancel one resource. DEFAULT: unsupported — providers omit
        deletion unless the capability is explicitly designed (Part 10:
        tasks delete disabled by default).
        """
        raise ProviderError(
            ProviderErrorCategory.AUTHORIZATION_DENIED,
            f"{self.capabilities.provider} does not support delete_resource",
        )

    # ── helper ────────────────────────────────────────────────────────────────

    def spec(self) -> ResourceSpec:
        return self.capabilities.resource
