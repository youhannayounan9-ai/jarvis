# jarvis/integrations/__init__.py
"""
jarvis/integrations — v0.29 Personal Integrations & Workflow Automation.

The secure, extensible framework for connecting external personal services:

  scopes.py       exact scope vocabulary + enforcement (Part 5)
  errors.py       normalized provider errors (Part 15)
  sanitize.py     credential redaction + untrusted-content framing (Parts 4/18)
  base.py         the abstraction: capabilities, identity, provider seam (Parts 2/3/6/12)
  credentials.py  honest local-dev credential boundary (Part 4)
  validation.py   no-silent-time-guessing event validation (Part 8)
  manager.py      connect/verify/authorize/execute + audit (Parts 3/13/19)
  providers/      calendar (reference) + tasks (narrow) local providers (Parts 7/10)

Boundary invariants (do not regress):
  - The runtime talks to normalized capabilities only — never provider SDKs.
  - Identity is explicit and persisted; never inferred from model text.
  - Credentials never reach prompts, tool output, logs, or descriptions.
  - Every external side effect flows through the EXISTING permission
    tiers, confirmation parking, and action ledger.
"""

from jarvis.integrations.base import (
    AuthState,
    ConnectedAccount,
    IntegrationProvider,
    Operation,
    ProviderCapabilities,
    ProviderResource,
    ResourceSpec,
    SideEffectRisk,
    static_tier_for,
)
from jarvis.integrations.errors import (
    ProviderError,
    ProviderErrorCategory,
)
from jarvis.integrations.manager import (
    IntegrationManager,
    IntegrationManagerError,
)
from jarvis.integrations.scopes import (
    ALL_SCOPES,
    PROVIDER_SCOPE_MENUS,
    InsufficientScopeError,
    UnknownScopeError,
    require_scope,
    validate_scope_set,
)

__all__ = [
    "AuthState",
    "ConnectedAccount",
    "IntegrationProvider",
    "Operation",
    "ProviderCapabilities",
    "ProviderResource",
    "ResourceSpec",
    "SideEffectRisk",
    "static_tier_for",
    "ProviderError",
    "ProviderErrorCategory",
    "IntegrationManager",
    "IntegrationManagerError",
    "ALL_SCOPES",
    "PROVIDER_SCOPE_MENUS",
    "InsufficientScopeError",
    "UnknownScopeError",
    "require_scope",
    "validate_scope_set",
]
