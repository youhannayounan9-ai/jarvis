"""
jarvis/browser/__init__.py
──────────────────────────
v0.28 — Safe Browser & Computer Interaction.

This package is the SAFETY CORE for browser automation:

  url_policy.py    deny-by-default URL validation (Part 6)
  risk.py          action risk classification (Part 9) — maps into the
                   EXISTING permission tiers, never a second system
  observations.py  observation identity + freshness (Part 8)
  limits.py        deterministic pacing / rate limits (Part 12)
  emergency.py     external emergency stop (Part 11)
  redaction.py     credential/secret redaction (Part 19)
  downloads.py     bounded, never-executed download handling (Part 18)
  injection.py     UNTRUSTED PAGE CONTENT framing (Part 14)
  verification.py  OBSERVE-ACT-VERIFY outcome statuses (Parts 16/17)
  controller.py    the only stateful browser-session controller (Part 7)
  driver.py        the BrowserDriver boundary + Playwright/Simulated impls

Design invariants (contract):
  - The model never receives raw coordinates, cookies, or credentials.
  - Every action references a fresh observation ID; stale IDs are rejected.
  - Risk escalation is computed by the RUNTIME from validated arguments.
  - Authorization flows ONLY through the existing PermissionGuard /
    confirmation parking / dispatch ledger / result-cache layering — this
    package adds policy INPUTS, never a second confirmation implementation.
  - Every driver capability fails closed on any uncertainty.
"""

from jarvis.browser.url_policy import (
    URLPolicy,
    URLPolicyError,
    validate_url,
)
from jarvis.browser.risk import (
    ActionRisk,
    classify_click_risk,
    classify_fill_risk,
    risk_to_permission_tier,
)
from jarvis.browser.observations import ObservationStore, Observation
from jarvis.browser.emergency import (
    EmergencyStop,
    EmergencyStopTriggered,
    get_emergency_stop,
)
from jarvis.browser.verification import (
    VERIFICATION_STATUSES,
    VerificationStatus,
)

__all__ = [
    "URLPolicy",
    "URLPolicyError",
    "validate_url",
    "ActionRisk",
    "classify_click_risk",
    "classify_fill_risk",
    "risk_to_permission_tier",
    "ObservationStore",
    "Observation",
    "EmergencyStop",
    "EmergencyStopTriggered",
    "get_emergency_stop",
    "VERIFICATION_STATUSES",
    "VerificationStatus",
]
