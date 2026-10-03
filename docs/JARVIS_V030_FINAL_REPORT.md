# JARVIS v0.30.0 Final Report — Real OAuth & Account Connectivity

**Date:** October 2, 2026 · **Baseline:** v0.29.0 · **Version:** 0.30.0 ·
**Source control:** none performed (no commit / push / PR).

This report covers the v0.30 phase: making the v0.29 personal-integration
framework **really connectable** via a provider-neutral OAuth 2.0
authorization-code + PKCE flow with a full token lifecycle — without
redesigning the framework and without a second security model.

---

## 1. Implementation summary

v0.29 built the integration *boundary* (opaque account handles, exact scope
enforcement, exactly-once writes, confirmation parking, read-back verification,
per-account cache, audit, grounding) but its providers were local-dev only and
`connect()` stored a credential as given — there was no real authorization, no
token exchange, and no token lifetime.

v0.30 adds the missing authorization layer:

- A **provider-neutral** OAuth seam (`OAuthIntegrationProvider`) so provider
  endpoints/scope strings/identity discovery live in the adapter and the
  orchestrator never sees them.
- A **durable one-time `state`** stored by SHA-256 hash, bound to
  `(provider, session, display_label, scopes, redirect_uri)`, expiring, consumed
  atomically once.
- **PKCE (S256)** on every flow.
- A real **authorization-code exchange** and **refresh lifecycle** with
  provider-verified account identity and best-effort revocation on disconnect.
- A new **`provider_schedule` grounding policy** for event/task times,
  timezones, and statuses.
- Bounded **audit retention**.
- Full surfaces (API / CLI / dashboard) plus a deterministic manual live eval.

All existing mechanisms are unchanged in behavior: permission tiers,
confirmation parking, the action ledger, idempotency/reissue, read-back
verification, grounding guard, result cache, audit logging, API auth, and the
dashboard controls.

## 2. Architecture changes

- **`jarvis/integrations/oauth.py`** (new) — the whole lifecycle. See §4.
- **`jarvis/integrations/base.py`** — `ConnectedAccount` gains
  `authorization_status`, `token_expires_at`, `token_updated_at`, and
  repr/compare-excluded `_access_token`/`_refresh_token` (+ `is_oauth_account`);
  `IntegrationProvider` gains `supports_oauth`. `public_metadata()` emits only
  safe `authorization_*` keys (never `token_*`).
- **`jarvis/integrations/manager.py`** — `begin_authorization`,
  `handle_callback`, `authorization_flow_outcome`,
  `_refresh_oauth_tokens_if_needed`; `disconnect()` revokes best-effort and
  purges cache; `refresh_auth_state()` refreshes first for OAuth accounts and
  fails closed when the row is gone; `_ensure_usable_auth` refuses stale handles.
- **`jarvis/integrations/providers/{calendar,tasks}.py`** — now
  `OAuthIntegrationProvider` mixins; OAuth logic is NOT duplicated per provider.
- **`jarvis/memory/session_store.py`** — `oauth_states` table + token columns +
  flow/token/retention/cache-invalidation methods.
- **`jarvis/core/grounding.py`** — `ProviderScheduleGroundingPolicy` registered
  in `POLICIES`.
- **`jarvis/tools/integration_tools.py`** — `_derived_schedule_lines` labeled
  output (local time from UTC + IANA zone).
- **`jarvis/config.py`**, **`jarvis/api/{app,schemas,client}.py`**,
  **`jarvis/main.py`**, **`jarvis/maintenance.py`**, **`ui/dashboard.py`** — the
  settings and surfaces.

## 3. Capability delta from v0.29

| Capability | v0.29 | v0.30 |
|---|---|---|
| Authorization | credential pasted in | real authorization-code + PKCE flow |
| CSRF/replay protection | n/a | one-time hashed `state`, session/provider/redirect-bound, expiring |
| Redirect | n/a | config-constructed URI, fixed-page callback, no open redirect, no token |
| Token lifetime | none | access + refresh + expiry, automatic margin-based refresh, atomic rotation |
| Auth failure states | static prefix rules | honest `AUTHENTICATION_REQUIRED` / `REVOKED`; revoked is never retried |
| Account identity | not verified | provider-discovered via introspection; label collisions impossible; in-place re-auth |
| Disconnect | local row delete | best-effort provider revocation → row delete → cache purge → fail-closed |
| Audit retention | unbounded | `INTEGRATION_AUDIT_RETENTION_DAYS` + bounded cleanup (UNKNOWN/RUNNING protected) |
| Schedule grounding | `count:` only | `provider_schedule` verifies time/weekday/timezone/status |

## 4. OAuth design

One normalized `AuthorizationStatus` state machine
(DISCONNECTED → AUTHORIZING → AUTHORIZED → TOKEN_EXPIRING → REFRESHING →
AUTHORIZED / AUTHENTICATION_REQUIRED / REVOKED / ERROR) mapped onto the existing
operational `AuthState` via `operational_auth_state()`, so the manager's
enforcement order is untouched.

- `OAuthFlowManager` — `begin_flow` / `consume_flow` / `record_outcome` /
  `purge_expired`.
- `OAuthTokenClient` ABC → `HttpOAuthTokenClient` (stdlib `urllib`, RFC 6749
  form posts, explicit timeout, **no blind retries**) and `LocalOAuthTokenClient`
  (deterministic `LocalAuthorizationServer`, dev/tests/live-eval only).
- `OAuthIntegrationProvider` mixin — `oauth_config`, `token_client`,
  `begin_authorization`, `handle_callback`, `discover_account_ref`,
  `revoke_remote`, `oauth_introspect`.
- PKCE S256 on every flow; verifier held server-side only.
- A bounded `OAuthFlowError` taxonomy
  (INVALID_STATE … REVOKED_GRANT, UNAVAILABLE) with `to_public_message()` —
  never raw provider payloads.

## 5. Credential model

Tokens are distinct access/refresh material with an expiry, stored in the SAME
local XOR+base64-obfuscated store as v0.29 credentials, keyed by the per-install
`.jarvis_integration_key`. **This is a privacy guard, not encryption**, and not
production-grade secret management — stated honestly in the module docstrings,
the user manual, the developer manual, and the security model. Production must
replace the store behind the same `CredentialStore` interface (OS keychain /
encrypted store).

## 6. Token lifecycle

- Refresh fires opportunistically **before** an operation inside
  `OAUTH_REFRESH_MARGIN_SECONDS` (default 300) of expiry.
- Success rotates ATOMICALLY; a provider that omits a new refresh token
  preserves the stored one.
- `REVOKED_GRANT` → `REVOKED` and is **never retried**; other refresh failures
  → `AUTHENTICATION_REQUIRED`, also not retried.
- The exchange performs exactly one attempt (one-time codes make retries
  meaningless).

## 7. Account identity

Identity is discovered **from the provider** after the exchange (introspection),
never inferred from chat text. Two accounts can share a display label without
collision; re-authorizing the same identity rotates the account in place; an
existing legacy raw-credential account is never overwritten by an OAuth flow.

## 8. Scope enforcement

Scopes remain connect-time constants bound to the flow; `require_scope` denies
by default; the model has no tool to widen them. A read-only grant refuses
writes/deletes; a provider that downgrades the grant at exchange is refused
(SCOPE_DOWNGRADED). No fallback to broader credentials or browser automation.

## 9. Calendar / task behavior

Unchanged externally: reads are SAFE and per-account cached (default 60 s),
writes run exactly once through the existing confirmation parking, and every
create/update is read-back verified. Reads now additionally emit labeled
`date:`/`time:`/`timezone:`/`duration_minutes:` lines (local time derived from
the UTC instant + IANA zone) for the new grounding policy.

## 10. Grounding hardening

`ProviderScheduleGroundingPolicy` (`provider_schedule`, registered last in
`POLICIES`) deterministically checks an answer's clock time, weekday, timezone
offset, and task status against those labeled lines. It detects a wrong time,
wrong weekday, wrong timezone offset (UTC+2 ≡ Africa/Cairo), and task-status
contradictions, and it **exempts read-pre-state observations**
(`calendar_list_events`, `task_list`) so state-change answers are not
false-rejected. Conservative for `status: open` vs claimed completion (fires
only when the resource id appears). High-confidence-only: a missed catch is
accepted, a false rejection is not.

## 11. Audit retention

`INTEGRATION_AUDIT_RETENTION_DAYS` (default 90) plus
`maintenance integrations-audit` — report-only by default, `--yes` performs a
bounded batch delete. Rows whose state is UNKNOWN or RUNNING are **always**
protected regardless of age. Audit rows still carry only
category/operation/op_id/idempotency-key/redacted-summary — never arguments or
payloads.

## 12. Security audit

Full threat model in `docs/JARVIS_V030_SECURITY_MODEL.md` (T1–T20). Highlights:
CSRF/forged callback (T1), state replay (T2), expired state (T3), open redirect
(T4), provider/session mismatch (T5), code interception (T6), token leakage
(T7), expired-token reuse (T8), rotation-replay revocation (T9), revoked-grant
hammering (T10), scope escalation (T11), identity spoofing (T12), stale handle
(T13), post-disconnect access (T14), prompt injection (T15), cross-account cache
leakage (T16), wrong grounded answer (T17), ambiguous write (T18), audit
growth/secret in trail (T19), exchange-timeout blind retry (T20). All failures
are fail-closed; token redaction is test-pinned.

## 13. Deterministic test results

- **`tests/test_oauth.py` — 83 tests**, categories A–V/AA/AC, all passing.
- Combined `test_oauth.py` + `test_integrations.py` = **186 passed**.
- Full suite: **1218 passed / 29 skipped** (baseline v0.29 was 1135 passed / 29
  skipped). The only change needed to an existing test was extending the
  `POLICIES` shape pin in `tests/test_grounding.py` to include the new
  `provider_schedule` policy.
- One pre-existing test (`test_session_concurrency.py::TestMultiProcessLease::
  test_takeover_race_across_processes_consistent`) is load-sensitive across
  processes (`time.sleep`-coordinated) and flaked once under full-suite CPU
  load; it passes in isolation and is untouched by v0.30. Not a regression.
- Note: `TestRegistryExtensibility::test_policies_registry_shape` initially
  failed and was updated to reflect the intentional new policy.

## 14. Controlled live OAuth results

`live_oauth_eval.py` — **manual-only, deterministic, no Ollama, no network**
(requires `ENABLE_INTEGRATIONS=true`; simulated local provider). **18/18 steps
passed**, written to `live_oauth_report_v030.json`. Phases:

- **oauth_flow (8):** start authorization → user grants scope → callback
  succeeds → identity discovered → credential stored → access-token refresh →
  disconnect → reauthorization restores access.
- **provider_api (1):** provider read succeeds.
- **integration_runtime (7):** calendar write + read + read-back verification;
  task read/write; confirmation required (risk=SYSTEM, parked before any side
  effect); post-disconnect rejected; cache hit.
- **grounding (2):** consistent time accepted; a shifted time (17:30 vs 15:30)
  rejected.

This cleanly distinguishes OAuth-flow success, provider-API success,
integration-runtime success, and grounding success. The model-driven chat path
remains measured separately by `live_provider_eval.py`.

## 15. Performance

Medians from `live_oauth_report_v030.json` (ms) — provider/OAuth latency only,
model generation excluded:

| Measurement | Median (ms) |
|---|---|
| authorization_start | 0.2669 |
| authorization_callback | 12.0671 |
| token_refresh | 1.6878 |
| provider_read | 0.5798 |
| provider_write_and_verify | 7.7340 |
| readback_verification | 0.9900 |
| cache_hit | 0.1921 |
| disconnect | 0.6557 |

The OAuth layer's cost is dominated by the callback (hash lookup + token
exchange + identity discovery); steady-state reads and cache hits are
sub-millisecond.

## 16. Known limitations

See `docs/JARVIS_DEVELOPER_MANUAL.md` §26 (v0.30 bullets). Summary: tokens are
obfuscated, not encrypted; the shipped adapters are still local-dev (real
third-party providers are future work behind the SAME mixin); refresh is
margin-based and lazy (no background daemon); rotation-replay revocation depends
on the provider; disconnect revocation is best-effort; the callback is
state-authenticated (standard OAuth); `provider_schedule` grounding is
labeled-field/high-confidence-only; retention cleanup protects UNKNOWN/RUNNING
rows forever.

## 17. Exact version

**0.30.0**, verified in all three sources this session:

- `pyproject.toml` (L3) — `version = "0.30.0"`
- `jarvis/__init__.py` (L17) — `__version__ = "0.30.0"`
- `jarvis/api/app.py` (L107) — FastAPI `version="0.30.0"`

The only remaining `0.29.0` strings are the unrelated `uvicorn>=0.29.0`
dependency pin, the v0.29 final report, and the v0.29 "What's New" heading.

## 18. Changed / new files and components

**New:** `jarvis/integrations/oauth.py`; `tests/test_oauth.py`;
`live_oauth_eval.py`; `live_oauth_report_v030.json`;
`docs/JARVIS_V030_SECURITY_MODEL.md`; `docs/JARVIS_V030_FINAL_REPORT.md`.

**Modified:** `jarvis/integrations/base.py`, `manager.py`,
`providers/calendar.py`, `providers/tasks.py`; `jarvis/memory/session_store.py`;
`jarvis/config.py`; `jarvis/tools/integration_tools.py`;
`jarvis/core/grounding.py`; `jarvis/api/schemas.py`, `app.py`, `client.py`;
`jarvis/main.py`; `jarvis/maintenance.py`; `ui/dashboard.py`;
`pyproject.toml`, `jarvis/__init__.py` (version); `tests/test_grounding.py`;
`README.md`, `AGENTS.md`, `deploy/README.md`, `docs/JARVIS_USER_MANUAL.md`,
`docs/JARVIS_DEVELOPER_MANUAL.md`, `docs/architecture.md`.

---

## The five required answers

**1. What can JARVIS now do with a real connected account that v0.29 could only
model or simulate?**

v0.29 could describe the *shape* of an account connection but always took the
credential as given: there was no authorization, no consent screen, no token
exchange, no token expiry, and no refresh. v0.30 performs a **real OAuth
authorization-code + PKCE exchange**: it sends the user to the provider, receives
a one-time code on a real callback endpoint, exchanges it for an access token and
(often) a refresh token, **discovers the account identity from the provider**,
and thereafter keeps the connection alive by **automatically refreshing** the
access token before it expires — all while the model still holds only an opaque
`account_id`. It can also **revoke** the grant at disconnect. These are now
actually executed against the provider interface (deterministically verified
end-to-end in `live_oauth_eval.py`), not simulated.

**2. What happens when a token expires?**

Before any operation, if the access token is expired or within
`OAUTH_REFRESH_MARGIN_SECONDS` (default 300 s) of expiry, JARVIS refreshes it
automatically, rotating the token pair atomically while preserving the account
identity and scopes, and proceeds — the user notices nothing. If the refresh
fails, the account moves to **AUTHENTICATION_REQUIRED** and the operation
**refuses** (fail closed) rather than using a dead token; the user is told to
reconnect. If the provider reports the grant is revoked, the account moves to
**REVOKED** and JARVIS **never retries** — reconnect is required.

**3. What happens when a user disconnects an account?**

Disconnect performs, in order: a best-effort **provider revocation**, deletion of
the local account row (tokens gone), purge of that account's cached reads, and a
state change that makes **every subsequent operation fail closed** (a stale
`account_id` handle is refused). If the provider is unreachable, the local
side is still cleared; only the remote grant may linger until it expires
(reported honestly). Reconnecting is the only way back.

**4. What happens when an integration action succeeds at the provider but
verification fails?**

The action result is reported as **not verified** (`ACTION_NOT_VERIFIED` /
verification-failed), never as success. Writes already run **exactly once**, so
there is no automatic re-execution. If the *outcome itself* is ambiguous (e.g. a
timeout after dispatch), the action is marked **UNKNOWN**, an audit row is
written, and it is **never silently re-run** — an explicit reissue flows through
the existing confirmation machinery, and a provider idempotency key collapses a
re-dispatch to the ORIGINAL resource so it cannot duplicate. The model is
required to report the uncertainty honestly rather than claim success.

**5. What happens when external provider data contains prompt injection?**

Every provider field is passed through `sanitize()` (redaction) and
`frame_external_content()` (framing + the browser injection detector) before the
model sees it; reads are wrapped in `UNTRUSTED PROVIDER CONTENT` and any
instruction-shaped text is flagged with an injection warning. Provider text is
**data, never instructions** — it cannot grant permissions, cannot change scopes,
cannot impersonate trusted evidence, and cannot trigger a tool. Risk is computed
from validated tool arguments only, and every injected instruction is still gated
by the existing confirmation/ledger boundary.

---

## Intentionally unsupported (unchanged or new)

- **Email (EMAIL_READ / EMAIL_SEND)** — declared-not-implemented; no tool exists
  and the tool policy honestly refuses email requests.
- **Real third-party OAuth providers** — the flow is real and provider-neutral,
  but only local-dev calendar/tasks adapters ship; real Google/Outlook/Todoist
  adapters are future work behind the SAME mixin + manager.
- **Production-grade secret storage** — tokens are obfuscated at rest, not
  encrypted; an OS keychain / encrypted store behind the same `CredentialStore`
  interface is the documented next step.
- **HTTP execute endpoint** — deliberately absent; the chat runtime is the only
  execution path.
- **Background token refresh daemon** — refresh is lazy (on use / on explicit
  refresh), not scheduled.
- **Retry queue for provider revocation** — disconnect revocation is
  best-effort only.
- **Arbitrary HTTP requests / model-supplied endpoints, headers, or auth** —
  structurally absent.
- **Full-duplex voice, wake words, computer/shell control** — unchanged
  boundaries from earlier versions.
