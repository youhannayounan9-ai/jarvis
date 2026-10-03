# JARVIS v0.29.0 Final Report — Personal Integrations & Workflow Automation

**Date:** October 2, 2026 · **Baseline:** v0.28.0 · **Version:** 0.29.0 · **Source control:** none performed

---

## 1. Implementation summary

v0.29 gives JARVIS a secure, extensible framework for connecting to external personal services and performing useful workflows — reusing, not duplicating, every existing reliability and security mechanism (permission tiers, durable confirmation parking, action ledger, evidence ledger, grounding guard, result cache, recovery/reissue).

Shipped:

- **`jarvis/integrations/`** — the framework (11 modules across the package + `providers/`): provider ABC + `ResourceSpec` (per-operation scope + risk), `IntegrationManager` (the ONLY path to a connected account), explicit credential store, scope registry, fault taxonomy, content sanitization/framing, event validation, audit support.
- **Two deliberately small reference integrations** — calendar (5 operations) and tasks (3 operations) — as local-dev providers (`LocalCalendarProvider`, `LocalTasksProvider`, `production_like=false`).
- **8 narrow tools** (`jarvis/tools/integration_tools.py`): `calendar_list_events`, `calendar_get_event`, `calendar_create_event`, `calendar_update_event`, `calendar_delete_event`, `task_list`, `task_create`, `task_complete`.
- **Opt-in surface** mirroring the browser pattern: `ENABLE_INTEGRATIONS=true` (default **off**) gates tools + API `/integrations*` + CLI `/integration*` + dashboard section.
- **103 deterministic tests** (`tests/test_integrations.py`, categories A–AD), a manual-only live eval (`live_provider_eval.py`), a baseline audit (`docs/JARVIS_V029_BASELINE_AUDIT.md`), a security model (`docs/JARVIS_V029_SECURITY_MODEL.md`), and updated user/developer/deploy/architecture docs.

The orchestrator's dispatch sites required **no changes**: dynamic risk escalation rides the existing `registry.effective_risk_level()` (the v0.28 extension point), so SYSTEM-tier writes flow into the existing confirmation parking automatically.

## 2. Architecture changes

- **New layer — integration framework** between tools and external services. `IntegrationManager.execute_read/execute_write` enforces IN ORDER: provider registered → provider available → **LIVE auth state** (EXPIRED/REVOKED refuse) → **exact scope** (`require_scope`, deny-by-default) → operation. Reads get bounded retries (≤2, {RATE_LIMITED, PROVIDER_OUTAGE} only); **writes run EXACTLY ONCE**.
- **Identity model** (`integration_accounts` table): provider, account label, auth state, granted scopes, connected/last-verified timestamps, credential stored separately. The model holds opaque `account_id` handles — never credentials, never provider endpoints/headers/auth values.
- **Credential boundary** (`credentials.py`): XOR+base64 obfuscation keyed by per-install `.jarvis_integration_key` (0600, next to `DB_PATH`; process-lifetime in-memory key for `:memory:` DBs), honestly documented as a local-dev privacy guard, not hard security.
- **Untrusted-data boundary** (`sanitize.py`): every provider field is redacted (`sanitize()`) and framed (`frame_external_content()`, reusing the browser injection detector); read outputs carry the `UNTRUSTED PROVIDER CONTENT` footer with a line-anchored `count:` the v0.26 grounding policy can check.
- **Risk mapping** (`base.py::static_tier_for`): READ_ONLY→SAFE, LOW/MEDIUM→NETWORK, HIGH→SYSTEM, IRREVERSIBLE→DESTRUCTIVE; tool-level `risk_for_args` escalates fully-valid writes to SYSTEM ⇒ the EXISTING confirmation parking. Validation failures degrade to the static tier and refuse BEFORE any side effect (no audit rows for refused ops).
- **Store** (`session_store.py`): `integration_accounts` + `integration_audit` tables in the same SQLite; audit rows carry category/operation/op_id/idempotency-key/redacted summary ONLY — never arguments or payloads.
- **Tool policy** (`tool_policy.py`): new `integrations` capability family + narrowing patterns (calendar/task nouns); `detect_unmet_capability` honestly refuses email requests (EMAIL_READ/EMAIL_SEND declared-not-implemented).
- **Cache** (`result_cache.py`): the 3 read tools declare session-scoped `CachePolicy`s with settings-driven TTL (`RESULT_CACHE_INTEGRATION_TTL_SECONDS`, default 60 s, per-account keys); writes declare NO policy ⇒ never cached (v0.24 contract).
- **API/CLI/dashboard**: `GET /integrations`, `GET /integrations/{provider}/scopes`, `POST /integrations/{provider}/connect`, `POST /integrations/accounts/{id}/disconnect`, `GET .../audit` — account management ONLY, deliberately **no HTTP execute endpoint**; CLI `/integrations`, `/integration-status`, `/integration-connect`, `/integration-disconnect`; dashboard Integrations section.

## 3. Capability delta from v0.28

| Dimension | v0.28 | v0.29 |
|---|---|---|
| Deterministic suite | 1032 passed / 29 skipped | **1135 passed / 29 skipped, 0 failed** |
| Integration tools | 0 | 8 (calendar ×5, tasks ×3) |
| API endpoints | — | +5 (`/integrations*`, management only) |
| CLI commands | — | +4 (`/integration*`) |
| Dashboard sections | Browser control | + Integrations section |
| Settings | — | `ENABLE_INTEGRATIONS`, `RESULT_CACHE_INTEGRATION_TTL_SECONDS` |
| Store tables | — | `integration_accounts`, `integration_audit` |
| Honest refusals | computer control; code exec on ineligible hosts | + email send/read (declared-not-implemented) |
| Security docs | v0.28 browser security model | + v0.29 security model + baseline audit |

## 4. Integration framework

- **`IntegrationProvider` ABC** (`base.py`): provider identity (`name`, `production_like`), availability, auth state, and per-`ResourceSpec` operations (list/get/create/update/delete) — adapters implement only what they support. The orchestrator never sees a provider SDK; the runtime interacts with normalized capabilities.
- **Scopes** (`scopes.py`): exact, separate capabilities — `CALENDAR_READ`, `CALENDAR_WRITE`, `TASKS_READ`, `TASKS_WRITE`. No `FULL_*_ACCESS` grants exist; scopes are bound at CONNECT time by the user and the model has **no tool** to change them (no dynamic scope expansion).
- **Fault taxonomy** (`errors.py`): `PROVIDER_NOT_REGISTERED/UNAVAILABLE`, `AUTH_EXPIRED/REVOKED`, `SCOPE_INSUFFICIENT`, `RATE_LIMITED`, `PROVIDER_OUTAGE`, `TIMEOUT`, `AMBIGUOUS_OUTCOME`, `VALIDATION_FAILED` — each mapping to a sanitized tool-level message via `to_tool_error()`, so provider text can never smuggle secrets into an ERROR line.
- **Writes dispatched exactly once** — no retries; ambiguity (TIMEOUT/AMBIGUOUS_OUTCOME) ⇒ UNKNOWN + audit row, never silent re-execution. Reissue flows through the existing confirmation machinery with a provider idempotency key so a replay returns the ORIGINAL resource.
- **Extending**: (1) a new `IntegrationProvider`, (2) `ResourceSpec` entries with per-operation scope+risk, (3) tools in `integration_tools.py` reusing the shared plumbing, (4) a `tool_policy` capability-family entry. Never per-phrase special cases; never a second manager.

## 5. Calendar integration (reference implementation)

- **Reads** (SAFE tier, session-cached): `calendar_list_events` (bounded `limit`; renders the line-anchored `count:`), `calendar_get_event` (`event: <id>` + framed fields).
- **Create** (`calendar_create_event`): `validate_event` requires explicit title, date, time, timezone, duration — **vague requests never silently become exact external actions** (no-silent-time-guessing: missing scheduling information is requested back, never invented; timezones normalized explicitly via ZoneInfo). Idempotency key = sha256(title | start_utc | duration | attendees); on key collision the provider returns the ORIGINAL event.
- **Update** (`calendar_update_event`): validated field changes; verification is a read-back comparison with UTC-instant normalization (`+02:00` ≡ `Z`).
- **Delete/cancel** (`calendar_delete_event`): HIGH side effect ⇒ SYSTEM ⇒ explicit confirmation parking; irreversible once confirmed.
- Every write: full confirmation preview (title/date/time/timezone/duration/attendees/side effect), action-ledger claim, audit row, read-back `verification: VERIFIED|ACTION_NOT_VERIFIED`.

## 6. Task integration (deliberately narrow)

- `task_list`, `task_create`, `task_complete` — list/create/complete only. **Delete is disabled by design** (no concrete need today; a stronger confirmation tier would be required if ever added).
- Same identity/scope/auth/cache/grounding machinery as calendar, with `TASKS_READ`/`TASKS_WRITE`; writes escalate identically.

## 7. Authentication & security

- **Explicit connected identity, never inferred from model text**: an external action names an `account_id`; the manager resolves it against the stored account and refuses on unknown id, expired/revoked auth, or missing scope.
- **Auth states**: ACTIVE / EXPIRED / REVOKED (local-dev providers simulate via credential-prefix rules); `refresh_auth_state()` re-validates labels — v0.29 has **no OAuth token exchange/refresh anywhere** (documented design gap, not hidden).
- **Credentials never reach the model**: not in prompts, not in tool output, not in logs (redacted), not in session memory, not embedded in action descriptions. Provider responses are sanitized before the model sees them.
- **Honest crypto posture**: XOR+base64 obfuscation with a per-install key file — a no-secrets-in-plaintext privacy guard for a local-first app, NOT protection against local malware running as the user (`credentials.py` documents this; OS-keychain/encrypted storage is the deliberate next step behind the same `CredentialStore` interface).
- **Threat model**: T1–T15 in `docs/JARVIS_V029_SECURITY_MODEL.md` (incl. T13 malicious-provider impersonation → `production_like` flag + ABC-gated registration).

## 8. Action / confirmation behavior

- Static tiers per operation (list/get → SAFE; create/update → LOW/MEDIUM; delete → HIGH) with **args-dependent escalation**: a fully-valid write on a REAL account escalates to SYSTEM ⇒ `PAUSED_FOR_CONFIRMATION` parking. The user confirms the actual external action; the model never confirms it itself.
- Confirmation preview contains: title, date, time, timezone, duration, target calendar, attendees, side effect.
- **At-most-once ledger**: `action_executions` claim → execute → record; a crash-window leaves UNKNOWN (reported, never auto-rerun); reissue is explicit, bounded, and idempotent (verified: replays the provider idempotency key and returns the original event id).
- REFUSED operations produce **no audit rows** (they had no effect); executed writes always do.

## 9. Verification behavior

- `calendar_create_event` / `calendar_update_event` verify by **read-back comparison** (deterministic, UTC-instant normalization for start/end): the output carries `verification: VERIFIED` or `ACTION_NOT_VERIFIED` beneath `ACTION_EXECUTED:` — an unverifiable write never reads as success.
- Failed observations never govern grounding; cache provenance (`source`/`cached_age`) applies to read evidence exactly as in v0.24–v0.26.

## 10. Cache behavior

- Exactly 3 read tools cached: `calendar_list_events`, `calendar_get_event`, `task_list` — **session-scoped** `CachePolicy`, TTL from `RESULT_CACHE_INTEGRATION_TTL_SECONDS` (default 60 s), per-account normalized keys (account id inside the key ⇒ no cross-account leakage).
- Writes declare **no policy ⇒ never cached** (pinned by the policy-consistency tests).
- `refresh=True` (the v0.25 client control) bypasses ONLY the cache lookup; permissions, confirmations, and the repeat guard are unchanged. Errors are never stored; hits are provenance-labeled untrusted evidence whose `count:` field is grounding-checkable.

## 11. Deterministic test results

- `tests/test_integrations.py`: **103 passed** — categories A–AD: disconnected/connected/expired/revoked auth, exact-scope + insufficient-scope enforcement, account-identity handling, credential redaction, provider-content prompt injection, read-only ops, calendar create/update, duplicate-create prevention (idempotency + UNKNOWN crash-window + reissue), delete confirmation, provider timeout/rate-limit/outage, verification success/failure, cache behavior, refresh bypass, action ledger, grounding (`count:`), no-silent-time-guessing, API authorization boundary, disconnect, multi-session, and no-credentials-to-model.
- **Full suite (final regression, this session): 1135 passed, 29 skipped, 0 failed** (244.6 s, Windows, `uv run pytest tests/ -q`). v0.28 baseline was 1032 passed / 29 skipped ⇒ **+103**.
- Infrastructure note: `tests/test_context_management.py`'s evidence-blob allowance was raised (+4500 → +5000) to absorb the v0.29 system-prompt integrations paragraph (prompt now 7,648 chars); the evidence clamp itself is unchanged.

## 12. Live / provider validation

- `live_provider_eval.py` (repo root, **manual-only**) exercises 6 cases against the local dev providers: grounded read, confirmed write (preview → approval → VERIFIED read-back), denied write (scope/auth refusal), email honest refusal, cross-turn repeat re-parking with idempotent re-dispatch capped at ONE create, and deterministic idempotent replay.
- Preconditions verified this session: the script refuses to run without `ENABLE_INTEGRATIONS=true` (guard exits 2) and compiles cleanly (`py_compile`).
- **Ollama was not running this session** (localhost:11434 no response), so LLM-driven live paths were NOT re-validated here; deterministic tool/manager/provider paths were smoke-verified directly. The script must be re-run manually with a local model when Ollama is up.
- **Mock vs real, clearly separated**: all evidence comes from `LocalCalendarProvider`/`LocalTasksProvider` (`production_like=false`). No claim is made about real Google/Microsoft-class providers — no real OAuth provider exists in v0.29.

## 13. Performance

Measured (median of 200 in-process reps, Windows, `:memory:` SQLite, local dev provider — provider I/O ≈ 0, so these isolate FRAMEWORK overhead; security model §6):

| Path | Median | Max |
|---|---|---|
| Tool read, session-cache hit (`calendar_list_events.run`) | **0.042 ms** | 0.124 ms |
| Manager enforcement only (`execute_read`) | **0.027 ms** | — |
| Event validation (`validate_event`) | **0.015 ms** | 8.7 ms (ZoneInfo first load) |
| Tool write path (validate → risk → manager → provider → audit → read-back verify) | **0.258 ms** | 5.1 ms (first-call warmup) |

- No LLM calls anywhere in the tool path; enforcement adds ~27 µs over a bare provider call.
- With a real networked provider, provider latency will dominate and must be attributed to the provider, not to model latency — tool elapsed timing is reported separately from synthesis timing in telemetry.
- System prompt grew by the integrations paragraph: 7,648 chars total (context-management allowance adjusted, test-pinned).

## 14. Security audit

- Full audit: `docs/JARVIS_V029_SECURITY_MODEL.md` (baseline audit table, threat model T1–T15, safety loop, trust boundaries, guarantee summary, measured performance, audit checklist, kill switches) plus the pre-implementation baseline in `docs/JARVIS_V029_BASELINE_AUDIT.md`.
- Audited areas and outcomes: OAuth state (absent — no flow shipped, gap documented), redirect handling (absent by design), secret storage (obfuscated at rest + redaction everywhere model-visible), token exposure (none — model never receives credentials), scope escalation (impossible — connect-time constants, no tool to change), account confusion (opaque account ids, deny-by-default resolution), prompt injection (sanitize + framing + injection detector; T-covered), provider-response trust (UNTRUSTED PROVIDER CONTENT, never instructions), side-effect duplication (at-most-once ledger + provider idempotency keys), UNKNOWN recovery (manual, bounded, idempotent reissue), cache leakage (session scope + per-account keys + no write caching), session isolation (existing leases/sessions untouched), authorization boundaries (shared `_AUTH`, no execute endpoint), logging/redaction (audit rows and telemetry carry categories/keys only).
- **Residual risks (documented, accepted for a local-dev milestone)**: obfuscation ≠ encryption; no real OAuth; local-dev providers only; audit rows unbounded in count; scopes fixed at connect time; grounding coverage on integration answers is `count:`-first (titles/dates are narrative).

## 15. Known limitations

Summarized here; the canonical list is developer manual §26:

1. Credentials are obfuscated (XOR+base64, per-install key file), not encrypted-at-rest — a privacy guard, not malware protection.
2. Providers are local-dev only (`production_like=false`); no OAuth token exchange/refresh exists anywhere in v0.29.
3. Audit rows are bounded in size (redacted summaries only) but not pruned by count — retention policy is future work.
4. Scope sets are connect-time constants: changing scopes = disconnect + reconnect; no scope-diff consent flow.
5. Grounding verifies only the line-anchored `count:` on integration reads; titles/dates/times in answers are narrative (high-confidence-only design).
6. Email/messaging is deliberately unimplemented (declared-not-implemented refusal), pending the documented safety model (recipient verification, content preview, attachment handling, send confirmation, duplicate-send prevention).

## 16. Exact version

**0.29.0**, verified in all three sources this session:

- `pyproject.toml` (L3) — `version = "0.29.0"`
- `jarvis/__init__.py` (L17) — `__version__ = "0.29.0"`
- `jarvis/api/app.py` (L104) — FastAPI `version="0.29.0"`

No stale `0.28.0` remains in any version source (the only other `0.29.0` string is the unrelated `uvicorn>=0.29.0` dependency pin).

## 17. Changed components / files

**Created (v0.29):**

- `jarvis/integrations/` — `base.py`, `manager.py`, `credentials.py`, `scopes.py`, `errors.py`, `sanitize.py`, `validation.py`, `__init__.py`, plus `providers/{__init__,calendar,tasks}.py`
- `jarvis/tools/integration_tools.py` (8 tools)
- `tests/test_integrations.py` (103 tests)
- `live_provider_eval.py` (manual-only live eval)
- `docs/JARVIS_V029_BASELINE_AUDIT.md`, `docs/JARVIS_V029_SECURITY_MODEL.md`, this report

**Modified (v0.29):**

- `jarvis/config.py` (integration settings), `jarvis/runtime.py` (opt-in registration block), `jarvis/core/tool_policy.py` (capability family + narrowing + email refusal), `jarvis/core/result_cache.py` (read-TTL overrides), `jarvis/memory/session_store.py` (`integration_accounts` + `integration_audit`), `jarvis/api/app.py` (+5 endpoints, version), `jarvis/api/schemas.py`, `jarvis/api/client.py`, `jarvis/main.py` (CLI commands), `ui/dashboard.py` (Integrations section)
- `pyproject.toml`, `jarvis/__init__.py` (version), `.gitignore` (`.jarvis_integration_key`)
- Docs: `README.md`, `AGENTS.md`, `deploy/README.md`, `docs/architecture.md`, `docs/JARVIS_USER_MANUAL.md`, `docs/JARVIS_DEVELOPER_MANUAL.md`
- `tests/test_context_management.py` (evidence-blob allowance +5000)

Deliberately **unchanged**: `jarvis/core/orchestrator.py` dispatch sites, PermissionGuard, confirmation parking, action ledger, grounding policies, and the browser layer — integration risk rides the existing `registry.effective_risk_level()` extension point.

---

## The two required answers

**1. What real-world personal workflow can JARVIS now complete safely that required manual work before?**

Plan-and-confirm personal scheduling and to-dos through plain chat. Before v0.29 this meant opening the calendar/task app yourself; now, with `ENABLE_INTEGRATIONS=true` and a connected account:

- *"What's on my calendar Friday?"* → `calendar_list_events`: a LIVE-auth + exact-scope enforced read, session-cached for 60 s, with a grounding-checkable `count:` line.
- *"Create an event Friday 15:00, 'Dentist', 45 minutes"* → the event is fully validated (vague requests are asked back, never guessed), risk escalates to SYSTEM, and JARVIS parks for your confirmation with the full preview (title/date/time/timezone/duration/attendees/side effect). On approval the action ledger claims at-most-once, the write runs EXACTLY ONCE, a read-back comparison reports `verification: VERIFIED`, and an audit row records the operation. A crash mid-write leaves UNKNOWN — never auto-rerun; a reissue replays the idempotency key and returns the ORIGINAL event.
- *"Add 'call the plumber' to my tasks"* and *"mark task … complete"* work the same way.

Net: read + confirmation-gated write workflows over calendar and tasks with a per-account audit trail — previously manual app work, now one conversation, with the system (not the LLM) holding every boundary.

**2. What external action will JARVIS refuse to perform automatically, and why?**

**Sending an email or message.** EMAIL_READ/EMAIL_SEND are DECLARED-NOT-IMPLEMENTED: no tool exists, and the tool policy refuses the request honestly instead of approximating one. Why: unsolicited outbound communication is a high-side-effect, effectively irreversible disclosure to third parties, exposed to recipient spoofing and prompt injection through provider content — doing it safely requires recipient verification, content preview, attachment handling, duplicate-send prevention, and sensitive-information handling that v0.29 deliberately does not ship (the safety model is documented for later milestones).

Also refused automatically: any external write without explicit user confirmation (SYSTEM-tier confirmation parking); event deletion without its HIGH-risk confirmation; arbitrary HTTP/API calls (no such tool exists, and the model can never supply endpoints, headers, or authorization values); scope expansion (scopes are connect-time constants); and the automatic re-run of an UNKNOWN action (ambiguity ⇒ park + audit row; only a human reissue — which is idempotent — may resolve it).
