# JARVIS v0.29 — Part 1 Baseline Audit (v0.28 → v0.29)

Date: 2026-10-01. Scope: everything v0.29 "Personal Integrations & Workflow
Automation" touches, verified by reading the current code (not from memory).

## Deterministic v0.28 status

`uv run python -m pytest tests/ -q` on the exact pre-v0.29 tree:

```
1032 passed, 29 skipped in 228.55s
```

This matches the recorded v0.28 final status exactly. All v0.29 work must
keep this suite green.

## Real-browser status

NOT RUN. `live_browser_eval.py` is manual-only by design (v0.28 contract):
it requires the operator to provision Playwright Chromium
(`uv run playwright install chromium`), run with
`BROWSER_DRIVER=playwright` + `BROWSER_ALLOW_LOCAL_NETWORK=true`, and serve
the controlled local pages on port 8931. The v0.28 deterministic suite
(124 tests, `tests/test_browser_control.py`) covers the SimulatedDriver
contract; the live script exists and is operator-ready, but no live
Chromium run happened in this environment. Same posture carries into v0.29:
live provider validation is scripted but manual-only.

## Current external-service boundaries (pre-v0.29)

| Surface | Class | Boundary |
|---|---|---|
| `web_search` | NETWORK read | DuckDuckGo, no auth, no credentials |
| `web_scrape` | NETWORK read | Playwright fetch of a user-named URL, no auth |
| `wikipedia_summary` | NETWORK read | public API, no auth |
| `vision_analyze` | local | Ollama llava, no network egress |
| browser tools (opt-in) | NETWORK + dynamic risk | URL policy / pacing / confirmation; automation profile holds NO credentials |
| everything else | local | files (sandboxed), calculator, memory, knowledge base |

There are NO OAuth flows, NO stored credentials, NO personal-service
integrations, and NO authenticated third-party API clients anywhere in the
tree today. v0.29 introduces the first ones.

## Architecture facts v0.29 builds on (verified in code)

- **Permission tiers** (`jarvis/core/permissions.py`): `_ALLOWED_RISKS` =
  {SAFE, NETWORK, FILE_READ} auto-allowed; `_CONFIRMATION_RISKS` =
  {SYSTEM, DESTRUCTIVE} → confirmation; FILE_WRITE blocked outright.
- **Dynamic risk** (v0.28, `base.py::risk_for_args` +
  `registry.effective_risk_level()`): validates args WITHOUT dispatch,
  escalates only toward the confirmation tier, degrades to the static
  ceiling on any classifier error. Consulted by the orchestrator before
  parking. **This is the exact mechanism v0.29 reuses for external
  side effects** — a calendar write declares static NETWORK and escalates
  to SYSTEM on validated write args → existing confirmation parking →
  existing action ledger. No new confirmation machinery.
- **Action ledger** (`session_store.py`): `save_pending_confirmation`
  creates the PENDING row in the same transaction; `claim_action_execution`
  is at-most-once (PENDING→RUNNING atomic claim); states
  PENDING/RUNNING/SUCCEEDED/FAILED/UNKNOWN; UNKNOWN never auto-reruns —
  explicit `request_action_reissue` (idempotent per request_id, ceiling 3).
  v0.29 external writes ride this unchanged.
- **Result cache** (v0.24): per-tool class-level `CachePolicy` opt-in;
  no policy ⇒ never cached; errors never stored; session vs global scope;
  freshness requests + `refresh=True` bypass. v0.29 calendar/task reads
  declare short-TTL session-scoped policies; write tools declare none.
- **Evidence + grounding** (v0.25/v0.26): successful observations →
  bounded ledger → `_synthesize` final-user-message block → deterministic
  `check_grounding`. The existing `StructuredFieldGroundingPolicy` checks
  any `label: value` scalar — calendar/task outputs will emit exactly such
  labels (`count:`, `id:`, `start:`), so provider facts get grounding
  coverage without touching the policy engine.
- **Injection framing** (v0.28 `browser/injection.py`):
  `is_injection_attempt()` + wrap-with-contract pattern — v0.29 reuses both
  for provider content (event/task titles are data, never instructions).
- **Redaction** (v0.28 `browser/redaction.py`): secret-shaped substring
  redaction — v0.29 extends the same idea to token-shaped provider
  responses (its own `sanitize` module so the browser contract stays
  frozen).
- **Runtime assembly** (`runtime.py`): `_TOOL_FACTORIES` static list +
  opt-in conditional registration (code execution needs Docker AND config;
  browser needs `ENABLE_BROWSER_CONTROL`). v0.29 integrations follow the
  identical opt-in pattern (`ENABLE_INTEGRATIONS`, default False).
  `describe().disabled_tools` is pinned by `tests/test_runtime.py` to
  exactly {execute_python_code, computer_control} — integration tools are
  opt-in exactly like browser tools and never enter that list.
- **Tool policy** (`core/tool_policy.py`): a new capability family must
  appear in ALL of `CAPABILITY_TOOLS`, `_CAPABILITY_KEYWORDS`,
  `_NARROW_PATTERNS` (the v0.28 KeyError regression came from a missing
  keyword entry).
- **Store extension pattern**: `_init_db` uses `CREATE TABLE IF NOT EXISTS`
  + PRAGMA-based column migrations; all access behind `self._lock`; the
  store is the single DB owner. v0.29 integration state lives here
  (new tables), not in a second storage system.
- **API pattern**: every endpoint takes `_AUTH` + `_enforce_rate_limit`;
  safe metadata only in responses. `api/app.py` carries a hardcoded
  `version=` (one of the three version sources).
- **Eval isolation**: every `evaluation/` entry calls
  `_bootstrap.isolate()` before jarvis imports; tests get `:memory:` DB via
  conftest; offline LLM guard blocks real `litellm.completion`.

## Design decisions locked by this audit

1. **Integration state in SessionStore** (tables `integration_accounts`,
   `integration_audit`) — same DB, same lock, no second store.
2. **Provider manager assembled in `runtime.py`** (like the browser
   registry), injected into tool constructors — tools never build
   providers themselves.
3. **Risk mapping onto EXISTING tiers**: provider READ_ONLY → SAFE static;
   provider write operations → NETWORK static + `risk_for_args` escalation
   to SYSTEM on validated args (escalation requires the explicit account_id
   + operation fields; any validation error degrades to static — fail
   toward caution, v0.28 semantics).
4. **Verification reads** (v0.14 "executed ≠ verified") happen INSIDE the
   write tools: after a confirmed create/update, the tool performs a safe
   scope-checked read and reports VERIFIED / ACTION_NOT_VERIFIED — it can
   never widen scopes to do so.
5. **No new dependencies** (AGENTS.md strict rule): providers are stdlib +
   dataclasses; the first provider is a deterministic local mock with a
   real provider-shaped interface (OAuth-capable adapters are designed and
   documented, not faked).
