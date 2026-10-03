# deploy/README.md — Service Deployment Guide (v0.16)

How to run JARVIS as a real service: sandbox image, API hardening, and the
runtime/interface split. Security posture first: **everything fails closed.**

---

## 1. The sandbox image

JARVIS code execution runs untrusted Python in one-shot containers. The
default fallback image is a tag-pinned distro image; production deployments
should build the dedicated minimal image instead.

### Why a dedicated image

`deploy/Dockerfile.sandbox` contains only what untrusted Python needs —
the interpreter, the stdlib, and a dedicated unprivileged user
(`jarvis-sbx`, uid/gid 2000). Compared to a distro image it removes the
attack surface and the weight: no shell-only tooling of value, no
curl/wget/ssh, no package managers, ~75MB → ~50MB, faster cold starts.

### Build, pin, pre-pull

```bash
# 1. Verify and pin the base image digest for your architecture first
#    (edit deploy/Dockerfile.sandbox FROM line), then:
docker build -t jarvis-sandbox:1.0.0 -f deploy/Dockerfile.sandbox deploy/

# 2. Pin by digest so the runtime boundary is immutable:
docker images --digests jarvis-sandbox
#   .env:  SANDBOX_IMAGE=jarvis-sandbox:1.0.0@sha256:<digest>

# 3. Pre-pull on the host that runs JARVIS (the sandbox never pulls at
#    runtime — a missing image is a denial, not a download):
docker pull jarvis-sandbox:1.0.0@sha256:<digest>
```

#### Base digest: verified state and re-verification procedure

**CONFIRMED (2026-09-27, real Linux daemon):** the `python:3.12-slim` base
digest in `deploy/Dockerfile.sandbox` was resolved and verified by an actual
`docker pull python:3.12-slim` against a live Linux engine:

```
sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
```

The digest was then confirmed pullable *by ref*
(`docker pull python:3.12-slim@sha256:f77ac9e4…`) and the production image
was built from it. To re-verify after a future base update (or on a different
architecture):

```bash
docker pull python:3.12-slim
docker images --digests python          # copy the new digest
# edit the FROM line in deploy/Dockerfile.sandbox
# then re-run the integration suite (below) before deploying
```

If the digest is ever reverted to `REPLACE_WITH_VERIFIED_DIGEST`, the build
script and CI both fail closed, and the Docker integration tests skip with a
printed reason.

#### Real-Docker verification suite (v0.16)

`tests/test_sandbox_integration.py` verifies **observed behavior** against a
real Linux engine — not flag construction: it builds the production image and
asserts, from inside real containers, the non-root uid, dropped capabilities,
read-only rootfs, noexec tmpfs, no network, bounded PIDs, memory-cap kills,
workload-killing timeouts (exit 124, no orphans), the layer-C force-removal
mechanism, and output caps. Run it wherever a Linux daemon exists:

```bash
uv run pytest tests/test_sandbox_integration.py -v
```

Prerequisites are checked first; the whole module **skips cleanly** (never
fails, never pretends) when the docker CLI, a Linux engine, or the pinned
digest is unavailable.

The sandbox's image validator enforces this policy mechanically:
bare names and `latest` are rejected (`latest` can be re-pushed with
different contents — a mutable security boundary); concrete tags and
`sha256:` digest refs are accepted.

### Enabling code execution

```env
ENABLE_CODE_EXECUTION=true
SANDBOX_IMAGE=jarvis-sandbox:1.0.0@sha256:<digest>
```

The tool joins the LLM surface only if `is_available()` verifies docker CLI,
daemon, and the image locally. Every run is still one-shot, network-less,
read-only-rootfs, capability-less, non-root, and resource-capped — and still
`SYSTEM` risk, so a durable confirmation is required before execution.

### Enabling personal integrations (v0.29)

```env
ENABLE_INTEGRATIONS=true
```

This gates the 8 integration tools (calendar list/get/create/update/delete,
task list/create/complete), the `/integrations*` API endpoints, the CLI
`/integration*` commands, and the dashboard Integrations section. Providers
ship local-dev only (`LocalCalendarProvider` / `LocalTasksProvider`; the API
surfaces `production_like=false`) — v0.30 adds a real OAuth FLOW but still
ships no real third-party provider adapter, so do not expose integrations on
a public deployment. The credential key file `.jarvis_integration_key` is
created next to `DB_PATH` — treat it as a secret and never commit it. There
is deliberately NO HTTP execute endpoint: the chat runtime is the only
execution path.

### OAuth setup & redirect requirements (v0.30)

v0.30 adds a provider-neutral OAuth authorization-code + PKCE flow. No new
dependency is required (the token client uses the stdlib `urllib`). Configure
it through the same env surface:

```env
OAUTH_REDIRECT_BASE_URL=http://127.0.0.1:8000   # MUST match the provider's registered callback exactly
OAUTH_STATE_TTL_SECONDS=600                     # one-time authorization state lifetime
OAUTH_REFRESH_MARGIN_SECONDS=300                # refresh access tokens this long before expiry
OAUTH_ACCESS_TOKEN_TTL_SECONDS=3600             # local-dev token lifetime
OAUTH_CODE_TTL_SECONDS=120                       # local-dev authorization-code lifetime
OAUTH_CALENDAR_CLIENT_ID=local-dev-calendar
OAUTH_CALENDAR_CLIENT_SECRET=local-dev-calendar-secret
OAUTH_TASKS_CLIENT_ID=local-dev-tasks
OAUTH_TASKS_CLIENT_SECRET=local-dev-tasks-secret
INTEGRATION_AUDIT_RETENTION_DAYS=90             # bound the audit trail
```

The callback the provider must redirect to is
**`{OAUTH_REDIRECT_BASE_URL}/integrations/oauth/callback/{provider}`** (the
session id is appended as a query parameter by the authorization URL).
Deployment notes and honest limits:

- The callback endpoint is **deliberately not API-key-gated** — it is visited
  by the user's browser after the provider redirect; the one-time `state` IS
  the authenticator (standard OAuth). It returns a fixed local HTML page, never
  a redirect and never a token. Do not put it behind an auth proxy that would
  break the browser redirect.
- The redirect base must be reachable by the user's browser. For a Docker
  deployment, `127.0.0.1:8000` works only when the API port is published to the
  host (`-p 8000:8000`); otherwise set it to the externally reachable URL.
- Client id/secret above are **local-dev placeholders** for the simulated
  provider. Real providers require their own id/secret and a fixed registered
  redirect URI.
- Tokens are stored with the same local **obfuscation** as v0.29 credentials —
  a privacy guard, **not** encryption. Do not treat a deployment's token store
  as production-grade secret management; a real OS keychain / encrypted store
  is the documented next step behind the same `CredentialStore` interface.
- Prune the audit trail with `maintenance integrations-audit` (report-only by
default; `--yes` performs a bounded deletion). Rows whose state is
  UNKNOWN or RUNNING are **always** protected from cleanup.

### Timeout enforcement (three distinct layers)

Since v0.15 the time limit is enforced **at the workload boundary** (v0.16
additionally verified it at runtime — see §1), not only
at the host:

- **B — container timeout (primary):** the container entrypoint is
  `timeout <cap>s python3 script.py` (coreutils). The workload is killed
  *inside* the container (exit 124) — definitive termination.
- **A — host process timeout (secondary):** the host stops waiting on the
  docker CLI at cap + 5 s slack. By itself this only stops *waiting* — which
  is exactly why layer B exists.
- **C — actual termination:** if layer A fires first (hung CLI/daemon), the
  container is force-removed (`docker rm -f <container>`) so no orphan
  keeps consuming CPU.

`ExecutionResult.timeout_layer` reports which layer fired (`container`,
`host_kill`, or None), and `denial_reason="timeout"` on both timeout paths.
The sandbox never pulls images at runtime and keeps every isolation flag
listed above.

---

## 2. Running the service

```bash
uv sync --extra dev
cp .env.example .env        # set OLLAMA_MODEL, JARVIS_API_KEY, sandbox vars

# API (primary service surface)
uv run uvicorn jarvis.api.app:app --host 127.0.0.1 --port 8000
# production: --workers 2+ behind a reverse proxy; set JARVIS_API_KEY

# CLI (same runtime, same surface)
uv run jarvis

# Dashboard (now an API client, not an in-process runtime)
uv run streamlit run ui/dashboard.py
```

### Hardening checklist for exposure beyond loopback

- `JARVIS_API_KEY` set (constant-time enforced on every endpoint but `/health`)
- reverse proxy (TLS, real rate limiting, request size caps) in front
- `DB_PATH` on a persistent volume; backups of `jarvis.db`
- `ENABLE_CODE_EXECUTION` left off unless the host is Linux/WSL2 with the
  dedicated image pre-pulled
- `ENABLE_INTEGRATIONS` left off unless personal integrations are wanted
  (local-dev providers only; keep `.jarvis_integration_key` out of backups
  that leave the host)
- computer control remains unconditionally disabled — there is no flag

---

## 3. API protection (v0.17: process-local default, database-backed option)

- **Rate limiting** — sliding-window limiter keyed by API key (or client IP
  when auth is off). `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` in
  `.env` (default 60 req / 60 s; `0` disables). Over-limit → `429` with
  `Retry-After`. `/health` is exempt. Two backends behind one class
  (`jarvis/api/ratelimit.py`):

  - **In-memory (deployment default):** per-process. For multi-replica
    deployments keep the reverse proxy as the authoritative limiter.
  - **SQLite-backed (opt-in):** `make_durable_limiter(store)` from
    `jarvis.api.ratelimit` records hits in the shared JARVIS database
    (`rate_limit_events`). Every check is one `BEGIN IMMEDIATE` write
    transaction, so all processes pointing at the same `DB_PATH` enforce
    ONE limit per client; each check also deletes expired events (the table
    stays at ~clients × max_requests rows). Costs a DB write per request —
    enable it when you run more than one JARVIS process against one volume.

  No Redis or external service is involved in either backend.
- **Per-session serialization — two layers (v0.17)** — one chat turn per
  session at a time; concurrent turns on the same session return `409
  Conflict` instead of interleaving history or double-resolving
  confirmations. Different sessions are fully parallel. Layer 1 is the
  process-local mutex (fast). Layer 2 is a **database-backed session
  lease** (`session_leases` table): TTL 300 s, owner-checked renew/release,
  fencing token bumped on takeover — two JARVIS processes sharing one
  `DB_PATH` can no longer both process the same session, and a crashed
  process's lease self-expires (no indefinite lock). Residual race: a single
  turn that outlives the 300 s TTL near its boundary.
- **Confirmation continuation (v0.15) + execution ledger (v0.17)** — an
  approved/denied high-risk action RESUMES the original task, even if the
  service restarted while the action was pending. Every protected execution
  is additionally recorded in the `action_executions` ledger (PENDING →
  RUNNING → SUCCEEDED/FAILED, with UNKNOWN for crash ambiguity): repeated
  approvals return the recorded outcome instead of re-executing, and actions
  whose outcome is unknown are never automatically re-run
  (`ACTION_EXECUTION_STATE_UNKNOWN` report; see docs/JARVIS_DEVELOPER_MANUAL.md §8).
- **Operational endpoints (v0.18)** — `GET /actions`, `GET /actions/{id}`,
  `GET /sessions/leases` are read-only introspection (safe metadata; owner
  tokens redacted; tool arguments and result bodies never exposed).
  `POST /actions/{id}/reissue` is the ONE mutating operational endpoint:
  it deliberately re-issues an UNKNOWN action as a NEW action id (idempotent
  per `request_id`, max 3 per original, audited in `action_reissues`). It
  sits behind the same auth + rate limiting as every other mutating
  endpoint — with `JARVIS_API_KEY` set, an unauthenticated caller cannot
  trigger a reissue (401). Reading state is deliberately easier than
  reissuing it: never expose a reissue-capable key to a read-only consumer.
- **Auth** — see README (`JARVIS_API_KEY`, Bearer/X-API-Key, constant-time).

---

## 4. Interface / runtime split

`jarvis/api/client.py` is a dependency-free HTTP client over the REST API
(stdlib `urllib`). The Streamlit dashboard uses it: the UI is now a pure
client and can run on a different machine than the agent runtime. The CLI
still embeds the runtime directly (lowest latency for local use) — both
surfaces share `build_runtime()` semantics through the API contract.

Env for the dashboard:

```env
JARVIS_API_URL=http://127.0.0.1:8000   # default
JARVIS_API_KEY=                        # only if the server requires auth
```

---

## 5. CI trust boundary (public repository — read this before adding a runner)

The GitHub repository is **public**. That makes the CI trust model a
security control, not an optimization:

- **`push`/`pull_request` jobs run on GitHub-hosted ephemeral runners.**
  Untrusted pull-request code never reaches your machines. This is the only
  place PR-triggered code may execute.
- **The nightly live-model eval job (`nightly-evals`) runs on a self-hosted
  runner and is SCHEDULE-ONLY** (`if: github.event_name == 'schedule'`). It
  executes whatever is on `main` at 03:00 UTC — it never runs pull-request
  workflow code. Never remove that guard.
- **Self-hosted runner requirements** (all of them, not optional):
  - A **dedicated machine/VM** that hosts nothing else — a personal
    workstation is NOT an acceptable runner; it holds your credentials,
    browser sessions, and files.
  - An **unprivileged service account**, never your login user.
  - **No Docker socket mounted** into the runner, and no credential
    material beyond what the eval needs (a local Ollama endpoint).
  - Prefer an **ephemeral** runner (re-registered per job) so any compromise
    does not persist between runs.
- **Workflow changes are owner-reviewed**: `CODEOWNERS` covers
  `.github/workflows/`, `deploy/`, and container files. Enable *Require
  review from Code Owners* on `main` branch protection. Anyone can open a
  PR in a public repo; your approval of workflow changes is the trust
  boundary that keeps the runner from executing attacker-chosen code.
- The workflow runs with `permissions: contents: read` (least privilege);
  the nightly job additionally checks out with `persist-credentials: false`.

## 6. Operational notes

- `GET /health` reports `auth_enabled`, `code_execution` posture, tool
  surface, and version — wire your liveness probe to it.
- Structured logs: every API request logs method/path/status/duration
  (`api_request`); chat turns log the plan and each step (`plan_ready`,
  `step_execute_start/done`) — aggregate with any JSON log shipper.
- `cleanup_old_sessions(max_age_days=30)` exists on the store; schedule it
  (cron/OS task) — the service does not do background jobs itself.
- **Tool-selection policy (v0.21)** — on by default; `JARVIS_DISABLE_TOOL_POLICY=true`
  restores exact v0.20 behavior. New log events to ship alongside
  `plan_ready`/`step_execute_*`: `tool_policy_applied` (forced_tool /
  force_tool_round per turn), `no_tool_direct_answer` (model chose no tool),
  `forced_tool_round_unfulfilled` (the safety net fired its one recovery
  round), `deterministic_tool_fallback` (JARVIS executed the calculator
  itself after a zero-tool-turn on plain arithmetic — same PermissionGuard
  and schema validation as model-initiated calls). These answer "why did
  JARVIS call / not call tool X?" from logs alone. The deterministic
  tool-selection benchmark (`evaluation/tool_selection_benchmark.py`, 27
  cases / 13 categories) is pytest-wired and runs offline in CI; the live
  A/B harness (`live_tool_eval.py`) is manual-only and must NEVER run
  against a shared production Ollama instance unattended.
- **Multi-step planning (v0.22)** — new log events to ship:
  `plan_validated` (raw_steps / steps / issues[] — issues list every
  structural problem found: unknown tools, duplicates, forward references),
  `plan_step_failed` (step + required_tools), and the existing
  `plan_ready`/`step_execute_*`. Plans are validated structurally before
  execution (shape, bounds, registry truth, duplicates, forward-reference
  rejection) but validation is NOT authorization — every step execution
  still flows through PermissionGuard, schema validation, and the action
  ledger. The deterministic multi-step benchmark
  (`evaluation/multistep_benchmark.py`, 7 cases) is pytest-wired and runs
  offline in CI; the live multi-step harness (`live_multistep_eval.py`)
  is manual-only with the same precautions as `live_tool_eval.py`.
- **Repeat semantics (v0.23)** — new log events to ship alongside the
  plan events: `duplicate_tool_call_suppressed` (tool + fingerprint only,
  never arguments — fired when an identical successful call repeats within
  one turn; see `jarvis/core/dispatch_guard.py`), `plan_step_satisfied`
  (step + required_tools — the step met its evidence requirement),
  `redundant_plan_step` (an identical later step was skipped), and
  `plan_completed` (planned/completed/failed steps, suppressed duplicates,
  complete=bool — the one-line "did the plan actually finish?" signal).
- **Cross-turn result cache (v0.24)** — read-only retrieval tools reuse
  still-valid results across turns. Operational notes:
  - storage is the SAME SQLite `jarvis.db` (`result_cache` table), bounded
    by `RESULT_CACHE_MAX_ENTRIES` (default 200) — expired rows purge at
    store time; oldest-evict only when still over the cap. WAL backups
    include the cache automatically; deleting the DB clears it.
  - kill switch: `JARVIS_DISABLE_RESULT_CACHE=true` restores always-fresh
    behavior (test-pinned).
  - TTLs are env-tunable: `RESULT_CACHE_WEB_TTL_SECONDS` (300),
    `RESULT_CACHE_WIKI_TTL_SECONDS` (86400),
    `RESULT_CACHE_CALC_TTL_SECONDS` (604800),
    `RESULT_CACHE_DEFAULT_TTL_SECONDS` (900).
  - maintenance (bounded; never touches valid entries by default):
    `uv run python -m jarvis.maintenance cache stats` — entries / hits /
    expired / per-tool counts; `cache inspect` — fingerprint-only listing
    (payloads are never shown); `cache cleanup [--expire-older-than-days N]`
    — expired rows first, then (with `--yes` or a confirm prompt) rows older
    than N days, at most 500 per invocation.
  - the API exposes counts only: `GET /ops/cache/stats` (auth'd) — no
    fingerprints, no payloads. The dashboard Operations → Result cache
    section renders the same safe metadata.
  - **v0.27 (multimodal):** `POST /chat/multimodal` accepts multipart
    text + optional image (JPEG/PNG/WebP/GIF ≤10 MB) + optional audio
    (WAV/MP3 ≤25 MB), content-sniffed (magic bytes) and pixel-bounded;
    uploads are stored under random names in `multimodal_upload_dir`
    (default `./jarvis_data/uploads`, inside the file sandbox) and deleted
    after the request — session history keeps only the derived text.
    STT is LOCAL Whisper (needs ffmpeg on PATH for audio upload handling);
    TTS is NETWORK-BACKED Edge TTS (`TTS_ENABLED=false` or CLI `--no-tts`
    to disable); vision is LOCAL llava (`vision_model`). Same auth, rate
    limits and session leases as `/chat` — no separate security model.
  - **v0.26 (grounding guard):** every grounded synthesis turn records
    daily counters (checks / contradictions / corrections / fallbacks,
    per tool) into `grounding_metrics_daily` (same SQLite, same
    retention discipline). Surfaced via `GET /ops/grounding/stats`
    (aggregates only — never answers or evidence text),
    `maintenance cache stats` (`grounding today:`), and the dashboard's
    Answer-grounding (last 14 days) table. Kill switch:
    `JARVIS_DISABLE_GROUNDING_GUARD=true` restores v0.25 synthesis
    behavior (no check, no correction); `GROUNDING_METRICS_RETENTION_DAYS`
    (default 30) bounds the table.
  - **v0.25:** every dispatch also records daily hit/miss/stale/bypass/
    store counters into `cache_metrics_daily` (same SQLite, per-tool
    deltas merged atomically), pruned past
    `RESULT_CACHE_METRICS_RETENTION_DAYS` (default 30). Surfaced via
    `GET /ops/cache/stats/history?days=&limit=` (aggregates only),
    `maintenance cache stats` (`today:` counters + per-tool deltas), and
    the dashboard's Daily cache activity (last 14 days) table. Clients
    can force fresh runs past the cache: `POST /chat` with
    `"refresh": true`, CLI `--refresh` / `/refresh`, or the dashboard
    Refresh-mode toggle. Refresh skips ONLY the cache lookup —
    permissions, confirmations, and repeat suppression still apply.
- **Bounded replanning (v0.24)** — a plan step that structurally failed
  (ERROR result, or a required tool whose every attempt failed) triggers at
  most ONE validated replan per turn within the remaining tool budget.
  New log/SSE signals: `plan_step_failed` (now with `reason=`:
  `step_result_error` or `required_tool_all_attempts_failed`),
  `replan_triggered`, `replan_validated`, `replan_rejected_empty`,
  `plan_complete` SSE (complete=bool, failed_steps, replans). A failed
  replan never masquerades as success: synthesis receives an explicit
  incompleteness directive and must name the unfinished work.
  `plan_ready` now also carries `quality=` — a deterministic plan-quality
  block (steps, toolless steps, unique/repeated tools, forward references,
  duplicate steps, estimated LLM calls, issues) from
  `jarvis/core/plan_quality.py`; alert on high `estimated_llm_calls` or
  repeated-tool issues if you care about efficiency. Suppression is
  per-turn and success-only by design: failed calls are never recorded
  (retry works), state-coupled tools (`get_current_datetime`,
  `recall_facts`, `remember_fact`) are exempt, and the ledger is fresh
  every turn. The benchmark is now 11 cases with a repeat_semantics
  category; the live harness reports per-case suppression counts.
- **Maintenance CLI (v0.18/v0.19)** — schedule `python -m jarvis.maintenance
  doctor` (exit code 0/1 by check results) and alert on non-zero. For
  recovery: `actions --state UNKNOWN`, `unknown-actions`, `sessions
  --expired`, `inspect --session <id> / --action <id>` are read-only;
  `reissue --action <id> --request-id <unique> [--yes]` is the one
  deliberate mutation. Retention: schedule `cleanup --operational
  --dry-run` weekly to preview, then without `--dry-run` (defaults:
  terminal ledger rows 30 d, reissue audit rows 90 d but only when both
  linked actions are gone, expired/orphaned leases 30 d; PENDING/RUNNING/
  UNKNOWN and chain-linked rows are always protected). New v0.18/v0.19 log
  events to ship: `action_marked_unknown`, `action_reissue_requested`,
  `action_reissue_created`, `action_reissue_duplicate_request`,
  `recovered_action_resolved`, `reissue_context_corrupt_degrades`,
  `session_lease_recovered_from_stale`, `cleanup_operational_records`,
  `doctor_check`. New schema: `action_reissues` table (`CREATE TABLE IF NOT
  EXISTS`) and `action_executions.pause_context_json` (`ALTER TABLE` on
  next startup — no manual migration needed).
- **Operator dashboard (v0.19)** — the Streamlit Operations view is a
  client of the same API: give it `JARVIS_CLIENT_API_KEY` (same value as
  the server's `JARVIS_API_KEY`) when auth is enabled, or every Operations
  read renders an authentication message. The dashboard performs no
  background jobs and no writes on render; its only mutation is reissue,
  which requires two explicit interactions and then behaves exactly like
  the API/CLI path (idempotent, ceiling-bounded, audited).
- **Personal knowledge base (v0.20)** — documents ingest into
  `VECTOR_DB_PATH` under a dedicated `knowledge_base` Chroma collection
  (plus `knowledge_documents` rows in `DB_PATH`; both created lazily —
  no manual migration). Path boundary: ingestion accepts only files whose
  resolved path is inside `FILE_READER_ALLOWED_DIR` (default `.` — set it
  to a dedicated documents directory in production so the agent cannot
  index the whole checkout). Credential-like files are refused outright.
  New dependency: `pypdf` (PDF text extraction, page metadata). New API
  surface: `/knowledge/documents`, `/knowledge/documents/{id}`,
  `/knowledge/ingest`, `/knowledge/search` — all behind the same auth +
  rate limiting; ingest/delete are mutating and 401 without the key.
  New log events: `knowledge_ingested`, `knowledge_ingest_unchanged`,
  `knowledge_reindex_stale_chunks_removed`, `knowledge_document_removed`,
  `knowledge_chunks_added/deleted`, `knowledge_search`,
  `knowledge_api_ingest/remove`, `knowledge_ingest_refused`.

## 7. Live-model evaluation procedure

The 32-case harness grades the real agent loop against a live Ollama model
(tool side effects mocked; the permission guard stays real). A case timeout
is an ABANDONMENT (thread join), not a kill of the model call — budget
per-case time accordingly (`--timeout`, default 120 s; allow ≥240 s for a
cold CPU-only model whose first calls include load time).

```bash
# 0. Preconditions
ollama serve &                      # or the desktop app
ollama pull qwen2.5:7b              # settings.ollama_model

# 1. Preflight + inventory (no model calls)
uv run python evaluation/run_evals.py --list

# 2. Full live run with a machine-readable report
uv run python evaluation/run_evals.py --timeout 240 --json eval-report.json

# 3. Compare against a prior run (regression diff)
uv run python evaluation/run_evals.py --timeout 240 \
    --json eval-report-new.json --compare eval-report.json
```

Exit codes: `0` all cases passed, `1` failures (details per case incl.
expected-vs-called tools and grader verdicts), `2` usage error or Ollama
unreachable (fail-fast, no 32-failure cascade).

**What the numbers prove:** tool-selection and refusal/continuity behavior
under lexical graders — deterministic, cheap, but NOT semantic correctness.
There is no LLM judge; a passing response can still be subtly wrong in ways
the textual patterns miss. Treat the suite as a regression tripwire, not a
quality guarantee.
