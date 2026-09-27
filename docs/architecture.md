# JARVIS v0.16 — Architecture

JARVIS is a local-first AI assistant organized as **one agent runtime and several thin interfaces**. This document describes the system as it actually exists in v0.16: the layered structure, the request lifecycle, the context-management and safety models, the code-execution sandbox, the service surface, and the deployment path.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Interfaces (choose one per process)                                  │
│   CLI (main.py)   REST API (api/app.py, FastAPI)   Dashboard (Streamlit)   Voice │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  build_runtime() + JarvisRuntime
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ JarvisRuntime (runtime.py) — single assembly point                   │
│   start_session / chat / handle_confirmation / describe              │
└───────┬──────────────────────┬──────────────────────┬────────────────┘
        ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
│ Orchestrator     │  │ ToolRegistry     │  │ PermissionGuard         │
│ (core/)          │  │ (tools/)         │  │ (core/permissions.py)   │
│ route → plan →   │  │ 11 active tools  │  │ SAFE/NETWORK/FILE_READ  │
│ execute → synth  │  │                  │  │ auto; SYSTEM/DESTR need │
└──────┬───────────┘  └──────────────────┘  │ confirmation            │
       ▼                                    └─────────────────────────┘
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
│ ContextManager   │  │ SessionStore     │  │ LLM client              │
│ (memory/)        │  │ (memory/, SQLite)│  │ (llm/) LiteLLM → Ollama │
└──────────────────┘  └──────────────────┘  └─────────────────────────┘
```

---

## 1. Layers

| Layer | Location | Responsibility | Talks to |
|---|---|---|---|
| Interfaces | `main.py`, `jarvis/api/`, `ui/`, `voice/` | User I/O only. Build a runtime, translate between UI events and `JarvisRuntime` methods. | `runtime.py` |
| Runtime assembly | `jarvis/runtime.py` | Wire store + tools + guard + orchestrator; own session lifecycle; expose introspection (`describe()`). | all core layers |
| Agent core | `jarvis/core/` | Orchestration (route → plan → execute → synthesize), planning, permissions. | memory, tools, llm |
| Tools | `jarvis/tools/` | Capabilities: search, scrape, files, calculator, memory, vision. Each tool declares a `risk_level`. | filesystem / network |
| Memory | `jarvis/memory/` | SQLite session persistence (thread-safe), `ContextManager` (windowing/anchoring/summarization/clamping), ChromaDB vector store. | disk |
| LLM | `jarvis/llm/` | LiteLLM wrapper. Every model call goes through `chat_completion`. | Ollama / any LiteLLM provider |

**The rule that keeps this clean:** interfaces never instantiate the Orchestrator, SessionStore, or tools directly. They call `build_runtime()` and use the returned `JarvisRuntime`. This means the CLI, REST API, and Streamlit dashboard are guaranteed to present the identical capability surface, because there is exactly one wiring function.

Legacy exception: `main._build_orchestrator()` still exists as a shim returning the runtime's components for old callers (evals, some tests); it is not how new code should wire JARVIS.

---

## 2. Request Lifecycle

Every user message, regardless of interface, takes the same path:

1. **Persist** — the user turn is saved to SQLite before any model call.
2. **Load context** — `SessionStore.load_history(limit=max_context_messages * 3)` fetches a window wider than the prompt budget; `ContextManager.build_messages` compacts it (windowing → anchoring → summarization) and injects the system prompt.
3. **Route intent** — `Orchestrator.route_intent` is a heuristic, zero-cost keyword/length router:
   - `simple` → skip the planner, run a short ReAct loop directly (`max_rounds=2`).
   - `complex` → full Plan → Execute → Synthesize.
   - Known trade-off (documented in AGENTS.md): keyword matching has false positives and false negatives. This intentionally trades classification accuracy for zero latency and zero token cost.
4. **Plan** (complex only) — the `Planner` decomposes the request into ≤ `MAX_PLAN_STEPS` (5) discrete steps with `required_tools`, validated against the actual registry.
5. **Execute** — each step runs a mini ReAct loop with its own budget (`MAX_TOOL_ROUNDS_PER_STEP = 2`) inside a global budget (`MAX_TOOL_ROUNDS = 5`). Tool calls within a round dispatch concurrently (`asyncio.gather`). Tool errors trigger a self-correction sub-loop (`MAX_SELF_CORRECTION_ATTEMPTS = 2`) with a recovery hint injected into the prompt.
6. **Synthesize** — one final tool-free call turns step results into the user-facing answer, with quality rules baked into the prompt (lead with the answer, quote tool numbers exactly, never paper over failed steps).
7. **Persist** — assistant message saved; text returned to the interface.

### Confirmation flow (high-risk tools) — pausable turns (v0.15)

When a tool's risk tier requires confirmation (see §4), dispatch does **not** execute it. Instead the turn PAUSES:

1. The pending action **plus a durable resume context** (original request, pending plan, completed steps, remaining round budget, execution mode) is persisted to SQLite (`save_pending_confirmation(context=...)`).
2. The orchestrator returns `PAUSED_FOR_CONFIRMATION` to the interface — control returns to the user; nothing else about the turn is lost.
3. The user confirms or denies via any interface (`/confirm` `/deny` in CLI, `POST /sessions/{id}/confirm` in the API, buttons in the dashboard).
4. `handle_confirmation` pops the confirmation (atomically — restart-safe), executes the tool on approval (or records the denial), persists the result as a normal `tool` message, restores the pause context, executes the REMAINING plan steps, and synthesizes the final answer.

Handled states: approval, denial, approved-but-failed execution (the error flows into synthesis like any tool error), expired confirmation (cannot execute), rows without context (pre-v0.15 — raw-result reply), corrupt context (graceful fallback), and a SECOND high-risk action during resume (re-parks with refreshed durable state). The turn never blocks waiting for approval mid-loop; control returns immediately.

Durable schema: `pending_confirmations.context_json` (added via `ALTER TABLE` migration for pre-0.15 databases).

### Lifecycle events (observability seam)

`Orchestrator.chat` accepts an optional `on_event` callback. It fires synchronously at each transition: `intent` → (`plan` → `step_start` / `tool_calls` / `step_done` per step) → `synthesis` on the complex path; a single `intent` on the fast path. Callback exceptions are swallowed — observation must never break execution. The SSE endpoint (`POST /chat/stream`) and structured logging both consume this seam; `response_ready` logs carry `duration_ms` per turn.

### Quality loop (evaluation/run_evals.py)
The harness runs the real orchestrator against a live Ollama model with tool dispatch mocked — it grades *decisions*, not network access. Runner features: Ollama pre-flight (exit 2 with a clear message — no 32-failure cascade), per-case timeouts, `--filter`/`--category` selection, a category pass-rate breakdown (ambiguity, continuity, injection, instructions, multi_step, no_tool, recovery, refusal, robustness, tool_selection), per-failure details (expected vs actually-called tools, grader verdicts), and `--json` reports (schema v2: per-case `case_id` content digest, `expected` summary, `failure_reason`/`failure_detail`, durations). `--compare PRIOR_JSON` diffs two reports deterministically (newly failing / newly passing / unchanged failures, keyed by the case-content digest so renames survive).

**Timeout semantics (honest):** the per-case timeout is a thread `join()` — it ABANDONS the case, it does NOT terminate the in-flight model call (Python cannot kill a running thread). The worker closes its runtime in its `finally` and dies with the process. For a single-process evaluation harness this is acceptable and deliberate; it is not workload termination (the code-execution sandbox, by contrast, terminates for real — see §4).

**Grader honesty:** all graders are deterministic/LEXICAL string heuristics — there is no LLM judge in this suite. A grader proves a textual pattern ("the computed number appears", "a refusal marker is present"), not semantic correctness; false positives and false negatives on both sides are possible and documented per grader in `tests/test_eval_harness.py::TestGraderAudit`. This is an intentional trade-off: zero grader cost and variance, at the price of no semantic judgment.

CI runs the harness nightly on self-hosted Ollama runners as a non-blocking signal.

---

## 3. Context Management (memory/context_manager.py)

Long sessions must not blow up the prompt. The ContextManager enforces four mechanisms, applied when building any LLM-facing message list:

- **Windowing** — only the most recent `max_context_messages` (24) turns are fed to the model, regardless of session length.
- **Anchoring** — the session's original task is pinned so the model retains the overarching goal even when early turns fall out of the window.
- **Rolling summarization** — dropped older turns are compacted into a `[Context summary]` digest injected as a system message. Summarization is deterministic and offline (no extra LLM call).
- **Tool-output clamping** — any tool result exceeding `max_tool_output_chars` (6000) is head+tail-clamped with an explicit `…[N characters omitted]…` marker *for the prompt only*; the full payload is already in SQLite.

Failure-mode note: summarization is deterministic (rule-based), not LLM-based, so it never fails, never costs tokens, but is less fluent than an LLM summary. That is the current trade-off.

---

## 4. Safety Model

Defense runs in four independent layers. The guiding principle: **dangerous capability must be absent or gated, never merely discouraged via prompt.**

### Layer 1 — Absent tools
`computer_control` is **never registered** — it is a pure placeholder (no OS automation imports) that fails closed if instantiated. `execute_python_code` joins the registry **only** when both config and reality agree:

1. `ENABLE_CODE_EXECUTION=true` in config (default **false**), and
2. `DockerCodeSandbox.is_available()` returns True — docker CLI present, daemon reachable, supported host, and the image already pulled locally.

If enabled but Docker is unusable, the tool stays **absent from the registry**: the LLM never sees its schema, and a startup error is logged (`code_execution_enabled_but_docker_unavailable`).

### Layer 2 — Fail-closed tool gate
`CodeExecutionTool` refuses any sandbox whose `provides_isolation` is not True. Defense in depth: even a constructed-with-unavailable-Docker sandbox is degraded to `DisabledSandbox` at tool construction (`is_available()` is verified at build time; a False result swaps in the disabled sandbox).

### Layer 3 — PermissionGuard risk tiers
Every tool declares a `risk_level`:

| Tier | Meaning | Behavior |
|---|---|---|
| `SAFE` | No side effects, no network | Auto-allowed |
| `NETWORK` | External HTTP | Auto-allowed |
| `FILE_READ` | Reads local files (sandboxed to `FILE_READER_ALLOWED_DIR`) | Auto-allowed |
| `FILE_WRITE` | Modifies local files | **Blocked** (`is_allowed` → False) |
| `SYSTEM` | Shell / OS automation | **Confirmation required** |
| `DESTRUCTIVE` | Irreversible | **Confirmation required** |

Note the FILE_WRITE subtlety: the guard auto-blocks it outright (returns "ERROR: Tool 'write_file' is not permitted"), while only SYSTEM/DESTRUCTIVE enter the interactive confirmation flow. `REQUIRE_CONFIRMATION_FOR_HIGH_RISK=True` gates the confirmation path globally in `config.py`.

### Layer 4 — Docker-isolated code execution (jarvis/core/sandbox.py)
The sandbox is a contract plus two implementations:

- `ExecutionRequest` / `ExecutionResult` — validated contract; `to_report()` renders an LLM-readable report; denials become `ERROR: Code execution denied (reason)`.
- `CodeSandbox` ABC — `provides_isolation: bool`; `execute()` returns denials instead of raising for expected refusals.
- `DisabledSandbox` — denies everything (`denial_reason="disabled"`); the default.
- `DockerCodeSandbox` — **real container isolation**, implemented over the docker CLI (no SDK dependency):

| Property | Enforcement |
|---|---|
| No network | `--network none` |
| Immutable root filesystem | `--read-only` + noexec tmpfs `/tmp` |
| No privilege | `--cap-drop ALL`, `--security-opt no-new-privileges`, non-root `--user 65534:65534` |
| Resource ceilings | `--memory` = `--memory-swap` (no swap escape), `--cpus`, `--pids-limit 64` |
| One-shot | `--rm`; fresh container per execution |
| Code delivery | temp file (mode 0600) bind-mounted `:ro` — never shell interpolation |
| Output cap | stdout/stderr truncated to `max_output_bytes` |

Fail-closed ladder inside `execute()`: docker CLI missing → `docker_unavailable`; image not pulled locally → `docker_unavailable` (the sandbox **never pulls at runtime**); request invalid → specific denial reason; timeout → `timeout`; any unexpected exception → `sandbox_error: ...`. It never executes on the host.

**Timeout enforcement (since v0.15, three distinct layers):**
- **B — container timeout (primary):** the entrypoint is `timeout <cap>s python3 script.py` (coreutils). The WORKLOAD is killed inside the container — exit code 124, definitive termination. This closes the classic gap where a host-side wait leaves the container burning CPU.
- **A — host process timeout (secondary):** the host abandons the docker CLI wait at cap + 5 s slack (guards against CLI/daemon hangs).
- **C — actual termination on layer A:** the container is force-removed (`docker rm -f jarvis-sbx-exec`), so even a layer-A timeout cannot leave an orphan workload.

`ExecutionResult.timeout_layer` distinguishes `container` (workload killed at the boundary) from `host_kill` (host gave up and cleaned up); `denial_reason` is `timeout` in both cases.

**Image policy (enforced by the validator):** bare names and `latest` are rejected — a mutable tag can be re-pushed with different contents, silently changing what a security boundary runs. Concrete tags (`ubuntu:24.04`) and `sha256:` digest refs (recommended) are accepted. A dedicated minimal image ships in `deploy/Dockerfile.sandbox`: interpreter + stdlib + a dedicated unprivileged user only, built from a **digest-pinned base** (the `python:3.12-slim` digest was resolved by an actual `docker pull` against a Linux daemon on 2026-09-27 — see `deploy/README.md` §1).

Host limitations: on Windows, `is_available()` returns False (Docker Desktop cannot enforce the Linux hardening flags for Windows containers; use WSL2-backed docker on a Linux host or in CI). Availability checks shell out, so they run once at startup — not per execution.

#### Verification status of the sandbox boundary (v0.16)

The three claims below are deliberately separated. Do not blur them when editing this file.

- **CONFIRMED (real execution, Linux Docker daemon, 2026-09-27):** the *production* image built from `deploy/Dockerfile.sandbox` was observed, from inside real containers, enforcing: non-root uid 65534; `setuid(0)`/`chown(0,0)` refused (capabilities dropped + no-new-privileges); read-only rootfs (`/etc`, `/usr`, `/home`, the `:ro` code mount and traversal paths all refuse writes); `/tmp` writable but noexec; no outbound network (connect fails); a tiny container-local `/proc`; the `pids-limit` ceiling bounding concurrent forks (≤64); a memory-hog process killed under `--memory`/`--memory-swap`; the container `timeout` wrapper killing an infinite loop at the cap (exit 124) with no orphan container left; the layer-C `docker rm -f` helper actually removing a live container; output caps and per-run tmpfs isolation (no file leakage between executions).
- **TEST-VERIFIED (mocked construction):** the exact docker CLI argument construction, validation ladder, fail-closed denials, and timeout-layer reporting are unit-tested offline (`tests/test_docker_sandbox.py`, `tests/test_sandbox_timeout_layers.py`) without requiring Docker.
- **ENVIRONMENT-BLOCKED:** the Windows development host cannot run JARVIS's code-execution *tool* end-to-end by design — `is_available()` refuses win32 hosts even with a working Linux-engine daemon (observed: it returned False with the daemon up). Runtime verification above was performed by driving `DockerCodeSandbox` and raw `docker run` directly, not through the enabled registry tool.

Remaining gaps are listed in `deploy/README.md` §1 (what one machine's verification cannot prove: kernel/container-runtime CVEs, host kernel hardening, multi-tenant pressure, other runtimes).

**Rule for contributors:** no `exec()` / `eval()` on model-influenced strings, ever. The v0.7-era in-process sandbox was removed for exactly this reason.

---

## 5. Service Architecture

v0.10 hardens the service direction: the agent runtime is a library object; interfaces are swappable.

### JarvisRuntime (jarvis/runtime.py)
`build_runtime()` assembles `SessionStore` → 11 tools (+ conditionally the code-execution tool) → `PermissionGuard` → `Orchestrator` and returns a `JarvisRuntime` exposing:

- `start_session()` — create a session, bind long-term memory to it
- `chat(session_id, user_input, on_event=None)` — one full turn, optionally observed
- `handle_confirmation(session_id, confirmed)` — resolve pending high-risk action
- `get_pending_confirmation(session_id)`
- `session_exists(session_id)`
- `describe()` — version, model, planner model, tool list, disabled tools, DB path (feeds `/health` and startup banners)
- Context-manager support (`with build_runtime() as rt: ...`) for clean teardown

### REST API (jarvis/api/)
FastAPI app (`jarvis.api.app:app`), thin over the runtime:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Runtime introspection + posture (`auth_enabled`, `code_execution`) — no auth |
| `/sessions` | POST | Start a session → `{session_id}` |
| `/chat` | POST | `{message, session_id}` → `{response, pending_confirmation?}` |
| `/chat/stream` | POST | **SSE** stream: `begin` → `intent`/`plan`/`step_*`/`synthesis` → `done` (or `error`); events interleaved live, 15s keepalive comments |
| `/sessions/{id}/history` | GET | Persisted message history |
| `/sessions/{id}/confirm` | GET | Check pending confirmation |
| `/sessions/{id}/confirm` | POST | `{confirmed: bool}` → resolve it |
| `/tools` | GET | Active tool surface with risk tiers |

Design decisions:
- **Authentication (opt-in):** set `JARVIS_API_KEY` to require `Authorization: Bearer <key>` (or `X-API-Key`) on every endpoint except `/health` and OpenAPI metadata. Unset = local-only trust (the default). Comparison is constant-time (`hmac.compare_digest`); failures log the client and return 401 with `WWW-Authenticate: Bearer`.
- **Rate limiting (in-process):** sliding window per client — keyed by API key when auth is on, else client IP. `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` (default 60/60s; `0` disables). Over-limit → `429` with `Retry-After`. `/health` is exempt. Per-process only: multi-replica deployments should enforce limits at the reverse proxy.
- **Per-session serialization:** `JarvisRuntime.chat` holds a non-blocking per-session mutex. A second concurrent turn on the same session raises immediately, mapped to **409 Conflict** by the API — history can no longer interleave and confirmations cannot double-resolve. Different sessions are fully parallel. (Runtime-local only; multi-process deployments need an external lock.)
- **Dependency injection via `set_runtime()`/`get_runtime()`** — tests inject a runtime built with an in-memory store and a mocked LLM; production calls `build_runtime()` at startup.
- **Error mapping** — unknown session → 404; busy session → 409; runtime/LLM failure → 503 with a readable message; the SSE endpoint carries errors in-band as a terminal `error` event (HTTP 200).
- **Schemas** (`api/schemas.py`) are explicit Pydantic models — the API contract is typed, not inferred.
- **Thread safety** — the API is multi-threaded (uvicorn workers); `SessionStore` serializes all SQLite access with an `RLock` (and clears its history cache inside the lock, so readers never see stale windows) and opens connections with `check_same_thread=False`.
- **Observability** — one structured log line per request (`api_request`: request_id, method, path, status, duration_ms) via middleware; every response echoes `X-Request-ID` and `/chat` bodies carry the same `request_id`. Runtime errors are **sanitized**: clients get a generic message pointing at the request id; exception internals (paths, hostnames) stay in the log under that id. On top of the plan/step logs and the `on_event` lifecycle seam.
- **Deep health** — `/health` executes a real DB read+write round-trip (probe row inserted and deleted; residue swept) and returns **503** with `status: degraded` when persistence is broken. LLM reachability is deliberately *not* part of `/health` — sessions, history, and confirmations still work while Ollama is down; the `maintenance doctor` command checks it separately.

### API client (jarvis/api/client.py)
A dependency-free (stdlib `urllib`) client over the whole surface, including SSE parsing, typed errors (`SessionConflict` for 409, `RateLimitedError` with `retry_after` for 429, `JarvisClientError` otherwise), and a `get_confirmation()` that maps 404 → `None`. **The Streamlit dashboard is built on it:** with `JARVIS_API_URL` set, the UI is a pure client and the agent runtime runs only in the API server process — the dashboard can live on a different machine. Empty URL falls back to the legacy in-process runtime for single-machine setups. The CLI still embeds the runtime directly (lowest local latency); both consume the same underlying agent.

Run it:

```bash
uv run uvicorn jarvis.api.app:app --port 8000
# Interactive docs at http://localhost:8000/docs

# Streaming example:
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Authorization: Bearer $JARVIS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Plan a trip and then remember it"}'
```

The CLI (`jarvis` / `python -m jarvis.main`) and Streamlit dashboard (`streamlit run ui/dashboard.py`) keep working unchanged — they consume the same runtime.

### Deployment path (v0.12)

- **`Dockerfile`** — the API service image: slim base, uv-managed deps, non-root `jarvis` user, `/data` volume conventions (`DB_PATH=/data/jarvis.db`, `VECTOR_DB_PATH=/data/chroma_db`), built-in healthcheck against `/health`.
- **`docker-compose.yml`** — `api` service plus an optional `local-llm` profile running Ollama; `jarvis-data` volume persists the DB and vector store across restarts; healthchecks on both services; a documented (commented-out, Linux-only) docker-socket mount for enabling sandboxed code execution from the container.
- **`deploy/Dockerfile.sandbox`** + **`deploy/build-sandbox-image.sh`** — the dedicated sandbox image and its digest-pinning build script (the validator rejects mutable tags; the script prints the exact `SANDBOX_IMAGE` value and refuses to build until the base digest placeholder is replaced with a verified one).
- **`.github/workflows/ci.yml`** — offline pytest matrix (3.11/3.12), API image build, sandbox image build gated on the digest pin.
- **`jarvis/maintenance.py`** — ops CLI (`stats`, `cleanup --days N`, `expire-confirmations`, `doctor`) to schedule and triage against a deployed volume; the server deliberately runs no background jobs. `doctor` verifies DB read+write, Ollama reachability, sandbox posture, and pending-confirmation state, exiting non-zero on failure for alerting.
- **WAL persistence** — file-backed SQLite runs in WAL mode (`synchronous=NORMAL`): history reads no longer block during chat-turn commits, and the DB survives hard process death. `:memory:` keeps the default journal mode.

Quick start: `cp .env.example .env` (set `JARVIS_API_KEY`), then `docker compose up -d api` with `OLLAMA_BASE_URL` pointing at the host Ollama — or `docker compose --profile local-llm up -d` for a self-contained stack.

---

## 6. Adding a New Tool (Checklist)

1. Create `jarvis/tools/my_tool.py`; subclass `BaseTool`.
2. Define `name`, `description`, `parameters` (JSON Schema), `risk_level`, and `run()` (return a string; errors start with `ERROR:`).
3. Add the class to `_TOOL_FACTORIES` in `jarvis/runtime.py` — **this is the single registration point**, not `main.py`. (Capability-gated tools like code execution instead go through a builder such as `_build_code_execution_tool`.)
4. Export it from `jarvis/tools/__init__.py`.
5. If the model needs usage guidance, add one line to the Tool Selection Guide in `jarvis/config.py::system_prompt`.
6. Write tests in `tests/test_tools.py` (behavior + risk level).

Nothing else changes: the CLI, API, and dashboard pick it up automatically because they all share `build_runtime()`.

---

## 7. Testing Philosophy

- Tests are **offline** by default: no Ollama, no network. `chat_completion` is patched at the orchestrator boundary (`patch("jarvis.core.orchestrator.chat_completion")`); LiteLLM response objects are faked with minimal shape-compatible classes.
- `tests/conftest.py` points `DB_PATH` at `:memory:` and `VECTOR_DB_PATH` at a throwaway path; API/runtime tests patch `jarvis.runtime.get_vector_store` so no ChromaDB directory is created.
- Safety-critical behavior (permission tiers, disabled tools, sandbox fail-closed contract) has dedicated tests in `tests/test_security_critical.py` — these must stay green before any change to the safety model.
- **Docker integration tests are a separate layer**: `tests/test_sandbox_integration.py` drives a REAL Linux Docker engine (builds the production image, asserts observed container behavior from inside). They skip cleanly with a printed reason when the docker CLI, a Linux daemon, or the pinned digest is missing — a skip is never a pass.
- **The evaluation harness has its own offline unit tests**: `tests/test_eval_harness.py` covers the case inventory, grader behavior, preflight/CLI exits, report shape, the abandonment-timeout contract, and the regression comparison — the live model itself is never contacted.

---

*Documented for v0.16.0. If this file and the code disagree, fix one of them in the same commit.*
