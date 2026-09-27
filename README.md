# JARVIS

> A **Local-first, privacy-oriented, and free of paid model APIs by default** AI assistant — powered by Ollama. No cloud.

**JARVIS v0.20.0** — a **local-first agentic AI assistant with a closed, *inspectable*, *recoverable* reliability loop — now grounded in your own documents**. One agent runtime (Plan-and-Execute loop, 12 tools, permission tiers, *resumable* durable confirmations, context management, self-correction) behind a hardened **REST API (FastAPI)**: auth, rate limiting, per-session serialization, request-ID correlation, deep health checks, truly interleaved SSE streaming — deployed via docker-compose with WAL-backed persistence, a **runtime-verified** workload-timeout-enforced sandbox, and CI with a documented trust boundary. v0.20 adds a **personal knowledge base**: explicitly ingest your PDFs/Markdown/text/code/JSON into a dedicated Chroma collection (separate from personal memory), retrieve with `search_knowledge`, and get answers with real citations — with document text treated strictly as untrusted evidence, never as instructions. Computer control remains structurally disabled.

---

## ✨ What's New in v0.20.0

- 📚 **Personal knowledge base + RAG** — a clear architectural split: **personal memory** (facts about you, `remember_fact`/`recall_facts`, unchanged) vs **document knowledge** (files you explicitly ingest into a dedicated `knowledge_base` Chroma collection with its own SQLite registry).
- 🖹 **Safe ingestion** — explicit paths only (never crawling): TXT, Markdown, source code, JSON (stdlib) and **PDF with page-level metadata** (new `pypdf` dependency). Resolved paths must live inside `FILE_READER_ALLOWED_DIR`; credential-like files (`.env`, `*.pem/*.key`, secret-named text) are refused unconditionally; unsupported/oversized/empty files fail with clear reasons.
- 🧩 **Deterministic chunking** — paragraph-first packing (target 1200 chars, overlap 150), lossless hard-split for oversized paragraphs, stable chunk ids, and full trace-back metadata (document_id, source, filename, page, line range, chunk index, section, content hash).
- 🔁 **Incremental by content hash** — unchanged file → skipped (no re-embedding); identical bytes at a new path → one shared index entry; changed file → old chunks deleted and replaced in one pass (no stale chunks, no orphaned registry rows).
- 🎯 **`search_knowledge` tool** — bounded retrieval (top_k ≤ 20, evidence ≤ 6000 chars) with citation-ready results. Document evidence is returned inside explicit `DOCUMENT EVIDENCE START/END` framing marked as data, never instructions — prompt-injection text stays quoted, and the tool honestly reports `NO_RELEVANT_EVIDENCE` instead of inviting fabrication.
- 🧪 **32 deterministic RAG tests** (`tests/test_knowledge_rag.py`) — ingestion lifecycle incl. duplicate/reindex/delete, PDF page metadata, citation derivation from metadata only, adversarial-document framing, memory/knowledge separation, path-security refusals, API lifecycle over a real uvicorn server, and auth on all knowledge routes. Plus a test-isolation hardening: an autouse offline guard now fails fast (with a clear diagnosis) if any test tries to reach the real LLM.
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.19.0

- 🖥️ **Operations dashboard** — the Streamlit UI gained an Operations view: recent actions (state/session filters), UNKNOWN actions with safe recovery guidance, session leases (redacted owners, staleness), and per-session causal timelines. Viewing never mutates; the ONLY mutation is reissue, behind an acknowledgement checkbox **plus** a final confirmation dialog. All API failures (unreachable, 401, 404, 409, 429) render as operator-level messages — the page never crashes.
- ♻️ **Full-context UNKNOWN recovery** — the resume context (original request, plan, completed steps, step number, remaining tool budget, mode) is now captured onto the ledger row at park time and copied onto a reissued action's confirmation (transitively across reissue chains). Approving a recovered action dispatches it exactly once, then **continues the original task** and synthesizes the final answer. Denial dispatches nothing and is recorded. Nested confirmations during recovery work and are themselves recoverable; a second UNKNOWN mid-recovery reissues safely.
- 📜 **Session timelines** — `GET /sessions/{id}/timeline`, `python -m jarvis.maintenance inspect --session <id> [--json]`, and a dashboard view merge messages, confirmation parks, action states, reissues, and leases into one bounded, chronological, safe-metadata-only stream (never tool args, result bodies, message content, or raw owner tokens).
- 🧹 **Retention + dry-run** — `cleanup --operational --dry-run` reports (without deleting) what would age out: terminal ledger rows past `--terminal-actions-days` (default 30), reissue audit rows whose BOTH linked actions are gone past `--reissues-days` (default 90), and expired/orphaned leases past `--leases-days` (default 30). PENDING/RUNNING/UNKNOWN actions, active confirmations, and any audit row still linking live actions are always protected.
- 🧪 **38 new tests** (`tests/test_full_recovery.py`, `tests/test_operator_experience.py`): the full recovery lifecycle against the real orchestrator (approve/deny/nested/second-UNKNOWN/budget/malformed context/missing session/duplicates/ceiling), timeline ordering/isolation/bounds/leak-proofing, retention protection rules incl. dry-run, the new endpoints served by a **real uvicorn server** consumed by the **real client**, and dashboard-backend parity. `reissue` now also accepts an optional `request_id` (server generates + echoes one otherwise).
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.18.0

- 🔎 **Action ledger introspection** — the execution ledger is now queryable: `GET /actions` (filter by `state`/`session_id`, bounded `limit` ≤ 500) and `GET /actions/{action_id}` return safe metadata only (ids, tool name, risk level, state, timestamps, reissue depth) — never tool arguments or result bodies. Store-level: `list_action_executions`, `count_action_executions_by_state`, deterministic newest-first ordering.
- 📟 **Session-lease introspection** — `GET /sessions/leases` (and `python -m jarvis.maintenance sessions`) shows which sessions are leased, by a **redacted** owner (`component:rand8` — raw `host:pid:…` tokens never leave the store), until when, at which fencing token, and whether the lease is active or stale.
- ♻️ **Explicit UNKNOWN-action recovery (reissue)** — an UNKNOWN action can be deliberately re-issued via `POST /actions/{id}/reissue` or `python -m jarvis.maintenance reissue`. This is **not** a retry and never mutates the original: it mints a **new action id** + durable confirmation (so the new action passes the normal permission/confirmation flow), records an audit row (`action_reissues`), is **idempotent per request id** (same request twice → same new action), and is **bounded** (`MAX_REISSUES_PER_ACTION = 3` per original). Requires auth exactly like any mutating endpoint.
- 🩺 **Expanded `doctor` + maintenance CLI** — `python -m jarvis.maintenance` gained `actions`, `unknown-actions`, `sessions` (all read-only, `--json` available) and the `reissue` mutator; `doctor` now also checks the configured Ollama model's availability, stale leases, and UNKNOWN-action counts, with hints pointing at the matching inspection command. Inspection never mutates; only `reissue` does.
- 🧪 **73 new tests** (`tests/test_operator_introspection.py`): introspection filters/pagination/safe-metadata, lease staleness + redaction, the full reissue lifecycle (new identity, original stays UNKNOWN, approval dispatches exactly once, denial never executes, idempotent replay, ceiling, audit chain, unauthenticated reissue → 401, wrong state → 409), doctor scenarios, and CLI output/exit codes.
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.17.0

- 🔁 **Durable action idempotency (execution ledger)** — every confirmed high-risk action now has a server-generated `action_id` and an explicit state machine in SQLite (`PENDING → RUNNING → SUCCEEDED/FAILED`, plus `UNKNOWN`). The execution claim is an atomic database conditional: duplicate approvals, retries, concurrent requests, and restarts can no longer double-dispatch a tool. A crash between dispatch and result recording surfaces as `UNKNOWN` — **never automatically re-executed** — with an explicit user-facing report. (At-most-once *automatic dispatch*; external side effects are honestly not claimed to be exactly-once.)
- 🗄️ **Multi-process session coordination without Redis** — per-session turns are now guarded by a two-layer scheme: the in-process mutex (fast) plus a database-backed **session lease** (TTL, owner-checked renew/release, fencing tokens, stale takeover after a crash). Two JARVIS processes sharing one SQLite file can no longer both process the same session; sessions recover automatically after a process dies.
- 🚦 **Database-backed rate limiting (opt-in)** — the sliding-window limiter gained an SQLite backend (`make_durable_limiter`): one `BEGIN IMMEDIATE` transaction per decision, so multiple processes on one database enforce a single limit per client. The in-memory default is unchanged.
- 🐳 **Per-run sandbox container identities** — every code execution runs in `jarvis-sbx-<random>` (previously one fixed name): concurrent executions no longer collide or serialize, timeout cleanup targets exactly the offending container, and `ExecutionResult.container_name` + structured logs make each run traceable. **Seccomp posture documented and verified**: containers run under Docker's builtin default seccomp profile (`name=seccomp,profile=builtin` — observed on the live engine), never weakened.
- 🧪 **Reliability regression suites** — 24 action-idempotency tests (including an 8-thread claim race and restart-between-park-and-approval), 10 session-concurrency tests (including true multi-process `ProcessPoolExecutor` lease tests), 13 rate-limiter tests (including cross-process limits), and new real-Docker concurrent-sandbox integration tests.
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.16.0

- 🐳 **Sandbox base image digest genuinely pinned** — the `python:3.12-slim` digest in `deploy/Dockerfile.sandbox` was resolved by an actual `docker pull` against a Linux daemon and the production image built from it (previously a fail-closed placeholder). CI now really builds and smoke-runs the sandbox image.
- 🔬 **Real-Docker integration suite** — `tests/test_sandbox_integration.py` drives a real Linux engine and asserts *observed* behavior from inside containers: non-root uid, dropped capabilities, read-only rootfs, noexec tmpfs, no network, bounded PIDs, memory-cap kills, workload-killing timeouts (exit 124, no orphans), and the layer-C force-removal mechanism. Skips cleanly (with a reason) when Linux Docker is unavailable; a skip is never a pass.
- 🤖 **Live-model evaluation actually run** — the full 32-case suite executed against a real `qwen2.5:7b` via Ollama 0.34.1: **23/32 passed (71.9%), zero timeouts** on the final run (20/32 before the harness/grader fixes — the regression diff is exactly what `--compare` now reports). Failures were classified (agent defects vs grader gaps vs model variance); the real defects were fixed: an absent-tool fabrication hole in the system prompt (model silently computed `print(2+2)` instead of refusing) and an eval-harness mock that masked unregistered-tool calls.
- 📊 **Sharper evaluation signal** — machine-readable reports upgraded (schema v2: per-case content digest, expected-vs-actual tool detail, failure reasons/details, durations); `--compare PRIOR_JSON` regression diffing (newly failing / newly passing / unchanged, rename-surviving via content digests); the context-budget check now measures the prompt-side window it always claimed to check. Grader semantics audited; lexical-≠-semantic limitation documented.
- ⏱️ **Honest timeout semantics** — the per-case eval timeout is documented as a thread-join abandonment (the in-flight model call is NOT terminated); the code-execution sandbox remains the real workload terminator (exit 124, force-removal — both now observed against a live daemon).
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md` § Verification status](docs/architecture.md).

## ✨ What's New in v0.15.0

- ⏱️ **Sandbox timeout enforced at the workload boundary** — the container entrypoint is `timeout <cap>s python3 …`, so a runaway snippet is killed *inside* the container (exit 124), not merely abandoned by the host. If the host-side wait times out first (hung CLI/daemon), the container is force-removed (`docker rm -f`). `ExecutionResult.timeout_layer` reports which layer fired: `container`, `host_kill`, or none. All isolation flags unchanged.
- 🔁 **Confirmation approval now resumes the original task** — approving a high-risk action executes it, persists the result, restores the paused plan from SQLite, executes the remaining steps, and synthesizes a final answer. Works after a process restart. Denials also continue the task (minus the denied action). Expired/legacy/corrupt pause states degrade safely.
- 🛡️ **CI trust boundary documented and enforced** — workflow `permissions: contents: read`, nightly eval job schedule-only with no persisted credentials, `CODEOWNERS` on workflows/deploy/container files; runner requirements (dedicated machine, unprivileged account, no docker.sock) written down in `deploy/README.md`.
- Full details of the resume architecture: [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.14.0

- 🔬 **Live-model eval harness** — `evaluation/run_evals.py` is now practical: Ollama pre-flight, per-case timeouts (a runaway generation can't wedge the run), `--filter`/`--category` selection, category pass-rate breakdown, failure summaries, and `--json` machine-readable reports for trend tracking.
- 🤖 **Nightly CI evals** — scheduled job on self-hosted Ollama runners (non-blocking, report uploaded as an artifact).
- 🌊 **True SSE interleaving** — `/chat/stream` now runs the turn in a worker thread and streams lifecycle events *as they happen*, with 15-second keepalive comments so proxies don't close idle connections during long generations.
- 🧠 **Planner tool filtering** — hallucinated `required_tools` names are filtered against the real registry before execution (fully-hallucinated steps degrade to safe reasoning steps).
- 📏 **Strict-grounding prompt rules** — explicit anti-fabrication policy (say what's missing instead of guessing; never compute silently when a tool exists) and referent-echo rule for multi-turn answers.
- ⏱️ **Turn durations in logs** — `response_ready` now carries `duration_ms` for both the fast and complex paths; removed a duplicate schema-serialization call per complex request.
- 🧹 **Test dedup completed** — all fakes now come from `tests/fakes.py`.

- 🚦 **API rate limiting** — in-process sliding window (default 60 req/60s, `RATE_LIMIT_*` envs, `0` disables), keyed by API key or client IP; over-limit → `429` + `Retry-After`; `/health` exempt.
- 🔒 **Per-session serialization** — concurrent turns on one session now return `409 Conflict` instead of interleaving history or double-resolving confirmations; different sessions stay fully parallel.
- 🖥️ **API-first dashboard** — the Streamlit UI consumes the REST API via a new stdlib-only client (`jarvis/api/client.py`); with `JARVIS_API_URL` set it can run on a different machine than the agent runtime (legacy in-process mode still available).
- 📦 **Dedicated sandbox image** — `deploy/Dockerfile.sandbox`: minimal interpreter-only image with a dedicated unprivileged user; mutable tags (`latest`, bare names) now rejected by the sandbox validator — pin tags or digests. Full build/pin/pre-pull guide in `deploy/README.md`.
- 👁️ **Observability** — structured per-request logs (`api_request`), `tool_calls` lifecycle events in the SSE stream and `on_event` seam.
- 🧪 **Eval harness at 26 cases** — adds tool-failure honesty, three-turn context chains across topic detours, and comparative multi-source synthesis.

- 🐳 **Real Docker-isolated code execution** — one-shot containers with `--network none`, read-only rootfs, `--cap-drop ALL`, `no-new-privileges`, non-root user, memory/CPU/PID ceilings, and read-only code mounts. Off by default (`ENABLE_CODE_EXECUTION=false`); the tool joins the LLM's surface only when the config flag is set **and** Docker is verified usable. Fail-closed on every uncertain step.
- 🔑 **API authentication (opt-in)** — set `JARVIS_API_KEY` to require `Authorization: Bearer <key>` (or `X-API-Key`) on every endpoint except `/health`. Constant-time comparison.
- 🌊 **SSE streaming chat** — `POST /chat/stream` emits the agent's lifecycle (`intent` → `plan` → `step_start`/`step_done` → `synthesis` → `done`) as server-sent events, with in-band errors.
- 👁️ **Observability seam** — `Orchestrator.chat(..., on_event=...)` exposes the same lifecycle events the stream emits; observer failures can never break execution.
- 🧱 **Stability fixes** — context-window floor (a zero-length window can no longer return the entire history), session-history cache invalidated inside the store lock (no stale reads for concurrent API workers), plus regression tests for concurrency and edge cases.
- ⚠️ **Honest capability claims** — see [Current Limitations](#current-limitations).

---

## Features

- 💬 **Conversational AI** — full session memory backed by SQLite (thread-safe)
- 🛠️ **Tool calling** — Plan-and-Execute loop with a fast path for simple intents; concurrent dispatch within a tool round
- 🐍 **Code execution (opt-in)** — Docker-isolated one-shot containers; off unless explicitly enabled and verified (see [Safe Code Execution](#safe-code-execution-docker))
- 🔍 **Web search & Scraping** — DuckDuckGo and Playwright browser integration for **dynamic web browsing and extraction**
- 👁️ **Vision** — Image understanding via Llava
- 🕐 **Current time/date** — instant, no network
- 📄 **Read files** — sandboxed to your allowed directory (see [Current Limitations](#current-limitations) for writes)
- 🧠 **Long-term memory** — `remember_fact` / `recall_facts` backed by ChromaDB
- 🧩 **Context management** — windowing, task anchoring, rolling summary, tool-output clamping
- 🛡️ **Permission tiers & durable confirmations** — high-risk actions require explicit approval, persisted across restarts
- 🔒 **Fully local** — your data never leaves your machine, providing **local inference without remote API network latency**
- 🌐 **REST API** — FastAPI service layer with optional API-key auth and SSE streaming
- 🏗️ **Modular** — adding a new tool takes ~30 lines and one registration line

### 1. Python 3.11+

```bash
python --version   # must be 3.11 or higher
```

### 2. uv (fast Python package manager)

```bash
# Install uv
pip install uv
# or on macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Ollama

Ollama runs LLMs locally on your machine.

**Install Ollama:**

| Platform | Command |
|----------|---------|
| macOS | `brew install ollama` or download from [ollama.com](https://ollama.com) |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Windows | Download installer from [ollama.com/download](https://ollama.com/download) |

**Pull a model** (choose one):

```bash
# Recommended — capable, supports tool calling
ollama pull qwen2.5:7b

# Lighter alternative — fast, less capable (~2 GB)
ollama pull phi3:mini

# Verify it works
ollama run qwen2.5:7b "Hello, are you working?"
```

**Start the Ollama server** (if not already running):

```bash
ollama serve
```

> Ollama starts automatically on macOS and Windows after installation.
> On Linux you may need to run `ollama serve` manually in a terminal.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd JARVIS

# 2. Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell)

# 3. Install dependencies
uv pip install -e ".[dev]"

# 4. Copy the environment template
cp .env.example .env
```

The `.env` file comes pre-configured to use Ollama locally. No edits are needed
unless you want to change the model or sandbox directory.

---

## Running JARVIS

### Docker (recommended for a service)

```bash
cp .env.example .env          # set JARVIS_API_KEY at minimum
# external Ollama on the host:
OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up -d api

# or fully self-contained (includes Ollama):
docker compose --profile local-llm up -d
docker compose exec ollama ollama pull qwen2.5:7b

curl http://localhost:8000/health
```

The `jarvis-data` volume persists `jarvis.db` and the vector store across restarts. Scheduled maintenance runs against the same volume:

```bash
docker compose run --rm api python -m jarvis.maintenance stats
docker compose run --rm api python -m jarvis.maintenance cleanup --days 30
```

See [`deploy/README.md`](deploy/README.md) for the sandbox image build and the full hardening checklist.

### Local CLI

Make sure Ollama is running in the background:

```bash
ollama serve
```

Then, you can start JARVIS in CLI mode:

```bash
jarvis
```

Or, if the script isn't on your PATH:

```bash
python -m jarvis.main
```

### Voice mode

```bash
jarvis --voice
```

Or from the text REPL: type `/voice`.

### REST API server (v0.10)

Start the FastAPI service (same runtime as the CLI):

```bash
uv run uvicorn jarvis.api.app:app --port 8000
```

Quick tour:

```bash
# Health & introspection (version, model, tools, auth + sandbox posture)
curl http://localhost:8000/health

# Start a session
curl -X POST http://localhost:8000/sessions
# → {"session_id": "..."}

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "message": "What time is it?"}'

# Stream one turn as SSE events (intent → plan → steps → done)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "message": "Research X and summarize it"}'

# Inspect history / resolve pending confirmations
curl http://localhost:8000/sessions/<id>/history
curl http://localhost:8000/sessions/<id>/confirm
curl -X POST http://localhost:8000/sessions/<id>/confirm \
  -H "Content-Type: application/json" -d '{"confirmed": true}'
```

Interactive OpenAPI docs are served at `http://localhost:8000/docs`.

### API authentication (opt-in)

By default the API trusts localhost. To require a key, set it in `.env` (or the environment):

```env
JARVIS_API_KEY=change-me-to-a-long-random-value
```

Every endpoint except `/health`, `/docs`, and `/openapi.json` then requires:

```bash
curl -H "Authorization: Bearer $JARVIS_API_KEY" ...      # preferred
curl -H "X-API-Key: $JARVIS_API_KEY" ...                 # alternative
```

Keys are compared in constant time; failed attempts log the client and return `401` with `WWW-Authenticate: Bearer`.

### Safe Code Execution (Docker)

Code execution ships **off**. To enable it:

```env
ENABLE_CODE_EXECUTION=true
SANDBOX_IMAGE=ubuntu:24.04          # tag-pinned; 'latest'/bare names are REJECTED
```

Production should build the dedicated minimal image instead (no shell tooling, no package managers, dedicated unprivileged user, **digest-pinned and pull-verified base**) — see [`deploy/README.md`](deploy/README.md):

```bash
docker build -t jarvis-sandbox:1.0.0 -f deploy/Dockerfile.sandbox deploy/
docker images --digests jarvis-sandbox   # then pin: SANDBOX_IMAGE=jarvis-sandbox:1.0.0@sha256:<digest>
```

The tool registers **only if** Docker verifies usable at startup (CLI + daemon + image present). Each execution runs in a one-shot container with: no network (`--network none`), a read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`, a non-root user, hard memory/CPU/PID ceilings, and the code mounted read-only. Time limits are enforced **inside the container** (`timeout` wrapper, exit 124) with a host-side wait as backup and force-removal if the host wait fires first. Any failure mode is a *denial* — never host execution. Windows hosts are refused (use WSL2/Linux); the sandbox never pulls images at runtime.

> The code-execution tool is `SYSTEM` risk: even when enabled, a call parks a durable confirmation that must be approved (`/confirm` in the CLI, `POST /sessions/{id}/confirm` in the API) before the container runs.

## Security & Safety

JARVIS employs a strict permission model to ensure your machine stays secure.
- **Structurally absent tools:** `computer_control` is never registered — a pure placeholder with no OS automation code. `execute_python_code` registers only behind the double gate described above.
- **Sandboxed File Reading:** `read_file` is strictly limited to paths relative to `FILE_READER_ALLOWED_DIR`.
- **Permission tiers:** `SAFE`/`NETWORK`/`FILE_READ` run automatically; `SYSTEM`/`DESTRUCTIVE` require explicit, durable user confirmation (`/confirm` in CLI, `POST /sessions/{id}/confirm` in the API, buttons in the dashboard); `FILE_WRITE` is currently blocked outright.
- **Hardened containers, not promises:** when code execution is enabled, every run is one-shot, network-less, capability-less, non-root, read-only-rootfs, and resource-capped. Failures deny — they never fall back to host execution.
- **No in-process `exec()`/`eval()`** on model-influenced strings — the v0.7-era in-process sandbox was removed. The calculator uses a restricted AST walker.
- **Bounded autonomy:** tool rounds are capped per request (5) and per step (2), with a self-correction sub-loop capped at 2 recovery attempts.
- **Optional API auth:** constant-time API-key enforcement whenever `JARVIS_API_KEY` is set.

Details: [`docs/architecture.md` § Safety Model](docs/architecture.md).

---

## Web UI Dashboard (v0.5+)

To launch the beautiful browser interface with thought process observability and image upload support, run:

```bash
streamlit run ui/dashboard.py
```

---

## Voice Mode (v0.4)

JARVIS can listen with **Whisper** (local STT) and speak with **Edge TTS** (free neural voices).

### Current Limitations

JARVIS advertises only what actually works today:

- **Code execution is opt-in and platform-gated** — disabled unless `ENABLE_CODE_EXECUTION=true` **and** Docker verifies usable; unavailable on Windows hosts (use WSL2/Linux), and the image must be pulled in advance. See [Safe Code Execution](#safe-code-execution-docker).
- **Computer control is disabled** — the tool is a pure placeholder with no OS automation code.
- **File writes are blocked** — `write_file` is registered but the permission guard blocks `FILE_WRITE` risk tier; only reads are permitted.
- **Voice TTS needs internet** — recognition is local (Whisper); speech output streams from Microsoft's free Edge TTS endpoint.
- **Auth is opt-in** — without `JARVIS_API_KEY` the API trusts localhost; set a key before exposing it beyond loopback.
- **Deterministic summaries** — context compaction is rule-based (no extra LLM call, zero cost, but less fluent than an LLM summary).
- **Lexical, not semantic, evaluation** — the 32 graders are deterministic string checks: great as a regression tripwire, but they do not prove semantic correctness (no LLM judge). See [`deploy/README.md` §7](deploy/README.md).

---

## Prerequisites

1. **ffmpeg** (required by Whisper)

| Platform | Install |
|----------|---------|
| Windows | `winget install FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` (Debian/Ubuntu) |

Verify:

```bash
ffmpeg -version
```

2. **Microphone** access allowed for your terminal / Python in OS privacy settings.

3. **Internet** for Edge TTS (speech synthesis is streamed from Microsoft Edge’s free TTS endpoint). Recognition itself is local via Whisper.

### Start voice mode

```bash
# Directly
jarvis --voice

# Or inside the text REPL
/voice
```

Say **“exit”** or **“quit”** (or press `Ctrl+C`) to leave voice mode.

### Configuration

```env
WHISPER_MODEL=base              # tiny | base | small | medium | large
TTS_VOICE=en-US-GuyNeural       # Edge neural voice
VOICE_RECORD_SECONDS=5          # Seconds of mic capture per turn
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Whisper fails to load / “ffmpeg not found” | Install ffmpeg and restart the terminal so `PATH` updates |
| No speech detected | Check mic permissions; speak during the “Listening…” window; raise `VOICE_RECORD_SECONDS` |
| TTS silent / network errors | Edge TTS needs internet; try another `TTS_VOICE` or check firewall |
| `playsound` errors on Windows | Install ffmpeg (`ffplay` is used as a fallback player) |

Text CLI mode remains fully available and is the default when you run `jarvis` without `--voice`.

---

## CLI Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/tools` | List registered tools |
| `/history` | Print this session's messages |
| `/voice` | Enter voice mode |
| `/new` | Start a fresh session |
| `/quit` | Exit JARVIS |

---

## Configuration

Edit `.env` to customise behaviour:

```env
OLLAMA_MODEL=qwen2.5:7b        # Which model to use
OLLAMA_BASE_URL=http://localhost:11434
MAX_TOKENS=2048
FILE_READER_ALLOWED_DIR=.      # Sandbox for read_file tool
LOG_LEVEL=INFO                 # DEBUG | INFO | WARNING | ERROR

# Context management
MAX_CONTEXT_MESSAGES=24        # LLM-facing window size (older turns are summarized)
MAX_TOOL_OUTPUT_CHARS=6000     # Per-message cap for tool results (head + tail kept)

# Voice (v0.4)
WHISPER_MODEL=base
TTS_VOICE=en-US-GuyNeural
VOICE_RECORD_SECONDS=5
```

---

## Running Tests

```bash
pytest
```

Tests run fully offline — no Ollama or network required.

---

## Project Structure

```
JARVIS/
├── jarvis/
│   ├── runtime.py          ← Assembly seam: build_runtime() + JarvisRuntime
│   ├── main.py             ← CLI entry point (thin over the runtime)
│   ├── api/                ← FastAPI service layer (app.py, schemas.py)
│   ├── config.py           ← All config, loaded from .env
│   ├── core/
│   │   ├── orchestrator.py ← Route → Plan → Execute → Synthesize
│   │   ├── planner.py      ← JSON plan generation with fallback
│   │   ├── permissions.py  ← Risk tiers + confirmation policy
│   │   └── sandbox.py      ← CodeSandbox ABC + fail-closed implementations
│   ├── llm/
│   │   └── client.py       ← LiteLLM → Ollama wrapper
│   ├── memory/
│   │   ├── session_store.py   ← SQLite sessions/messages/confirmations (thread-safe)
│   │   ├── context_manager.py ← Windowing, anchoring, summarization, clamping
│   │   └── vector_store.py    ← ChromaDB long-term memory
│   ├── tools/
│   │   ├── base.py        ← Abstract tool interface + risk levels
│   │   ├── registry.py    ← Tool registry + dispatcher
│   │   ├── datetime_tool.py
│   │   ├── web_search.py
│   │   └── …
│   └── utils/
│       └── logging.py     ← structlog setup
├── evaluation/             ← Offline eval harnesses
├── tests/
├── docs/
│   ├── architecture.md     ← Layers, lifecycle, safety model
│   ├── JARVIS_USER_MANUAL.md     ← Owner's guide: capabilities, startup, troubleshooting
│   ├── JARVIS_DEVELOPER_MANUAL.md ← Architecture, internals, extension guides
│   └── PROJECT_OVERVIEW.md ← Handoff overview
├── ui/dashboard.py         ← Streamlit dashboard
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Adding a New Tool

See [`docs/architecture.md` § Adding a New Tool](docs/architecture.md) for the full checklist.
In short: create a file in `jarvis/tools/`, subclass `BaseTool`, and add it to `_TOOL_FACTORIES` in `jarvis/runtime.py` — the single registration point that CLI, API, and dashboard all share.

---

## Roadmap & Versions

| Version | Highlights |
| :--- | :--- |
| **v0.1** | ✅ Basic CLI, simple tools, SQLite memory |
| **v0.2** | ✅ Long-term memory (ChromaDB + embeddings) |
| **v0.3** | ✅ Plan-and-Execute multi-step agent loop |
| **v0.4** | ✅ Voice input (Whisper) + voice output (Edge TTS) |
| **v0.5** | ✅ Web UI (Streamlit) + write_file tool + synthesis polish |
| **v0.6** | ✅ Vision (image understanding) + Playwright web scrape |
| **v0.8** | ✅ Permission tiers, durable confirmations, caching, context management |
| **v0.9** | ✅ Service architecture: JarvisRuntime seam, FastAPI REST API, thread-safe store, sandbox contract |
| **v0.10** | ✅ Real Docker-isolated code execution (opt-in), API auth, SSE streaming, observability seam, stability fixes |
| **v0.11** | ✅ Deploy-ready service: rate limiting, per-session serialization, dedicated sandbox image, API-first dashboard |
| **v0.12** | ✅ Ship path: docker-compose, service image, CI, request IDs, maintenance CLI, dispatch dedup |
| **v0.13** | ✅ Operate: deep health (503 on broken persistence), error sanitization + correlation, WAL persistence, `doctor` command |
| **v0.14** | ✅ Live-model quality loop: practical eval harness, nightly CI evals, true SSE interleaving, planner grounding |
| **v0.15** | ✅ Reliability & security closure: workload-boundary sandbox timeout, resumable confirmations, CI trust hardening |

---

## Troubleshooting

- **LiteLLM / Ollama Error:** Ensure Ollama is running in the background (`ollama serve`). If you're missing a model, run `ollama pull <model-name>`.
- **Playwright errors:** Ensure the chromium binaries are installed via `playwright install chromium`.
- **Voice errors:** Ensure `ffmpeg` is on your system `PATH`.
- **API returns 503 on `/chat`:** Ollama is unreachable from the server process; check `ollama serve` and `OLLAMA_BASE_URL`.
- **API returns 401:** `JARVIS_API_KEY` is set and your request is missing/has a wrong key; send `Authorization: Bearer <key>`.
- **API returns 409 on `/chat`:** another turn is already running on that session; wait for it to finish (or use a different session).
- **Chat response is `ACTION_REQUIRES_CONFIRMATION…`:** the turn is *paused*, not finished — approve/deny via `/confirm`/`/deny` (CLI) or `POST /sessions/{id}/confirm` (API) and the original task continues automatically, including after a restart.
- **API returns 429:** you hit the rate limit (`RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW_SECONDS`); honor the `Retry-After` header.
- **API returns 503 on `/health`:** the DB read/write probe failed — check volume mount and disk; `python -m jarvis.maintenance doctor` localizes it.
- **`/chat` returns a generic 503:** the real cause is in the server log under the response's `X-Request-ID` (also echoed in the JSON body as `request_id`); `python -m jarvis.maintenance doctor` checks Ollama reachability.
- **Code execution stays disabled:** with `ENABLE_CODE_EXECUTION=true` the tool still requires usable Docker (CLI, daemon, image pulled). Check the startup log for `code_execution_enabled_but_docker_unavailable`. Not supported on Windows hosts — use WSL2 or Linux. Also check `SANDBOX_IMAGE` is not `latest`/bare — mutable tags are rejected.


## License

MIT
