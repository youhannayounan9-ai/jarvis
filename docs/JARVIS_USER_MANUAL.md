# JARVIS User Manual

> **For:** the owner and operator of this JARVIS instance.
> **Repo state documented:** v0.16.0 (branch `main`). Every command in this manual was verified against the actual repository in September 2026.
> **Companion documents:** [`JARVIS_DEVELOPER_MANUAL.md`](JARVIS_DEVELOPER_MANUAL.md) (architecture and internals), [`architecture.md`](architecture.md) (layer detail), [`deploy/README.md`](../deploy/README.md) (deployment and sandbox hardening).

---

## 1. What JARVIS Is

JARVIS is a **local-first personal AI agent platform**: a system built around a local LLM that can plan multi-step tasks, call tools, remember facts across sessions, and execute Python code in an isolated container — all on your machine.

Mental model:

```
   LLM (e.g. qwen2.5:7b via Ollama)  =  the BRAIN
   JARVIS                            =  the AGENT SYSTEM around the brain
```

The LLM by itself can only produce text. JARVIS wraps it in:

- an **orchestrator** that decides between a fast answer and a planned multi-step task,
- a **planner** that decomposes complex requests into bounded steps,
- a **tool registry** of 11 active tools (search, files, calculator, memory, vision, …),
- a **permission system** that blocks or parks dangerous actions for your approval,
- **memory** — session history in SQLite, long-term facts in a local vector store,
- **interfaces**: a terminal CLI, a REST API with SSE streaming, and a web dashboard.

JARVIS is **not** fully autonomous: tool rounds are capped (5 per request, 2 per step, +2 recovery), every high-risk action stops and waits for your explicit approval, and all processing stays on your machine except tools that inherently need the network (web search/scrape, Wikipedia, TTS, vision model pulls).

---

## 2. What JARVIS Can Do Today

Status legend: **ACTIVE** = on by default · **OPTIONAL** = works after you enable/configure it · **ENVIRONMENT-DEPENDENT** = works only with specific prerequisites · **PLACEHOLDER** = exists in code but is intentionally inactive.

| Capability | Status | What it does | Prerequisites | Example request |
|---|---|---|---|---|
| Text conversation | ACTIVE | Chat with session memory | Ollama + model | "Explain what a mutex is in two sentences." |
| Local LLM inference | ACTIVE | qwen2.5:7b via Ollama | Ollama running, model pulled | — |
| Planner (Plan-and-Execute) | ACTIVE | Decomposes complex requests into ≤5 steps | Ollama | "Research X and then summarize it" |
| ReAct tool loop | ACTIVE | Tool calls + observation per step | Ollama | — |
| Self-correction | ACTIVE | Up to 2 recovery rounds after tool errors | Ollama | — |
| `get_current_datetime` | ACTIVE | Local date/time/day/timezone | None | "What time is it?" |
| `calculator` | ACTIVE | Exact arithmetic via a restricted AST parser (+, -, *, /, **, parentheses) | None | "Calculate 12 * (4 + 3) / 2.5" |
| `wikipedia_summary` | ACTIVE | Short encyclopedia summary | Internet | "Summarize the Alan Turing article" |
| `web_search` | ACTIVE | DuckDuckGo search, numbered excerpts | Internet | "Search for today's bitcoin price" |
| `web_scrape` | ACTIVE | Full text of a URL via headless Chromium | Internet + Playwright chromium installed | "Scrape https://example.com and summarize" |
| `read_file` | ACTIVE | Read a file inside the allowed directory (32 KB cap) | None | "Read my notes.txt" |
| `list_directory` | ACTIVE | List a directory inside the allowed directory | None | "List the files in ." |
| `write_file` | ⚠️ See note | Registered, but **blocked by the permission guard** (FILE_WRITE tier is refused outright) | — | — |
| `remember_fact` / `recall_facts` | ACTIVE | Save/search long-term memory | None (local ChromaDB) | "Remember that my project is due Friday" → later "When is my project due?" |
| `vision_analyze` | OPTIONAL | Image description via local llava model | `ollama pull llava`; image path inside allowed dir | (with image attached) "What's in this image?" |
| Speech-to-text | OPTIONAL | Local Whisper transcription | ffmpeg on PATH + microphone | `jarvis --voice`, then speak |
| Text-to-speech | OPTIONAL | Edge TTS neural voice | **Internet** + ffmpeg (ffplay) | Same voice session |
| `execute_python_code` | ENVIRONMENT-DEPENDENT | Python in a hardened one-shot Docker container | `ENABLE_CODE_EXECUTION=true` **and** Linux/WSL2 Docker with the sandbox image built+pulled; **also always parks a confirmation** | (after approval) runs user code, returns stdout |
| `computer_control` | PLACEHOLDER | Mouse/keyboard automation — **never registered**, returns refusal if somehow invoked | None (unusable by design) | — |
| FastAPI REST service | ACTIVE | Sessions, chat, SSE streaming, confirmations, history, health | Python deps | `uv run uvicorn jarvis.api.app:app --port 8000` |
| SSE streaming | ACTIVE | Live agent lifecycle events | API running | `POST /chat/stream` |
| API authentication | OPTIONAL | Bearer/X-API-Key on all endpoints except /health | `JARVIS_API_KEY` set | — |
| Rate limiting | ACTIVE | 60 req/60s sliding window per client (disable with `0`) | None | — |
| Streamlit dashboard | ACTIVE | Web chat UI + image upload + approval buttons | API running (or legacy in-process mode) | `streamlit run ui/dashboard.py` |
| CLI | ACTIVE | Terminal REPL with `/confirm`, `/deny`, `/voice`, … | Python deps | `jarvis` (or `python -m jarvis.main`) |
| Evaluation system | ACTIVE | 32-case lexical eval suite + reports + regression diff | Ollama for live runs | `uv run python evaluation/run_evals.py --list` |
| CI evaluation | OPTIONAL | Nightly scheduled live eval | Self-hosted Ollama runner | (GitHub Actions schedule) |
| Docker deployment | OPTIONAL | Compose file for API (+ optional Ollama) | Docker | `docker compose up -d api` |
| Logging/observability | ACTIVE | Structured JSON logs, request IDs, turn durations | None | `LOG_LEVEL=DEBUG` in `.env` |

**Note on `write_file`:** the tool is registered and the model can *request* it, but the permission guard blocks the `FILE_WRITE` risk tier outright (README's "blocked" is enforced in `_permission_decision`, not by absence). Expect an `ERROR: Tool 'write_file' is not permitted` refusal. This is the deliberate current posture.

---

## 3. What JARVIS Cannot Do Yet

- **No OS automation** — `computer_control` is a structurally disabled placeholder; JARVIS cannot move your mouse or type for you.
- **No file writes** — `write_file` is refused by the permission guard (FILE_WRITE tier blocked outright). File access is read-only.
- **No code execution on Windows hosts** — the sandbox refuses Windows by design (use WSL2 or Linux). On this Windows machine, Docker Desktop's Linux engine does run, but `sandbox.is_available()` still fails closed on `win32` — so even with Docker running, `execute_python_code` stays unregistered on this host.
- **No multi-process state sharing** — rate limiting, per-session locks, and the ChromaDB vector store are single-process. Run one API process (or front it with a reverse proxy for limits).
- **No semantic evaluation** — the 32 graders are deterministic/lexical: a great regression tripwire, not a semantic judge.
- **No distributed safety** — do not run multiple replicas against one SQLite DB expecting shared rate limits or shared session locks.
- **Model limits** — a 7B-class model makes planning/tool-order mistakes; live eval measured 23/32 (71.9%) on qwen2.5:7b.
- **TTS needs internet** — speech output streams from Microsoft's Edge TTS endpoint (STT is local).
- **Vision needs a second model** — `llava` is not pulled by default; `vision_analyze` errors until it is.
- **Confirmation TTL** — pending confirmations expire after 10 minutes; re-ask after expiry.

---

## 4. Prerequisites

### What MUST be running for normal chat (CLI/API/dashboard)

| Requirement | Why | Check |
|---|---|---|
| Python 3.11+ | App runtime | `python --version` |
| uv | Dependency management | `uv --version` |
| Ollama running | Local inference | `curl http://localhost:11434/api/version` |
| qwen2.5:7b pulled | The default model | `ollama list` |

Everything else in the table below is **only needed for special capabilities**.

### What is ONLY needed for special capabilities

| Capability | Extra requirement |
|---|---|
| Vision | `ollama pull llava` |
| Voice STT | ffmpeg on PATH + microphone; Whisper weights (auto-downloaded on first use) |
| Voice TTS | Internet (Edge TTS) + ffmpeg (ffplay) |
| `web_scrape` | `uv run playwright install chromium` |
| Code execution | Linux/WSL2 Docker + sandbox image built & pulled + `ENABLE_CODE_EXECUTION=true` |
| Dashboard | Streamlit dep (already in project) |
| Docker deployment | Docker / docker-compose |

### Install steps (verified against the repo)

```bash
# 1. Clone
git clone <your-repo-url> && cd JARVIS

# 2. Virtual environment
uv venv
source .venv/bin/activate        # Linux/macOS shell
.venv\Scripts\activate           # Windows PowerShell

# 3. Install
uv pip install -e ".[dev]"

# 4. Environment template
cp .env.example .env
```

> **Model note (verified):** `.env.example` lists `OLLAMA_MODEL=llama3.1:8b`, but the shipped code default in `jarvis/config.py` is **`qwen2.5:7b`** (both chat and planner). If you create `.env` from the template without editing it, you will actually run llama3.1:8b. Keep it consistent with what you pulled: either set `OLLAMA_MODEL=qwen2.5:7b` in `.env` or pull llama3.1:8b.

---

## 5. Starting JARVIS

### 0) Start Ollama (required for everything except health endpoints)

```bash
ollama serve        # usually already running on Windows/macOS after install
```

### CLI (text)

```bash
jarvis                          # if installed via `uv pip install -e .`
# or, without install:
uv run python -m jarvis.main
```

In the REPL: `/help` `/tools` `/history` `/voice` `/new` `/confirm` `/deny` `/quit` (`/exit`, `/q` also quit).

### REST API

```bash
uv run uvicorn jarvis.api.app:app --port 8000
```

- Interactive OpenAPI docs: `http://localhost:8000/docs`
- Health: `curl http://localhost:8000/health`
- **Auth is off by default** (localhost trust). Set `JARVIS_API_KEY` in `.env` before exposing beyond loopback.

### Dashboard (Streamlit)

```bash
streamlit run ui/dashboard.py
```

- With `JARVIS_API_URL` set (e.g. `http://127.0.0.1:8000`): dashboard talks to the API server.
- Empty → legacy in-process runtime in the dashboard process (single-machine fallback).

### Docker compose (service deployment)

```bash
cp .env.example .env
# external Ollama on the host:
OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up -d api
# or fully self-contained:
docker compose --profile local-llm up -d
docker compose exec ollama ollama pull qwen2.5:7b
curl http://localhost:8000/health
```

---

## 6. Ollama

**What it does:** hosts the local LLM. JARVIS sends the conversation + tool schemas to `http://localhost:11434` (configurable via `OLLAMA_BASE_URL`) through LiteLLM. The planner uses the same or a separate model (`PLANNER_MODEL`).

**Where configured:** `.env` (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `PLANNER_MODEL`, `MAX_TOKENS`) → `jarvis/config.py`.

**Check it:**

```bash
# Is the server up?
curl http://localhost:11434/api/version
# {"version":"0.34.1"} means up; connection refused means start it with `ollama serve`.

# Which models are installed?
ollama list

# Is the configured model present? (should list qwen2.5:7b by default)
ollama list | grep -i qwen
```

If the model is missing: `ollama pull qwen2.5:7b` (~4.7 GB; first load is slow on CPU — subsequent calls are warm).

---

## 7. CLI Usage

```bash
uv run python -m jarvis.main
```

Session banner shows version, model, and tool count. Commands:

| Command | Action |
|---|---|
| `/help` | Command reference |
| `/tools` | List the registered tool surface |
| `/history` | Print this session's messages |
| `/voice` | Enter voice mode (returns to text with "exit") |
| `/new` | Fresh session (new ID, cleared context) |
| `/confirm` | Approve a pending high-risk action |
| `/deny` | Deny a pending high-risk action |
| `/quit`, `/exit`, `/q` | Exit |

**A pause is normal, not an error.** If a high-risk action is requested you'll see `ACTION_REQUIRES_CONFIRMATION…` — the turn is parked. Answer `/confirm` or `/deny` and the original task continues automatically (even after a restart).

---

## 8. API Usage

Start it:

```bash
uv run uvicorn jarvis.api.app:app --port 8000
```

Endpoints (all except `/health`, `/docs`, `/openapi.json` require the key when `JARVIS_API_KEY` is set):

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Version, model, tool surface, auth + sandbox posture; 503 when the DB probe fails |
| POST | `/sessions` | Create a session → `{"session_id": ...}` |
| POST | `/chat` | One turn; auto-creates a session when `session_id` is omitted |
| POST | `/chat/stream` | SSE: `begin → intent → plan → step_start/tool_calls/step_done → synthesis → done/error` + 15s keepalives |
| GET | `/sessions/{id}/history` | Persisted history (`?limit=` optional) |
| GET | `/sessions/{id}/confirmation` | Pending high-risk action or 404 |
| POST | `/sessions/{id}/confirm` | `{"confirmed": true|false}` → executes/skips + resumes + synthesize |
| GET | `/tools` | Active tool surface with risk levels |

Minimal working examples:

```bash
# Health (no auth even when auth is on)
curl http://localhost:8000/health

# Create a session
SID=$(curl -s -X POST http://localhost:8000/sessions | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SID\", \"message\": \"What time is it?\"}"

# Stream a complex turn
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SID\", \"message\": \"Search for the latest Python release and summarize it\"}"

# Confirm a parked high-risk action
curl -X POST http://localhost:8000/sessions/$SID/confirm \
  -H "Content-Type: application/json" -d '{"confirmed": true}'
```

**Authentication behavior:** empty `JARVIS_API_KEY` = local trust. Set → `Authorization: Bearer <key>` (or `X-API-Key`) required on everything except `/health` + docs. Constant-time compare. Wrong/missing key → 401; whitespace-only key config → 503 (fail-closed). Rate limit: 60 req/60s per client → 429 + `Retry-After` (`0` disables). Same-session concurrent turns → 409.

**Errors carry correlation IDs:** every response has `X-Request-ID` (and `/chat` echoes it as `request_id`); the server log line under that ID holds the real error — clients never see internals.

---

## 9. Streamlit Dashboard

```bash
streamlit run ui/dashboard.py
```

- Point it at a running API via `JARVIS_API_URL` in `.env` (recommended); without it, it builds a legacy in-process runtime.
- Upload an image (sidebar) → attaches `[Attached Image: <path>]` to your next message for `vision_analyze`.
- Pending high-risk action → red approval panel with **Approve Action** / **Deny Action** buttons and an "Agent Thought Process" expander showing tool calls and results.

---

## 10. Memory

Two distinct layers:

| Layer | Storage | Lifetime | How it's used |
|---|---|---|---|
| **Session (short-term)** | SQLite (`jarvis.db`) | Per session | Every turn; context window of 24 messages with anchoring/summary of older turns; full tool payloads kept on disk, clamped in prompts |
| **Long-term (semantic)** | ChromaDB (`./jarvis_data/chroma_db`) | Forever (cross-session) | `remember_fact` writes; `recall_facts` similarity-searches (top 3) |

Examples:

```text
You: Remember that my favorite editor is Neovim.
JARVIS: (calls remember_fact) Got it — stored.

# Days later, new session:
You: What's my favorite editor?
JARVIS: (calls recall_facts) Your favorite editor is Neovim.
```

Session memory is what makes "what did I just ask?" work in-turn; long-term memory is what makes "what's my name?" work next week. Never guess facts: the system prompt forces `recall_facts` before answering questions about you.

---

## 11. Web Tools

| Tool | What | Needs |
|---|---|---|
| `web_search` | DuckDuckGo results as numbered excerpts (title/URL/snippet, 1–10 results, TTL-cached 1h) | Internet |
| `web_scrape` | Full text of one URL via headless Chromium (JS rendering), truncated to 4000 chars | Internet + `playwright install chromium` |
| `wikipedia_summary` | REST summary from Wikipedia (disambiguation-aware, 1200-char extract) | Internet |

**Untrusted-content rule:** everything these tools return is *data*, not instructions. A malicious page cannot grant JARVIS permissions — privileged actions are gated structurally by the PermissionGuard + durable confirmation, not by anything a page says. Still, treat scraped/search content as untrusted input when reading JARVIS's answers.

---

## 12. Files

- `read_file` / `list_directory` are sandboxed to **`FILE_READER_ALLOWED_DIR`** (default `.` — the directory where you start JARVIS). Absolute or `../` paths outside it are refused. `read_file` caps output at 32,000 bytes with a truncation notice.
- `write_file` is registered but **refused** (FILE_WRITE tier is blocked outright — no file writes today).
- `vision_analyze` reads images from the allowed dir only.

Set a tighter boundary in `.env`:

```env
FILE_READER_ALLOWED_DIR=C:\Users\youha\Documents\JARVIS-workspace   # Windows
FILE_READER_ALLOWED_DIR=/home/you/jarvis-workspace                  # Linux/macOS
```

---

## 13. Vision

```bash
ollama pull llava        # once
```

Then, in any interface, attach an image path inside the allowed directory:

```text
You: Describe the image at C:\Users\youha\Pictures\chart.png
```

The tool base64-encodes the image, sends it to `ollama_chat/llava` directly (bypasses the main model), and returns the description as the tool result. Requires the image to be inside `FILE_READER_ALLOWED_DIR` (path traversal refused) and llava installed, or you'll get `ERROR: Failed to analyze image`.

---

## 14. Voice

| Part | Backend | Network? |
|---|---|---|
| Speech-to-text | **OpenAI Whisper** (local, `base` model by default, ~140 MB weights auto-download) | No |
| Text-to-speech | **Microsoft Edge TTS** (`en-US-GuyNeural` default) | **Yes** |
| Playback | `ffplay` (ffmpeg) | — |

Requirements: ffmpeg on PATH, microphone permission for the terminal.

```bash
jarvis --voice          # start directly in voice mode
# or from the text REPL:
/voice
```

Say **"exit" / "quit" / "stop" / "goodbye"** to leave. Config: `WHISPER_MODEL`, `TTS_VOICE`, `VOICE_RECORD_SECONDS` in `.env`.

Known failure modes (from code): Whisper load failure → "Voice input is unavailable… Returning to text mode."; missing ffmpeg → explicit ERROR message; TTS/network failures are logged and the voice loop continues.

---

## 15. Docker Sandbox

**Read this first: Docker is NOT required to chat with JARVIS.** The LLM, planner, tools, memory, API, dashboard, and voice all run without Docker. Docker is used for exactly one thing: **sandboxed Python code execution** — and that feature ships **off**.

Key facts:

- **Image vs container vs app:** the *image* (`jarvis-sandbox:1.0.0`, built from `deploy/Dockerfile.sandbox`) is a minimal Python-only template with a dedicated unprivileged user. A *container* is one throwaway instance created per execution (`--rm`), created from that image, running your code, then removed. The *JARVIS application* itself can run directly on the host, in its own container, or via compose — it is separate from the sandbox image.
- **Your project files stay on the host.** The sandbox mounts only the user's code (read-only) and a small tmpfs `/tmp`. It has no view into your filesystem beyond that.
- **Windows host limitation (verified in v0.16):** `DockerCodeSandbox` refuses Windows hosts even when Docker Desktop's Linux engine is running. `execute_python_code` therefore never registers on this Windows machine. To use code execution, run JARVIS under WSL2 or on Linux.
- **Linux verification status (v0.16, CONFIRMED):** 25 integration tests ran against a real Linux Docker daemon asserting observed behavior: non-root uid, dropped capabilities, `no-new-privileges`, read-only rootfs, noexec tmpfs, no network, bounded PIDs (64), memory-cap OOM kill, timeout killing the workload (exit 124, no orphan containers), and force-removal. Digest `sha256:f77ac9e4…` resolved by real pull.
- **Resource limits (defaults):** 256 MB RAM, 5s CPU quota, 64 processes, 16 MB noexec tmpfs, read-only rootfs, `--rm`, no network.
- **Timeout behavior:** the container entrypoint is `timeout <cap>s python3 …` — a runaway snippet is killed *inside* the container (exit 124). If the host-side wait fires first (hung daemon/CLI), the container is force-removed. `timeout_layer` reports which fired. Not to be confused with the eval harness's thread-join abandonment timeout (see developer manual §17).
- **Security restrictions:** `--network none`, `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--user 65534:65534` (runs as nobody), non-root, code mounted `:ro`.
- **Why you may not see containers in Docker Desktop:** containers are one-shot with `--rm` and exist only while your code runs (≤ timeout). After completion or timeout they are removed automatically. Watching Docker Desktop during a run should show the container only briefly.
- **To enable (Linux/WSL2 only):**

```bash
# 1. Build the dedicated sandbox image (pinned digest base)
./deploy/build-sandbox-image.sh            # prints the SANDBOX_IMAGE value to use

# 2. Configure and verify
# .env:  ENABLE_CODE_EXECUTION=true
#        SANDBOX_IMAGE=<the digest-pinned ref from step 1>
python -m jarvis.maintenance doctor      # sandbox posture check

# 3. Even when enabled, each call parks a confirmation (SYSTEM risk)
```

The tool joins the LLM's surface **only** when config is true **and** Docker verifies usable at startup; otherwise it's absent entirely — the model never sees it.

---

## 16. Permissions and Confirmations

Every tool declares a risk tier:

| Tier | Behavior | Tools |
|---|---|---|
| SAFE / NETWORK / FILE_READ | Auto-run | calculator, datetime, wikipedia, web_search, web_scrape, read_file, list_directory, remember/recall |
| FILE_WRITE | **Blocked outright** (refused) | write_file |
| SYSTEM / DESTRUCTIVE | **Parked for your confirmation** | execute_python_code (when enabled) |

Flow for a high-risk action:

```
model requests tool → PermissionGuard check → tier needs confirmation?
  → action parked durably in SQLite (args + resume context)
  → you see ACTION_REQUIRES_CONFIRMATION (CLI/API/dashboard)
  → /confirm → action runs → remaining plan steps execute → final answer
  → /deny   → action skipped → task continues without it → final answer
```

Durable means restart-safe: the parked action + resume context live in SQLite (TTL 10 min). Kill JARVIS, restart it, then `/confirm` — the task resumes from where it paused.

---

## 17. Example Tasks

1. "What time is it?" — datetime tool.
2. "Calculate 198765 * 4321 / 77" — calculator.
3. "Summarize the Wikipedia article on photosynthesis." — wikipedia_summary.
4. "Search the web for the latest Python 3.13 release notes and summarize the headline features." — web_search.
5. "Scrape https://en.wikipedia.org/wiki/Johannes_Kepler and tell me his three laws." — web_scrape.
6. "Remember that my sister's name is Elena and her birthday is March 3." — remember_fact.
7. (new session) "When is my sister's birthday?" — recall_facts.
8. "List the files in the current directory." — list_directory.
9. "Read README.md and summarize what JARVIS v0.16 changed." — read_file.
10. "Describe the image at C:\Users\youha\Pictures\chart.png" — vision_analyze (needs llava).
11. (voice) Say "What's the weather in Tokyo?" — web_search via voice; "exit" to leave.
12. "First search for the population of Tokyo, then calculate what percentage of Japan's population that is." — planned multi-step (search → search → calculator).
13. "Remember my project deadline is October 15, then search whether that is a Japanese public holiday." — multi-step + memory.
14. "What are my favorite things?" (after you've taught JARVIS facts) — recall_facts; honest "no memories stored yet" if none.
15. (with code execution enabled on Linux) "Run Python code that prints the first 10 Fibonacci numbers." — confirmation flow → sandboxed container.
16. "Write my notes to output.txt" — honest refusal (FILE_WRITE blocked); JARVIS should say writes aren't permitted and offer to show content instead.

---

## 18. Troubleshooting

| Symptom | Cause / Fix (verified commands) |
|---|---|
| `LiteLLM ... Connection error` / 503 on `/chat` | Ollama not running or wrong URL. `curl http://localhost:11434/api/version`; if refused: `ollama serve`. Check `OLLAMA_BASE_URL` in `.env`. |
| `model not found` errors | `ollama list` — pull what's missing: `ollama pull qwen2.5:7b`. Remember the `.env.example`/code default mismatch (§4 note). |
| API `/health` returns 503 | DB probe failed — disk/volume issue. `python -m jarvis.maintenance doctor` localizes it. |
| API 401 | Key set but missing/wrong → send `Authorization: Bearer <key>`. |
| API 429 | Rate limit → honor `Retry-After`, or raise/disable in `.env` (`RATE_LIMIT_REQUESTS=0`). |
| API 409 on `/chat` | Another turn is running on that session — wait or use a different session. |
| `ACTION_REQUIRES_CONFIRMATION…` | Not an error — turn is parked. `/confirm` / `/deny` (CLI) or `POST /sessions/{id}/confirm` (API). TTL is 10 minutes. |
| Vision errors (`Failed to analyze image`) | `ollama pull llava`; check the path is inside `FILE_READER_ALLOWED_DIR`. |
| Voice: "ffmpeg not found" | `winget install FFmpeg` (Windows) then **restart the terminal** so PATH updates. Verify `ffmpeg -version`. |
| Voice: no speech detected | Mic permissions; speak during the "Listening…" window; raise `VOICE_RECORD_SECONDS`. |
| TTS silent / network errors | Edge TTS needs internet; check firewall; try another `TTS_VOICE`. |
| `web_scrape` fails with Playwright error | `uv run playwright install chromium`. |
| Code execution stays disabled | Windows host → use WSL2/Linux. Also check `ENABLE_CODE_EXECUTION=true`, `SANDBOX_IMAGE` pinned (no `latest`/bare tags), image pulled on the target daemon. Log line to look for: `code_execution_enabled_but_docker_unavailable`. |
| Evaluation: all cases fail fast / preflight exit 2 | Ollama unreachable — start it; the harness aborts cleanly with exit 2 instead of 32 cascade failures. |
| Evaluation: a case times out | Raise `--timeout`; note the timeout is an *abandonment* (the model call may still be running server-side). |
| Dashboard "Backend unreachable" | API not running or `JARVIS_API_URL` wrong/mismatched key. |

---

## 19. "What Should I Keep Running?"

| Component | Needed for | Keep running? |
|---|---|---|
| **Ollama** | All model inference (chat, planner, vision) | **Yes** — always |
| **JARVIS API** (uvicorn) | REST/dashboard/streaming/confirmations | Yes while using API/dashboard; not needed for CLI-only use |
| **Streamlit dashboard** | Web UI | Only while you use it; it's a client |
| **CLI process** | Terminal use | Foreground — keep the terminal open while you use it |
| **Docker Desktop** | Sandboxed code execution only | **No** for normal chat (and irrelevant on this Windows host anyway — the sandbox refuses win32) |
| **Random terminals opened by the coding agent** | Nothing, usually | No — they can be closed. Only terminals *hosting a foreground service* (CLI session, `ollama serve` if not auto-started, uvicorn, streamlit) need to stay open |

**Minimum for normal chat:** Ollama + one JARVIS interface (CLI *or* API *or* dashboard).

---

*Every command and behavior in this manual was verified against the actual repository (v0.16.0, September 2026). When code and docs disagree, code wins — and this manual was written from the code.*