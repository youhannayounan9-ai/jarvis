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
| Rate limiting | ACTIVE | 60 req/60s sliding window per client (disable with `0`); multi-process deployments can share one limit via the database-backed limiter | None | — |
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
| `/refresh` | Toggle refresh mode — force fresh (non-cached) tool runs for your next messages (v0.25) |
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

**Authentication behavior:** empty `JARVIS_API_KEY` = local trust. Set → `Authorization: Bearer <key>` (or `X-API-Key`) required on everything except `/health` + docs. Constant-time compare. Wrong/missing key → 401; whitespace-only key config → 503 (fail-closed). Rate limit: 60 req/60s per client → 429 + `Retry-After` (`0` disables). Same-session concurrent turns → 409 (v0.17: enforced across processes via a database-backed session lease, so a second JARVIS process cannot interleave with your session either; a crashed process's lease self-expires).

**Confirmed actions execute at most once (v0.17):** every approved high-risk action gets a durable execution record. Approving twice, retrying the request, or approving after a crash returns the already-recorded outcome instead of running the tool again. If a crash leaves the outcome genuinely unknown (executed, but the result was never recorded), JARVIS **refuses to re-run it automatically** and reports `ACTION_EXECUTION_STATE_UNKNOWN` — re-issue the request deliberately if you want it redone (see below). Denials never execute the tool at all.

**Inspecting and recovering actions (v0.18):** the action ledger and session leases are inspectable, and an UNKNOWN action can be deliberately re-issued. There is **no automatic recovery**: an UNKNOWN action stays UNKNOWN until a human decides.

```bash
# What actions exist? (add ?state=UNKNOWN&session_id=…&limit=…; bounded at 500)
curl http://localhost:8000/actions

# One action's safe metadata (ids, tool, state, timestamps — never tool arguments)
curl http://localhost:8000/actions/<action_id>

# Which sessions are leased, by whom (redacted), until when?
curl http://localhost:8000/sessions/leases

# Deliberately re-issue an UNKNOWN action (NEW action id; original stays UNKNOWN
# for audit; the new action goes through the normal approval flow).
curl -X POST http://localhost:8000/actions/<action_id>/reissue \
  -H "Content-Type: application/json" \
  -d '{"request_id": "recovery-2026-09-27-001"}'
```

Reissue semantics:

- **Explicit only.** Read endpoints never mutate anything; `POST /actions/{id}/reissue` and the `reissue` CLI command are the only mutating operations, and they refuse anything whose state is not exactly UNKNOWN.
- **Idempotent per request id.** Submitting the same `request_id` twice returns the **same** new action — an accidental double-submit cannot create two actions.
- **Bounded.** One original action can be re-issued at most `MAX_REISSUES_PER_ACTION` (3) times.
- **Normal permission flow.** The reissued action is parked as a fresh confirmation for its session; nothing executes until it is approved (or denied) like any other high-risk action. If that session already has an active confirmation awaiting an answer, reissue is refused (409) instead of silently replacing it.
- **Audited.** Every reissue is recorded (`action_reissues` table); the original row is never modified.
- **Authentication applies.** With `JARVIS_API_KEY` set, all `/actions*` and `/sessions/leases` endpoints — including reissue — require the key; unauthenticated calls get 401.

CLI equivalents (see `python -m jarvis.maintenance --help`):

```bash
python -m jarvis.maintenance actions --state UNKNOWN
python -m jarvis.maintenance unknown-actions          # what is ambiguous, why, what next
python -m jarvis.maintenance sessions --expired        # stale leases
python -m jarvis.maintenance reissue --action <id> --request-id <unique-id>
python -m jarvis.maintenance inspect --session <id>    # causal timeline (or --action <id> for one row)
python -m jarvis.maintenance cleanup --operational --dry-run   # retention preview
python -m jarvis.maintenance knowledge list            # v0.20 knowledge base
python -m jarvis.maintenance doctor                    # DB, Ollama + model, leases, UNKNOWN counts, sandbox, voice deps
```

**Operator dashboard (v0.19):** the Streamlit UI has an **Operations** view (sidebar → 🛠️ Operations): recent actions with state/session filters, UNKNOWN actions with recovery guidance, session leases, and per-session timelines. Viewing never changes anything; reissue is the only mutation and asks twice (an acknowledgement checkbox, then a final confirmation). Errors from the API (down, wrong key, 404, 409, 429) appear as messages — the page never crashes.

**Full-context recovery (v0.19):** since v0.19 the original task context (request, plan, completed steps, budgets) is captured when an action is parked. Reissuing an UNKNOWN action copies that context onto the new action's confirmation, so **approving the reissued action continues the original task** — remaining plan steps run and a final answer is synthesized — instead of stopping at a bare result. A denial still never executes the tool. Nothing runs without the normal approval; UNKNOWN is still never retried automatically.

---

## 8b. Personal Knowledge Base (v0.20)

Two different stores — keep them straight:

| | **Personal memory** | **Knowledge base (v0.20)** |
|---|---|---|
| What it is | Facts about you ("I prefer Python") | Documents you own (PDFs, notes, code) |
| Tools | `remember_fact` / `recall_facts` | `search_knowledge` (+ CLI/API management) |
| Storage | Chroma `long_term_memory` | Chroma `knowledge_base` + SQLite registry |
| Semantics | durable until you delete | re-ingest replaces older versions |

**Ingest a document (explicit paths only — JARVIS never crawls your computer):**

```bash
python -m jarvis.maintenance knowledge ingest --path docs/ai_roadmap.md
python -m jarvis.maintenance knowledge list
python -m jarvis.maintenance knowledge search --query "LangGraph phase five" --top-k 5
python -m jarvis.maintenance knowledge inspect --id <document_id>
python -m jarvis.maintenance knowledge reindex --id <document_id>
python -m jarvis.maintenance knowledge remove --id <document_id>      # asks for 'yes'
```

- **Supported formats:** `.txt`, `.md`, source code (`.py`, `.js`, `.ts`, `.java`, `.go`, …), `.json`, `.csv`, `.yaml` — and **PDF** (page numbers preserved) via the new `pypdf` dependency.
- **Path safety:** the resolved file must be inside `FILE_READER_ALLOWED_DIR` (same boundary as `read_file`); traversal and symlink escapes are rejected; `.env`, key/certificate files, and secret-named text files are refused outright.
- **Incremental:** re-ingesting an unchanged file does nothing (no re-embedding); a changed file replaces its old chunks completely.
- **Citations:** retrieval results carry `[Source: <file>, page N, chunk M]` derived only from stored metadata — never invented. If the knowledge base lacks evidence, JARVIS says so instead of guessing.
- **Injection safety:** retrieved document text is delivered to the model as clearly-marked *evidence data*, never as instructions. A document containing "ignore all previous instructions" is still just text to reason about.

Then just ask in chat:

- *"Search my AI Engineering roadmap for LangGraph."*
- *"What does my university PDF say about neural networks?"*
- *"Compare what my two roadmap documents say about phase 5."*
- *"Does my knowledge base contain anything about RAG evaluation?"*

The Streamlit dashboard gained a **📚 Knowledge** view: indexed documents, explicit ingest-by-path, and search with per-chunk citations.

**Errors carry correlation IDs:** every response has `X-Request-ID` (and `/chat` echoes it as `request_id`); the server log line under that ID holds the real error — clients never see internals.

---

## 8c. How JARVIS Decides When to Use Tools (v0.21)

JARVIS does not just dump every tool into the model and hope. Every request first passes a **capability classification** (zero-latency, no LLM call) that asks: does this need **no tool**, a **specific known capability** (calculation, knowledge retrieval, web, memory, files, vision, datetime), or **multiple steps**? The result decides the execution path:

- **Simple requests** ("What is 893 × 47?", "What does my AI roadmap say about LangGraph?") take the fast path, with the tool-selection contract added to the prompt. For clear single-intent **arithmetic** or **knowledge** asks, a safety net requires one tool round — the model may not silently substitute mental arithmetic or pretend it read your documents.
- **Multi-step requests** go to the planner; each planned step now states that tool-backed capabilities must actually use the tool.
- **If the model still refuses the calculator** on a plain arithmetic request, JARVIS runs the calculator itself (same permission checks, same validation), gives the exact result to the model, and lets it phrase the answer — you get the correct number instead of a confidently wrong one. This is deliberately calculator-only: knowledge retrieval stays model-driven.
- **Unavailable capabilities are refused honestly** (v0.16 rule, unchanged): asking it to run Python or control the computer yields a plain "I can't do that", never a pretended attempt — even when the request contains numbers like `print(2+2)`.
- **No-tool questions** ("Explain gradient descent") should answer directly. Small local models occasionally still reach for a tool on such questions — that residual variance is a model limitation, documented in the developer manual.

The result on the reference model (qwen2.5:7b, live A/B): correct tool selection rose from 69% to 81% of focused evaluation cases, with zero fabricated tool names. Date questions route to the calendar (not the calculator), "what does MY document say" reliably enters the knowledge-retrieval path, and every routing decision is visible in the logs (`tool_policy_applied`, `deterministic_tool_fallback`, `no_tool_direct_answer`).

---

## 8d. Multi-Step Tasks (v0.22)

When you ask for something with several parts — "Calculate 893 × 47 and remember the result" — JARVIS plans before it acts:

```
calculate          ↓                ↓
remember the result   ↓                ↓
answer with the real number
```

What v0.22 guarantees:

- **The planner sees what tools do.** Plans are built from a compact catalogue of your actual tools (purpose per tool), so steps name real capabilities — and any hallucinated tool name is removed before execution.
- **Plans are validated.** Step count, required fields, duplicate steps, and impossible dependencies ("use the result of step 4" in step 2) are caught deterministically before anything runs.
- **Results flow between steps.** Each step sees the exact (size-bounded) tool results of earlier steps — the memory step stores the calculator's actual number, not a paraphrase of it.
- **Tool steps use tools.** A step that names a required tool must attempt it; the executor may not quietly answer from memory.
- **Corrections stay bounded.** A failed tool call feeds one recovery round; a scripted-failure live test showed the agent recovering and completing the task.
- **Every execution still passes the same permission checks** — planning never authorizes anything.

Honest limits: the local 7B model still over-plans tools for pure-explanation questions occasionally; exact multi-step tool ordering is currently at ~0.6 of focused live runs (all *required* tools run in 1.0, order variance comes from extra calls — and since v0.23, redundant extra calls are usually suppressed before they run, see §8e). Single-capability tasks remain more reliable than chained ones.

---

## 8e. Repeated Work & Knowing When to Stop (v0.23)

**Why did JARVIS used to repeat itself?** A small local model drives every decision, and when a step doesn't produce the phrasing it hoped for, its instinct is to try the same tool again — identical search, identical query. That wastes time and clutters the answer.

**What stops it now:** within a single conversation turn, JARVIS remembers every tool call that **succeeded** — by tool *and* exact arguments, not just tool name. If the model emits the very same call again, JARVIS blocks it before it runs and tells the model: *that result is already above — use it.* So "search the web for X, then search the web for X again" runs one real search.

**What still repeats — on purpose:**

- **Different arguments are different work.** Searching for LangGraph *then* for checkpointing runs both searches; only identical argument repeats are blocked.
- **A failed call can be retried.** Only *successful* calls are remembered, so a transient error never locks the tool away.
- **Genuinely fresh information is exempt.** The clock and your memory (`get_current_datetime`, `recall_facts`, `remember_fact`) may legitimately be asked twice — the answer can change or matter twice. Web and knowledge searches are *not* exempt: an external page doesn't change between two identical searches seconds apart.
- **New turn, clean slate.** The ledger lives for one turn only; asking the same thing tomorrow really searches again.

**Redundant plan steps** are skipped too: if the plan contains two steps described identically, the second is marked *skipped: identical to an earlier completed step* instead of re-running.

**How completion is known:** every plan step must either produce its required tool evidence or legitimately not need one; JARVIS logs which steps were completed, failed, or skipped, and only synthesizes the final answer once the plan is done (or its failure budget is spent). The Dashboard's Operations page shows the same picture per session: executions, successes, failures, and a repeated-tool warning that notes repeats may be legitimate or suppressed.

Honest limits: suppression matches *exact arguments* — `"2+2"` and `"2 + 2"` are different strings and fingerprint differently; the model re-running a search with reworded arguments is doing different (legitimate) work by this rule. (For the *calculator* specifically, v0.24's cross-turn cache DOES close this gap — see §8f.)

---

## 8f. Remembering Across Turns & Recovering from Failure (v0.24)

v0.23 stopped repeated work *inside* one turn (§8e). v0.24 adds the two pieces that were still missing: useful retrieval is remembered **across turns**, and a half-finished plan gets **one honest second chance**.

### Cross-turn reuse (the result cache)

Ask the same question twice — even days apart — and JARVIS may not need to repeat the work:

- **What is reused:** read-only retrieval only — `web_search`, `wikipedia_summary`, `web_scrape`, `read_file`, `list_directory`, `search_knowledge`. Cacheability is declared per tool; anything that *does* something (writing files, vision, code execution, computer control) or is *supposed* to change (`get_current_datetime`, your memory) is **never** served from cache — it always really runs.
- **Freshness is per tool:** web results expire after ~5 minutes, Wikipedia after a day, the knowledge-base retrieval is invalidated the moment you ingest or remove a document, and file/directory answers are re-checked against the file's size + modification time. A changed file ⇒ the cache is not trusted.
- **You are told:** a reused answer is prefixed in the model's context with a provenance note — *"cached result: retrieved 5m ago … not a live re-run"* — so JARVIS can qualify it instead of presenting stale data as live.
- **You are in control:** ask for the *latest*, *current*, *today*, *breaking*, etc., and JARVIS skips time-sensitive cache entries and fetches fresh data. There is no configuration surface to learn — your wording is the switch.
- **Privacy:** your files and your knowledge base are cached **per session**; only public, deterministic content (e.g. calculator results, public web lookups) is shared between sessions. The maintenance CLI (`maintenance cache stats / inspect / cleanup`) shows counts and fingerprints, never payloads.
- **The calculator is exact:** the cache keys on the *evaluated value* using the calculator's own parser, so `2+2`, `2 + 2`, `(2+2)` and `5-1` all reuse one entry. Different values never collide.

Power users: set `JARVIS_DISABLE_RESULT_CACHE=true` to restore always-fresh behavior.

### The one bounded replan

When a multi-step task **structurally fails** — a required tool errored and could not be completed (not merely a verbose or odd-looking answer) — JARVIS now gets **one** automatic second chance:

1. the original plan runs; completed work is kept;
2. if a required step failed **and** tool budget remains, a *replan* is generated for **only the unfinished requirements** (already-done steps are explicitly excluded);
3. the replan goes through the same validation, permissions, and repeat guards as any plan, within the **remaining** budget — nothing is reset;
4. if the replan also fails, JARVIS **tells you plainly what remains unfinished** instead of pretending the task completed. The Dashboard's Operations page (`Result cache` section) and the structured logs (`replan_triggered`, `plan_completed … complete=true/false`) show all of this.

Recursion is impossible by construction: the replan path cannot trigger another replan. A plan is still a request, never an authorization. Every replan is also **diffed** against the plan that failed — the `replan_diff` log/SSE event shows which steps were added, removed, or retargeted (structure only, never tool arguments).

---

## 8g. Grounded Answers, Refresh Mode & Cache Metrics (v0.25)

### Answers come from the evidence, not the model's memory

When tools run during a turn, JARVIS now keeps a bounded **evidence ledger** of their raw results and hands it to the final answer step as **AUTHORITATIVE TOOL EVIDENCE**, with one instruction: transcribe tool-derived values exactly; never recompute them. This closes a real, live-observed gap — the model once announced 42071 for 893 × 47 even though the calculator had correctly produced 41971. The ledger is bounded (at most 16 items, 1200 characters each) and is framed as data, never as instructions.

### Refresh mode (client-controlled)

You can force JARVIS to ignore cached results and really re-run the tools:

- **CLI:** start with `--refresh`, or toggle `/refresh` during a session (state shown in the banner/help).
- **Dashboard:** the sidebar has a **Refresh mode (bypass cache)** toggle.
- **API:** `POST /chat` accepts `"refresh": true` (the Python client takes `refresh=`).

Refresh skips **only the result cache**. Permissions, confirmation gates, repeat-suppression, and schema validation all still apply — a refresh of a high-risk action still parks for your approval. Asking for the *latest/today/current* still bypasses time-sensitive entries automatically, as before.

### Cache metrics over time

Every turn records hit/miss/stale/bypass/store counters (per tool) into a small daily table kept for 30 days by default (`RESULT_CACHE_METRICS_RETENTION_DAYS`). View it three ways: the dashboard's **Daily cache activity (last 14 days)** table, the API `GET /ops/cache/stats/history`, or `maintenance cache stats`, which now prints today's counters and per-tool deltas.

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
| `ACTION_EXECUTION_STATE_UNKNOWN…` | A crash left the outcome ambiguous; JARVIS will NOT re-run it automatically. Inspect (`python -m jarvis.maintenance unknown-actions` or `GET /actions?state=UNKNOWN`), verify the real-world effect, then re-issue deliberately (`reissue --action <id> --request-id <unique-id>`, max 3 per action). |
| API 409 on `/actions/{id}/reissue` | The action is not UNKNOWN, the reissue limit (3) is reached, or the session already has an active confirmation — read the `detail` message. |
| Dashboard shows "Authentication required" | The API has `JARVIS_API_KEY` set — set `JARVIS_CLIENT_API_KEY` to the same value so the dashboard can read the Operations view. |
| `recovery: no durable resume context…` in `unknown-actions`/`inspect` | A pre-v0.19 UNKNOWN row (or a park without a plan) — reissue still works; the result is reported directly instead of resuming a plan. |
| `doctor` reports stale leases | Another process died holding a lease. `python -m jarvis.maintenance sessions --expired` lists them; they self-expire after the TTL (300 s) and are safe to take over. |
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