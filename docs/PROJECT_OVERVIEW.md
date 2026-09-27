# JARVIS — Project Overview

**Version:** 0.16.0
**License:** MIT
**Tagline:** A local-first AI assistant — one agent runtime, multiple interfaces. Powered by Ollama. No paid APIs. No cloud lock-in.

This is a handoff-style overview: what JARVIS is today, what it can and cannot do, and where it is headed. For layer-by-layer internals see [`architecture.md`](architecture.md).

---

## 1. Vision

JARVIS aims to become a **fully local personal AI assistant** that lives on the user's machine: private by default, extendable through tools, and able to *act* through tools rather than only chat — with a permission model that makes acting safe.

**Non-negotiables:**

- Runs locally (Ollama + local Python app)
- Free stack for core features (no required paid LLM or search APIs)
- Modular architecture: features grow without rewriting the core
- Privacy-first: conversation data stays on disk under the user's control
- **Safety is structural, not prompt-based:** dangerous capability is absent or gated, never merely discouraged

---

## 2. What Exists Today (v0.10)

| Capability | Status |
|---|---|
| Local LLM chat via Ollama (LiteLLM wrapper) | Done |
| Plan-and-Execute agent loop + fast ReAct path with intent routing | Done |
| Tool calling: web search, scrape, Wikipedia, files, calculator, vision, facts | Done (11 active tools) |
| Persistent session memory (SQLite, thread-safe) | Done |
| Context management: windowing, anchoring, rolling summary, tool-output clamping | Done |
| Long-term memory (ChromaDB) + `remember_fact` / `recall_facts` | Done |
| Permission tiers + durable SQLite confirmations for high-risk tools | Done |
| Resumable confirmations: approval/denial continues the original task (restart-safe) | Done |
| Sandbox timeouts enforced at the workload boundary (container `timeout` + host force-removal) | Done |
| Self-correction loop on tool errors (with recovery hints) | Done |
| Async/concurrent tool dispatch per round | Done |
| REST API (FastAPI): `/chat`, SSE `/chat/stream`, opt-in API-key auth, rate limiting, per-session serialization, sessions, confirmations, tool surface | Done |
| Stdlib API client (`jarvis/api/client.py`); dashboard consumes the API (optional legacy fallback) | Done |
| Dedicated minimal sandbox image + immutable-image policy (no `latest`) in `deploy/` | Done |
| Container deployment: `Dockerfile`, `docker-compose.yml` (+ optional Ollama profile), CI workflow, maintenance CLI, request IDs | Done |
| Operational readiness: deep health (503 on broken persistence), sanitized correlated errors, WAL persistence, `doctor` command | Done |
| CLI (Rich REPL) and Streamlit dashboard over the same runtime | Done |
| Voice mode (Whisper STT + Edge TTS) | Done (needs internet for TTS) |
| Agent-step observability (`on_event` hook + true interleaved SSE + keepalives) | Done |
| Live-model eval harness (32 cases, per-case timeouts, category breakdowns, JSON reports, nightly CI) | Done |
| **Code execution (`execute_python_code`)** | **Opt-in, Docker-isolated, off by default; registers only when Docker verifies usable (not on Windows hosts)** |
| **Computer control (`computer_control`)** | **Disabled — pure placeholder, no OS automation** |

**Default model:** `qwen2.5:7b` (tool-calling capable), planner on the same model (`planner_model` config).
**Runtime requirement:** Ollama running locally (`ollama serve` or the desktop app). The REST API does not need Ollama to boot, only to answer chats.

---

## 3. High-Level Architecture

```
        CLI (main.py)        REST API (jarvis/api)      Dashboard (Streamlit)     Voice
              │                       │                        │                  │
              └───────────────────────┼────────────────────────┘                  │
                                      ▼                                           │
                    JarvisRuntime (jarvis/runtime.py)  ◄──────────────────────────┘
                    build_runtime() → store + 11 tools + guard + orchestrator
                                      │
        ┌─────────────────┬───────────┼──────────────────┬─────────────────┐
        ▼                 ▼           ▼                  ▼                 ▼
  Orchestrator       ToolRegistry  PermissionGuard  SessionStore      LLM client
  route→plan→        (capabilities) (risk tiers +   (SQLite,          (LiteLLM →
  execute→synthesize               confirmations)   thread-safe)      Ollama)
```

The one rule that keeps this coherent: **interfaces never wire components themselves** — they call `build_runtime()` and talk to the returned `JarvisRuntime`. CLI, API, and dashboard therefore always expose the identical capability surface.

### Request lifecycle

1. User turn is persisted to SQLite.
2. Context is loaded and compacted (`ContextManager`: window → anchor → summarize → clamp).
3. `route_intent` classifies: `simple` → direct ReAct loop; `complex` → Plan → Execute → Synthesize.
4. Tools dispatch through `PermissionGuard` (auto-allow / block / confirm-and-persist).
5. Tool errors trigger a bounded self-correction loop with recovery hints.
6. A final synthesis call produces the answer.

Details (including the confirmation flow and context budget math): [`architecture.md`](architecture.md).

---

## 4. Repository Structure (v0.10)

```
JARVIS/
├── jarvis/
│   ├── runtime.py               # Assembly seam: build_runtime() + JarvisRuntime
│   ├── api/                     # FastAPI service layer
│   │   ├── app.py               #   /health /sessions /chat /chat/stream /confirm /tools
│   │   ├── auth.py              #   Opt-in API-key enforcement (constant-time)
│   │   └── schemas.py           #   Typed request/response models
│   ├── main.py                  # CLI entry + REPL (uses build_runtime())
│   ├── config.py                # Single settings surface (pydantic-settings)
│   ├── core/
│   │   ├── orchestrator.py      # Route → Plan → Execute → Synthesize
│   │   ├── planner.py           # JSON plan generation with fallback
│   │   ├── permissions.py       # PermissionGuard risk tiers
│   │   └── sandbox.py           # CodeSandbox ABC + fail-closed implementations
│   ├── llm/client.py            # LiteLLM → Ollama
│   ├── memory/
│   │   ├── session_store.py     # SQLite sessions/messages/confirmations (RLock)
│   │   ├── context_manager.py   # Windowing, anchoring, summary, clamping
│   │   └── vector_store.py      # ChromaDB long-term memory
│   ├── tools/                   # 11 active tools + disabled placeholders
│   └── voice/                   # Whisper + TTS
├── ui/dashboard.py              # Streamlit dashboard (consumes the runtime)
├── evaluation/                  # Offline eval harnesses (run_evals, tool_selection)
├── tests/                       # Offline pytest suite
├── docs/                        # architecture.md, this file
└── pyproject.toml
```

### Design principles

1. **One assembly point** — `build_runtime()` is the only place components are wired; interfaces stay thin.
2. **Dependency injection** — `Orchestrator` receives store/registry/guard; nothing constructs its own DB or LLM client.
3. **Explicit capability surface** — tools are registered in a visible tuple (`_TOOL_FACTORIES`), not auto-discovered.
4. **Tool contract** — every tool subclasses `BaseTool`, declares a `risk_level`, exposes JSON Schema, returns a string (`ERROR:` prefix marks failure). Exceptions are caught at the registry boundary.
5. **Config centralization** — `jarvis/config.py` is the single settings surface.
6. **Fail-closed dangerous paths** — disabled tools are absent from the registry *and* fail closed if ever instantiated.

---

## 5. Current Tools (Capability Surface)

| Tool name | Purpose | Risk tier |
|---|---|---|
| `get_current_datetime` | Local date, time, weekday, UTC offset | SAFE |
| `calculator` | Restricted AST math eval (no arbitrary code) | SAFE |
| `wikipedia_summary` | Encyclopedia summaries (free API) | NETWORK |
| `web_search` | DuckDuckGo via `ddgs`, TTL-cached | NETWORK |
| `web_scrape` | Playwright-based page extraction | NETWORK |
| `vision_analyze` | Image understanding (Llava via Ollama) | SAFE |
| `read_file` | Text file read, sandboxed to `FILE_READER_ALLOWED_DIR` | FILE_READ |
| `list_directory` | Directory listing, same sandbox | FILE_READ |
| `remember_fact` / `recall_facts` | Long-term personal memory (ChromaDB) | SAFE |
| `write_file` | File write | **FILE_WRITE — currently blocked by the guard** |
| `execute_python_code` | Code execution | **DISABLED — not registered; sandbox not implemented** |
| `computer_control` | OS automation | **DISABLED — not registered; pure placeholder** |

CLI meta-commands (not LLM tools): `/help`, `/tools`, `/history`, `/confirm`, `/deny`, `/new`, `/voice`, `/quit`.

---

## 6. Memory Model

### Short-term (SQLite)
- Sessions + messages tables; OpenAI-compatible roles; the system prompt is never persisted.
- Every read/write goes through a `threading.RLock`, so CLI, API workers, and the dashboard can share a store safely.
- History loading trims orphan `tool` messages so a window never starts mid tool-call chain.

### Prompt-facing (ContextManager)
- 24-message LLM window (`MAX_CONTEXT_MESSAGES`), older turns compacted into a deterministic `[Context summary]`, original task anchored, tool outputs clamped to 6000 chars (head+tail, full payload kept in SQLite).

### Long-term (ChromaDB)
- `remember_fact` / `recall_facts` give the agent durable personal memory; the vector store binds to a session on `start_session()`.

---

## 7. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Tooling ecosystem |
| Packaging | `pyproject.toml` (hatchling), run via `uv` | Reproducible envs |
| LLM runtime | Ollama | Local inference, no cloud bill |
| LLM glue | LiteLLM | Provider-portable model strings |
| API | FastAPI + uvicorn | Typed service layer, async-friendly |
| Persistence | sqlite3 | Zero-ops durability |
| Long-term memory | ChromaDB | Local vector store |
| Search / scrape | `ddgs`, Playwright | Free retrieval |
| CLI / UI | Rich / Streamlit | Terminal polish / quick dashboards |
| Config | pydantic-settings + `.env` | Typed, fail-fast |
| Tests | pytest | Fully offline suite |

---

## 8. How to Run

Prerequisites: Python 3.11+, Ollama installed with a model pulled (`ollama pull qwen2.5:7b`).

```bash
uv sync --extra dev          # install dependencies
cp .env.example .env         # defaults are fine for local Ollama

# CLI (primary interface)
uv run jarvis
# or: uv run python -m jarvis.main

# REST API (v0.10)
uv run uvicorn jarvis.api.app:app --port 8000
# interactive docs: http://localhost:8000/docs

# Dashboard
uv run streamlit run ui/dashboard.py

# Tests (offline, no Ollama needed)
uv run pytest -v

# Evaluation harness (needs a live Ollama; see flags with --help)
uv run python evaluation/run_evals.py --json report.json
```

---

## 9. Security & Safety Posture

- **Local-first:** chat history stays in local SQLite; ChromaDB lives under `jarvis_data/`.
- **Dangerous tools are gated, not prompted:** `computer_control` is never registered (pure placeholder). `execute_python_code` registers only behind the double gate (config flag + verified Docker), runs in a hardened one-shot container, and every failure mode is a denial — never host execution.
- **Permission tiers:** SAFE/NETWORK/FILE_READ auto-allowed; SYSTEM/DESTRUCTIVE require persisted confirmation; FILE_WRITE currently blocked outright by the guard.
- **Durable confirmations:** pending high-risk actions survive restarts and can be resolved from any interface.
- **No `exec()`/`eval()`** on model-influenced strings; the calculator uses a restricted AST walker; the code sandbox contract denies by default.
- **Hard loop caps:** 5 tool rounds per request, 2 per step, 2 self-correction attempts.

Full details: [`architecture.md` § Safety Model](architecture.md).

---

## 10. Extending the Project

### Add a tool
1. `jarvis/tools/my_tool.py` subclassing `BaseTool` (`name`, `description`, `parameters`, `risk_level`, `run()`).
2. Add to `_TOOL_FACTORIES` in `jarvis/runtime.py`.
3. Export from `jarvis/tools/__init__.py`; add tests; optionally add one prompt-guidance line in `config.py`.

### Read-first onboarding order
1. `README.md` — install & run
2. `docs/JARVIS_USER_MANUAL.md` — operating JARVIS: capability matrix, prerequisites, startup, troubleshooting
3. `docs/architecture.md` — layers, lifecycle, safety model
4. `docs/JARVIS_DEVELOPER_MANUAL.md` — architecture rationale, request lifecycle, extension guides, debugging flow
5. `jarvis/runtime.py` — the assembly seam
6. `jarvis/core/orchestrator.py` — the agent loop
7. `jarvis/memory/context_manager.py` + `session_store.py`
8. `jarvis/api/app.py` — service surface

---

## 11. Roadmap

| Version | Goal | Status |
|---|---|---|
| v0.1 | CLI + tools + SQLite memory | ✅ |
| v0.2 | ChromaDB long-term memory | ✅ |
| v0.3 | Plan-and-Execute loop | ✅ |
| v0.4 | Voice (Whisper + Edge TTS) | ✅ |
| v0.5 | Streamlit dashboard + write_file | ✅ |
| v0.6 | Vision + Playwright scrape | ✅ |
| v0.8 | Permission tiers + durable confirmations + caching | ✅ |
| v0.9 | Service-oriented: JarvisRuntime seam + FastAPI API layer | ✅ |
| v0.10 | Real Docker-isolated code execution (opt-in), API auth, SSE streaming, observability seam | ✅ |
| v0.11 | Deploy-ready service: rate limiting, per-session serialization, dedicated sandbox image, API-first dashboard | ✅ |
| v0.12 | Ship path: docker-compose, service image, CI, request IDs, maintenance CLI, dispatch dedup | ✅ |
| v0.13 | Operate: deep health, sanitized correlated errors, WAL persistence, `doctor`, shared test fakes | ✅ |
| v0.14 | Live-model quality loop: practical eval harness, nightly CI evals, true SSE interleaving, planner grounding | ✅ |
| v0.15 | Reliability & security closure: workload-timeout sandbox, resumable confirmations, CI trust boundary | ✅ |
| **v0.16** | **Production readiness & live validation: digest-pinned sandbox verified against a real Docker daemon, 32-case live-model evaluation run and classified, sharper eval signal (`--compare`, failure details), sandbox integration test suite** | ✅ **current** |
| v0.17 (recommended) | Confirmation/action idempotency keys; semantic eval layer (LLM judge as a *secondary*, non-gating signal); WebSocket token streaming; shared rate-limit store for multi-replica | Planned |
| Later | Long-term memory consolidation, scheduled tasks, multi-user sessions | Exploratory |

### End-state picture
A private assistant that talks in text and voice, sees images, searches and reads the web, reads/writes files and automates the computer **under explicit, durable permissions**, remembers important things long-term, and offers CLI, web, and API interfaces — while staying local-first and model-swappable.

---

## 12. One-Paragraph Summary

**JARVIS is a local-first AI assistant (v0.16) built in Python.** One agent runtime (Plan-and-Execute + fast ReAct path, 11 tools plus an opt-in Docker-isolated code-execution tool, permission tiers, SQLite sessions, ChromaDB memory, context management, self-correction) is shared by three interfaces: a Rich CLI, a FastAPI REST service with optional API-key auth and SSE streaming, and a Streamlit dashboard. It runs on Ollama via LiteLLM with no paid APIs. Computer control remains structurally disabled; code execution is real container isolation (digest-pinned image, runtime-verified boundary on Linux) but ships off unless explicitly enabled and Docker verifies usable. The roadmap continues toward action idempotency, a secondary semantic eval layer, and richer long-term memory.

---

*Current codebase version: **0.16.0**.*
