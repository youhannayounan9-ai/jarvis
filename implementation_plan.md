# JARVIS — Personal AI Assistant: Architectural Blueprint

> A phased, production-quality implementation plan for review before any code is written.

---

## A. High-Level Architecture

JARVIS is designed as a **modular, event-driven assistant platform**. At the highest level it has three tiers:

```
┌─────────────────────────────────────────────────────────┐
│                     INTERFACE TIER                       │
│  Web UI  │  Voice I/O  │  CLI  │  REST/WebSocket API    │
└──────────────────────┬──────────────────────────────────┘
                       │  (events / messages)
┌──────────────────────▼──────────────────────────────────┐
│                      CORE TIER                           │
│  Orchestrator  │  Session Manager  │  Tool Router        │
│  LLM Client    │  Memory Manager   │  Permission Guard   │
└──────────────────────┬──────────────────────────────────┘
                       │  (tool calls / data requests)
┌──────────────────────▼──────────────────────────────────┐
│                    SERVICES TIER                         │
│  Tools & Agents  │  RAG / Knowledge  │  Automation       │
│  File I/O        │  Web Retrieval    │  Vision           │
│  Data Analysis   │  Code Execution   │  IoT / Robotics   │
└─────────────────────────────────────────────────────────┘
```

**Key design principles baked into this architecture:**
- The Orchestrator is the single entry point for all requests; nothing bypasses it.
- Every tool is registered with explicit permissions; the Permission Guard enforces them at runtime.
- The LLM is treated as a *reasoning component*, not the whole system — it calls tools, it doesn't run them directly.
- All secrets flow through environment variables / a secrets manager, never through source code.

---

## B. Major Subsystems / Modules

| # | Subsystem | Responsibility |
|---|-----------|----------------|
| 1 | **Interface Layer** | All user-facing I/O: web UI, CLI, voice, REST API |
| 2 | **Orchestrator** | Routes incoming messages, manages conversation turns, decides when to call tools |
| 3 | **LLM Client** | Abstraction over any LLM provider (OpenAI, Anthropic, local models); handles prompt formatting, retries, streaming |
| 4 | **Session & Context Manager** | Maintains per-session conversation history, token budgeting, context compression |
| 5 | **Memory Manager** | Persistent memory: short-term (Redis/in-memory), long-term (vector DB), episodic recall |
| 6 | **Tool Registry & Router** | Registers tools with schemas, routes LLM tool-call requests to the correct implementation |
| 7 | **Permission Guard** | Enforces what tools are allowed per context (user-approved, sandboxed, etc.) |
| 8 | **Built-in Tools** | Web search, file read/write, code execution, data analysis, calendar, etc. |
| 9 | **RAG / Knowledge Engine** | Document ingestion, chunking, embedding, vector search, retrieval-augmented generation |
| 10 | **Agent Framework** | Multi-step planning agents (ReAct, Plan-and-Execute) for complex tasks |
| 11 | **Voice I/O** | STT (speech-to-text) and TTS (text-to-speech) pipelines |
| 12 | **Vision Module** | Image/video understanding, screen capture analysis |
| 13 | **Computer Interaction Module** | Controlled mouse/keyboard automation with explicit permission model |
| 14 | **Secrets & Config** | Centralized, environment-based secrets management; hot-reloadable config |
| 15 | **Observability** | Structured logging, tracing (conversation turns, tool calls), cost tracking |

---

## C. Recommended Technology Stack

### Runtime & Language

| Technology | Role | Justification |
|------------|------|---------------|
| **Python 3.11+** | Primary language | Best ecosystem for AI/ML, LLM SDKs, tool libraries. Async-first with `asyncio`. Widely documented. |
| **`uv`** | Package / env manager | Dramatically faster than pip/venv; drop-in compatible; reproducible lockfiles. Replaces pip + virtualenv. |

### Web & API Server

| Technology | Role | Justification |
|------------|------|---------------|
| **FastAPI** | REST + WebSocket server | Async, typed, auto-generates OpenAPI docs. First-class WebSocket support for streaming LLM responses. Widely adopted in AI services. |
| **Uvicorn** | ASGI server | Production-grade async server, pairs naturally with FastAPI. |

### LLM Integration

| Technology | Role | Justification |
|------------|------|---------------|
| **LiteLLM** | LLM abstraction layer | Single unified interface for 100+ providers (OpenAI, Anthropic, Gemini, Ollama, local). Prevents vendor lock-in from day one. |
| **OpenAI-compatible tools spec** | Tool/function calling schema | Industry standard for structured tool calls; supported across all major providers. |

### Memory & Persistence

| Technology | Role | Justification |
|------------|------|---------------|
| **SQLite → PostgreSQL** | Structured persistence (sessions, users, logs) | SQLite for v0.1 (zero-ops), migrate to PostgreSQL when scaling is needed. |
| **ChromaDB** | Vector store (RAG / long-term memory) | Embeds locally with no infrastructure in v0.1; same API scales to hosted/cloud later. |
| **Redis (later)** | Short-term/session cache | Introduced only when horizontal scaling is needed; not in v0.1. |

### Frontend (Web UI)

| Technology | Role | Justification |
|------------|------|---------------|
| **React + TypeScript** | Web interface | Strong type safety for a complex UI. Component model is well-suited to a chat + tool-result rendering system. |
| **Vite** | Build tool | Fast dev server, modern ESM-first bundling. |
| **CSS Modules / Vanilla CSS** | Styling | Avoids framework churn; gives full control over the futuristic visual design. |

### Voice

| Technology | Role | Justification |
|------------|------|---------------|
| **OpenAI Whisper (local)** | Speech-to-Text | Free, runs offline, high quality. Can fall back to cloud Whisper API. |
| **Pyttsx3 / Edge TTS (v0.x) → ElevenLabs (later)** | Text-to-Speech | Pyttsx3 is zero-cost/offline for early versions; ElevenLabs for premium voice quality in later phases. |

### Automation & Vision

| Technology | Role | Justification |
|------------|------|---------------|
| **PyAutoGUI / `pywinauto`** | Computer control | Cross-platform mouse/keyboard control. Introduced only when needed with strict permission gating. |
| **Playwright** | Browser automation | More reliable than Selenium for web tasks; async-native. |
| **OpenCV / PIL** | Image processing | Standard libraries; no vendor dependency. |

### Security & Config

| Technology | Role | Justification |
|------------|------|---------------|
| **`python-dotenv`** | Local secret loading | Industry standard `.env` approach; secrets stay out of code. |
| **Pydantic Settings** | Config validation | Type-safe config parsing; integrates with FastAPI naturally. |

### Observability

| Technology | Role | Justification |
|------------|------|---------------|
| **`structlog`** | Structured logging | JSON-first logs; easier to search and parse than `logging`. |
| **OpenTelemetry (later)** | Distributed tracing | Standard; introduced when system grows beyond a single process. |

---

## D. Phased Development Roadmap

### Phase 0 — Foundation (v0.1)
**Goal:** A minimal, working text-based assistant that can converse and call basic tools.

- ✅ Project structure, secrets management, config system
- ✅ FastAPI backend with a `/chat` endpoint (streaming SSE)
- ✅ LiteLLM integration (OpenAI, Anthropic, or local model)
- ✅ Basic tool calling: `web_search`, `get_time`, `read_file`
- ✅ Simple React chat UI (text-in, streamed text-out)
- ✅ SQLite session storage (conversation history)
- ✅ Permission Guard skeleton
- ✅ Structured logging

---

### Phase 1 — Intelligence Layer (v0.2 – v0.3)
**Goal:** Make JARVIS smarter about remembering and retrieving information.

- Persistent memory (ChromaDB vector store)
- RAG pipeline: document ingestion, chunking, embedding, retrieval
- Improved context compression (summarize old history, stay within token budget)
- Additional tools: `run_python_code`, `read_directory`, `write_file`
- Basic agent loop (ReAct pattern for multi-step tasks)
- Improved UI: tool-call visualization, memory panel

---

### Phase 2 — Voice & Vision (v0.4 – v0.5)
**Goal:** Make interaction multimodal.

- Voice Input: Whisper STT integration
- Voice Output: TTS pipeline
- Wake word detection (local, e.g., `pvporcupine` or `openwakeword`)
- Vision: image understanding via multimodal LLM
- Screen capture + describe
- UI: voice waveform visualization, image display

---

### Phase 3 — Automation (v0.6 – v0.7)
**Goal:** JARVIS can act on the computer under controlled conditions.

- Playwright browser automation tool
- PyAutoGUI computer control (with explicit user-approval flow for each action)
- Sandboxed code execution environment (Docker)
- Advanced agent: Plan-and-Execute for multi-step autonomous tasks
- Security hardening: per-tool permission levels, audit log

---

### Phase 4 — Ecosystem & Scale (v0.8+)
**Goal:** Productionize and extend.

- PostgreSQL migration
- Redis session cache
- Multi-user support with auth
- Plugin/extension system for third-party tools
- Mobile-friendly UI / PWA
- IoT/robotics integration hooks
- CI/CD pipeline, containerization (Docker Compose)
- ElevenLabs or equivalent high-quality voice

---

## E. Recommended Scope for v0.1

> The first version should be **shippable in 1–2 weeks** and prove the architecture works end-to-end.

### In Scope for v0.1:
- [x] Project scaffolding (folders, config, secrets)
- [x] FastAPI backend: `/chat` endpoint with streaming
- [x] LiteLLM integration (plug in any API key)
- [x] 3–4 basic tools: `get_datetime`, `web_search` (DuckDuckGo API, no key), `read_file`, `list_directory`
- [x] Tool calling loop (parse LLM tool requests → run tool → return result)
- [x] SQLite conversation history (per session)
- [x] React + Vite frontend: chat interface with streaming display
- [x] `.env`-based secrets, Pydantic config validation
- [x] Structured logging with `structlog`
- [x] README with setup instructions

### Out of Scope for v0.1:
- ❌ Voice (any form)
- ❌ Vision / image understanding
- ❌ RAG / vector memory
- ❌ Autonomous agents
- ❌ Computer automation
- ❌ User authentication
- ❌ Docker / deployment

---

## F. Proposed Project Folder Structure

```
JARVIS/
│
├── .env.example                  # Template for secrets (never .env itself)
├── .gitignore
├── README.md
├── pyproject.toml                # Python project config (uv)
├── uv.lock                       # Reproducible lockfile
│
├── jarvis/                       # Python backend package
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Pydantic Settings — config & secrets loading
│   │
│   ├── api/                      # HTTP / WebSocket route handlers
│   │   ├── __init__.py
│   │   ├── chat.py               # /chat endpoint (streaming)
│   │   └── health.py             # /health endpoint
│   │
│   ├── core/                     # Business logic, no framework dependencies
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Central request handler
│   │   ├── session.py            # Conversation session management
│   │   └── permissions.py        # Permission guard
│   │
│   ├── llm/                      # LLM abstraction layer
│   │   ├── __init__.py
│   │   ├── client.py             # LiteLLM wrapper
│   │   └── streaming.py          # Streaming response handling
│   │
│   ├── tools/                    # Tool implementations
│   │   ├── __init__.py
│   │   ├── registry.py           # Tool registry & router
│   │   ├── base.py               # Base tool interface / schema
│   │   ├── datetime_tool.py
│   │   ├── web_search.py
│   │   ├── file_reader.py
│   │   └── directory_lister.py
│   │
│   ├── memory/                   # Memory subsystem (expanded in Phase 1)
│   │   ├── __init__.py
│   │   └── session_store.py      # SQLite-backed session history
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py            # structlog setup
│
├── ui/                           # React + Vite frontend
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── ChatWindow.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── InputBar.tsx
│       │   └── ToolCallDisplay.tsx
│       ├── hooks/
│       │   └── useChat.ts        # Manages streaming SSE connection
│       ├── styles/
│       │   └── global.css
│       └── types/
│           └── index.ts
│
├── tests/                        # Backend tests (pytest)
│   ├── conftest.py
│   ├── test_tools.py
│   ├── test_orchestrator.py
│   └── test_api.py
│
└── docs/                         # Architecture docs
    ├── architecture.md
    └── tool_spec.md
```

**Key decisions in this structure:**
- `jarvis/core/` is framework-agnostic; FastAPI only lives in `jarvis/api/`. This means the core can be tested without a running server.
- `jarvis/tools/` is a flat, easily discoverable collection; each tool is its own file.
- `ui/` is a completely separate concern — could be replaced with a CLI or mobile app without touching the backend.
- Tests mirror the `jarvis/` module structure.

---

## G. Major Risks & Engineering Concerns

### 🔴 Critical

| Risk | Detail | Mitigation |
|------|--------|------------|
| **Secret leakage** | API keys in source control or logs | `.env` only, `.gitignore` enforced, log scrubbing for keys |
| **Unbounded LLM costs** | A runaway agent loop can make thousands of API calls | Hard token/call limits per session from day one; cost tracking |
| **Computer automation safety** | Automation tools can cause irreversible damage (delete files, send emails) | Permission Guard with explicit user-approval step before any destructive action; introduced only in Phase 3 |
| **Prompt injection** | Malicious content in tool results hijacking the LLM | Tool output sanitization; system prompt hardening |

### 🟡 Important

| Risk | Detail | Mitigation |
|------|--------|------------|
| **LLM vendor lock-in** | Hard dependency on one provider | LiteLLM abstraction from day one |
| **Context window exhaustion** | Long conversations exceed model limits | Session summarization strategy planned for Phase 1 |
| **Tool failure cascades** | A failing tool causes the agent to loop or hallucinate | Every tool returns a structured result including error state; orchestrator handles gracefully |
| **Scope creep** | Temptation to build "everything" in v0.1 | Strict phase gating; each phase has an explicit done-state |

### 🟢 Watch Points

| Risk | Detail | Mitigation |
|------|--------|------------|
| **Database migration pain** | SQLite → PostgreSQL can be messy | Use SQLAlchemy ORM from the start so the migration is a config change |
| **Frontend/Backend coupling** | Tight API contracts break refactoring | Version the API (`/v1/chat`); use TypeScript types generated from the OpenAPI schema |
| **Test coverage decay** | AI projects are notoriously hard to test | Integration tests for every tool from day one; mock LLM calls in unit tests |
| **Voice/Vision latency** | These are slow pipelines | Introduce only in Phase 2 with async streaming; never block the main response thread |

---

## Open Questions for Your Review

> [!IMPORTANT]
> Please review these before we proceed to implementation.

1. **LLM Provider for v0.1**: Do you have an OpenAI API key, an Anthropic key, or do you prefer to start with a **free local model** (via Ollama)? This affects the first config we build.

2. **Operating System target**: Is JARVIS primarily for **Windows**, or should it be cross-platform from the start? This affects voice and automation tooling choices.

3. **Web Search tool**: DuckDuckGo has a free unofficial API (no key needed). Alternatively, Tavily or Brave Search have official APIs. Which do you prefer for v0.1?

4. **Auth in scope?**: Should v0.1 have any form of authentication, or is this single-user/local-only for now?

5. **SQLAlchemy vs raw SQLite**: Do you want a full ORM from the start (more boilerplate, easier migration), or raw SQLite queries in v0.1 (simpler now, refactor later)?
