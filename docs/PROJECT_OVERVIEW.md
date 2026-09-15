# JARVIS — Project Overview

**Version:** 0.1.0  
**License:** MIT  
**Tagline:** A local, modular personal AI assistant — powered by Ollama. No paid APIs. No cloud lock-in.

This document is a handoff-style overview: what JARVIS is, how it is built today, what it can already do, and where it is headed. It is meant for collaborators, reviewers, or anyone who needs to understand the project without reading every source file.

---

## 1. Vision

JARVIS aims to become a **fully local personal AI assistant** that lives on the user’s machine: private by default, extendable through tools, and capable enough to help with research, files, calculation, and (later) voice, vision, and computer control.

The name is intentional — not as a movie clone, but as a design target: a capable assistant that feels coherent, remembers context, and can *act* through tools rather than only chat.

**Non-negotiables for the current phase:**

- Runs locally (Ollama + local Python app)
- Free stack for core features (no required paid LLM or search APIs)
- Modular architecture so features can grow without rewriting the core
- Privacy-first: conversation data stays on disk under the user’s control

---

## 2. What Exists Today (v0.1)

JARVIS v0.1 is a **text-based CLI assistant** with:

| Capability | Status |
|---|---|
| Local LLM chat via Ollama | Done |
| Autonomous tool calling (ReAct-style loop) | Done |
| Persistent short-term session memory (SQLite) | Done |
| Free web search (DuckDuckGo via `ddgs`) | Done |
| Wikipedia topic summaries | Done |
| Sandboxed file read / directory listing | Done |
| Safe calculator | Done |
| Current date/time | Done |
| Rich terminal UX (panels, Markdown, commands) | Done |
| Offline unit tests for tools & memory | Done |
| Permission framework (stub, ready to harden) | Scaffolded |
| Vector / long-term memory | Not yet (planned v0.2) |
| Multi-step agent planner beyond tool loop | Not yet (planned v0.3) |
| Voice / vision / browser automation / web UI | Later roadmap |

**Default model:** `qwen2.5:7b` (tool-calling capable).  
**Runtime requirement:** Ollama must be running (`ollama serve` or the Ollama desktop app).

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User (CLI)                          │
│                    Rich terminal interface                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     jarvis/main.py                          │
│         Wires components + REPL (/help, /tools, …)          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Orchestrator (core/orchestrator.py)            │
│         ReAct loop: think → tool calls → answer             │
└───────┬──────────────────┬──────────────────┬───────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐
│ SessionStore  │  │ Permission    │  │ LLM Client            │
│ (SQLite)      │  │ Guard (stub)  │  │ LiteLLM → Ollama      │
└───────────────┘  └───────────────┘  └───────────┬───────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │ ToolRegistry          │
                                      │ + concrete tools      │
                                      └───────────────────────┘
```

### Request lifecycle (one user message)

1. User types a message in the CLI.
2. `main.py` calls `orchestrator.chat(session_id, user_input)`.
3. Orchestrator saves the user message to SQLite.
4. It loads a **recent history window**, injects the system prompt (+ a short memory cue), and calls the LLM with tool schemas.
5. If the model returns **tool calls**:
   - PermissionGuard checks each tool
   - ToolRegistry dispatches execution
   - Results are saved and fed back to the model
   - Loop continues (capped at 5 tool rounds)
6. If the model returns **plain text**, that is the final answer.
7. CLI renders the answer as Markdown in a Rich panel.

This is the classic **ReAct (Reason + Act)** pattern, kept deliberately simple and testable.

---

## 4. Repository Structure

```
JARVIS/
├── jarvis/                      # Application package
│   ├── main.py                  # CLI entry + dependency wiring
│   ├── config.py                # Settings from .env (pydantic-settings)
│   ├── __init__.py              # Version (__version__ = 0.1.0)
│   ├── core/
│   │   ├── orchestrator.py      # Brain: ReAct tool-calling loop
│   │   └── permissions.py       # PermissionGuard (permissive stub in v0.1)
│   ├── llm/
│   │   └── client.py            # LiteLLM wrapper → Ollama
│   ├── memory/
│   │   └── session_store.py     # SQLite sessions + messages
│   ├── tools/
│   │   ├── base.py              # BaseTool abstract interface
│   │   ├── registry.py          # Register / schema export / dispatch
│   │   ├── datetime_tool.py     # get_current_datetime
│   │   ├── web_search.py        # web_search (ddgs / DuckDuckGo)
│   │   ├── wikipedia_summary.py # wikipedia_summary (Wikipedia API)
│   │   ├── file_reader.py       # read_file (sandboxed)
│   │   ├── directory_lister.py  # list_directory (sandboxed)
│   │   └── calculator.py        # calculator (safe expression eval)
│   └── utils/
│       └── logging.py           # structlog setup
├── tests/
│   ├── conftest.py
│   ├── test_tools.py
│   └── test_session_store.py
├── docs/
│   ├── architecture.md          # Architecture notes
│   └── PROJECT_OVERVIEW.md      # This document
├── pyproject.toml               # Package metadata + dependencies
├── README.md                    # Install / run guide
├── .env.example                 # Config template
└── jarvis.db                    # Local SQLite DB (created at runtime)
```

### Design principles baked into the layout

1. **Dependency injection** — `main.py` builds real objects; Orchestrator never constructs its own DB/LLM/tools. This keeps unit testing easy.
2. **Manual tool registration** — tools are registered explicitly in `main.py` (no magic auto-discovery), so the active capability surface is obvious.
3. **Provider abstraction** — all model calls go through LiteLLM. Switching providers later is mostly a model-string / config change.
4. **Tool contract** — every tool subclasses `BaseTool`, exposes OpenAI-style JSON Schema, and returns a **string** (success or `ERROR: …`). Exceptions inside tools are caught at the registry boundary so the loop stays stable.
5. **Config centralization** — `jarvis/config.py` is the single settings surface (model, DB path, sandbox dir, history window, system prompt).

---

## 5. Current Tools (Capability Surface)

| Tool name | Purpose | Notes |
|---|---|---|
| `get_current_datetime` | Local date, time, weekday, UTC offset | No network |
| `web_search` | Live web search via DuckDuckGo (`ddgs`) | Free, no API key; returns numbered excerpts for synthesis |
| `wikipedia_summary` | Short encyclopedia summary | Free Wikipedia public API; preferred for topic overviews |
| `read_file` | Read a text file | Sandboxed to `FILE_READER_ALLOWED_DIR` |
| `list_directory` | List folder contents | Same sandbox rules |
| `calculator` | Evaluate math expressions | Restricted AST eval (no arbitrary code) |

**CLI meta-commands** (not LLM tools): `/help`, `/tools`, `/history`, `/new`, `/quit`.

---

## 6. Memory Model (Today vs Tomorrow)

### Today — short-term session memory

- Conversations are stored in **SQLite** (`jarvis.db` by default).
- Tables: `sessions`, `messages` (OpenAI-compatible roles: user / assistant / tool / system).
- On each turn, JARVIS loads the **most recent N messages** (default `MAX_HISTORY_MESSAGES=40`), not the oldest — so long chats keep fresh context.
- Leading orphan `tool` messages are trimmed so a history window never starts mid tool-call chain.
- Large old tool payloads are truncated when reloaded, to leave room for dialogue.
- The system prompt is **not** stored in the DB; it is injected fresh every call.
- A short “memory cue” system message reminds the model that prior turns are available for follow-ups.

### Not yet — long-term / semantic memory

Vector DB (e.g. ChromaDB), RAG over past sessions, and cross-session recall are **planned for v0.2**, not implemented now.

---

## 7. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Clear tooling ecosystem for agents |
| Packaging | `pyproject.toml` + hatchling; run via `jarvis` script | Standard, installable package |
| LLM runtime | Ollama | Local models, no cloud bill for inference |
| LLM API glue | LiteLLM | One interface; portable to other providers later |
| Search | `ddgs` (DuckDuckGo) | Free, no key |
| Knowledge lookup | Wikipedia REST / OpenSearch | Free, reliable summaries |
| Config | pydantic-settings + `.env` | Typed, fail-fast |
| Persistence | sqlite3 | Zero ops for v0.1 |
| CLI UX | Rich | Panels, Markdown, tables, spinners |
| Logging | structlog | Structured, readable logs |
| Tests | pytest | Offline unit tests for tools & memory |

---

## 8. How to Run (Quick)

Prerequisites: Python 3.11+, Ollama installed, model pulled (e.g. `ollama pull qwen2.5:7b`).

```bash
# Install
uv venv
# activate venv, then:
uv pip install -e ".[dev]"
cp .env.example .env

# Ensure Ollama is running
ollama serve   # if not already running via the app

# Start JARVIS
jarvis
# or: python -m jarvis.main

# Tests
pytest
```

If chat fails with connection refused / WinError 10061, Ollama is not listening on `http://localhost:11434`.

---

## 9. Roadmap — What It Will Have

The project is intentionally versioned as a progressive build-out:

| Version | Goal | Intent |
|---|---|---|
| **v0.1 (current)** | CLI + Ollama + tool calling + SQLite short-term memory | Solid core loop and modular tool system |
| **v0.2** | Persistent **vector memory** (e.g. ChromaDB) + RAG | Remember facts across sessions; retrieve relevant past context |
| **v0.3** | Stronger **multi-step agent loop** for complex tasks | Plan → act → verify beyond simple tool rounds |
| **v0.4** | **Voice** input (Whisper) + TTS output | Hands-free interaction |
| **v0.5** | **Vision** (image understanding) | Screenshots / photos as context |
| **v0.6** | Browser + computer automation with a real **permission model** | Act on the OS/web safely, with confirmation for risky actions |
| **v0.7** | **Web UI** (React + FastAPI) | Richer interface while keeping the same core |

### Directional product picture (end state)

A private assistant that can:

- Talk in text and voice
- See images
- Search the web and reference knowledge sources
- Read/write files and automate the computer under explicit permissions
- Remember important things long-term
- Offer both CLI and web interfaces
- Stay swappable at the model layer (local first; cloud optional later via LiteLLM)

---

## 10. Security & Safety Posture (Current)

- **Local-first:** chat history stays in local SQLite.
- **File tools are sandboxed** to an allowed directory.
- **Calculator is restricted** (no shell/code execution).
- **PermissionGuard exists** but currently allows all registered tools — intentional scaffold for v0.6-style risk levels (SAFE / NETWORK / FILE_WRITE / SYSTEM / DESTRUCTIVE).
- Tool loop has a **hard cap** (`MAX_TOOL_ROUNDS = 5`) to prevent runaway tool calling.
- No FastAPI/React/voice surface yet — attack surface is intentionally small in v0.1.

---

## 11. Extending the Project

### Add a new tool (checklist)

1. Create `jarvis/tools/my_tool.py` subclassing `BaseTool`
2. Define `name`, `description`, `parameters` (JSON Schema), and `run()`
3. Register in `jarvis/main.py`
4. Export from `jarvis/tools/__init__.py`
5. Add tests in `tests/test_tools.py`
6. Mention it in the system prompt tool policy (`config.py`) if the model should know when to use it

Nothing else in the orchestrator needs to change for a normal tool.

### Key files to read first (onboarding order)

1. `README.md` — install & run  
2. `docs/PROJECT_OVERVIEW.md` — this file  
3. `docs/architecture.md` — lifecycle details  
4. `jarvis/main.py` — wiring  
5. `jarvis/core/orchestrator.py` — agent loop  
6. `jarvis/tools/` — capability implementations  
7. `jarvis/memory/session_store.py` — short-term memory  

---

## 12. One-Paragraph Summary (for quick sharing)

**JARVIS is a local personal AI assistant (v0.1) built in Python.** It chats through a Rich CLI, runs models via Ollama (through LiteLLM), calls tools in a ReAct loop, and stores short-term conversation history in SQLite. Today it can search the web (DuckDuckGo), summarize Wikipedia topics, read sandboxed files, list directories, calculate, and tell the time — all without paid APIs. The architecture is modular on purpose: tools plug in cleanly, permissions are stubbed for future hardening, and the roadmap grows toward long-term vector memory, stronger multi-step agency, voice, vision, computer control, and a web UI — while staying local- and privacy-first.

---

*Generated for project handoff / collaboration. Current codebase version: **0.1.0**.*
