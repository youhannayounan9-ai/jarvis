# JARVIS

> A local, modular AI assistant — powered by Ollama. No paid APIs. No cloud.

**JARVIS v0.1** — text-based CLI assistant with tool calling, persistent conversation history, and a clean modular architecture designed for progressive expansion.

---

## Features (v0.1)

- 💬 **Conversational AI** — full session memory backed by SQLite
- 🛠️ **Tool calling** — LLM can autonomously call tools to answer questions
- 🔍 **Web search** — DuckDuckGo (free, no API key)
- 🕐 **Current time/date** — instant, no network
- 📄 **Read files** — sandboxed to your project directory
- 🔒 **Fully local** — your data never leaves your machine
- 🏗️ **Modular** — adding a new tool takes ~30 lines

---

## Prerequisites

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

Make sure Ollama is running (`ollama serve`), then:

```bash
jarvis
```

Or, if the script isn't on your PATH:

```bash
python -m jarvis.main
```

### Example session

```
╭─ J.A.R.V.I.S ──────────────────────────────╮
│ v0.1 — Local AI Assistant                   │
│ Model: qwen2.5:7b   |  Type /help for commands │
╰─────────────────────────────────────────────╯

You: What time is it?
⠋ Thinking…

╭─ JARVIS ────────────────────────────────────╮
│ It's Monday, September 15, 2026 at 22:42:00 │
│ (UTC+0300).                                  │
╰─────────────────────────────────────────────╯

You: Search for the latest news on local LLMs
⠙ Thinking…

╭─ JARVIS ────────────────────────────────────╮
│ Here's what I found…                        │
╰─────────────────────────────────────────────╯
```

---

## CLI Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/tools` | List registered tools |
| `/history` | Print this session's messages |
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
│   ├── main.py            ← CLI entry point + component wiring
│   ├── config.py          ← All config, loaded from .env
│   ├── core/
│   │   ├── orchestrator.py  ← ReAct tool-calling loop
│   │   └── permissions.py   ← Permission guard (stub in v0.1)
│   ├── llm/
│   │   └── client.py      ← LiteLLM → Ollama wrapper
│   ├── memory/
│   │   └── session_store.py ← SQLite conversation history
│   ├── tools/
│   │   ├── base.py        ← Abstract tool interface
│   │   ├── registry.py    ← Tool registry + dispatcher
│   │   ├── datetime_tool.py
│   │   ├── web_search.py
│   │   └── file_reader.py
│   └── utils/
│       └── logging.py     ← structlog setup
├── tests/
├── docs/architecture.md
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Adding a New Tool

See [`docs/architecture.md`](docs/architecture.md) for the full checklist.
In short: create a file in `jarvis/tools/`, subclass `BaseTool`, register it in `main.py`.

---

## Roadmap

| Version | Goal |
|---------|------|
| **v0.1** | ✅ CLI + Ollama + tool calling + SQLite history |
| v0.2 | Persistent vector memory (ChromaDB) + RAG |
| v0.3 | Agent loop for multi-step tasks |
| v0.4 | Voice input (Whisper) + voice output (TTS) |
| v0.5 | Vision (image understanding) |
| v0.6 | Browser + computer automation (with permission model) |
| v0.7 | Web UI (React + FastAPI) |

---

## License

MIT
