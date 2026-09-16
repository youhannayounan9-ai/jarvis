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

### Voice mode

```bash
jarvis --voice
```

Or from the text REPL: type `/voice`.

---

## Voice Mode (v0.4)

JARVIS can listen with **Whisper** (local STT) and speak with **Edge TTS** (free neural voices).

### Prerequisites

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
| **v0.2** | ✅ Persistent vector memory (ChromaDB) + RAG |
| **v0.3** | ✅ Plan-and-Execute multi-step agent loop |
| **v0.4** | ✅ Voice input (Whisper) + voice output (Edge TTS) |
| v0.5 | Vision (image understanding) |
| v0.6 | Browser + computer automation (with permission model) |
| v0.7 | Web UI (React + FastAPI) |

---

## License

MIT
