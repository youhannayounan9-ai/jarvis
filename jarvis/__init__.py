# jarvis/__init__.py
"""
JARVIS — Local-first, multimodal, agentic AI assistant.

Architecture layers (see docs/architecture.md):
  core/    — orchestrator, planner, permissions, sandbox contract
  tools/   — capability implementations behind a common interface
  memory/  — SQLite sessions + ChromaDB long-term memory + context manager
  llm/     — LiteLLM → Ollama abstraction
  api/     — FastAPI service layer: auth, rate limiting, SSE, stdlib client
  voice/   — Whisper STT + Edge TTS

The API is the primary service surface; the dashboard consumes it via
jarvis.api.client, and the CLI embeds the runtime directly.
"""

__version__ = "0.16.0"
