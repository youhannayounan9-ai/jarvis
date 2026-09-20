# JARVIS Project Brain

## Project Overview
JARVIS is a local-first, multimodal, agentic AI assistant.

## Architecture Map
- `core/`: The brain of the system, orchestrator, and permission enforcement.
- `tools/`: The capabilities available to the agent (e.g., computer automation, web scraping).
- `memory/`: The state and session history (SQLite, ChromaDB).
- `llm/`: The wrapper for local LLM inference (LiteLLM/Ollama).
- `utils/`: Utilities like logging and system configuration.
- `voice/`: Whisper and TTS integration for voice mode.

## Strict Rules & Constraints
- Never add new dependencies without explicit instruction.
- Always run `uv run pytest -v` after modifying core logic.
- Treat the LLM as an untrusted reasoning engine; enforce PermissionGuard for SYSTEM/DESTRUCTIVE tools.
- Maintain dependency injection in the Orchestrator.

## Testing & Verification
- Run tests: `uv run pytest -v`
- Enforce linting/formatting according to project standards.

## Tech Stack
- Ollama
- LiteLLM
- ChromaDB
- SQLite
- Streamlit
- Playwright

## Intent Router Architecture & Trade-offs
- **Mechanism:** Heuristic keyword + length matching (zero latency).
- **Benefits:** Skips the heavy Plan-and-Execute loop for simple queries, saving ~2-3 LLM calls.
- **Known Limitations (Edge Cases):** False Positives (complex queries containing simple keywords) and False Negatives (simple queries missing keywords). This is an intentional trade-off prioritizing speed and reduced token cost over perfect classification accuracy.
