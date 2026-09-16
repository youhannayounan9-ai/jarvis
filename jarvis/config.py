"""
jarvis/config.py
────────────────
Central configuration, loaded from environment variables / .env file.

All application code imports settings from here — no os.environ scattered
throughout the codebase. This makes the config surface explicit and auditable.

Usage:
    from jarvis.config import settings
    print(settings.ollama_model)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic BaseSettings automatically:
      1. Reads values from the .env file (if present).
      2. Falls back to actual environment variables.
      3. Falls back to the default values defined here.
      4. Validates types at startup — bad config fails fast with a clear error.
    """

    model_config = SettingsConfigDict(
        # Load from .env in the current working directory (where `jarvis` is run).
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore any extra env vars that are not declared here.
        extra="ignore",
        # Case-insensitive env var matching (OLLAMA_MODEL == ollama_model).
        case_sensitive=False,
    )

    # ── LLM / Ollama ──────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Optional separate model for the v0.3 planner (defaults to the main model).
    planner_model: str = "qwen2.5:7b"
    max_tokens: int = 2048
    ollama_num_ctx: int = 8192  # Capped context window to prevent OOM on Qwen 2.5

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str = "jarvis.db"
    # How many recent messages to feed the model (keeps context fresh + bounded).
    max_history_messages: int = 40
    # Persistent ChromaDB path for long-term / RAG memory (v0.2).
    vector_db_path: str = "./jarvis_data/chroma_db"

    # ── Tools ─────────────────────────────────────────────────────────────────
    # read_file is sandboxed to this directory tree.
    # Default "." resolves to wherever `jarvis` is launched from.
    file_reader_allowed_dir: str = "."

    # ── Voice (v0.4) ───────────────────────────────────────────────────────────
    whisper_model: str = "base"  # tiny | base | small | medium | large
    tts_voice: str = "en-US-GuyNeural"
    voice_record_seconds: float = 5.0  # Mic capture length per listen()

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived helpers (not from env) ────────────────────────────────────────
    @property
    def litellm_model(self) -> str:
        """
        LiteLLM model string for Ollama chat endpoint.
        The 'ollama_chat/' prefix tells LiteLLM to use the /api/chat route,
        which supports proper message formatting and tool calling.
        """
        return f"ollama_chat/{self.ollama_model}"

    @property
    def litellm_planner_model(self) -> str:
        """LiteLLM model string used by the Plan-and-Execute planner."""
        return f"ollama_chat/{self.planner_model}"

    @property
    def file_reader_allowed_path(self) -> Path:
        """Resolved absolute path used for sandboxing the read_file tool."""
        return Path(self.file_reader_allowed_dir).resolve()

    @property
    def system_prompt(self) -> str:
        """The persistent system prompt injected at the start of every session."""
        return (
            "You are JARVIS, a capable local AI personal assistant. "
            "Tone: professional, concise, lightly dry wit. "
            "You run entirely on the user's machine — their data stays private.\n\n"

            "## Conversation memory (critical)\n"
            "- The messages below ARE this session's short-term memory. "
            "Treat earlier user and assistant turns as known facts.\n"
            "- On follow-ups ('that', 'it', 'the second one', 'continue'), "
            "resolve references from prior turns. Do not re-ask for details "
            "the user already gave.\n"
            "- Never invent memory files or call tools to 'recall' chat history. "
            "History is already in context.\n\n"

            "## Tool policy\n"
            "Default to answering without tools. Tools cost time — use them only "
            "when they clearly improve the answer.\n"
            "## Available Tools & Strict Usage Rules\n"
            "- get_current_datetime: ONLY when asked for the current time or date.\n"
            "- web_search: ONLY for real-time information (news, weather, recent events). When summarizing results, extract specific facts, data, and dates. Do NOT give vague, generic summaries.\n"
            "- read_file: ONLY when the user explicitly names a specific file to read.\n"
            "- list_directory: ONLY when the user asks to see what files or folders exist in a specific path.\n"
            "- calculator: ONLY when you need to accurately compute a mathematical expression.\n"
            "- remember_fact: You MUST call this whenever the user shares personal information (name, age, preference, project, etc). Do NOT just say you remembered it — actually call the tool.\n"
            "- recall_facts: You MUST call this before answering any question about the user's personal details or preferences. Never guess — search first.\n\n"

            "## Memory & Anti-Hallucination Rules (CRITICAL)\n"
            "- If the user shares personal information, you MUST call the `remember_fact` tool.\n"
            "- If the user asks about their own information, you MUST call the `recall_facts` tool.\n"
            "- Never hallucinate facts. If you do not know something, use a tool to find it.\n\n"

            "## Do not use tools for\n"
            "Greetings, identity questions, general knowledge you already know, "
            "coding help, or follow-ups already answered in this chat.\n\n"

            "Be concise. Use clean Markdown. Do not repeat the user's question. "
            "If unsure, say so plainly. Never invent tool names."
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
# The rest of the application imports this single instance.
# If .env is missing or malformed, this line raises a clear Pydantic error
# at import time — not silently mid-run.
settings = Settings()
