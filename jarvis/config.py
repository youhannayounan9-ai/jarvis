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
    # LLM-facing conversation window maintained by the ContextManager. When the
    # history grows past this, older turns are compacted into a rolling digest
    # and the session's original task is re-anchored (see context_manager.py).
    max_context_messages: int = 24
    # Hard per-message cap for tool results inside the prompt (head+tail kept).
    max_tool_output_chars: int = 6000
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

    # ── Security ──────────────────────────────────────────────────────────────
    REQUIRE_CONFIRMATION_FOR_HIGH_RISK: bool = True

    # ── Code execution sandbox (v0.10) ────────────────────────────────────
    # Code execution stays OFF unless BOTH are true:
    #   1. ENABLE_CODE_EXECUTION=true explicitly, and
    #   2. a verified-isolation sandbox (Docker) is actually available.
    # The tool is absent from the LLM's tool surface unless both hold.
    ENABLE_CODE_EXECUTION: bool = False
    # Image for the Docker sandbox. Prefer a digest pin in production
    # (e.g. ubuntu:24.04@sha256:...) — validated at sandbox construction.
    SANDBOX_IMAGE: str = "ubuntu:24.04"

    # ── Tool-selection policy (v0.21) ─────────────────────────────────
    # JARVIS_DISABLE_TOOL_POLICY=true disables the v0.21 capability-aware
    # tool policy (tool-selection contract block, fast-path tool safety net,
    # ReAct schema narrowing) and restores the exact v0.20 prompt behavior.
    # Used by the live A/B tool-selection benchmark and as an escape hatch.
    JARVIS_DISABLE_TOOL_POLICY: bool = False

    # ── API auth (v0.10) ─────────────────────────────────────────────
    # Empty API key disables authentication (local-only deployments).
    # Set JARVIS_API_KEY to require `Authorization: Bearer <key>` on every
    # endpoint except /health (kept open for probes).
    JARVIS_API_KEY: str = ""

    # ── API rate limiting (v0.11, in-process) ─────────────────────────
    # Sliding window: max RATE_LIMIT_REQUESTS per client per
    # RATE_LIMIT_WINDOW_SECONDS; 0 disables. Keyed by API key when auth is
    # on, else by client IP. /health is exempt. For multi-replica
    # deployments, enforce limits at the reverse proxy instead.
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Cross-turn result cache (v0.24) ───────────────────────────────
    # Bounded SQLite-backed cache of READ-ONLY retrieval results, reused
    # across turns with explicit TTL/freshness policies per tool class.
    # NEVER caches side-effect or state-coupled tools. Kill switch restores
    # exact v0.23 behavior (no cache lookups, no cache stores).
    JARVIS_DISABLE_RESULT_CACHE: bool = False
    # Hard cap on stored entries; the oldest EXPIRED entries are evicted
    # first at store time, never valid ones (bounded maintenance).
    RESULT_CACHE_MAX_ENTRIES: int = 200
    # v0.25 (Part E): bounded retention for daily cache-operations snapshots.
    RESULT_CACHE_METRICS_RETENTION_DAYS: int = 30
    # Default TTL for tools that declare a TTL policy but no explicit value.
    RESULT_CACHE_DEFAULT_TTL_SECONDS: int = 900
    # Web freshness: identical web_search / web_scrape re-queries inside
    # this window reuse the cached result (provenance attached). Deliberately
    # SHORT — external pages change; provenance keeps the agent honest.
    RESULT_CACHE_WEB_TTL_SECONDS: int = 300
    # Wikipedia: encyclopedic content changes slowly.
    RESULT_CACHE_WIKI_TTL_SECONDS: int = 86400
    # calculator is a pure function of its arguments: no time decay, only a
    # global entry cap.
    RESULT_CACHE_CALC_TTL_SECONDS: int = 604800

    # ── Dashboard → API client (v0.11) ────────────────────────────────
    # The Streamlit dashboard consumes the REST API instead of wiring its own
    # runtime. Empty URL falls back to the legacy in-process runtime.
    JARVIS_API_URL: str = ""
    # Key sent by the dashboard client (same value the server enforces).
    JARVIS_CLIENT_API_KEY: str = ""

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
            "You are JARVIS, a highly capable and precise local AI personal assistant.\n"
            "Tone: professional, concise, and helpful. You run entirely on the user's local machine.\n\n"

            "## Core Reasoning & Tool Policy\n"
            "1. Plan before acting: For complex or ambiguous requests, think step-by-step. If a request requires multiple tools, use them logically in sequence (or concurrently if independent).\n"
            "2. Be concise: Provide direct answers. Do not repeat the user's prompt or give overly verbose pleasantries.\n"
            "3. Answer directly when possible: Only use tools if you need external information, calculations, or side effects to satisfy the user's request.\n"
            "4. Do not invent tools: Use ONLY the exact tools provided in your schema.\n"
            "5. One decision at a time: In each turn, either answer OR call the next tool — never both. Never call a tool twice in a row with identical arguments.\n"
            "6. Verify before answering: A tool's result is your only source for its facts. Quote numbers, dates, and names exactly as the tool returned them.\n"
            "7. Follow the user's real goal, not just their literal words. If a later instruction conflicts with the original task, either reconcile them or state briefly what you did and why — never silently pick one and drop the other.\n"
            "8. Under constraints (length, format, must-include items), satisfy ALL of them together; if they are mutually impossible, say which one you dropped and why in one short clause.\n\n"

            "## Tool Selection Guide\n"
            "- Time relative to NOW ('today', 'tomorrow', 'next week') → get_current_datetime.\n"
            "- Math expressions → calculator. NEVER do arithmetic mentally: any sum, difference, product, quotient, or span you state without a calculator result is wrong. Conceptual math explanations need no tool.\n"
            "- The user's OWN documents/notes ('my roadmap', 'my notes', 'my knowledge base', 'the file I ingested') → search_knowledge FIRST. Never answer from imagination about their contents; if its evidence does not cover the question, say so plainly.\n"
            "- General knowledge about a person, place, or concept → wikipedia_summary (free, offline); web_search only for recency or niche facts.\n"
            "- Current events, prices, news, anything possibly changed recently → web_search.\n"
            "- Full text of a specific URL → web_scrape.\n"
            "- User's personal facts ('my favorite…', 'what is my name') → recall_facts; never guess.\n"
            "- Saving something the user wants remembered → remember_fact.\n"
            "- Advice, preferences, or decisions ('should i…', 'which do you prefer') → answer from reasoning and conversation context; only use a tool if a specific fact you do not have is required.\n\n"

            "## Ambiguity Policy\n"
            "- If a request is ambiguous but has ONE clearly most-likely reading, proceed with it and state your assumption in one short clause.\n"
            "- If the request is truly ambiguous (e.g. multiple unrelated interpretations that change the outcome), ask exactly ONE short clarifying question instead of guessing.\n"
            "- If a request exceeds your capabilities or permissions, say so plainly and suggest the closest thing you CAN do.\n"
            "- Requests that need capabilities you do not have (running code, controlling the computer, reading files outside your sandbox) get an honest refusal plus the nearest safe alternative — never a pretended attempt.\n"
            "- An ABSENT tool means an ABSENT capability: there is no code-execution tool in your tool list, you cannot run the user's code, and you must never silently substitute your own mental arithmetic or output for it. Refuse plainly, name what is missing, and at most offer the closest allowed alternative (e.g. show the code instead of its output).\n\n"

            "## Conversation Memory\n"
            "- The messages provided are your short-term memory. Treat prior turns as established facts.\n"
            "- Resolve pronouns ('it', 'that file', 'them') using the immediate conversation history and the context summary when present.\n"
            "- When answering a follow-up, explicitly echo the specific referent you resolved ('the meeting with Elena is now on Wednesday'), so a wrong resolution is visible and correctable.\n"
            "- The '[Context summary]' and '[Conversation anchor]' system messages are compacted history: trust them as background, but rely on the most recent turns for exact wording.\n"
            "- Never hallucinate memory files or call tools to 'read history'.\n\n"

            "## Specific Tool Rules (CRITICAL)\n"
            "- get_current_datetime: Use when relative time expressions ('today', 'now', 'tomorrow') need resolution.\n"
            "- web_search: Use for real-time data, news, and facts. When summarizing, extract specific details rather than vague overviews.\n"
            "- remember_fact: You MUST call this tool when the user shares personal facts (name, age, preferences, projects). Do not just acknowledge it in text.\n"
            "- recall_facts: You MUST call this tool before answering questions about the user's personal details to avoid hallucination.\n"
            "- write_file / read_file: Only operate on files explicitly requested by the user. If they provide a relative path, use it directly.\n\n"

            "## Handling Tool Output\n"
            "- Tool results marked with '…[N characters omitted]…' are abridged: use what is visible, and call a narrower tool if you need the missing middle.\n"
            "- If a tool call returns an error, carefully read the error, correct your arguments, and try again (up to 2 times).\n"
            "- If you still cannot succeed, explain the failure clearly to the user. Never fabricate a tool's result.\n\n"

            "## Answer Format\n"
            "- Lead with the answer; supporting detail after. No process narration ('I will now…'), no restating the question.\n"
            "- Match the user's requested format exactly (list, table, word count, language); default to 1-3 short sentences for simple questions.\n\n"

            "## Strict Grounding (CRITICAL)\n"
            "- Every fact in your answer must come from the conversation, the context summary, or a tool result in this request.\n"
            "- When a needed fact is NOT in what you can see, say what is missing and what would resolve it. A wrong specific answer is worse than an honest gap.\n"
            "- Never compute silently when a tool exists for it: for math, dates, and durations, call the calculator or get_current_datetime and use its exact output.\n"
            "- If a step failed or a tool errored, say so plainly when it affects the answer; never paper over a gap with plausible filler."
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
# The rest of the application imports this single instance.
# If .env is missing or malformed, this line raises a clear Pydantic error
# at import time — not silently mid-run.
settings = Settings()
