"""
jarvis/memory/session_store.py
───────────────────────────────
Thin SQLite wrapper for persisting conversation history.

Design decisions:
  - Raw sqlite3 (no ORM) as agreed in v0.1 scope.
  - All DB access goes through this module. The rest of the codebase only
    calls save_message() and load_history() — never raw SQL.
  - Messages are stored as typed dicts matching the OpenAI message format
    so they can be fed directly back to LiteLLM without transformation.

Schema:
  sessions   (id TEXT PK, created_at TEXT)
  messages   (id INTEGER PK, session_id TEXT FK, role TEXT,
              content TEXT, tool_call_id TEXT, name TEXT,
              tool_calls_json TEXT, created_at TEXT)
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ── Message roles we store (mirrors OpenAI's role set) ────────────────────────
_VALID_ROLES = {"user", "assistant", "tool", "system"}

# Soft cap on stored tool-result content when reloading history so older
# bulky search/file payloads do not crowd out recent dialogue.
_MAX_STORED_TOOL_CONTENT = 2000


class SessionStore:
    """
    Manages conversation sessions and message history.

    Usage:
        store = SessionStore()
        session_id = store.create_session()
        store.save_message(session_id, {"role": "user", "content": "Hello"})
        history = store.load_history(session_id)
    """

    def __init__(self) -> None:
        self._conn = _get_connection()
        _init_db(self._conn)
        log.info("session_store_ready", db=settings.db_path)

    # ── Sessions ───────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        """
        Create a new conversation session and return its ID.
        Session IDs are random UUIDs — no sequential integers that reveal counts.
        """
        session_id = str(uuid.uuid4())
        now = _utcnow()
        self._conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        self._conn.commit()
        log.info("session_created", session_id=session_id)
        return session_id

    def list_sessions(self) -> list[dict[str, str]]:
        """Return all sessions sorted newest-first."""
        rows = self._conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def message_count(self, session_id: str) -> int:
        """Return how many messages are stored for a session."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    # ── Messages ───────────────────────────────────────────────────────────────

    def save_message(self, session_id: str, message: dict[str, Any]) -> None:
        """
        Persist one message to the database.

        Args:
            session_id: The session this message belongs to.
            message:    An OpenAI-format message dict.
                        Must have at least {"role": ..., "content": ...}
                        Tool results also have {"tool_call_id": ..., "name": ...}
                        Assistant tool calls have {"tool_calls": [...]}
        """
        role = message.get("role", "")
        if role not in _VALID_ROLES:
            log.warning("save_message_invalid_role", role=role)

        tool_calls = message.get("tool_calls")
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        self._conn.execute(
            """
            INSERT INTO messages
                (session_id, role, content, tool_call_id, name, tool_calls_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                message.get("content"),
                message.get("tool_call_id"),
                message.get("name"),
                tool_calls_json,
                _utcnow(),
            ),
        )
        self._conn.commit()

    def load_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load recent messages for a session in chronological order.

        Returns OpenAI-format message dicts ready for LiteLLM.

        Uses the *most recent* `limit` messages (default:
        settings.max_history_messages), not the oldest — so long sessions
        keep fresh context. Leading orphan tool results are trimmed so the
        window never starts mid tool-call chain.
        """
        if limit is None:
            limit = settings.max_history_messages
        limit = max(1, int(limit))

        # Newest-first fetch, then reverse to chronological order.
        rows = self._conn.execute(
            """
            SELECT role, content, tool_call_id, name, tool_calls_json
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in reversed(rows):
            msg: dict[str, Any] = {"role": row["role"]}

            content = row["content"]
            if content is not None:
                if row["role"] == "tool":
                    content = _trim_tool_content(content)
                msg["content"] = content

            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]

            if row["name"]:
                msg["name"] = row["name"]

            if row["tool_calls_json"]:
                msg["tool_calls"] = json.loads(row["tool_calls_json"])

            messages.append(msg)

        messages = _trim_orphan_tool_prefix(messages)

        log.debug(
            "history_loaded",
            session_id=session_id,
            messages=len(messages),
            limit=limit,
        )
        return messages

    def close(self) -> None:
        """Close the database connection cleanly."""
        self._conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    db_path = Path(settings.db_path)
    # ":memory:" must not be passed through Path (would become a relative file).
    if settings.db_path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL REFERENCES sessions(id),
            role            TEXT    NOT NULL,
            content         TEXT,
            tool_call_id    TEXT,
            name            TEXT,
            tool_calls_json TEXT,
            created_at      TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages (session_id, id);
    """)
    conn.commit()


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _trim_tool_content(content: str) -> str:
    """Keep tool payloads in history, but bound their size."""
    if len(content) <= _MAX_STORED_TOOL_CONTENT:
        return content
    return (
        content[: _MAX_STORED_TOOL_CONTENT - 1].rstrip()
        + "…\n[truncated — older tool output shortened for context]"
    )


def _trim_orphan_tool_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop leading tool-role messages that lack their parent assistant tool_calls.

    A sliding window can otherwise start mid tool-call chain, which confuses
    the model and some providers.
    """
    i = 0
    while i < len(messages) and messages[i].get("role") == "tool":
        i += 1
    return messages[i:] if i else messages
