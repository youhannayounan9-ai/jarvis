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
import threading
import uuid
from datetime import datetime, timezone, timedelta
from functools import lru_cache
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

# How long a pending confirmation is valid before it is considered expired.
CONFIRMATION_TTL_MINUTES: int = 10


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
        # The store is shared across threads once the FastAPI service layer
        # runs sync endpoints in FastAPI's worker pool. SQLite connections are
        # single-threaded by default; serialize access explicitly.
        self._lock = threading.RLock()
        log.info("session_store_ready", db=settings.db_path)

    # ── Sessions ───────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        """
        Create a new conversation session and return its ID.
        Session IDs are random UUIDs — no sequential integers that reveal counts.
        """
        session_id = str(uuid.uuid4())
        now = _utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, now),
            )
            self._conn.commit()
        log.info("session_created", session_id=session_id)
        return session_id

    def list_sessions(self) -> list[dict[str, str]]:
        """Return all sessions sorted newest-first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def message_count(self, session_id: str) -> int:
        """Return how many messages are stored for a session."""
        with self._lock:
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

        with self._lock:
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
            # Invalidate INSIDE the lock: clearing after release lets another
            # thread re-cache a snapshot that misses this row (stale history).
            self.load_history.cache_clear()

    @lru_cache(maxsize=100)
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
        with self._lock:
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

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Delete sessions older than max_age_days."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()

        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE created_at < ?)",
                (cutoff_iso,)
            )

            cursor = self._conn.execute(
                "DELETE FROM sessions WHERE created_at < ?",
                (cutoff_iso,)
            )
            deleted = cursor.rowcount
            self._conn.commit()
        log.info("cleanup_old_sessions", deleted=deleted, max_age_days=max_age_days)
        return deleted

    # ── Pending confirmations ──────────────────────────────────────────────────

    def save_pending_confirmation(
        self,
        session_id: str,
        tool_name: str,
        tool_args: str,
        tool_call_id: str,
        risk_level: str,
        context: dict[str, Any] | None = None,
        ttl_minutes: int = CONFIRMATION_TTL_MINUTES,
    ) -> None:
        """
        Persist a pending confirmation request to SQLite.

        ``context`` carries the durable agent state needed to RESUME the
        original workflow after resolution (original request, pending plan,
        completed steps, execution mode). It is stored as JSON so the pause
        survives process restarts exactly like the rest of the row.

        Any previous pending confirmation for the same session is replaced
        (one active confirmation per session at a time).
        """
        expires_at = (
            datetime.now(tz=timezone.utc) + timedelta(minutes=ttl_minutes)
        ).isoformat()
        context_json = json.dumps(context or {}, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO pending_confirmations
                    (session_id, tool_name, tool_args, tool_call_id, risk_level,
                     created_at, expires_at, completed_at, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    tool_name    = excluded.tool_name,
                    tool_args    = excluded.tool_args,
                    tool_call_id = excluded.tool_call_id,
                    risk_level   = excluded.risk_level,
                    created_at   = excluded.created_at,
                    expires_at   = excluded.expires_at,
                    completed_at = NULL,
                    context_json = excluded.context_json
                """,
                (session_id, tool_name, tool_args, tool_call_id, risk_level,
                 _utcnow(), expires_at, context_json),
            )
            self._conn.commit()
        log.info(
            "pending_confirmation_saved",
            session_id=session_id,
            tool_name=tool_name,
            risk_level=risk_level,
            expires_at=expires_at,
        )

    def load_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        """
        Return the active pending confirmation for a session, or None.

        Returns None if:
        - No record exists for this session.
        - The record has already been completed.
        - The record has expired (TTL elapsed).
        Expired records are deleted on access.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT session_id, tool_name, tool_args, tool_call_id,
                       risk_level, created_at, expires_at, context_json
                FROM pending_confirmations
                WHERE session_id = ? AND completed_at IS NULL
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                return None

            # Check TTL
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now(tz=timezone.utc) > expires_at:
                log.warning(
                    "pending_confirmation_expired",
                    session_id=session_id,
                    tool_name=row["tool_name"],
                )
                self._conn.execute(
                    "DELETE FROM pending_confirmations WHERE session_id = ?",
                    (session_id,),
                )
                self._conn.commit()
                return None

            data = dict(row)
            # Deserialize the resume context; a corrupt/absent payload must
            # not crash confirmation handling — resume degrades gracefully.
            raw_ctx = data.pop("context_json", None)
            try:
                data["context"] = json.loads(raw_ctx) if raw_ctx else {}
            except json.JSONDecodeError:
                log.warning("pending_confirmation_context_corrupt", session_id=session_id)
                data["context"] = {}
            return data

    def complete_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        """
        Atomically mark a pending confirmation as completed and return its data.

        Returns the confirmation data dict if one existed and was not expired,
        or None otherwise. This is the "pop" equivalent of the old in-memory
        _pending_confirmations.pop(session_id, None).
        """
        data = self.load_pending_confirmation(session_id)
        if data is None:
            return None

        with self._lock:
            self._conn.execute(
                """
                UPDATE pending_confirmations
                   SET completed_at = ?
                 WHERE session_id = ? AND completed_at IS NULL
                """,
                (_utcnow(), session_id),
            )
            self._conn.commit()
        log.info(
            "pending_confirmation_completed",
            session_id=session_id,
            tool_name=data["tool_name"],
        )
        return data

    def cleanup_expired_confirmations(self) -> int:
        """
        Delete all expired or completed confirmation rows.
        Returns the number of rows removed.
        """
        now = _utcnow()
        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM pending_confirmations
                WHERE completed_at IS NOT NULL
                   OR expires_at < ?
                """,
                (now,),
            )
            self._conn.commit()
            removed = cursor.rowcount
        if removed:
            log.info("cleanup_expired_confirmations", removed=removed)
        return removed

    def close(self) -> None:
        """Close the database connection cleanly."""
        with self._lock:
            self._conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection.

    File-backed databases run in WAL mode with synchronous=NORMAL: writers no
    longer block readers (the API serves history while a chat turn commits),
    and the DB survives sudden process death without corruption. WAL only
    works on real files, so :memory: keeps the default journal mode.
    """
    db_path = Path(settings.db_path)
    # ":memory:" must not be passed through Path (would become a relative file).
    if settings.db_path == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
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

        CREATE TABLE IF NOT EXISTS pending_confirmations (
            session_id   TEXT    PRIMARY KEY,
            tool_name    TEXT    NOT NULL,
            tool_args    TEXT    NOT NULL,
            tool_call_id TEXT    NOT NULL,
            risk_level   TEXT    NOT NULL,
            created_at   TEXT    NOT NULL,
            expires_at   TEXT    NOT NULL,
            completed_at TEXT,
            context_json TEXT
        );
    """)
    # Lightweight migration for pre-v0.15 databases: older installations
    # created this table without the resume-context column.
    existing_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(pending_confirmations)").fetchall()
    }
    if "context_json" not in existing_cols:
        conn.execute("ALTER TABLE pending_confirmations ADD COLUMN context_json TEXT")
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
