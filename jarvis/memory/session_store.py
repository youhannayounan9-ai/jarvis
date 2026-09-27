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
import socket
import sqlite3
import threading
import uuid
from dataclasses import dataclass
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

# ── Execution ledger (v0.17): durable state machine for protected actions ────
#
#   PENDING    → eligible to execute (created when a confirmation is parked)
#   RUNNING    → an execution attempt was atomically claimed; dispatch in flight
#   SUCCEEDED  → tool result durably recorded
#   FAILED     → tool was attempted and failed with a known result
#   UNKNOWN    → the process may have died after the tool was dispatched but
#                before the result was durably recorded. The side effect may or
#                may not have happened — the action MUST NOT be re-executed
#                automatically.
#
# This gives at-most-once AUTOMATIC execution (duplicate-dispatch protection),
# not "exactly once" external side effects: JARVIS cannot transactionally
# couple an arbitrary tool's side effect to the database record.
ACTION_STATE_PENDING = "PENDING"
ACTION_STATE_RUNNING = "RUNNING"
ACTION_STATE_SUCCEEDED = "SUCCEEDED"
ACTION_STATE_FAILED = "FAILED"
ACTION_STATE_UNKNOWN = "UNKNOWN"
_TERMINAL_ACTION_STATES = frozenset(
    {ACTION_STATE_SUCCEEDED, ACTION_STATE_FAILED, ACTION_STATE_UNKNOWN}
)


@dataclass(frozen=True)
class ActionExecution:
    """One durable record of a protected (confirmation-gated) action."""

    action_id: str
    session_id: str
    confirmation_id: str
    tool_name: str
    tool_args: str
    risk_level: str
    state: str
    attempt: int
    result: str | None
    created_at: str
    claimed_at: str | None
    finished_at: str | None
    owner: str | None
    # v0.19: durable resume context captured when the action was parked
    # (JSON, may be None on legacy rows). Enables full-context recovery on
    # an explicit UNKNOWN reissue. Never exposed via the API surface.
    pause_context_json: str | None = None


# ── Session lease (v0.17): database-backed per-session turn coordination ─────
#
# Replaces the process-local per-session mutex so two JARVIS processes
# sharing the same SQLite file cannot execute turns on the same session
# concurrently. A dead owner's lease is recovered after the TTL; a fencing
# counter increments on every ownership change so stale owners can detect
# (and callers can record) that they lost the session mid-turn.
SESSION_LEASE_TTL_SECONDS: int = 300


def new_owner_token(component: str = "runtime") -> str:
    """A durable, human-traceable owner token: host:pid:component:random."""
    return (
        f"{socket.gethostname()}:{__import__('os').getpid()}:{component}:"
        f"{uuid.uuid4().hex[:8]}"
    )


def redact_owner(owner_token: str | None) -> str:
    """
    Operator-safe rendering of a lease/action owner token.

    The raw token (``host:pid:component:random``) is an internal identity;
    only its component and short random suffix are disclosable. None/empty
    renders as ``none`` (e.g. an action that was never claimed).
    """
    if not owner_token:
        return "none"
    parts = str(owner_token).split(":")
    if len(parts) >= 4:
        return f"{parts[2]}:{parts[3]}"
    if len(parts) >= 1 and parts[0]:
        return parts[-1][:16]
    return "none"


# Deliberate reissue of an UNKNOWN action creates a NEW action identity;
# this bounds how many times one original action may be reissued (no loops).
MAX_REISSUES_PER_ACTION: int = 3

# Hard upper bound for ledger introspection queries (memory safety).
_MAX_ACTION_LIST_LIMIT: int = 500


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

    def cleanup_operational_records(
        self,
        terminal_actions_days: int = 30,
        reissues_days: int = 90,
        leases_days: int = 30,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        v0.19 retention for operational records — bounded, conservative,
        transactional. Only TERMINAL ledger states age out (SUCCEEDED, plus
        FAILED rows whose confirmation is no longer active); PENDING, RUNNING
        and UNKNOWN are NEVER deleted here, active confirmations are never
        orphaned, and every still-referenced audit row is kept (a reissue
        whose original or new action still exists survives regardless of
        age, so chains never dangle). Also purges leases of sessions that no
        longer exist. Returns per-table candidate counts; with ``dry_run``
        nothing is mutated (count-only pass, explicit contract, tested).
        """
        cutoff_a = (
            datetime.now(tz=timezone.utc) - timedelta(days=terminal_actions_days)
        ).isoformat()
        cutoff_r = (
            datetime.now(tz=timezone.utc) - timedelta(days=reissues_days)
        ).isoformat()
        cutoff_l = (
            datetime.now(tz=timezone.utc) - timedelta(days=leases_days)
        ).isoformat()
        with self._lock:
            # ── 1. Terminal action candidates (age + state gates) ───────────
            # Protected audit rows: any action referenced by a reissue must
            # survive, or the audit chain would dangle (FK violation).
            reissue_referenced = {
                str(r[0])
                for r in self._conn.execute(
                    "SELECT original_action_id FROM action_reissues "
                    "UNION SELECT new_action_id FROM action_reissues"
                ).fetchall()
            }
            # Confirmations not yet completed: their ledger rows are the
            # at-most-once identity for an approval that may still arrive.
            live_confirmations = {
                str(r[0])
                for r in self._conn.execute(
                    """
                    SELECT confirmation_id FROM pending_confirmations
                     WHERE completed_at IS NULL
                    """
                ).fetchall()
            }
            candidates: set[str] = set()
            confirmation_by_action: dict[str, str] = {}
            for r in self._conn.execute(
                """
                SELECT action_id, confirmation_id, state FROM action_executions
                 WHERE created_at < ? AND state IN (?, ?)
                """,
                (cutoff_a, ACTION_STATE_SUCCEEDED, ACTION_STATE_FAILED),
            ).fetchall():
                confirmation_by_action[str(r["action_id"])] = str(r["confirmation_id"])
                candidates.add(str(r["action_id"]))
            candidates -= reissue_referenced
            # A FAILED action whose confirmation is still awaiting resolution
            # is recoverable state — keep it.
            candidates -= {
                a for a, cid in confirmation_by_action.items()
                if cid in live_confirmations
            }

            # ── 2. Reissue audit rows (only when BOTH ends are gone) ────────
            # Compute the post-cleanup action set so a reissue whose ends are
            # removed in THIS pass is also collected; anything whose original
            # or new action still exists is kept, however old (chain integrity).
            final_action_rows = {
                str(r["action_id"])
                for r in self._conn.execute(
                    "SELECT action_id FROM action_executions"
                ).fetchall()
            } - candidates
            reissue_candidates: list[int] = []
            for r in self._conn.execute(
                "SELECT id, original_action_id, new_action_id, created_at FROM action_reissues"
            ).fetchall():
                if str(r["created_at"]) >= cutoff_r:
                    continue
                if (
                    str(r["original_action_id"]) not in final_action_rows
                    and str(r["new_action_id"]) not in final_action_rows
                ):
                    reissue_candidates.append(int(r["id"]))

            # ── 3. Stale leases (expired past retention or orphaned) ────────
            stale_lease_rows = [
                str(r["session_id"])
                for r in self._conn.execute(
                    """
                    SELECT session_id FROM session_leases
                     WHERE expires_at < ?
                        OR session_id NOT IN (SELECT id FROM sessions)
                    """,
                    (cutoff_l,),
                ).fetchall()
            ]

            # ── 4. Mutate (one transaction) or just report ─────────────────
            if not dry_run:
                if candidates:
                    marks = ",".join("?" for _ in candidates)
                    self._conn.execute(
                        f"DELETE FROM action_executions WHERE action_id IN ({marks})",
                        tuple(candidates),
                    )
                if reissue_candidates:
                    marks = ",".join("?" for _ in reissue_candidates)
                    self._conn.execute(
                        f"DELETE FROM action_reissues WHERE id IN ({marks})",
                        tuple(reissue_candidates),
                    )
                if stale_lease_rows:
                    marks = ",".join("?" for _ in stale_lease_rows)
                    self._conn.execute(
                        f"DELETE FROM session_leases WHERE session_id IN ({marks})",
                        tuple(stale_lease_rows),
                    )
                self._conn.commit()
        if not dry_run:
            log.info(
                "cleanup_operational_records",
                actions_removed=len(candidates),
                reissues_removed=len(reissue_candidates),
                leases_removed=len(stale_lease_rows),
            )
        return {
            "actions_removed": len(candidates),
            "reissues_removed": len(reissue_candidates),
            "stale_leases_removed": len(stale_lease_rows),
            "dry_run": int(dry_run),
        }

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
            # Drop leases of removed sessions so the table cannot grow with
            # dead session ids (one row per live session otherwise).
            self._conn.execute(
                """
                DELETE FROM session_leases
                 WHERE session_id IN (SELECT id FROM sessions WHERE created_at < ?)
                    OR session_id NOT IN (SELECT id FROM sessions)
                """,
                (cutoff_iso,),
            )
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
    ) -> str:
        """
        Persist a pending confirmation request to SQLite.

        ``context`` carries the durable agent state needed to RESUME the
        original workflow after resolution (original request, pending plan,
        completed steps, execution mode). It is stored as JSON so the pause
        survives process restarts exactly like the rest of the row.

        Any previous pending confirmation for the same session is replaced
        (one active confirmation per session at a time).

        Returns:
            The server-generated ``confirmation_id`` that durably identifies
            this pending action (and pairs it with its execution-ledger row).
        """
        confirmation_id = uuid.uuid4().hex
        expires_at = (
            datetime.now(tz=timezone.utc) + timedelta(minutes=ttl_minutes)
        ).isoformat()
        context_json = json.dumps(context or {}, separators=(",", ":"))
        with self._lock:
            # A replaced pending confirmation (one active per session) leaves
            # its ledger row PENDING forever unless explicitly closed out:
            # mark the superseded action FAILED so it can never be claimed.
            previous = self._conn.execute(
                "SELECT confirmation_id FROM pending_confirmations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if previous is not None and previous["confirmation_id"]:
                self._conn.execute(
                    """
                    UPDATE action_executions
                       SET state = ?, result = ?, finished_at = ?
                     WHERE confirmation_id = ? AND state = ?
                    """,
                    (ACTION_STATE_FAILED,
                     "superseded: a newer action replaced this confirmation",
                     _utcnow(), previous["confirmation_id"], ACTION_STATE_PENDING),
                )
            self._conn.execute(
                """
                INSERT INTO pending_confirmations
                    (confirmation_id, session_id, tool_name, tool_args, tool_call_id,
                     risk_level, created_at, expires_at, completed_at, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    confirmation_id = excluded.confirmation_id,
                    tool_name    = excluded.tool_name,
                    tool_args    = excluded.tool_args,
                    tool_call_id = excluded.tool_call_id,
                    risk_level   = excluded.risk_level,
                    created_at   = excluded.created_at,
                    expires_at   = excluded.expires_at,
                    completed_at = NULL,
                    context_json = excluded.context_json
                """,
                (confirmation_id, session_id, tool_name, tool_args, tool_call_id,
                 risk_level, _utcnow(), expires_at, context_json),
            )
            # v0.17: every parked action immediately gets a durable ledger row
            # (PENDING) in the SAME transaction — the identity that the
            # approval path later claims at-most-once.
            # v0.19: the resume context is ALSO captured here (same
            # transaction), so it survives the confirmation pop and any crash
            # window; an explicit UNKNOWN reissue can then restore the
            # original paused task verbatim.
            self._conn.execute(
                """
                INSERT INTO action_executions
                    (action_id, session_id, confirmation_id, tool_name, tool_args,
                     risk_level, state, attempt, result, created_at, pause_context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (uuid.uuid4().hex, session_id, confirmation_id, tool_name,
                 tool_args, risk_level, ACTION_STATE_PENDING, _utcnow(),
                 context_json),
            )
            self._conn.commit()
        log.info(
            "pending_confirmation_saved",
            session_id=session_id,
            tool_name=tool_name,
            risk_level=risk_level,
            expires_at=expires_at,
            confirmation_id=confirmation_id,
        )
        return confirmation_id

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
                SELECT confirmation_id, session_id, tool_name, tool_args, tool_call_id,
                       risk_level, created_at, expires_at, context_json
                FROM pending_confirmations
                WHERE session_id = ? AND completed_at IS NULL
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                return None

            # Check TTL
            expires_at = _parse_ts(row["expires_at"])
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
        Atomically pop the session's pending confirmation.

        v0.17 hardening: the claim is a single locked sequence (read → TTL
        check → conditional UPDATE with rowcount verification), so two
        concurrent resolvers can never both receive the same confirmation —
        the loser observes rowcount 0 and gets None.

        Returns the confirmation data dict if one existed and was not expired,
        or None otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT confirmation_id, session_id, tool_name, tool_args, tool_call_id,
                       risk_level, created_at, expires_at, context_json
                FROM pending_confirmations
                WHERE session_id = ? AND completed_at IS NULL
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None

            expires_at = _parse_ts(row["expires_at"])
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

            # The conditional UPDATE is the actual claim: exactly one caller
            # flips completed_at from NULL; every other concurrent caller's
            # UPDATE matches zero rows.
            cursor = self._conn.execute(
                """
                UPDATE pending_confirmations
                   SET completed_at = ?
                 WHERE session_id = ? AND completed_at IS NULL
                """,
                (_utcnow(), session_id),
            )
            if cursor.rowcount != 1:  # pragma: no cover - defensive
                self._conn.rollback()
                return None
            self._conn.commit()

        data = dict(row)
        raw_ctx = data.pop("context_json", None)
        try:
            data["context"] = json.loads(raw_ctx) if raw_ctx else {}
        except json.JSONDecodeError:
            log.warning("pending_confirmation_context_corrupt", session_id=session_id)
            data["context"] = {}
        log.info(
            "pending_confirmation_completed",
            session_id=session_id,
            tool_name=data["tool_name"],
            confirmation_id=data.get("confirmation_id"),
        )
        return data

    # ── Execution ledger (v0.17) ──────────────────────────────────────────────

    def get_action_execution(self, action_id: str) -> ActionExecution | None:
        """Return one ledger row by action_id, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM action_executions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        return self._action_row_to_dataclass(row) if row else None

    def get_action_execution_by_confirmation(
        self, confirmation_id: str
    ) -> ActionExecution | None:
        """Return the ledger row paired with a parked confirmation, or None.

        None also means "parked before v0.17" (legacy row without a ledger
        pair) — callers fall back to direct dispatch for compatibility.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM action_executions WHERE confirmation_id = ?",
                (confirmation_id,),
            ).fetchone()
        return self._action_row_to_dataclass(row) if row else None

    def get_last_action_execution(self, session_id: str) -> ActionExecution | None:
        """Most recent ledger row for a session (duplicate-approval reporting)."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM action_executions
                WHERE session_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._action_row_to_dataclass(row) if row else None

    # ── Action ledger introspection (v0.18, read-only) ───────────────────────

    def list_action_executions(
        self,
        state: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        newer_than: str | None = None,
    ) -> list[ActionExecution]:
        """
        Bounded, deterministic read-only view of the ledger, newest first.

        Filters: exact ``state`` (must be a valid ledger state when given),
        exact ``session_id``, and ``newer_than`` (ISO timestamp, exclusive —
        the cursor boundary for pagination). Always ``LIMIT``-bounded so a
        large history never floods memory; ties broken by rowid for a stable
        order.
        """
        if state is not None and state not in (
            ACTION_STATE_PENDING,
            ACTION_STATE_RUNNING,
            ACTION_STATE_SUCCEEDED,
            ACTION_STATE_FAILED,
            ACTION_STATE_UNKNOWN,
        ):
            raise ValueError(f"invalid action state filter: {state!r}")
        limit = max(1, min(int(limit), _MAX_ACTION_LIST_LIMIT))
        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if newer_than is not None:
            clauses.append("created_at > ?")
            params.append(newer_than)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM action_executions {where}
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [self._action_row_to_dataclass(r) for r in rows]

    def count_action_executions_by_state(self) -> dict[str, int]:
        """One bounded aggregate row per ledger state (zero rows for empty states)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM action_executions GROUP BY state"
            ).fetchall()
        return {str(r["state"]): int(r["n"]) for r in rows}

    # ── UNKNOWN recovery: explicit reissue (v0.18) ────────────────────────
    #
    # Reissue NEVER flips an UNKNOWN action back to RUNNING under the same
    # identity. It creates a NEW PENDING action (fresh action_id) that flows
    # through the normal permission/confirmation/dispatch machinery, and
    # records an audit row linking original → reissued.

    def count_reissues_for_action(self, action_id: str) -> int:
        """How many reissues already descend from this action (bounded loops)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM action_reissues WHERE original_action_id = ?",
                (action_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def request_action_reissue(
        self,
        action_id: str,
        request_id: str,
        max_reissues: int = MAX_REISSUES_PER_ACTION,
    ) -> str:
        """
        Deliberately re-issue an UNKNOWN action as a NEW action identity.

        One transaction does everything, so concurrent duplicate reissue
        requests (same request_id, two processes) still produce exactly one
        new action:

          1. Load the original; refuse unless it is UNKNOWN.
          2. Return early when this exact request already reissued the
             action (idempotent: same request → same new action) — even
             after the reissue ceiling was reached.
          3. Enforce the reissue ceiling (MAX_REISSUES_PER_ACTION).
          4. Refuse when the session already holds an ACTIVE pending
             confirmation: one active confirmation per session, and reissue
             never silently replaces what the user is currently being asked
             to approve. Completed/expired leftovers are cleaned up and
             their stranded PENDING ledger rows closed, exactly like
             save_pending_confirmation's replace semantics.
          5. INSERT the new PENDING ledger row AND its durable pending
             confirmation (confirmation_id ``reissue:{request_id}:{action}``)
             so the reissued action resolves through the ordinary
             handle_confirmation flow — approval, denial, and the claim
             machinery are unchanged. Empty resume context: the outcome is
             reported directly, like any freshly parked action.
          6. INSERT the audit row last — the (request_id, action_id) UNIQUE
             index is the cross-process idempotency gate; on that conflict
             the reissue this same request already created is returned.

        Nothing executes here: the new action is PENDING with attempt 0 and
        runs only after an explicit approval, exactly like any other
        protected action. The original UNKNOWN row is never modified.

        Raises:
            ValueError: unknown action, wrong state, reissue limit reached,
                or the session already has an active pending confirmation.
        """
        if not request_id or not request_id.strip():
            raise ValueError("reissue request_id must be non-empty")
        session_id = ""
        with self._lock:
            try:
                original = self._conn.execute(
                    "SELECT * FROM action_executions WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if original is None:
                    raise ValueError(f"unknown action_id: {action_id}")
                if str(original["state"]) != ACTION_STATE_UNKNOWN:
                    raise ValueError(
                        f"action {action_id} is {original['state']}, not UNKNOWN — "
                        "reissue is only for ambiguous (UNKNOWN) actions"
                    )
                session_id = str(original["session_id"])
                # Idempotent replay first (C4): this exact request already
                # reissued this action → return the same new action without
                # touching anything. Must precede the ceiling and the
                # active-confirmation guard so accidental duplicate
                # submissions never fail spuriously.
                seen = self._conn.execute(
                    """
                    SELECT new_action_id FROM action_reissues
                    WHERE request_id = ? AND original_action_id = ?
                    """,
                    (request_id, action_id),
                ).fetchone()
                if seen is not None:
                    log.info("action_reissue_duplicate_request", action_id=action_id)
                    return str(seen["new_action_id"])
                already = self._conn.execute(
                    """
                    SELECT COUNT(*) AS n FROM action_reissues
                    WHERE original_action_id = ?
                    """,
                    (action_id,),
                ).fetchone()
                if int(already["n"]) >= max_reissues:
                    raise ValueError(
                        f"reissue limit reached for action {action_id} "
                        f"({int(already['n'])}/{max_reissues})"
                    )
                now = _utcnow()
                # The reissued action must be resolvable through the normal
                # confirmation flow, so park a durable pending confirmation
                # for the session in the SAME transaction. One active
                # confirmation per session (PK on session_id): refuse rather
                # than silently replace what the user is currently being
                # asked about; clean up completed/expired leftovers exactly
                # like save_pending_confirmation's replace semantics.
                previous = self._conn.execute(
                    """
                    SELECT confirmation_id, completed_at, expires_at
                    FROM pending_confirmations WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if previous is not None:
                    if (
                        previous["completed_at"] is None
                        and _parse_ts(str(previous["expires_at"]))
                        > datetime.now(tz=timezone.utc)
                    ):
                        raise ValueError(
                            f"session {session_id} already has an active "
                            "pending confirmation — resolve it before "
                            "reissuing actions for this session"
                        )
                    self._conn.execute(
                        """
                        UPDATE action_executions
                           SET state = ?, result = ?, finished_at = ?
                         WHERE confirmation_id = ? AND state = ?
                        """,
                        (ACTION_STATE_FAILED,
                         "superseded: a newer action replaced this confirmation",
                         now, previous["confirmation_id"], ACTION_STATE_PENDING),
                    )
                    self._conn.execute(
                        "DELETE FROM pending_confirmations WHERE session_id = ?",
                        (session_id,),
                    )
                new_action_id = uuid.uuid4().hex
                # Deterministic per (request_id, original action): replays
                # rebuild the same id, and one request_id reused across two
                # different originals can never collide on the
                # confirmation_id → ledger-row lookup.
                reissue_confirmation_id = f"reissue:{request_id}:{action_id}"
                self._conn.execute(
                    """
                    INSERT INTO action_executions
                        (action_id, session_id, confirmation_id, tool_name, tool_args,
                         risk_level, state, attempt, result, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
                    """,
                    (
                        new_action_id,
                        session_id,
                        reissue_confirmation_id,
                        original["tool_name"],
                        original["tool_args"],
                        original["risk_level"],
                        ACTION_STATE_PENDING,
                        now,
                    ),
                )
                # Durable confirmation row: approval/denial flows through the
                # ordinary handle_confirmation path (permission architecture
                # unchanged). v0.19 full-context recovery: the ORIGINAL
                # action's captured pause context (recorded at park time in
                # the same transaction that created the ledger row) is copied
                # verbatim onto the new confirmation, so approval continues
                # the original paused task — remaining plan steps, budgets,
                # mode — instead of stopping at a bare result. Malformed or
                # absent context degrades to the pre-v0.19 raw-result reply
                # (load_pending_confirmation already handles both). A fresh
                # recovery marker lets the orchestrator label the resumed
                # step truthfully ("recovered action") in synthesis.
                original_ctx_raw = original["pause_context_json"]
                # v0.19 transitive recovery: when reissuing a reissued action
                # whose own confirmation was already resolved (and therefore
                # popped), the task context lives on the CHAIN ORIGIN's ledger
                # row. Walk back through the audit rows (bounded by the
                # reissue ceiling) so the new action always carries the
                # original task context — chain depth stays ≤ 3.
                if not (original_ctx_raw and original_ctx_raw.strip()):
                    cursor_origin: str | None = action_id
                    seen_origins: set[str] = set()
                    while cursor_origin:
                        if cursor_origin in seen_origins:  # pragma: no cover
                            break
                        seen_origins.add(cursor_origin)
                        prev = self._conn.execute(
                            """
                            SELECT original_action_id FROM action_reissues
                             WHERE new_action_id = ?
                            """,
                            (cursor_origin,),
                        ).fetchone()
                        if prev is None:
                            break
                        cursor_origin = str(prev["original_action_id"])
                        ancestor = self._conn.execute(
                            """
                            SELECT pause_context_json FROM action_executions
                             WHERE action_id = ?
                            """,
                            (cursor_origin,),
                        ).fetchone()
                        if ancestor is not None and ancestor["pause_context_json"]:
                            original_ctx_raw = str(ancestor["pause_context_json"])
                            break
                if original_ctx_raw:
                    try:
                        recovery_ctx: dict[str, Any] = dict(
                            json.loads(original_ctx_raw)
                        )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        log.warning(
                            "reissue_context_corrupt_degrades",
                            action_id=action_id,
                        )
                        recovery_ctx = {}
                    recovery_ctx["recovered_from_action"] = action_id
                    recovery_ctx["recovered_request_id"] = request_id
                else:
                    recovery_ctx = {}
                    recovery_ctx["recovered_from_action"] = action_id
                    recovery_ctx["recovered_request_id"] = request_id
                self._conn.execute(
                    """
                    INSERT INTO pending_confirmations
                        (confirmation_id, session_id, tool_name, tool_args, tool_call_id,
                         risk_level, created_at, expires_at, completed_at, context_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        reissue_confirmation_id,
                        session_id,
                        str(original["tool_name"]),
                        str(original["tool_args"]),
                        f"reissue-{request_id}-{action_id[:12]}",
                        str(original["risk_level"]),
                        now,
                        (
                            datetime.now(tz=timezone.utc)
                            + timedelta(minutes=CONFIRMATION_TTL_MINUTES)
                        ).isoformat(),
                        json.dumps(recovery_ctx, separators=(",", ":")),
                    ),
                )
                # Audit row last: its UNIQUE index is the cross-process
                # idempotency gate (C4).
                self._conn.execute(
                    """
                    INSERT INTO action_reissues
                        (request_id, original_action_id, new_action_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (request_id, action_id, new_action_id, now),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                # Same request_id for the same original action reissued by a
                # concurrent process — return the existing result instead of
                # failing (C4). A pending_confirmations PK conflict under a
                # concurrent same-session reissue surfaces as a clear 409.
                self._conn.rollback()
                row = self._conn.execute(
                    """
                    SELECT new_action_id FROM action_reissues
                    WHERE request_id = ? AND original_action_id = ?
                    """,
                    (request_id, action_id),
                ).fetchone()
                if row is not None:
                    log.info("action_reissue_duplicate_request", action_id=action_id)
                    return str(row["new_action_id"])
                if "pending_confirmations" in str(e):
                    raise ValueError(
                        f"session {session_id} already has an active pending "
                        "confirmation — resolve it before reissuing actions "
                        "for this session"
                    ) from e
                raise
            except Exception:
                self._conn.rollback()
                raise
        log.warning(
            "action_reissue_created",
            original_action_id=action_id,
            new_action_id=new_action_id,
            request_id=request_id,
            confirmation_id=reissue_confirmation_id,
        )
        return new_action_id

    def get_reissue_chain(self, action_id: str) -> list[dict[str, str]]:
        """Audit relationship: every reissue descending from this action."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT request_id, original_action_id, new_action_id, created_at
                FROM action_reissues
                WHERE original_action_id = ?
                ORDER BY created_at, id
                """,
                (action_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_reissue_origin(self, new_action_id: str) -> str | None:
        """The UNKNOWN action this one was reissued from, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT original_action_id FROM action_reissues WHERE new_action_id = ?",
                (new_action_id,),
            ).fetchone()
        return str(row["original_action_id"]) if row else None

    def get_action_recovery_preview(self, action_id: str) -> dict[str, Any]:
        """
        Safe, bounded recovery preview for an action (operator decision
        support; NEVER the raw context — tool payloads stay in the store).

        Returns: has_context, original_request (truncated), mode,
        step_number, pending_steps, completed_steps, remaining_rounds,
        recoverable, recoverable_reason. Unknown/legacy/corrupt rows are
        reported honestly rather than guessed.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT pause_context_json FROM action_executions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            return {"recoverable": False, "recoverable_reason": "unknown action"}
        raw = row["pause_context_json"]
        if not raw:
            return {
                "recoverable": False,
                "recoverable_reason": (
                    "no durable resume context was captured for this action "
                    "(pre-v0.19 row or a fast-path park without a plan); "
                    "reissue still works and reports the result directly"
                ),
                "has_context": False,
            }
        try:
            ctx = json.loads(raw)
            if not isinstance(ctx, dict):
                raise ValueError("context is not an object")
        except (json.JSONDecodeError, TypeError, ValueError):
            return {
                "recoverable": False,
                "recoverable_reason": (
                    "captured context is corrupt; reissue will degrade to "
                    "reporting the result directly (pre-v0.19 behavior)"
                ),
                "has_context": True,
                "corrupt": True,
            }
        pending = [
            s for s in (ctx.get("pending_plan") or [])
            if str(s.get("description") or "").strip()
            and int(s.get("step_number") or 0) > int(ctx.get("step_number") or 0)
        ]
        original_request = str(ctx.get("original_request") or "")
        return {
            "recoverable": bool(original_request.strip()),
            "has_context": True,
            "recoverable_reason": (
                "approval will execute the tool once, then continue the "
                "original task from its paused step"
                if original_request.strip()
                else "context exists but carries no original_request; "
                "reissue reports the result directly"
            ),
            "original_request": original_request[:300],
            "mode": str(ctx.get("mode") or "complex"),
            "step_number": int(ctx.get("step_number") or 0),
            "pending_steps": len(pending),
            "completed_steps": len(ctx.get("completed_steps") or []),
            "remaining_rounds": int(ctx.get("remaining_rounds") or 0),
        }

    # ── v0.19: causal session timeline (read-only) ────────────────────────

    def get_session_timeline(
        self, session_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        """
        Causal, read-only timeline of one session's reliability events,
        merged in timestamp order:

          message / confirmation_parked / action_state / reissue /
          confirmation_resolved / lease

        Bounded (LIMIT clamped to _MAX_ACTION_LIST_LIMIT); stable ordering by
        (ts, seq). NEVER includes tool_args, tool result bodies, message
        content, or raw owner tokens — safe metadata only. Unknown sessions
        return [] (callers decide 404 vs empty).
        """
        limit = max(1, min(int(limit), _MAX_ACTION_LIST_LIMIT))
        events: list[dict[str, Any]] = []
        with self._lock:
            for row in self._conn.execute(
                """
                SELECT id, role, created_at FROM messages
                 WHERE session_id = ? ORDER BY id LIMIT ?
                """,
                (session_id, limit),
            ):
                events.append({
                    "ts": row["created_at"], "seq": int(row["id"]),
                    "kind": "message", "role": row["role"],
                })
            for row in self._conn.execute(
                """
                SELECT tool_name, risk_level, created_at, completed_at
                  FROM pending_confirmations WHERE session_id = ? LIMIT ?
                """,
                (session_id, limit),
            ):
                events.append({
                    "ts": row["created_at"], "seq": 0,
                    "kind": "confirmation_parked",
                    "tool": row["tool_name"], "risk_level": row["risk_level"],
                    "resolved": row["completed_at"] is not None,
                })
            for row in self._conn.execute(
                """
                SELECT action_id, confirmation_id, tool_name, risk_level, state,
                       attempt, created_at, claimed_at, finished_at
                  FROM action_executions WHERE session_id = ? LIMIT ?
                """,
                (session_id, limit),
            ):
                events.append({
                    "ts": row["created_at"], "seq": 0,
                    "kind": "action_state", "action_id": row["action_id"],
                    "confirmation_id": row["confirmation_id"],
                    "tool": row["tool_name"], "risk_level": row["risk_level"],
                    "state": row["state"], "attempt": int(row["attempt"]),
                    "claimed_at": row["claimed_at"],
                    "finished_at": row["finished_at"],
                })
            for row in self._conn.execute(
                """
                SELECT r.request_id, r.original_action_id, r.new_action_id,
                       r.created_at, a.tool_name
                  FROM action_reissues r
                  JOIN action_executions a ON a.action_id = r.original_action_id
                 WHERE a.session_id = ? LIMIT ?
                """,
                (session_id, limit),
            ):
                events.append({
                    "ts": row["created_at"], "seq": 0,
                    "kind": "reissue", "request_id": row["request_id"],
                    "original_action_id": row["original_action_id"],
                    "new_action_id": row["new_action_id"],
                    "tool": row["tool_name"],
                })
            for row in self._conn.execute(
                """
                SELECT owner_token, acquired_at, expires_at, fencing
                  FROM session_leases WHERE session_id = ? LIMIT 1
                """,
                (session_id,),
            ).fetchall():
                events.append({
                    "ts": row["acquired_at"], "seq": 0,
                    "kind": "lease",
                    "owner": redact_owner(row["owner_token"]),
                    "fencing": int(row["fencing"]),
                    "active": _parse_ts(row["expires_at"]) > datetime.now(tz=timezone.utc),
                })
        events.sort(key=lambda e: (str(e.get("ts") or ""), int(e.get("seq") or 0)))
        return events[-limit:]

    # ── v0.20: personal knowledge base — document registry (read/write) ───

    def upsert_knowledge_document(
        self,
        document_id: str,
        source_path: str,
        filename: str,
        media_type: str,
        size_bytes: int,
        content_hash: str,
        chunk_count: int,
        parser_version: str,
        modified_at: str | None = None,
    ) -> None:
        """Insert or refresh one document's registry row (one tx)."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO knowledge_documents
                    (document_id, source_path, filename, media_type, size_bytes,
                     content_hash, chunk_count, parser_version, ingested_at, modified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_path    = excluded.source_path,
                    filename       = excluded.filename,
                    media_type     = excluded.media_type,
                    size_bytes     = excluded.size_bytes,
                    content_hash   = excluded.content_hash,
                    chunk_count    = excluded.chunk_count,
                    parser_version = excluded.parser_version,
                    ingested_at    = excluded.ingested_at,
                    modified_at    = excluded.modified_at
                """,
                (
                    document_id, source_path, filename, media_type, int(size_bytes),
                    content_hash, int(chunk_count), parser_version, _utcnow(),
                    modified_at,
                ),
            )
            self._conn.commit()
        log.info(
            "knowledge_document_upserted",
            document_id=document_id,
            filename=filename,
            chunk_count=chunk_count,
        )

    def get_knowledge_document(self, document_id: str) -> dict[str, Any] | None:
        """One document's registry row, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_knowledge_document_by_path(self, source_path: str) -> dict[str, Any] | None:
        """Registry row for an exact absolute source path, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM knowledge_documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return dict(row) if row else None

    def list_knowledge_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        """Bounded, newest-first document registry listing."""
        limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM knowledge_documents
                 ORDER BY ingested_at DESC, rowid DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_knowledge_document(self, document_id: str) -> bool:
        """Remove one document's registry row. True when it existed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM knowledge_documents WHERE document_id = ?",
                (document_id,),
            )
            self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("knowledge_document_deleted", document_id=document_id)
        return deleted

    def find_knowledge_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        """Any document with identical content (same-bytes reuse), or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM knowledge_documents WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
        return dict(row) if row else None

    def claim_action_execution(self, action_id: str, owner: str) -> str:
        """
        Atomically claim the single execution attempt for an action.

        Returns one of:
          "claimed"                       — this caller owns the execution now
          "already_running"               — another claimant holds RUNNING
          "already_terminal:<STATE>"      — the action already reached a
                                            terminal state and must NOT run
                                            again (its recorded result, if
                                            any, is the outcome to report)

        The claim is one SQL UPDATE constrained on the previous state, so the
        database (not a process lock) provides the coordination — two
        processes sharing the file cannot both win.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM action_executions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if row is None:
                log.error("action_claim_missing_row", action_id=action_id)
                return "already_terminal:UNKNOWN"
            state = str(row["state"])
            if state != ACTION_STATE_PENDING:
                return f"already_terminal:{state}" if state in _TERMINAL_ACTION_STATES else "already_running"
            cursor = self._conn.execute(
                """
                UPDATE action_executions
                   SET state = ?, attempt = attempt + 1, claimed_at = ?, owner = ?
                 WHERE action_id = ? AND state = ?
                """,
                (ACTION_STATE_RUNNING, _utcnow(), owner, action_id, ACTION_STATE_PENDING),
            )
            if cursor.rowcount != 1:  # pragma: no cover - defensive under lock
                self._conn.rollback()
                return "already_running"
            self._conn.commit()
        log.info("action_claimed", action_id=action_id, owner=owner)
        return "claimed"

    def finish_action_execution(
        self, action_id: str, state: str, result: str | None
    ) -> None:
        """
        Record the outcome of an action: RUNNING → terminal (normal dispatch)
        or PENDING → terminal (denial / superseded — closed without ever
        executing; ``attempt`` stays 0 in that case).
        """
        if state not in _TERMINAL_ACTION_STATES or (
            state == ACTION_STATE_UNKNOWN and result is None
        ):
            raise ValueError(f"invalid terminal state for finish: {state!r}")
        with self._lock:
            self._conn.execute(
                """
                UPDATE action_executions
                   SET state = ?, result = ?, finished_at = ?
                 WHERE action_id = ? AND state IN (?, ?)
                """,
                (
                    state,
                    result,
                    _utcnow(),
                    action_id,
                    ACTION_STATE_RUNNING,
                    ACTION_STATE_PENDING,
                ),
            )
            self._conn.commit()
        log.info(
            "action_finished", action_id=action_id, state=state,
            result_chars=len(result or ""),
        )

    def mark_action_unknown(self, action_id: str, reason: str) -> None:
        """Explicitly record crash ambiguity for a RUNNING action."""
        with self._lock:
            self._conn.execute(
                """
                UPDATE action_executions
                   SET state = ?, finished_at = ?
                 WHERE action_id = ? AND state = ?
                """,
                (ACTION_STATE_UNKNOWN, _utcnow(), action_id, ACTION_STATE_RUNNING),
            )
            self._conn.commit()
        log.warning(
            "action_marked_unknown", action_id=action_id, reason=reason,
        )

    def recover_unknown_action_executions(self) -> int:
        """
        Startup crash-recovery sweep: any action left RUNNING by a previous
        process becomes UNKNOWN. It is reported, never re-executed.

        Returns the number of rows transitioned.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT action_id, owner FROM action_executions WHERE state = ?
                """,
                (ACTION_STATE_RUNNING,),
            ).fetchall()
            if not rows:
                return 0
            self._conn.execute(
                """
                UPDATE action_executions
                   SET state = ?, finished_at = ?
                 WHERE state = ?
                """,
                (ACTION_STATE_UNKNOWN, _utcnow(), ACTION_STATE_RUNNING),
            )
            self._conn.commit()
        for row in rows:
            log.warning(
                "action_recovered_as_unknown",
                action_id=row["action_id"],
                previous_owner=row["owner"],
            )
        return len(rows)

    def cleanup_old_action_executions(self, max_age_days: int = 30) -> int:
        """Delete terminal ledger rows older than max_age_days (bounded growth)."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM action_executions
                WHERE state IN ('SUCCEEDED', 'FAILED', 'UNKNOWN')
                  AND created_at < ?
                """,
                (cutoff,),
            )
            self._conn.commit()
            removed = cursor.rowcount
        if removed:
            log.info("cleanup_old_action_executions", removed=removed)
        return removed

    @staticmethod
    def _action_row_to_dataclass(row: sqlite3.Row) -> ActionExecution:
        return ActionExecution(
            action_id=row["action_id"],
            session_id=row["session_id"],
            confirmation_id=row["confirmation_id"],
            tool_name=row["tool_name"],
            tool_args=row["tool_args"],
            risk_level=row["risk_level"],
            state=row["state"],
            attempt=row["attempt"],
            result=row["result"],
            created_at=row["created_at"],
            claimed_at=row["claimed_at"],
            finished_at=row["finished_at"],
            owner=row["owner"],
            pause_context_json=(
                row["pause_context_json"]
                if "pause_context_json" in row.keys() else None
            ),
        )

    # ── Session leases (v0.17) ────────────────────────────────────────────────

    def acquire_session_lease(
        self,
        session_id: str,
        owner_token: str,
        ttl_seconds: int = SESSION_LEASE_TTL_SECONDS,
    ) -> tuple[bool, int]:
        """
        Try to acquire the per-session turn lease (database-backed).

        Returns:
            (acquired, fencing) — ``fencing`` is a monotonically increasing
            token that changes on every ownership change; a reentrant acquire
            by the SAME owner refreshes the TTL and keeps its fencing value.

        A lease held by a dead process is recoverable once its TTL elapses —
        there is no indefinite lock.
        """
        now = _utcnow()
        expires_at = (
            datetime.now(tz=timezone.utc) + timedelta(seconds=ttl_seconds)
        ).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT owner_token, expires_at, fencing FROM session_leases WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                active = _parse_ts(row["expires_at"]) > datetime.now(tz=timezone.utc)
                if active and row["owner_token"] != owner_token:
                    log.info(
                        "session_lease_blocked",
                        session_id=session_id,
                        holder=row["owner_token"],
                    )
                    return (False, int(row["fencing"]))
                if active:  # same owner: reentrant refresh
                    self._conn.execute(
                        "UPDATE session_leases SET expires_at = ? WHERE session_id = ? AND owner_token = ?",
                        (expires_at, session_id, owner_token),
                    )
                    self._conn.commit()
                    return (True, int(row["fencing"]))
                # Expired lease: take over with a NEW fencing value so any
                # stale owner still operating can detect the loss.
                new_fencing = int(row["fencing"]) + 1
                self._conn.execute(
                    """
                    UPDATE session_leases
                       SET owner_token = ?, acquired_at = ?, expires_at = ?, fencing = ?
                     WHERE session_id = ?
                    """,
                    (owner_token, now, expires_at, new_fencing, session_id),
                )
                self._conn.commit()
                log.warning(
                    "session_lease_recovered_from_stale",
                    session_id=session_id,
                    previous_owner=row["owner_token"],
                    fencing=new_fencing,
                )
                return (True, new_fencing)
            self._conn.execute(
                """
                INSERT INTO session_leases
                    (session_id, owner_token, acquired_at, expires_at, fencing)
                VALUES (?, ?, ?, ?, 1)
                """,
                (session_id, owner_token, now, expires_at),
            )
            self._conn.commit()
        log.debug("session_lease_acquired", session_id=session_id, owner=owner_token)
        return (True, 1)

    def renew_session_lease(self, session_id: str, owner_token: str, ttl_seconds: int = SESSION_LEASE_TTL_SECONDS) -> bool:
        """Extend the lease if (and only if) still owned; False means lost."""
        expires_at = (
            datetime.now(tz=timezone.utc) + timedelta(seconds=ttl_seconds)
        ).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE session_leases SET expires_at = ?
                 WHERE session_id = ? AND owner_token = ?
                """,
                (expires_at, session_id, owner_token),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def release_session_lease(self, session_id: str, owner_token: str) -> bool:
        """Release the lease if owned by ``owner_token`` (only ever our own)."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM session_leases WHERE session_id = ? AND owner_token = ?",
                (session_id, owner_token),
            )
            self._conn.commit()
            released = cursor.rowcount == 1
        if released:
            log.debug("session_lease_released", session_id=session_id)
        return released

    def get_session_lease(self, session_id: str) -> dict[str, Any] | None:
        """Lease row for introspection/tests: owner, expiry, fencing."""
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, owner_token, acquired_at, expires_at, fencing FROM session_leases WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    # ── Lease introspection (v0.18, read-only) ────────────────────────────

    def list_session_leases(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        # Bounded, deterministic read-only view of session leases, oldest
        # acquisition first. ``active`` is computed here so every consumer
        # agrees on staleness. (The API layer redacts owner tokens.)
        limit = max(1, min(int(limit), _MAX_ACTION_LIST_LIMIT))
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, owner_token, acquired_at, expires_at, fencing"
                " FROM session_leases ORDER BY acquired_at, session_id LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["active"] = _parse_ts(r["expires_at"]) > now
            result.append(d)
        return result

    def count_session_leases(self) -> dict[str, int]:
        """{'active': n, 'expired': m} in one bounded aggregate query."""
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            rows = self._conn.execute(
                "SELECT owner_token, expires_at, fencing FROM session_leases"
            ).fetchall()
        active = sum(1 for r in rows if _parse_ts(r["expires_at"]) > now)
        return {"active": active, "expired": len(rows) - active}

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
        # Multi-process coordination (session leases, durable rate limiting)
        # relies on SQLite file locking: wait briefly for a competing writer
        # instead of failing instantly with "database is locked".
        conn.execute("PRAGMA busy_timeout=5000")
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

        -- v0.17: durable ledger for protected (confirmation-gated) actions.
        -- One row per parked action; the approval path claims it at-most-once.
        CREATE TABLE IF NOT EXISTS action_executions (
            action_id       TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            confirmation_id TEXT NOT NULL,
            tool_name       TEXT NOT NULL,
            tool_args       TEXT NOT NULL,
            risk_level      TEXT NOT NULL,
            state           TEXT NOT NULL,
            attempt         INTEGER NOT NULL DEFAULT 0,
            result          TEXT,
            created_at      TEXT NOT NULL,
            claimed_at      TEXT,
            finished_at     TEXT,
            owner           TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_action_exec_session
            ON action_executions (session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_action_exec_confirmation
            ON action_executions (confirmation_id);

        -- v0.17: database-backed per-session turn leases (multi-process
        -- coordination). fencing increments on every ownership change.
        CREATE TABLE IF NOT EXISTS session_leases (
            session_id  TEXT PRIMARY KEY,
            owner_token TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            fencing     INTEGER NOT NULL DEFAULT 1
        );

        -- v0.18: explicit UNKNOWN-action reissue audit trail. One row per
        -- deliberate operator/user reissue; ``request_id`` is the caller's
        -- idempotency key, so the same request twice returns the same new
        -- action instead of creating two.
        CREATE TABLE IF NOT EXISTS action_reissues (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id        TEXT NOT NULL,
            original_action_id TEXT NOT NULL REFERENCES action_executions(action_id),
            new_action_id     TEXT NOT NULL REFERENCES action_executions(action_id),
            created_at        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_action_reissues_original
            ON action_reissues (original_action_id, created_at);
        -- Idempotency: one reissue per (request_id, original action).
        CREATE UNIQUE INDEX IF NOT EXISTS idx_action_reissues_idempotency
            ON action_reissues (request_id, original_action_id);

        -- v0.20: personal knowledge base — one durable row per ingested
        -- document (chunks + embeddings live in the dedicated Chroma
        -- ``knowledge_base`` collection; this table is the identity/registry
        -- layer that makes incremental ingestion and removal safe).
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            document_id     TEXT PRIMARY KEY,
            source_path     TEXT NOT NULL,
            filename        TEXT NOT NULL,
            media_type      TEXT NOT NULL,
            size_bytes      INTEGER NOT NULL,
            content_hash    TEXT NOT NULL,
            chunk_count     INTEGER NOT NULL DEFAULT 0,
            parser_version  TEXT NOT NULL,
            ingested_at     TEXT NOT NULL,
            modified_at     TEXT
        );
    """)
    # Lightweight migrations for pre-v0.15 / pre-v0.17 databases —
    # older installations lack these columns; existing data is preserved.
    existing_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(pending_confirmations)").fetchall()
    }
    if "context_json" not in existing_cols:
        conn.execute("ALTER TABLE pending_confirmations ADD COLUMN context_json TEXT")
    if "confirmation_id" not in existing_cols:
        conn.execute("ALTER TABLE pending_confirmations ADD COLUMN confirmation_id TEXT")        # Backfill: legacy rows get a durable id so v0.17 code paths (and a
        # paired ledger row) exist for already-parked actions too.
        conn.execute(
            """
            UPDATE pending_confirmations
               SET confirmation_id = hex(randomblob(16))
             WHERE confirmation_id IS NULL
            """
        )
        conn.execute(
            """
            INSERT INTO action_executions
                (action_id, session_id, confirmation_id, tool_name, tool_args,
                 risk_level, state, attempt, result, created_at)
            SELECT hex(randomblob(16)), session_id, confirmation_id, tool_name,
                   tool_args, risk_level,
                   CASE WHEN completed_at IS NULL THEN 'PENDING' ELSE 'UNKNOWN' END,
                   0, NULL, created_at
            FROM pending_confirmations
            WHERE confirmation_id IS NOT NULL
            """
        )
    # v0.19: durable resume context on the ledger row itself. Captured at
    # park time (same transaction), it survives the confirmation pop and any
    # crash window, so an explicit UNKNOWN reissue can restore the original
    # paused task. NULL on legacy rows / fresh parks without a plan.
    ledger_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(action_executions)").fetchall()
    }
    if "pause_context_json" not in ledger_cols:
        conn.execute(
            "ALTER TABLE action_executions ADD COLUMN pause_context_json TEXT"
        )
    conn.commit()


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime:
    """
    Parse an ISO-8601 timestamp defensively: rows written by hand-edited or
    pre-v0.15 databases may carry naive timestamps — treat those as UTC so
    comparisons never raise.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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
