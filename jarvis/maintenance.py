"""
jarvis/maintenance.py
─────────────────────
Operational maintenance CLI for a deployed JARVIS service.

The API server intentionally runs no background jobs; operators schedule
this instead (cron / systemd timer / CI):

    uv run python -m jarvis.maintenance cleanup --days 30
    uv run python -m jarvis.maintenance stats
    uv run python -m jarvis.maintenance expire-confirmations --minutes 60

Exit code is 0 on success, 1 on failure — suitable for alerting.
"""

from __future__ import annotations

import argparse
import sys

from jarvis.config import settings
from jarvis.memory.session_store import SessionStore
from jarvis.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def _open_store() -> SessionStore:
    return SessionStore()


def cmd_cleanup(store: SessionStore, days: int) -> int:
    removed = store.cleanup_old_sessions(max_age_days=days)
    print(f"Removed {removed} session(s) older than {days} day(s).")
    return 0


def cmd_expire_confirmations(store: SessionStore) -> int:
    # TTL is enforced at save time (CONFIRMATION_TTL_MINUTES); this purges
    # already-expired and completed rows from the table.
    expired = store.cleanup_expired_confirmations()
    print(f"Purged {expired} expired/completed confirmation row(s).")
    return 0


def cmd_doctor(store: SessionStore) -> int:
    """
    Pre-flight / triage probe for a deployed instance: DB read+write,
    Ollama reachability, sandbox posture, and pending-confirmation state.
    Fail-closed: any exception is reported as a failed check, never raised.
    """
    from jarvis.api.health import check_ollama, deep_health
    from jarvis.config import settings as s

    failures = 0

    # ── DB read + write round-trip ──────────────────────────────────────
    probe = deep_health(type("StoreHolder", (), {'store': store})())
    ok = probe["status"] == "ok"
    failures += 0 if ok else 1
    db = probe["db"]
    print(f"[{ 'OK' if ok else 'FAIL' }] database   read={db['read']} write={db['write']} path={s.db_path}")

    # ── Ollama reachability (warn only: chats fail but the service lives) ──
    ollama_ok = check_ollama(s.ollama_base_url)
    if not ollama_ok:
        failures += 1
    print(
        f"[{ 'OK' if ollama_ok else 'FAIL' }] ollama     {s.ollama_base_url} "
        f"(model={s.ollama_model})"
    )

    # ── Sandbox posture (informational: disabled is the SAFE default) ──────
    if not s.ENABLE_CODE_EXECUTION:
        print("[ OK ] sandbox    disabled (safe default)")
    else:
        from jarvis.core.sandbox import DockerCodeSandbox

        try:
            sandbox = DockerCodeSandbox(image=s.SANDBOX_IMAGE)
            available = sandbox.is_available()
        except ValueError as e:
            print(f"[FAIL] sandbox    invalid image ref: {e}")
            failures += 1
        else:
            if available:
                print(f"[ OK ] sandbox    docker-isolated, image={s.SANDBOX_IMAGE}")
            else:
                print(
                    f"[FAIL] sandbox    enabled but unavailable "
                    f"(image={s.SANDBOX_IMAGE}) - tool will stay unregistered"
                )
                failures += 1

    # ── Pending confirmations ───────────────────────────────────────────
    with store._lock:
        pending = store._conn.execute(
            "SELECT COUNT(*) FROM pending_confirmations WHERE completed_at IS NULL"
        ).fetchone()[0]
    print(f"[INFO] state      {pending} pending confirmation(s) awaiting resolution")

    print()
    if failures:
        print(f"doctor: {failures} check(s) FAILED.")
        return 1
    print("doctor: all checks passed.")
    return 0


def cmd_stats(store: SessionStore) -> int:
    with store._lock:
        conn = store._conn
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM pending_confirmations WHERE completed_at IS NULL"
        ).fetchone()[0]
    print(f"db_path:            {settings.db_path}")
    print(f"sessions:           {sessions}")
    print(f"messages:           {messages}")
    print(f"pending confirmations: {pending}")
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(
        prog="jarvis.maintenance",
        description="Operational maintenance for the JARVIS service",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cleanup = sub.add_parser("cleanup", help="Delete sessions older than N days")
    p_cleanup.add_argument("--days", type=int, default=30)

    sub.add_parser(
        "expire-confirmations",
        help="Purge expired/completed pending-confirmation rows",
    )

    sub.add_parser("stats", help="Print session/message counts")
    sub.add_parser(
        "doctor",
        help="Pre-flight checks: DB write, Ollama reachability, sandbox posture",
    )

    args = parser.parse_args(argv)

    store = None
    try:
        # Opened inside try: a locked/missing DB is an operational error that
        # must map to exit code 1, not a traceback.
        store = _open_store()
        if args.command == "cleanup":
            return cmd_cleanup(store, args.days)
        if args.command == "expire-confirmations":
            return cmd_expire_confirmations(store)
        if args.command == "stats":
            return cmd_stats(store)
        if args.command == "doctor":
            return cmd_doctor(store)
        parser.error(f"unknown command: {args.command}")
        return 2
    except Exception as e:
        log.error("maintenance_failed", command=args.command, error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
