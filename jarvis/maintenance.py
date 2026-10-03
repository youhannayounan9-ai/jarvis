"""
jarvis/maintenance.py
─────────────────────
Operational maintenance CLI for a deployed JARVIS service.

The API server intentionally runs no background jobs; operators schedule
this instead (cron / systemd timer / CI):

    uv run python -m jarvis.maintenance doctor
    uv run python -m jarvis.maintenance actions --state UNKNOWN
    uv run python -m jarvis.maintenance unknown-actions
    uv run python -m jarvis.maintenance sessions --expired
    uv run python -m jarvis.maintenance cleanup --days 30
    uv run python -m jarvis.maintenance stats
    uv run python -m jarvis.maintenance cache stats
    uv run python -m jarvis.maintenance cache inspect
    uv run python -m jarvis.maintenance cache cleanup [--expire-older-than-days N]
    uv run python -m jarvis.maintenance expire-confirmations --minutes 60

v0.18: ``actions`` / ``unknown-actions`` / ``sessions`` are strictly
read-only introspection. The ONE mutating command is ``reissue``, which
deliberately re-issues an UNKNOWN action as a NEW action id (with an
idempotency key and an interactive confirmation; the original row stays
UNKNOWN for audit). Nothing here automatically retries actions.

Exit code is 0 on success, 1 on failure — suitable for alerting.
"""

from __future__ import annotations

import argparse
import json as _json
import shutil
import sys
from datetime import datetime, timezone

from jarvis.config import settings
from jarvis.memory.session_store import (
    MAX_REISSUES_PER_ACTION,
    SessionStore,
    redact_owner,
)
from jarvis.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def _maintenance_parse_ts(value: str) -> float:
    """Parse a store timestamp for display; failures return 0 (never raise)."""
    try:
        from jarvis.memory.session_store import _parse_ts

        return _parse_ts(value).timestamp()
    except Exception:
        return 0.0


# One-shot flag so the remove-guard can honor --yes without threading a
# parameter through every cmd_ signature.
_REMOVE_YES: dict[str, bool] = {"skip": False}

# States the operator can filter the action report by.
_ACTION_STATES = ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN")


def _check_ollama_model(base_url: str, model: str, timeout: float = 3.0) -> bool:
    """True when ``model`` is installed on the reachable Ollama endpoint."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags", timeout=timeout
        ) as resp:
            import json as _parse

            tags = _parse.loads(resp.read().decode("utf-8", errors="replace"))
        return any(m.get("name", model) == model or m.get("name", "").startswith(f"{model}:") for m in tags.get("models", []))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _open_store() -> SessionStore:
    return SessionStore()


def cmd_cleanup(
    store: SessionStore,
    days: int,
    operational: bool = False,
    terminal_actions_days: int = 30,
    reissues_days: int = 90,
    leases_days: int = 30,
    dry_run: bool = False,
) -> int:
    """Session cleanup, plus v0.19 bounded operational-record retention.

    ``--operational`` enables the retention pass over the execution ledger,
    the reissue audit table, and stale leases. It NEVER touches PENDING,
    RUNNING or UNKNOWN actions, active confirmations, or any audit row whose
    original/new action still exists. ``--dry-run`` reports candidates
    without deleting anything (verified by tests).
    """
    if operational:
        report = store.cleanup_operational_records(
            terminal_actions_days=terminal_actions_days,
            reissues_days=reissues_days,
            leases_days=leases_days,
            dry_run=dry_run,
        )
        scope = "WOULD REMOVE (dry run)" if dry_run else "Removed"
        print(
            f"{scope}: {report['actions_removed']} terminal action(s) "
            f"(older than {terminal_actions_days}d; PENDING/RUNNING/UNKNOWN "
            "always protected)"
        )
        print(
            f"{scope}: {report['reissues_removed']} reissue audit row(s) "
            f"(older than {reissues_days}d; kept while either linked action "
            "exists)"
        )
        print(
            f"{scope}: {report['stale_leases_removed']} stale/orphaned "
            f"session lease(s) (expired {leases_days}d ago or dead session)"
        )
        if dry_run:
            print("Dry run: nothing was deleted.")
        return 0
    if dry_run:
        print("NOTE: --dry-run applies to --operational; session cleanup ran normally.")
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

    # ── Configured model availability (v0.18) ────────────────────────────
    if ollama_ok:
        model_ok = _check_ollama_model(s.ollama_base_url, s.ollama_model)
        if not model_ok:
            print(
                f"[FAIL] ollama     model '{s.ollama_model}' not installed "
                f"(run: ollama pull {s.ollama_model})"
            )
            failures += 1
        else:
            print(f"[OK] ollama     model '{s.ollama_model}' available")
    else:
        print("[WARN] ollama     model check skipped (endpoint unreachable)")

    # ── Reliability state: stale leases + UNKNOWN actions (v0.18) ─────────
    lease_counts = store.count_session_leases()
    stale = lease_counts.get("expired", 0)
    if stale:
        failures += 1
    print(
        f"[{ 'OK' if not stale else 'FAIL' }] leases     "
        f"{lease_counts.get('active', 0)} active, {stale} stale "
        + (f"(inspect: python -m jarvis.maintenance sessions --expired)" if stale else "")
    )
    action_counts = store.count_action_executions_by_state()
    unknown = action_counts.get("UNKNOWN", 0)
    if unknown:
        failures += 1
    pending_actions = action_counts.get("PENDING", 0)
    print(
        f"[{ 'OK' if not unknown else 'FAIL' }] actions    "
        f"{unknown} UNKNOWN, {pending_actions} PENDING, "
        f"{action_counts.get('RUNNING', 0)} RUNNING"
        + (f" (inspect: python -m jarvis.maintenance unknown-actions)" if unknown else "")
    )

    # ── Sandbox posture (informational: disabled is the SAFE default) ──────
    if not s.ENABLE_CODE_EXECUTION:
        print("[OK] sandbox    disabled (safe default)")
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
                print(f"[OK] sandbox    docker-isolated, image={s.SANDBOX_IMAGE}")
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

    # ── Optional voice dependencies (informational) ─────────────────────
    if shutil.which("ffmpeg") is None:
        print("[WARN] voice      ffmpeg not on PATH (voice mode unavailable)")
    else:
        print("[OK] voice      ffmpeg present")

    log.info(
        "doctor_check",
        failures=failures,
        stale_leases=stale,
        unknown_actions=unknown,
    )

    print()
    if failures:
        print(f"doctor: {failures} check(s) FAILED.")
        return 1
    print("doctor: all checks passed.")
    return 0


def _age(created_at: str) -> str:
    """Human-readable age of an ISO timestamp (coarse, operator-friendly)."""
    from jarvis.memory.session_store import _parse_ts

    try:
        delta = datetime.now(tz=timezone.utc) - _parse_ts(created_at)
    except Exception:  # pragma: no cover - malformed timestamps degrade
        return "?"
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h {minutes % 60}m"
    return f"{delta.days}d"


def _print_action_row(a, verbose: bool = False) -> None:
    """One ledger row, safe fields only (never tool_args / result bodies)."""
    print(
        f"  action: {a.action_id}  state: {a.state:9s} tool: {a.tool_name} "
        f"risk: {a.risk_level} attempt: {a.attempt}"
    )
    print(
        f"    session: {a.session_id}  confirmation: {a.confirmation_id}"
    )
    print(
        f"    created: {a.created_at} ({_age(a.created_at)} ago)  "
        f"claimed: {a.claimed_at or '-'}  finished: {a.finished_at or '-'}"
    )
    if verbose:
        print(f"    owner: {redact_owner(a.owner)}")


def cmd_actions(
    store: SessionStore,
    state: str | None,
    session: str | None,
    limit: int,
    as_json: bool,
) -> int:
    """Bounded, read-only execution-ledger report."""
    try:
        rows = store.list_action_executions(state=state, session_id=session, limit=limit)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if as_json:
        print(
            _json.dumps(
                [
                    {
                        "action_id": r.action_id,
                        "session_id": r.session_id,
                        "confirmation_id": r.confirmation_id,
                        "tool_name": r.tool_name,
                        "risk_level": r.risk_level,
                        "state": r.state,
                        "attempt": r.attempt,
                        "created_at": r.created_at,
                        "claimed_at": r.claimed_at,
                        "finished_at": r.finished_at,
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        return 0
    title = f"ACTIONS (state={state})" if state else "ACTIONS (all states)"
    print(title)
    print("-" * len(title))
    if not rows:
        print("  (none)")
        return 0
    for r in rows:
        _print_action_row(r)
    counts = store.count_action_executions_by_state()
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"\n  showing {len(rows)} row(s); ledger totals: {summary}")
    return 0


def cmd_unknown_actions(store: SessionStore, limit: int, as_json: bool) -> int:
    """Focused UNKNOWN report: what is ambiguous, why, and what to do next."""
    rows = store.list_action_executions(state="UNKNOWN", limit=limit)
    if as_json:
        print(
            _json.dumps(
                [
                    {
                        "action_id": r.action_id,
                        "session_id": r.session_id,
                        "tool_name": r.tool_name,
                        "created_at": r.created_at,
                        "attempt": r.attempt,
                        "reissue_depth": store.count_reissues_for_action(r.action_id),
                        "max_reissues": MAX_REISSUES_PER_ACTION,
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        return 0
    print("UNKNOWN ACTIONS")
    print("---------------")
    if not rows:
        print("  (none — no ambiguous actions await resolution)")
        return 0
    for r in rows:
        chain = store.get_reissue_chain(r.action_id)
        rec = store.get_action_recovery_preview(r.action_id)
        print(
            f"  action: {r.action_id}  session: {r.session_id}  tool: {r.tool_name}"
        )
        print(
            f"    age: {_age(r.created_at)}  attempt: {r.attempt}  "
            f"last owner: {redact_owner(r.owner)}"
        )
        print(
            "    reason: outcome not durably recorded "
            "(crash/restart between dispatch and result) — the side effect "
            "may or may not have happened"
        )
        if rec.get("has_context") and rec.get("recoverable"):
            print(
                f"    recovery: original task context available — reissue "
                f"will continue step {rec.get('step_number')} "
                f"({rec.get('pending_steps')} step(s) pending after it)"
            )
            if rec.get("original_request"):
                print(f"      task: {rec['original_request'][:160]}")
        else:
            print(f"    recovery: {rec.get('recoverable_reason', '?')}")
        print(
            f"    reissues: {len(chain)}/{MAX_REISSUES_PER_ACTION} "
            + (f"-> {chain[-1]['new_action_id']}" if chain else "")
        )
    print(
        "\n  JARVIS will NOT re-run these automatically. To re-issue "
        "deliberately (NEW action id, normal permission flow):"
    )
    print(
        "    python -m jarvis.maintenance reissue --action <action_id> "
        "--request-id <unique-id>"
    )
    return 0


def cmd_sessions(store: SessionStore, expired_only: bool, limit: int, as_json: bool) -> int:
    """Session/lease report: holders (redacted), expiry, fencing, staleness."""
    leases = store.list_session_leases(limit=limit)
    if expired_only:
        leases = [l for l in leases if not l["active"]]
    if as_json:
        print(_json.dumps(leases, indent=2, default=str))
        return 0
    title = "STALE SESSION LEASES (expired)" if expired_only else "SESSION LEASES"
    print(title)
    print("-" * len(title))
    if not leases:
        print("  (none)" if not expired_only else "  (none — no stale leases)")
        return 0
    for l in leases:
        status = "ACTIVE" if l["active"] else "STALE (expired)"
        print(
            f"  session: {l['session_id']}  {status:16s} fencing: {l['fencing']}"
        )
        print(
            f"    owner: {redact_owner(l['owner_token'])}  "
            f"acquired: {l['acquired_at']}  expires: {l['expires_at']}"
        )
    counts = store.count_session_leases()
    print(f"\n  showing {len(leases)} lease(s); totals: {counts}")
    return 0


def cmd_reissue(store: SessionStore, action_id: str, request_id: str, assume_yes: bool) -> int:
    """Explicit, deliberate UNKNOWN-action reissue (mutating CLI operation)."""
    row = store.get_action_execution(action_id)
    if row is None:
        print(f"ERROR: unknown action_id: {action_id}", file=sys.stderr)
        return 1
    if row.state != "UNKNOWN":
        print(
            f"ERROR: action {action_id} is {row.state}, not UNKNOWN — "
            "reissue is only for ambiguous (UNKNOWN) actions.",
            file=sys.stderr,
        )
        return 1
    chain = store.get_reissue_chain(action_id)
    print(
        f"Re-issue UNKNOWN action {action_id} (tool: {row.tool_name}, "
        f"attempt: {row.attempt}, existing reissues: {len(chain)}/{MAX_REISSUES_PER_ACTION})?"
    )
    print(
        "  This creates a NEW action identity (the original stays UNKNOWN "
        "for audit) which will follow the normal confirmation flow."
    )
    if not assume_yes:
        answer = input("Type 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            print("Aborted (no changes made).")
            return 1
    try:
        new_id = store.request_action_reissue(action_id, request_id)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    reused = any(r["request_id"] == request_id for r in chain)
    print(f"Reissued: {action_id} -> {new_id} (state PENDING)")
    if reused:
        print("  (idempotent replay: this request_id had already been processed)")
    print(
        "  Next: resolve it through the normal flow (CLI /confirm, API "
        "POST /sessions/{id}/confirm, or dashboard) after any user approval."
    )
    return 0


def cmd_inspect(store: SessionStore, session: str | None, action: str | None,
                limit: int, as_json: bool) -> int:
    """Read-only causal timeline / action deep-inspection (v0.19)."""
    if action:
        row = store.get_action_execution(action)
        if row is None:
            print(f"ERROR: unknown action_id: {action}", file=sys.stderr)
            return 1
        payload = {
            "action": {
                "action_id": row.action_id,
                "session_id": row.session_id,
                "confirmation_id": row.confirmation_id,
                "tool_name": row.tool_name,
                "risk_level": row.risk_level,
                "state": row.state,
                "attempt": row.attempt,
                "created_at": row.created_at,
                "claimed_at": row.claimed_at,
                "finished_at": row.finished_at,
                "reissue_origin": store.get_reissue_origin(row.action_id),
                "reissue_chain": store.get_reissue_chain(row.action_id),
            },
            "recovery": store.get_action_recovery_preview(row.action_id),
        }
        if as_json:
            print(_json.dumps(payload, indent=2, default=str))
            return 0
        a = payload["action"]
        rec = payload["recovery"]
        print(f"ACTION {a['action_id']}")
        print("-" * (7 + len(a["action_id"])))
        print(
            f"  state: {a['state']}  tool: {a['tool_name']}  risk: {a['risk_level']} "
            f"attempt: {a['attempt']}"
        )
        print(f"  session: {a['session_id']}  confirmation: {a['confirmation_id']}")
        print(
            f"  created: {a['created_at']} ({_age(a['created_at'])} ago)  "
            f"finished: {a['finished_at'] or '-'}"
        )
        origin = a["reissue_origin"]
        print(f"  reissued from: {origin or '-'}")
        if a["reissue_chain"]:
            print("  reissue chain:")
            for c in a["reissue_chain"]:
                print(
                    f"    {c['request_id']} -> {c['new_action_id']} "
                    f"({c['created_at']})"
                )
        print(f"  recovery: {rec.get('recoverable_reason', '?')}")
        if rec.get("original_request"):
            print(f"  original task: {rec['original_request'][:200]}")
        if rec.get("has_context"):
            print(
                f"  plan: step {rec.get('step_number')}, "
                f"{rec.get('completed_steps')} completed, "
                f"{rec.get('pending_steps')} pending, "
                f"{rec.get('remaining_rounds')} rounds left"
            )
        return 0
    if not session:
        print("ERROR: provide --session SESSION_ID or --action ACTION_ID", file=sys.stderr)
        return 2
    events = store.get_session_timeline(session, limit=limit)
    if as_json:
        print(_json.dumps(events, indent=2, default=str))
        return 0
    title = f"TIMELINE (session={session})"
    print(title)
    print("-" * len(title))
    if not events:
        print("  (no events recorded for this session)")
        return 0
    for e in events:
        ts = str(e.get("ts", ""))[:19]
        kind = e.get("kind", "?")
        if kind == "message":
            detail = f"role={e.get('role')}"
        elif kind == "confirmation_parked":
            detail = f"tool={e.get('tool')} risk={e.get('risk_level')} resolved={e.get('resolved')}"
        elif kind == "action_state":
            detail = (
                f"action={e.get('action_id')} tool={e.get('tool')} "
                f"state={e.get('state')} attempt={e.get('attempt')}"
            )
        elif kind == "reissue":
            detail = (
                f"request={e.get('request_id')} "
                f"{str(e.get('original_action_id'))[:12]}.. -> "
                f"{str(e.get('new_action_id'))[:12]}.."
            )
        else:  # lease
            detail = (
                f"owner={e.get('owner')} fencing={e.get('fencing')} "
                f"active={e.get('active')}"
            )
        print(f"  {ts}  {kind:22s} {detail}")
    print(f"\n  {len(events)} event(s) (bounded at {limit})")
    return 0


def _knowledge_service(store: SessionStore):
    """Build the knowledge service bound to the CLI's store instance."""
    from jarvis.memory.knowledge import get_knowledge_service
    from jarvis.memory.vector_store import get_vector_store

    return get_knowledge_service(store=store, vector_store=get_vector_store())


def cmd_knowledge(
    store: SessionStore,
    action: str,
    target: str | None,
    query: str | None,
    limit: int,
    top_k: int,
    as_json: bool,
) -> int:
    """v0.20 personal knowledge base management (explicit + bounded)."""
    from jarvis.memory.knowledge import DEFAULT_TOP_K, MAX_TOP_K

    svc = _knowledge_service(store)

    if action == "list":
        docs = svc.list_documents(limit=limit)
        if as_json:
            print(_json.dumps(docs, indent=2, default=str))
            return 0
        print(f"KNOWLEDGE BASE ({len(docs)} document(s))")
        print("-" * 30)
        if not docs:
            print("  (empty — ingest with: knowledge ingest <path>)")
            return 0
        for d in docs:
            print(
                f"  {d['document_id'][:16]}…  {d['filename']}  "
                f"chunks: {d['live_chunk_count']}/{d['chunk_count']}  "
                f"{d['media_type']}  {d['size_bytes']}B"
            )
            print(
                f"    source: {d['source_path']}  ingested: {d['ingested_at'][:19]}"
            )
        return 0

    if action == "inspect":
        if not target:
            print("ERROR: knowledge inspect requires --id DOCUMENT_ID", file=sys.stderr)
            return 2
        doc = svc.inspect_document(target)
        if doc is None:
            print(f"ERROR: unknown document_id: {target}", file=sys.stderr)
            return 1
        if as_json:
            print(_json.dumps(doc, indent=2, default=str))
            return 0
        for k in ("document_id", "filename", "source_path", "media_type",
                  "size_bytes", "content_hash", "chunk_count",
                  "live_chunk_count", "parser_version", "ingested_at",
                  "modified_at"):
            print(f"  {k}: {doc.get(k)}")
        return 0

    if action == "ingest":
        if not target:
            print("ERROR: knowledge ingest requires --path PATH", file=sys.stderr)
            return 2
        report = svc.ingest(target)
        if as_json:
            print(_json.dumps(report, indent=2, default=str))
        else:
            if report["status"] == "error":
                print(f"ERROR: {report.get('reason')}", file=sys.stderr)
                return 1
            print(
                f"{report['status']}: {report['filename']} → "
                f"{report['document_id'][:16]}… ({report['chunk_count']} chunk(s))"
            )
            if report.get("same_content_as"):
                print(f"  (identical content already indexed from {report['same_content_as']})")
        return 0 if report["status"] != "error" else 1

    if action == "reindex":
        if not target:
            print("ERROR: knowledge reindex requires --id DOCUMENT_ID", file=sys.stderr)
            return 2
        report = svc.reindex_document(target)
        if as_json:
            print(_json.dumps(report, indent=2, default=str))
            return 0 if report.get("status") != "error" else 1
        if report.get("status") == "error":
            print(f"ERROR: {report.get('reason')}", file=sys.stderr)
            return 1
        ing = report.get("ingest") or {}
        print(f"reindexed: {target[:16]}… ({ing.get('status')}, {ing.get('chunk_count')} chunk(s))")
        return 0

    if action == "remove":
        if not target:
            print("ERROR: knowledge remove requires --id DOCUMENT_ID", file=sys.stderr)
            return 2
        doc = svc.inspect_document(target)
        if doc is None:
            print(f"ERROR: unknown document_id: {target}", file=sys.stderr)
            return 1
        if not assume_yes_removal(skip=_REMOVE_YES.get("skip", False)):
            return 1
        report = svc.remove_document(target)
        if as_json:
            print(_json.dumps(report, indent=2, default=str))
            return 0
        print(
            f"removed: {target[:16]}… ({report['chunks_removed']} chunk(s), "
            f"registry row: {'yes' if report['registry_row_removed'] else 'no'})"
        )
        return 0

    if action == "search":
        if not query:
            print("ERROR: knowledge search requires --query TEXT", file=sys.stderr)
            return 2
        report = svc.search(query, top_k=max(1, min(top_k, MAX_TOP_K)))
        if as_json:
            print(_json.dumps(report, indent=2, default=str))
            return 0
        results = report["results"]
        if not results:
            print("NO_RELEVANT_EVIDENCE: the knowledge base has no matching chunks.")
            return 0
        print(f"KNOWLEDGE SEARCH: {len(results)} hit(s) of {report['total_chunks']} chunk(s)")
        print("-" * 40)
        for i, r in enumerate(results, start=1):
            dist = r.get("distance")
            dist_s = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
            print(f"  [{i}] {r['citation']}  distance={dist_s}")
            print(f"      {r['snippet'][:180]}")
        return 0

    print(f"ERROR: unknown knowledge action: {action}", file=sys.stderr)
    return 2


def assume_yes_removal(skip: bool = False) -> bool:
    """Interactive guard for explicit document removal."""
    if skip:
        return True
    answer = input("Type 'yes' to remove this document from the knowledge base: ")
    if answer.strip().lower() != "yes":
        print("Aborted (nothing removed).")
        return False
    return True


def cmd_cache(
    store: SessionStore,
    action: str,
    *,
    limit: int = 25,
    expire_older_than_days: int | None = None,
    yes: bool = False,
) -> int:
    """v0.24 cross-turn result cache operations (Part H).

    ``stats``   — entries / hits / expired / per-tool counts (read-only).
    ``inspect`` — metadata-only listing (fingerprint, tool, scope, age,
                  hits). NEVER prints cached payloads (Part Q).
    ``cleanup`` — bounded deletion: expired rows only by default;
                  ``--expire-older-than-days N`` additionally expires and
                  removes rows older than N days (still bounded by --limit).
    """
    if action == "stats":
        stats = store.cache_stats()
        print(f"entries: {stats['entries']}  hits: {stats['hits']}  expired: {stats['expired']}")
        for row in stats["per_tool"]:
            print(f"  {row['tool_name']:<22} entries={row['entries']:<6} hits={row['hits']}")
        if stats["entries"] == 0:
            print("(result cache is empty)")
        # v0.25 (Part E4): today's daily aggregates — counts only, never
        # payloads or queries.
        today_rows = store.cache_metrics_history(days=1, limit=1)
        if today_rows:
            t = today_rows[0]
            print(
                f"today: hits={t['hits']} misses={t['misses']} "
                f"stale={t['stale']} bypass={t['bypass']} stores={t['stores']}"
            )
            for tool, deltas in sorted(t.get("per_tool", {}).items()):
                parts = ", ".join(f"{k}={v}" for k, v in sorted(deltas.items()))
                print(f"  {tool:<22} {parts}")
        else:
            print("today: no cache activity recorded")
        # v0.26 (Part 11): today's grounding-guard aggregates — counts only.
        g_rows = store.grounding_metrics_history(days=1, limit=1)
        if g_rows:
            g = g_rows[0]
            print(
                f"grounding today: checks={g['checks']} contradictions={g['contradictions']} "
                f"corrections={g['corrections']} ok={g['corrections_ok']} "
                f"failed={g['corrections_failed']} fallbacks={g['fallbacks']}"
            )
            for tool, deltas in sorted(g.get("per_tool", {}).items()):
                parts = ", ".join(f"{k}={v}" for k, v in sorted(deltas.items()))
                print(f"  {tool:<22} {parts}")
        else:
            print("grounding today: no activity recorded")
        return 0

    if action == "inspect":
        rows = store.list_cache_entries(limit=limit)
        now = datetime.now(timezone.utc)
        for r in rows:
            try:
                age_s = max(0.0, now.timestamp() - _maintenance_parse_ts(r["created_at"]))
                age = f"{int(age_s // 60)}m" if age_s < 86400 else f"{int(age_s // 86400)}d"
            except Exception:
                age = "?"
            exp = r.get("expires_at") or "—"
            print(
                f"{r['cache_key'][:16]}  {r['tool_name']:<22} scope={r['scope']:<7} "
                f"hits={r['hit_count']:<4} age={age:<5} expires={exp}"
            )
        if not rows:
            print("(no cache entries)")
        else:
            print(f"({len(rows)} entry(ies), payloads withheld — keys are fingerprints, not content)")
        return 0

    if action == "cleanup":
        removed_expired = store.cleanup_result_cache(expired_only=True, limit=500)
        print(f"Removed {removed_expired} expired cache row(s) (bounded: 500/invocation).")
        if expire_older_than_days is not None:
            if not yes:
                reply = input(
                    f"Also expire+remove cache rows older than {expire_older_than_days}d? [y/N] "
                )
                if reply.strip().lower() not in ("y", "yes"):
                    print("Aborted; only expired rows were removed.")
                    return 0
            cutoff = (
                datetime.now(timezone.utc).timestamp() - expire_older_than_days * 86400
            )
            cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
            with store._lock:
                cursor = store._conn.execute(
                    """
                    DELETE FROM result_cache WHERE cache_key IN (
                        SELECT cache_key FROM result_cache
                        WHERE created_at <= ? LIMIT 500
                    )
                    """,
                    (cutoff_iso,),
                )
                removed_old = cursor.rowcount or 0
                store._conn.commit()
            print(f"Removed {removed_old} row(s) older than {expire_older_than_days}d (bounded: 500/invocation).")
        stats = store.cache_stats()
        print(f"remaining entries: {stats['entries']}")
        return 0

    print(f"ERROR: unknown cache action '{action}'", file=sys.stderr)
    return 2


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


def cmd_integration_audit_cleanup(
    store: SessionStore,
    *,
    days: int | None = None,
    dry_run: bool = False,
    limit: int = 500,
    yes: bool = False,
) -> int:
    """
    v0.30 Part 21: conservative retention for integration_audit rows.

    Deletes rows older than the retention window EXCEPT state UNKNOWN /
    RUNNING rows — the manual-recovery record is protected unconditionally.
    Deletion is bounded per run; the default is report-only unless ``--yes``.
    """
    from jarvis.config import settings as _s

    retention = int(
        days if days is not None else getattr(_s, "INTEGRATION_AUDIT_RETENTION_DAYS", 90)
    )
    if dry_run or not yes:
        preview = store.cleanup_integration_audit(retention_days=retention, dry_run=True)
        print(f"integration audit retention: {retention} day(s)")
        print(
            f"  would delete: {preview['would_delete']}   "
            f"protected (UNKNOWN/RUNNING): {preview['protected']}"
        )
        print(
            "  dry run — nothing deleted." if dry_run
            else "  re-run with --yes to delete (bounded per run)."
        )
        return 0
    result = store.cleanup_integration_audit(
        retention_days=retention, dry_run=False, batch_limit=limit
    )
    print(
        f"integration audit retention: {retention} day(s); "
        f"deleted {result['deleted']} row(s); protected {result['protected']}"
    )
    if result["deleted"] >= max(1, int(limit)):
        print("  note: batch limit reached — run again to continue (bounded deletion).")
    log.info(
        "integration_audit_cleanup",
        deleted=result["deleted"],
        protected=result["protected"],
        retention_days=retention,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(
        prog="jarvis.maintenance",
        description="Operational maintenance for the JARVIS service",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cleanup = sub.add_parser(
        "cleanup",
        help="Delete sessions older than N days (add --operational for retention)",
    )
    p_cleanup.add_argument("--days", type=int, default=30)
    p_cleanup.add_argument(
        "--operational",
        action="store_true",
        help="Also apply v0.19 operational-record retention (ledger, reissue audit, stale leases)",
    )
    p_cleanup.add_argument(
        "--terminal-actions-days", type=int, default=30,
        help="Retention for terminal (SUCCEEDED/FAILED) ledger rows (default 30)",
    )
    p_cleanup.add_argument(
        "--reissues-days", type=int, default=90,
        help="Retention for reissue audit rows whose BOTH linked actions are gone (default 90)",
    )
    p_cleanup.add_argument(
        "--leases-days", type=int, default=30,
        help="Retention for expired/orphaned session leases (default 30)",
    )
    p_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="Report retention candidates without deleting anything",
    )

    p_inspect = sub.add_parser(
        "inspect",
        help="Read-only causal timeline for a session, or deep detail for one action",
    )
    p_inspect.add_argument("--session", default=None, help="Session id to inspect")
    p_inspect.add_argument("--action", default=None, help="Action id to inspect")
    p_inspect.add_argument("--limit", type=int, default=200)
    p_inspect.add_argument("--json", action="store_true", help="Machine-readable output")

    sub.add_parser(
        "expire-confirmations",
        help="Purge expired/completed pending-confirmation rows",
    )

    sub.add_parser("stats", help="Print session/message counts")
    sub.add_parser(
        "doctor",
        help="Pre-flight checks: DB, Ollama + model, stale leases, UNKNOWN actions, sandbox, voice deps",
    )

    p_actions = sub.add_parser("actions", help="Read-only execution-ledger report")
    p_actions.add_argument(
        "--state",
        choices=_ACTION_STATES,
        default=None,
        help="Filter by ledger state",
    )
    p_actions.add_argument("--session", default=None, help="Filter by session id")
    p_actions.add_argument("--limit", type=int, default=50)
    p_actions.add_argument("--json", action="store_true", help="Machine-readable output")

    p_unknown = sub.add_parser(
        "unknown-actions", help="Report actions whose outcome is ambiguous (UNKNOWN)"
    )
    p_unknown.add_argument("--limit", type=int, default=50)
    p_unknown.add_argument("--json", action="store_true", help="Machine-readable output")

    p_sessions = sub.add_parser(
        "sessions", help="Session-lease report (holders redacted, expiry, fencing)"
    )
    p_sessions.add_argument(
        "--expired", action="store_true", help="Show only stale (expired) leases"
    )
    p_sessions.add_argument("--limit", type=int, default=100)
    p_sessions.add_argument("--json", action="store_true", help="Machine-readable output")

    p_reissue = sub.add_parser(
        "reissue",
        help="Explicitly re-issue an UNKNOWN action as a NEW action id (mutating; confirm prompt)",
    )
    p_reissue.add_argument("--action", required=True, help="The UNKNOWN action id")
    p_reissue.add_argument(
        "--request-id",
        required=True,
        help="Idempotency key: the same id twice returns the same new action",
    )
    p_reissue.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation"
    )

    p_knowledge = sub.add_parser(
        "knowledge",
        help="Personal knowledge base: list/inspect/ingest/reindex/remove/search",
    )
    p_knowledge.add_argument(
        "action",
        choices=["list", "inspect", "ingest", "reindex", "remove", "search"],
    )
    p_knowledge.add_argument("--id", dest="target_id", default=None,
                             help="Document id (inspect/reindex/remove)")
    p_knowledge.add_argument("--path", dest="target_path", default=None,
                             help="Explicit file path (ingest; no crawling)")
    p_knowledge.add_argument("--query", default=None, help="Search query")
    p_knowledge.add_argument("--top-k", type=int, default=4)
    p_knowledge.add_argument("--limit", type=int, default=100)
    p_knowledge.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation (remove)",
    )
    p_knowledge.add_argument("--json", action="store_true", help="Machine-readable output")

    p_integration_audit = sub.add_parser(
        "integrations-audit",
        help=(
            "v0.30 conservative retention for integration_audit rows "
            "(UNKNOWN/RUNNING always protected)"
        ),
    )
    p_integration_audit.add_argument(
        "--days", type=int, default=None,
        help="Retention window in days (default: INTEGRATION_AUDIT_RETENTION_DAYS)",
    )
    p_integration_audit.add_argument(
        "--dry-run", action="store_true", help="Report only (default without --yes)"
    )
    p_integration_audit.add_argument(
        "--limit", type=int, default=500, help="Max rows deleted per run (default 500)"
    )
    p_integration_audit.add_argument(
        "--yes", action="store_true", help="Perform the bounded deletion"
    )

    p_cache = sub.add_parser(
        "cache",
        help="v0.24 cross-turn result cache: stats / inspect / cleanup (bounded)",
    )
    p_cache.add_argument(
        "action", choices=["stats", "inspect", "cleanup"]
    )
    p_cache.add_argument("--limit", type=int, default=25, help="Rows shown by inspect (default 25)")
    p_cache.add_argument(
        "--expire-older-than-days", type=int, default=None,
        help="cleanup: ALSO expire+remove rows older than N days (bounded, confirm prompt)",
    )
    p_cache.add_argument("--yes", action="store_true", help="Skip the cleanup confirm prompt")

    args = parser.parse_args(argv)

    store = None
    try:
        # Opened inside try: a locked/missing DB is an operational error that
        # must map to exit code 1, not a traceback.
        store = _open_store()
        if args.command == "cleanup":
            return cmd_cleanup(
                store,
                args.days,
                operational=args.operational,
                terminal_actions_days=args.terminal_actions_days,
                reissues_days=args.reissues_days,
                leases_days=args.leases_days,
                dry_run=args.dry_run,
            )
        if args.command == "inspect":
            return cmd_inspect(
                store, args.session, args.action, args.limit, args.json
            )
        if args.command == "knowledge":
            target = args.target_id or args.target_path
            if args.action == "ingest" and not args.target_path:
                print("ERROR: knowledge ingest requires --path PATH", file=sys.stderr)
                return 2
            if args.action in ("inspect", "reindex", "remove") and not args.target_id:
                print(f"ERROR: knowledge {args.action} requires --id DOCUMENT_ID", file=sys.stderr)
                return 2
            _REMOVE_YES["skip"] = bool(args.yes)
            return cmd_knowledge(
                store,
                args.action,
                target,
                args.query,
                args.limit,
                args.top_k,
                args.json or False,
            )
        if args.command == "expire-confirmations":
            return cmd_expire_confirmations(store)
        if args.command == "cache":
            return cmd_cache(
                store,
                args.action,
                limit=args.limit,
                expire_older_than_days=args.expire_older_than_days,
                yes=args.yes,
            )
        if args.command == "integrations-audit":
            return cmd_integration_audit_cleanup(
                store,
                days=args.days,
                dry_run=args.dry_run,
                limit=args.limit,
                yes=args.yes,
            )
        if args.command == "stats":
            return cmd_stats(store)
        if args.command == "doctor":
            return cmd_doctor(store)
        if args.command == "actions":
            return cmd_actions(store, args.state, args.session, args.limit, args.json)
        if args.command == "unknown-actions":
            return cmd_unknown_actions(store, args.limit, args.json)
        if args.command == "sessions":
            return cmd_sessions(store, args.expired, args.limit, args.json)
        if args.command == "reissue":
            return cmd_reissue(store, args.action, args.request_id, args.yes)
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
