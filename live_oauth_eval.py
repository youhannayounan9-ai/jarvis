"""
live_oauth_eval.py
──────────────────
v0.30 CONTROLLED LIVE VALIDATION of real OAuth authorization + token
lifecycle — MANUAL-ONLY, deterministic, and safe by construction.

It needs NO Ollama and NO external network and NO real account: it drives the
REAL authorization machinery (durable one-time state, PKCE S256, code
exchange, refresh-token rotation, revocation, cached reads) against the
bundled SIMULATED provider, i.e. exactly the "controlled development/test
account" the spec allows. The only thing it does not exercise is the model's
own tool choice — that path is measured by `live_provider_eval.py`.

Requires ENABLE_INTEGRATIONS=true (the surface is opt-in).

Setup:
    set ENABLE_INTEGRATIONS=true
    uv run python live_oauth_eval.py --json live_oauth_report_v030.json

The 14 spec steps, reported with an explicit PHASE each so the four
success layers stay distinguishable:

    oauth_flow          — authorization/state/token lifecycle
    provider_api        — the (simulated) provider's own API surface
    integration_runtime — the JARVIS tool/manager/permission path
    grounding           — the v0.26/v0.30 answer-integrity guard

  1 start authorization          8 calendar read/write
  2 user grants requested scope  9 task read/write
  3 callback succeeds           10 confirmation required for writes
  4 account identity discovered 11 read-back verification
  5 credential stored           12 disconnect (provider revocation)
  6 provider read succeeds      13 post-disconnect action rejected
  7 access token refresh        14 reauthorization restores access

Exit code 0 iff every step passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

_REPO_ROOT = "/".join(os.path.abspath(__file__).replace("\\", "/").split("/")[:-1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ISOLATION (v0.25 Part G contract): private temp DB BEFORE any jarvis import.
from evaluation import _bootstrap as _eval  # noqa: E402

_eval.isolate()


# ── helpers ───────────────────────────────────────────────────────────────────


def _ms(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return round(median * 1000.0, 4)


def _time_it(reps: int, fn) -> tuple[float, Any]:
    samples: list[float] = []
    result: Any = None
    for _ in range(max(1, reps)):
        started = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - started)
    return _ms(samples), result


def _first_id(text: str, kind: str) -> str:
    match = re.search(rf"^{kind}:\s*(\S+)", text, re.MULTILINE)
    return match.group(1) if match else ""


def _render_time(text: str) -> str:
    match = re.search(r"^time:\s*(\d{2}:\d{2})\s*$", text, re.MULTILINE)
    return match.group(1) if match else ""


# ── the run ───────────────────────────────────────────────────────────────────


def _run() -> int:  # noqa: C901 - one linear, explicit scenario
    from jarvis.config import settings as _s

    if not bool(_s.ENABLE_INTEGRATIONS):
        print(
            "ERROR: run with ENABLE_INTEGRATIONS=true "
            "(local simulated provider; no external network).",
            file=sys.stderr,
        )
        return 2

    from jarvis.core.grounding import check_grounding
    from jarvis.core.orchestrator import PAUSED_FOR_CONFIRMATION, Orchestrator
    from jarvis.core.permissions import PermissionGuard
    from jarvis.core.result_cache import ResultCache
    from jarvis.integrations.base import Operation
    from jarvis.integrations.manager import IntegrationManager
    from jarvis.integrations.oauth import (
        local_authorization_server,
        local_simulate_consent,
    )
    from jarvis.integrations.providers import LocalCalendarProvider, LocalTasksProvider
    from jarvis.integrations.scopes import (
        CALENDAR_DELETE,
        CALENDAR_READ,
        CALENDAR_WRITE,
        TASKS_READ,
        TASKS_WRITE,
    )
    from jarvis.tools.integration_tools import build_integration_tools
    from jarvis.tools.registry import ToolRegistry

    parser = argparse.ArgumentParser(prog="live_oauth_eval")
    parser.add_argument("--json", dest="json_out", default=None)
    args, _rest = parser.parse_known_args()

    steps: list[dict[str, Any]] = []
    perf: dict[str, float] = {}

    def step(name: str, phase: str, ok: bool, detail: str = "") -> bool:
        steps.append(
            {"name": name, "phase": phase, "ok": bool(ok), "detail": str(detail)[:300]}
        )
        print(f"  [{'ok ' if ok else 'FAIL'}] {phase:19s} {name} {detail}"[:160])
        return bool(ok)

    store = _eval.fresh_store()
    manager = IntegrationManager(store)
    manager.register_provider(LocalCalendarProvider())
    manager.register_provider(LocalTasksProvider())
    server = local_authorization_server()
    session = "live-oauth-eval"
    full_calendar = {CALENDAR_READ, CALENDAR_WRITE, CALENDAR_DELETE}

    tools = {t.name: t for t in build_integration_tools(manager)}

    print("v0.30 live OAuth validation (simulated provider; deterministic)")
    print("─" * 72)

    # 1. start authorization
    flow_ms, start = _time_it(
        20,
        lambda: manager.begin_authorization(
            provider="calendar",
            session_id=session,
            display_label="Eval Calendar",
            scopes=full_calendar,
        ),
    )
    perf["authorization_start_ms"] = flow_ms
    url = start.get("authorization_url", "")
    step(
        "1_start_authorization",
        "oauth_flow",
        "state=" in url and "code_challenge=" in url and "code_challenge_method=S256" in url,
        f"url_len={len(url)}",
    )

    # 2. user grants the requested scope (simulated consent screen)
    consent = local_simulate_consent(url, subject="eval-user@example.com")
    step("2_user_grants_scope", "oauth_flow", consent.granted, f"code_len={len(consent.code)}")

    # 3. callback succeeds (state consumed; code exchanged; PKCE verified)
    # reps=1: the one-time state is CONSUMED by the first call (a repeat would
    # be — correctly — a replay refusal, so it is not a repeatable operation).
    callback_ms, account = _time_it(
        1,
        lambda: manager.handle_callback(
            provider="calendar",
            code=consent.code,
            state=consent.state,
            session_id=session,
            redirect_uri=consent.redirect_uri,
        ),
    )
    perf["authorization_callback_ms"] = callback_ms
    step(
        "3_callback_succeeds",
        "oauth_flow",
        account.authorization_status == "AUTHORIZED"
        and account.auth_state.value == "AUTHENTICATED",
        f"account={account.account_id} status={account.authorization_status}",
    )

    # 4. account identity discovered from the provider (not from chat text)
    step(
        "4_account_identity_discovered",
        "oauth_flow",
        account.provider_account_ref == "eval-user@example.com",
        f"ref={account.provider_account_ref}",
    )

    # 5. credential stored inside the boundary (obfuscated; never plaintext)
    row = store.get_integration_account(account.account_id) or {}
    step(
        "5_credential_stored",
        "oauth_flow",
        bool(row.get("token_access_obfuscated"))
        and account._access_token not in json.dumps(row)
        and account._refresh_token not in json.dumps(row),
        "obfuscated at rest; no plaintext material",
    )

    # 6. provider read succeeds
    read_ms, events = _time_it(200, lambda: manager.execute_read(account, Operation.LIST))
    perf["provider_read_ms"] = read_ms
    step("6_provider_read_succeeds", "provider_api", isinstance(events, list), f"count={len(events)}")

    # 7. access token refresh (expired locally AND at the provider)
    with store._lock:
        store._conn.execute(
            "UPDATE integration_accounts SET token_expires_at = ? WHERE account_id = ?",
            ("2000-01-01T00:00:00+00:00", account.account_id),
        )
        store._conn.commit()
    server.advance_clock(4000)
    before = account._access_token
    # reps=1: refresh ROTATES the grant, so a repeat measures a different path.
    refresh_ms, refreshed = _time_it(
        1, lambda: manager.refresh_auth_state(manager.get_account(account.account_id))
    )
    perf["token_refresh_ms"] = refresh_ms
    step(
        "7_access_token_refresh",
        "oauth_flow",
        refreshed._access_token != before and refreshed.authorization_status == "AUTHORIZED",
        "rotated access+refresh tokens; account identity preserved",
    )

    # 8. calendar read/write (with read-back verification)
    # reps=1: the write is a single side effect, not a repeatable sample.
    write_ms, created = _time_it(
        1,
        lambda: tools["calendar_create_event"].run(
            account_id=account.account_id,
            title="Dentist",
            date="2026-10-05",
            start_time="15:30",
            timezone="Africa/Cairo",
            duration_minutes=45,
        ),
    )
    perf["provider_write_and_verify_ms"] = write_ms
    ok_write = "ACTION_EXECUTED" in created and "verification: VERIFIED" in created
    step("8_calendar_write", "integration_runtime", ok_write, "created + verified")
    step(
        "8b_calendar_read",
        "integration_runtime",
        "count: 1" in tools["calendar_list_events"].run(account_id=account.account_id),
        "listing reflects the write",
    )

    # 11. read-back verification on a fresh read of the created event
    event_id = _first_id(created, "event")
    verification_ms, got = _time_it(
        200,
        lambda: tools["calendar_get_event"].run(
            account_id=account.account_id, event_id=event_id
        ),
    )
    perf["readback_verification_ms"] = verification_ms
    step(
        "11_readback_verification",
        "integration_runtime",
        "timezone: Africa/Cairo" in got and "time: 15:30" in got,
        "got matches what was sent",
    )

    # grounding: the derived labeled fields are deterministically checkable
    evidence = [
        {"step_number": 1, "tool": "calendar_get_event", "status": "ok", "result": got}
    ]
    shown = _render_time(got)
    consistent = not check_grounding(f"Your Dentist event is at {shown} Cairo time.", evidence).contradiction
    hour, minute = (int(x) for x in shown.split(":"))
    shifted = f"{(hour + 2) % 24:02d}:{minute:02d}"
    wrong = check_grounding(f"Your Dentist event is at {shifted} Cairo time.", evidence).contradiction
    step("G_grounding_consistent", "grounding", consistent, f"time={shown}")
    step("G_grounding_contradiction", "grounding", wrong, f"shifted={shifted} rejected")

    # 9. tasks read/write (same OAuth machinery, second provider)
    tasks_start = manager.begin_authorization(
        provider="tasks",
        session_id=session,
        display_label="Eval Tasks",
        scopes={TASKS_READ, TASKS_WRITE},
    )
    tasks_consent = local_simulate_consent(tasks_start["authorization_url"])
    tasks_account = manager.handle_callback(
        provider="tasks",
        code=tasks_consent.code,
        state=tasks_consent.state,
        session_id=session,
        redirect_uri=tasks_consent.redirect_uri,
    )
    task_out = tools["task_create"].run(
        account_id=tasks_account.account_id, title="Call plumber"
    )
    task_id = _first_id(task_out, "task")
    done_out = tools["task_complete"].run(
        account_id=tasks_account.account_id, task_id=task_id
    )
    step(
        "9_task_read_write",
        "integration_runtime",
        "verification: VERIFIED" in task_out
        and "verification: VERIFIED" in done_out
        and "status: done" in tools["task_list"].run(account_id=tasks_account.account_id),
        "create + complete + read",
    )

    # 10. confirmation remains required for external writes
    risk = tools["calendar_create_event"].risk_for_args(
        {
            "account_id": account.account_id,
            "title": "Dentist",
            "date": "2026-10-05",
            "start_time": "15:30",
            "timezone": "Africa/Cairo",
            "duration_minutes": 45,
        }
    )
    registry = ToolRegistry()
    for tool in build_integration_tools(manager):
        registry.register(tool)
    orch = Orchestrator(store, registry, PermissionGuard())

    import asyncio

    async def _dispatch() -> str:
        orch._dispatch_ledger = type(orch._dispatch_ledger)()
        return await orch._dispatch_with_permissions_async(
            session,
            "calendar_create_event",
            json.dumps(
                {
                    "account_id": account.account_id,
                    "title": "Standup",
                    "date": "2026-10-06",
                    "start_time": "09:00",
                    "timezone": "Africa/Cairo",
                    "duration_minutes": 15,
                }
            ),
            "c1",
        )

    parked = asyncio.run(_dispatch())
    step(
        "10_confirmation_required",
        "integration_runtime",
        risk == "SYSTEM" and PAUSED_FOR_CONFIRMATION in str(parked),
        f"risk={risk}; parked before any side effect",
    )

    # 12. disconnect (provider-side revocation + local removal)
    live_token = manager.get_account(account.account_id)._access_token
    disconnect_ms, removed = _time_it(1, lambda: manager.disconnect(account.account_id))
    perf["disconnect_ms"] = disconnect_ms
    step(
        "12_disconnect",
        "oauth_flow",
        removed
        and manager.get_account(account.account_id) is None
        and server.introspect(live_token)["active"] is False,
        "provider grant revoked; local row removed",
    )

    # 13. post-disconnect action is rejected (fail closed)
    after = tools["calendar_list_events"].run(account_id=account.account_id)
    step(
        "13_post_disconnect_rejected",
        "integration_runtime",
        after.startswith("ERROR: INTEGRATION_REFUSED"),
        after[:60],
    )

    # 14. reauthorization restores access
    re_start = manager.begin_authorization(
        provider="calendar",
        session_id=session,
        display_label="Eval Calendar",
        scopes=full_calendar,
    )
    re_consent = local_simulate_consent(re_start["authorization_url"])
    fresh = manager.handle_callback(
        provider="calendar",
        code=re_consent.code,
        state=re_consent.state,
        session_id=session,
        redirect_uri=re_consent.redirect_uri,
    )
    step(
        "14_reauthorization_restores_access",
        "oauth_flow",
        fresh.authorization_status == "AUTHORIZED"
        and isinstance(manager.execute_read(fresh, Operation.LIST), list),
        f"account={fresh.account_id}",
    )

    # cache-hit latency (warmed entry; no provider call)
    cache = ResultCache(store)
    policy = tools["calendar_list_events"].cache_policy
    cache_args = json.dumps({"account_id": fresh.account_id})
    key = ResultCache.cache_key("calendar_list_events", policy, cache_args)
    with store._lock:
        store._conn.execute(
            """
            INSERT OR REPLACE INTO result_cache
                (cache_key, tool_name, args_json, result, scope, session_id,
                 created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                key, "calendar_list_events", cache_args, "cached payload",
                "session", session, datetime.now(tz=timezone.utc).isoformat(),
                (datetime.now(tz=timezone.utc).replace(year=2099)).isoformat(),
            ),
        )
        store._conn.commit()
    cache_ms, decision = _time_it(
        200,
        lambda: cache.lookup(
            tool=tools["calendar_list_events"],
            tool_name="calendar_list_events",
            tool_args_json=cache_args,
            session_id=session,
        ),
    )
    perf["cache_hit_ms"] = cache_ms
    step("P_cache_hit", "integration_runtime", bool(decision.hit), decision.reason)

    passed = sum(1 for s in steps if s["ok"])
    failed = len(steps) - passed
    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "mode": (
            "deterministic — simulated local provider, no Ollama, no network; "
            "the model-driven chat path is measured separately by "
            "live_provider_eval.py"
        ),
        "provider": {
            "calendar": "local simulated OAuth provider (development only)",
            "tasks": "local simulated OAuth provider (development only)",
            "production_like": False,
        },
        "steps": steps,
        "performance_ms": perf,
        "passed": passed,
        "failed": failed,
        "phases": {
            phase: [s["name"] for s in steps if s["phase"] == phase]
            for phase in sorted({s["phase"] for s in steps})
        },
    }
    try:
        from jarvis import __version__

        report["version"] = __version__
    except Exception:  # noqa: BLE001
        report["version"] = "unknown"

    print("─" * 72)
    print(f"steps: {passed} passed, {failed} failed")
    for name, value in sorted(perf.items()):
        print(f"  {name:34s} {value:8.4f} ms (median)")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"report → {args.json_out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run())
