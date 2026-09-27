"""
tests/test_operator_introspection.py
────────────────────────────────────
v0.18 Tracks A–D + F: operator-facing reliability tooling.

Proves:
  - action ledger introspection: get / list / filter by state / filter by
    session / bounded pagination / deterministic ordering
  - safe-metadata-only projection: tool_args and result bodies never reach
    the API layer (ActionInfo schema), owner tokens never reach clients
  - session-lease introspection: active vs expired, list, takeover fencing,
    owner redaction
  - UNKNOWN recovery: explicit reissue mints a NEW action id, the original
    stays UNKNOWN for audit, the reissued action resolves through the NORMAL
    confirmation flow (approval dispatches exactly once; denial does not
    execute), duplicate request_ids are idempotent, the reissue ceiling is
    enforced, the audit chain is preserved, unauthenticated reissue is
    rejected, non-UNKNOWN reissue is refused
  - doctor: healthy / ollama down / model missing / stale leases / UNKNOWN
    actions (mockable, no network)
  - maintenance CLI: actions / unknown-actions / sessions / reissue with
    --json and human output, exit codes
All offline: DB is per-test tmp_path SQLite; Ollama and Docker are mocked.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jarvis.api.app import app, set_runtime
from jarvis.api.schemas import ActionInfo, LeaseInfo
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import (
    ACTION_STATE_FAILED,
    ACTION_STATE_PENDING,
    ACTION_STATE_RUNNING,
    ACTION_STATE_SUCCEEDED,
    ACTION_STATE_UNKNOWN,
    CONFIRMATION_TTL_MINUTES,
    MAX_REISSUES_PER_ACTION,
    SessionStore,
    redact_owner,
)
from jarvis.tools.base import BaseTool
from jarvis.tools.registry import ToolRegistry

# ── Helpers ───────────────────────────────────────────────────────────────────


def _stdout_json(capsys):
    """Parse the CLI's JSON payload from captured stdout.

    Structured log lines (timestamp-prefixed) may interleave with command
    output; the JSON payload always starts at column 0 with '[' or '{'.
    """
    out = capsys.readouterr().out
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("[") or line.startswith("{"):
            return json.loads("\n".join(lines[i:]))
    raise AssertionError(f"no JSON payload in captured stdout: {out!r}")


def _report_after(out: str, marker: str) -> str:
    """Return only the human report from ``marker`` on (drops log lines)."""
    idx = out.find(marker)
    assert idx != -1, f"marker {marker!r} not found in output: {out!r}"
    return out[idx:]


class CountingCalcTool(BaseTool):
    """Real BaseTool subclass so ToolRegistry.dispatch can validate args."""

    name = "calculator"
    description = "counting calculator for tests"
    parameters = {"type": "object", "properties": {}, "required": []}
    risk_level = "SAFE"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs) -> str:
        self.calls += 1
        return "42"

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from jarvis.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "v018_ops.db"))
    s = SessionStore()
    yield s
    s.close()


@pytest.fixture()
def client(store):
    """TestClient wired to the same isolated store (no auth by default)."""
    with patch("jarvis.runtime.get_vector_store"):
        rt = build_runtime_for(store)
    set_runtime(rt)
    with TestClient(app) as c:
        yield c
    set_runtime(None)
    rt.close()


def build_runtime_for(store):
    """Real runtime with its store rewired to the test's isolated DB.

    The /confirm endpoint needs runtime.orchestrator (lease + pop + claim),
    so a real build_runtime() is used; its SessionStore is swapped for the
    per-test file-backed one (same db_path the fixture created).
    """
    from jarvis.runtime import build_runtime

    with patch("jarvis.runtime.get_vector_store"):
        rt = build_runtime()
    rt.store = store
    rt.orchestrator._store = store
    return rt


def _park(store, sid, tool="calculator", args='{"expression": "2+2"}',
          risk="SAFE", call_id="call_1"):
    return store.save_pending_confirmation(
        session_id=sid,
        tool_name=tool,
        tool_args=args,
        tool_call_id=call_id,
        risk_level=risk,
    )


def _action_id_for(store, confirmation_id):
    return store.get_action_execution_by_confirmation(confirmation_id).action_id


def _to_unknown(store, action_id, owner="host:999:maint:deadbeef"):
    """RUNNING → UNKNOWN the way startup recovery does (claim first)."""
    claim = store.claim_action_execution(action_id, owner)
    assert claim == "claimed"
    store.mark_action_unknown(action_id, "test: simulated crash")
    return owner


# ── Track A: action introspection (store layer) ───────────────────────────────


class TestActionIntrospectionStore:
    def test_get_action(self, store):
        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        row = store.get_action_execution(aid)
        assert row is not None
        assert row.state == ACTION_STATE_PENDING
        assert row.tool_name == "calculator"
        assert row.attempt == 0

    def test_get_action_missing(self, store):
        assert store.get_action_execution("nope") is None

    def test_list_actions_deterministic_order(self, store):
        for i in range(5):
            _park(store, f"s{i}", call_id=f"call_{i}")
        rows = store.list_action_executions()
        assert len(rows) == 5
        created = [r.created_at for r in rows]
        assert created == sorted(created, reverse=True), "newest first"

    def test_filter_by_state(self, store):
        c1 = _park(store, "s1")
        a1 = _action_id_for(store, c1)
        c2 = _park(store, "s2")
        a2 = _action_id_for(store, c2)
        # Before the claim both are PENDING.
        assert {r.action_id for r in store.list_action_executions(state="PENDING")} == {a1, a2}
        store.claim_action_execution(a1, "host:1:t:o")
        # After the claim a1 is RUNNING and only a2 remains PENDING.
        assert {r.action_id for r in store.list_action_executions(state="PENDING")} == {a2}
        assert {r.action_id for r in store.list_action_executions(state="RUNNING")} == {a1}

    def test_filter_by_session(self, store):
        _park(store, "alpha")
        _park(store, "beta")
        rows = store.list_action_executions(session_id="alpha")
        assert len(rows) == 1
        assert rows[0].session_id == "alpha"

    def test_filter_state_and_session_combined(self, store):
        c = _park(store, "alpha")
        aid = _action_id_for(store, c)
        store.claim_action_execution(aid, "host:1:t:o")
        _park(store, "beta")
        rows = store.list_action_executions(state="RUNNING", session_id="alpha")
        assert [r.action_id for r in rows] == [aid]

    def test_limit_is_bounded_and_respected(self, store):
        for i in range(7):
            _park(store, f"s{i}", call_id=f"call_{i}")
        assert len(store.list_action_executions(limit=3)) == 3
        assert len(store.list_action_executions(limit=1)) == 1

    def test_limit_clamped_to_internal_max(self, store):
        for i in range(3):
            _park(store, f"s{i}", call_id=f"call_{i}")
        # Absurd limit is clamped, not honored (bounded queries requirement).
        assert len(store.list_action_executions(limit=10_000)) == 3

    def test_invalid_state_rejected(self, store):
        with pytest.raises(ValueError):
            store.list_action_executions(state="EXPLODED")

    def test_count_by_state(self, store):
        c1 = _park(store, "s1")
        c2 = _park(store, "s2")
        store.claim_action_execution(_action_id_for(store, c1), "host:1:t:o")
        store.finish_action_execution(
            _action_id_for(store, c2), ACTION_STATE_SUCCEEDED, "4"
        )
        counts = store.count_action_executions_by_state()
        assert counts.get("RUNNING") == 1
        assert counts.get("SUCCEEDED") == 1
        assert "FAILED" not in counts  # zero-state rows are absent, not 0

    def test_newer_than_boundary(self, store):
        _park(store, "old")
        cutoff = datetime.now(tz=timezone.utc)
        time.sleep(0.02)  # ensure the next row's timestamp is strictly later
        _park(store, "new", call_id="call_2")
        rows = store.list_action_executions(newer_than=cutoff.isoformat())
        assert [r.session_id for r in rows] == ["new"]
        # No filter → both rows, newest first.
        assert [r.session_id for r in store.list_action_executions()] == ["new", "old"]


# ── Track A: action introspection (API layer, safe metadata only) ─────────────


class TestActionIntrospectionAPI:
    def test_list_actions_endpoint(self, client, store):
        _park(store, "s1")
        resp = client.get("/actions")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["tool_name"] == "calculator"
        assert body[0]["state"] == "PENDING"

    def test_state_filter_endpoint(self, client, store):
        c = _park(store, "s1")
        aid = _action_id_for(store, c)
        store.claim_action_execution(aid, "host:1:t:o")
        resp = client.get("/actions", params={"state": "RUNNING"})
        assert resp.status_code == 200
        assert [a["action_id"] for a in resp.json()] == [aid]

    def test_invalid_state_is_422(self, client):
        assert client.get("/actions", params={"state": "NOPE"}).status_code == 422

    def test_get_single_action_404(self, client):
        assert client.get("/actions/missing").status_code == 404

    def test_safe_metadata_only_no_tool_args(self, client, store):
        """tool_args and result bodies must never appear in API output."""
        secret = "SECRET_PAYLOAD_SHOULD_NOT_LEAK"
        _park(store, "s1", args=json.dumps({"expression": secret}))
        body = client.get("/actions").json()
        assert len(body) == 1
        assert secret not in json.dumps(body)
        assert "tool_args" not in body[0]
        assert "result" not in body[0]
        # Schema-level guarantee, not just response shape:
        assert "tool_args" not in ActionInfo.model_fields
        assert "result" not in ActionInfo.model_fields

    def test_limit_validation(self, client, store):
        for i in range(3):
            _park(store, f"s{i}", call_id=f"call_{i}")
        ok = client.get("/actions", params={"limit": 2})
        assert ok.status_code == 200
        assert len(ok.json()) == 2
        assert client.get("/actions", params={"limit": 0}).status_code == 422
        assert client.get("/actions", params={"limit": 501}).status_code == 422

    def test_owner_not_exposed_via_actions(self, client, store):
        c = _park(store, "s1")
        aid = _action_id_for(store, c)
        store.claim_action_execution(aid, "host:123:runtime:abcd1234")
        body = client.get(f"/actions/{aid}").json()
        assert body["state"] == "RUNNING"
        assert "host:123" not in json.dumps(body)


# ── Track B: session/lease introspection ──────────────────────────────────────


class TestLeaseIntrospectionStore:
    def test_active_lease_visible(self, store):
        acquired, fencing = store.acquire_session_lease("s1", "host:1:runtime:aaaa")
        assert acquired is True
        leases = store.list_session_leases()
        assert len(leases) == 1
        l = leases[0]
        assert l["session_id"] == "s1"
        assert l["active"] is True
        assert l["fencing"] == fencing
        assert l["owner_token"] == "host:1:runtime:aaaa"

    def test_expired_lease_flagged_inactive(self, store):
        store.acquire_session_lease("s1", "host:1:runtime:aaaa", ttl_seconds=-1)
        leases = store.list_session_leases()
        assert leases[0]["active"] is False
        counts = store.count_session_leases()
        assert counts["expired"] == 1 and counts["active"] == 0

    def test_takeover_increments_fencing(self, store):
        store.acquire_session_lease("s1", "host:1:runtime:aaaa", ttl_seconds=-1)
        ok, fencing = store.acquire_session_lease("s1", "host:2:runtime:bbbb")
        assert ok is True
        assert fencing == 2
        l = store.list_session_leases()[0]
        assert l["owner_token"] == "host:2:runtime:bbbb"
        assert l["active"] is True

    def test_redact_owner(self):
        assert redact_owner("host:123:runtime:abcd1234") == "runtime:abcd1234"
        assert redact_owner(None) == "none"
        # Malformed tokens degrade without leaking host/pid components.
        assert redact_owner("totally-not-a-host-token") == "totally-not-a-ho"


class TestLeaseIntrospectionAPI:
    def test_list_leases_endpoint(self, client, store):
        store.acquire_session_lease("s1", "host:1:runtime:aaaa")
        resp = client.get("/sessions/leases")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["session_id"] == "s1"
        assert body[0]["active"] is True
        assert body[0]["fencing"] == 1

    def test_owner_redacted_in_api(self, client, store):
        store.acquire_session_lease("s1", "host:555:runtime:abcd1234")
        body = client.get("/sessions/leases").json()
        assert body[0]["owner"] == "runtime:abcd1234"
        assert "host:555" not in json.dumps(body)
        # Raw token absent from schema entirely:
        assert "owner_token" not in LeaseInfo.model_fields

    def test_limit_validation(self, client):
        assert client.get("/sessions/leases", params={"limit": 0}).status_code == 422
        assert client.get("/sessions/leases", params={"limit": 501}).status_code == 422

    def test_empty_list(self, client):
        assert client.get("/sessions/leases").json() == []


# ── Track C: UNKNOWN recovery via explicit reissue ────────────────────────────


class TestUnknownRecoveryStore:
    def _setup_unknown(self, store, sid="s-unk"):
        cid = _park(store, sid, tool="calculator")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)  # pop (real flow pops first)
        _to_unknown(store, aid)
        return aid

    def test_unknown_requires_prior_running(self, store):
        """A second claim of the same action returns already_terminal, not
        claimed — so the assert inside _to_unknown fires for PENDING rows."""
        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        claim = store.claim_action_execution(aid, "host:1:t:o")
        assert claim == "claimed"
        store.mark_action_unknown(aid, "legit")
        # From terminal (UNKNOWN) a further transition is impossible:
        claim2 = store.claim_action_execution(aid, "host:1:t:o")
        assert claim2.startswith("already_terminal")
        with pytest.raises(AssertionError):
            _to_unknown(store, aid)

    def test_reissue_mints_new_identity(self, store):
        aid = self._setup_unknown(store)
        new_id = store.request_action_reissue(aid, "req-1")
        assert new_id != aid
        new_row = store.get_action_execution(new_id)
        assert new_row.state == ACTION_STATE_PENDING
        assert new_row.attempt == 0
        assert new_row.session_id == store.get_action_execution(aid).session_id
        assert new_row.tool_name == store.get_action_execution(aid).tool_name
        assert new_row.risk_level == store.get_action_execution(aid).risk_level

    def test_original_stays_unknown(self, store):
        aid = self._setup_unknown(store)
        new_id = store.request_action_reissue(aid, "req-1")
        assert store.get_action_execution(aid).state == ACTION_STATE_UNKNOWN
        assert store.get_reissue_origin(new_id) == aid

    def test_reissue_parks_resolvable_confirmation(self, store):
        aid = self._setup_unknown(store)
        store.request_action_reissue(aid, "req-1")
        pending = store.load_pending_confirmation(store.get_action_execution(aid).session_id)
        assert pending is not None
        assert pending["confirmation_id"] == (
            store.get_action_execution(store.get_reissue_chain(aid)[0]["new_action_id"]).confirmation_id
        )

    def test_duplicate_request_idempotent(self, store):
        aid = self._setup_unknown(store)
        first = store.request_action_reissue(aid, "req-1")
        second = store.request_action_reissue(aid, "req-1")
        assert first == second
        assert len(store.get_reissue_chain(aid)) == 1
        # And exactly ONE new ledger row exists.
        assert len(store.list_action_executions(session_id=store.get_action_execution(aid).session_id)) == 2

    def test_reissue_limit(self, store):
        aid = self._setup_unknown(store)
        sid = store.get_action_execution(aid).session_id
        for i in range(MAX_REISSUES_PER_ACTION):
            store.request_action_reissue(aid, f"req-{i}")
            # Resolve each parked confirmation so the next reissue is not
            # blocked by the one-active-confirmation-per-session rule.
            store.complete_pending_confirmation(sid)
        with pytest.raises(ValueError, match="reissue limit reached"):
            store.request_action_reissue(aid, "req-over")

    def test_reissue_refused_for_non_unknown(self, store):
        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        with pytest.raises(ValueError, match="not UNKNOWN"):
            store.request_action_reissue(aid, "req-1")

    def test_reissue_refused_for_unknown_action(self, store):
        with pytest.raises(ValueError, match="unknown action_id"):
            store.request_action_reissue("missing", "req-1")

    def test_audit_chain_preserved(self, store):
        aid = self._setup_unknown(store)
        sid = store.get_action_execution(aid).session_id
        ids = []
        for i in range(2):
            ids.append(store.request_action_reissue(aid, f"req-{i}"))
            # Resolve the reissued confirmation so the next reissue is not
            # blocked by the one-active-confirmation-per-session rule.
            store.complete_pending_confirmation(sid)
        chain = store.get_reissue_chain(aid)
        assert [c["new_action_id"] for c in chain] == ids
        assert [c["request_id"] for c in chain] == ["req-0", "req-1"]
        assert all(store.get_reissue_origin(i) == aid for i in ids)

    def test_active_confirmation_blocks_reissue(self, store):
        sid = "s-active"
        cid = _park(store, sid, tool="calculator")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)
        _to_unknown(store, aid)
        # Someone parks a NEW action for the same session (still unapproved).
        _park(store, sid, tool="calculator", call_id="call_2")
        with pytest.raises(ValueError, match="active pending confirmation"):
            store.request_action_reissue(aid, "req-1")
        # Refused transactionally: no audit row, no NEW ledger row from the
        # reissue (the second park itself legitimately added one).
        assert store.get_reissue_chain(aid) == []
        assert len(store.list_action_executions(session_id=sid)) == 2
        assert store.load_pending_confirmation(sid) is not None

    def test_completed_confirmation_leftover_cleaned(self, store):
        sid = "s-leftover"
        cid = _park(store, sid)
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)
        _to_unknown(store, aid)
        # A stale completed row is fine — it is cleaned, not a blocker.
        new_id = store.request_action_reissue(aid, "req-1")
        assert store.load_pending_confirmation(sid) is not None
        assert store.get_action_execution(new_id).state == ACTION_STATE_PENDING

    def test_reissued_action_follows_normal_permission_flow(self, store):
        """Approval through handle_confirmation dispatches exactly once."""
        sid = "s-flow"
        reg = ToolRegistry()
        calc = CountingCalcTool()
        reg._tools["calculator"] = calc
        orch = Orchestrator(store, reg, PermissionGuard())

        cid = _park(store, sid, tool="calculator", args="{}")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)
        _to_unknown(store, aid)

        new_id = store.request_action_reissue(aid, "req-1")
        assert new_id != aid
        # The session now has a resolvable confirmation (reissue parked it).
        out = orch.handle_confirmation(sid, True)
        assert calc.calls == 1, f"expected exactly one dispatch, got {calc.calls}"
        assert store.get_action_execution(new_id).state == ACTION_STATE_SUCCEEDED
        assert store.get_action_execution(aid).state == ACTION_STATE_UNKNOWN
        # Second approval must not re-execute.
        orch.handle_confirmation(sid, True)
        assert calc.calls == 1

    def test_denied_reissue_does_not_execute(self, store):
        sid = "s-deny"
        reg = ToolRegistry()
        calc = CountingCalcTool()
        reg._tools["calculator"] = calc
        orch = Orchestrator(store, reg, PermissionGuard())

        cid = _park(store, sid, args="{}")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)
        _to_unknown(store, aid)
        store.request_action_reissue(aid, "req-1")
        orch.handle_confirmation(sid, False)
        assert calc.calls == 0
        assert store.get_action_execution(
            store.get_reissue_chain(aid)[0]["new_action_id"]
        ).state == ACTION_STATE_FAILED


class TestUnknownRecoveryAPI:
    def _setup_unknown_via_api(self, client, store, sid="s-api"):
        cid = _park(store, sid)
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)
        _to_unknown(store, aid)
        return aid

    def test_reissue_endpoint_happy_path(self, client, store):
        aid = self._setup_unknown_via_api(client, store)
        resp = client.post(
            f"/actions/{aid}/reissue", json={"request_id": "req-api-1"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["original_action_id"] == aid
        assert body["new_action_id"] != aid
        assert body["state"] == "PENDING"
        assert body["reissue_depth"] == 1
        assert body["reused_existing"] is False
        new_row = store.get_action_execution(body["new_action_id"])
        assert new_row.confirmation_id.startswith("reissue:")

    def test_reissue_endpoint_idempotent_replay(self, client, store):
        aid = self._setup_unknown_via_api(client, store)
        r1 = client.post(f"/actions/{aid}/reissue", json={"request_id": "req-replay"})
        r2 = client.post(f"/actions/{aid}/reissue", json={"request_id": "req-replay"})
        assert r1.json()["new_action_id"] == r2.json()["new_action_id"]
        assert r2.json()["reused_existing"] is True
        assert len(store.get_reissue_chain(aid)) == 1

    def test_reissue_endpoint_404_unknown_action(self, client):
        resp = client.post(
            "/actions/missing/reissue", json={"request_id": "req-404-xx"}
        )
        assert resp.status_code == 404

    def test_reissue_endpoint_409_wrong_state(self, client, store):
        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        resp = client.post(
            f"/actions/{aid}/reissue", json={"request_id": "req-409-xx"}
        )
        assert resp.status_code == 409
        assert "not UNKNOWN" in resp.json()["detail"]

    def test_reissue_endpoint_409_limit(self, client, store):
        aid = self._setup_unknown_via_api(client, store)
        sid = store.get_action_execution(aid).session_id
        for i in range(MAX_REISSUES_PER_ACTION):
            assert client.post(
                f"/actions/{aid}/reissue", json={"request_id": f"limit-req-{i}"}
            ).status_code == 200
            # Resolve the parked confirmation so the next reissue passes the
            # one-active-confirmation guard (mirrors the operator loop).
            assert client.post(
                f"/sessions/{sid}/confirm", json={"confirmed": False}
            ).status_code == 200
        resp = client.post(
            f"/actions/{aid}/reissue", json={"request_id": "limit-req-over"}
        )
        assert resp.status_code == 409
        assert "reissue limit" in resp.json()["detail"]

    def test_reissue_endpoint_409_active_confirmation(self, client, store):
        sid = "s-block"
        cid = _park(store, sid)
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(sid)
        _to_unknown(store, aid)
        _park(store, sid, call_id="call_2")  # new unapproved action
        resp = client.post(
            f"/actions/{aid}/reissue", json={"request_id": "req-block-1"}
        )
        assert resp.status_code == 409
        assert "active pending confirmation" in resp.json()["detail"]

    def test_unauthenticated_reissue_rejected(self, store):
        """Critical (Part F): reissue is NEVER reachable without the key."""
        with patch("jarvis.runtime.get_vector_store"):
            rt = build_runtime_for(store)
        set_runtime(rt)
        try:
            with patch("jarvis.api.app.auth_enabled", return_value=True), patch(
                "jarvis.api.auth.settings"
            ) as auth_settings:
                auth_settings.JARVIS_API_KEY = "sekrit"
                with TestClient(app) as c:
                    cid = _park(store, "s-auth")
                    aid = _action_id_for(store, cid)
                    store.complete_pending_confirmation("s-auth")
                    _to_unknown(store, aid)
                    no_key = c.post(
                        f"/actions/{aid}/reissue", json={"request_id": "req-auth-1"}
                    )
                    assert no_key.status_code == 401
                    wrong = c.post(
                        f"/actions/{aid}/reissue",
                        json={"request_id": "req-auth-1"},
                        headers={"Authorization": "Bearer wrong"},
                    )
                    assert wrong.status_code == 401
                    ok = c.post(
                        f"/actions/{aid}/reissue",
                        json={"request_id": "req-auth-1"},
                        headers={"X-API-Key": "sekrit"},
                    )
                    assert ok.status_code == 200
        finally:
            set_runtime(None)
            rt.close()

    def test_read_endpoints_require_auth_too(self, store):
        with patch("jarvis.runtime.get_vector_store"):
            rt = build_runtime_for(store)
        set_runtime(rt)
        try:
            with patch("jarvis.api.app.auth_enabled", return_value=True), patch(
                "jarvis.api.auth.settings"
            ) as auth_settings:
                auth_settings.JARVIS_API_KEY = "sekrit"
                with TestClient(app) as c:
                    assert c.get("/actions").status_code == 401
                    assert c.get("/sessions/leases").status_code == 401
        finally:
            set_runtime(None)
            rt.close()

    def test_request_id_validated(self, client, store):
        aid = self._setup_unknown_via_api(client, store)
        too_short = client.post(f"/actions/{aid}/reissue", json={"request_id": "ab"})
        assert too_short.status_code == 422


# ── Track D: doctor + CLI ─────────────────────────────────────────────────────


class TestDoctor:
    def _run_doctor(self, store, monkeypatch, ollama_ok=True, model_ok=True,
                    sandbox_available=True, code_exec=True):
        from jarvis import maintenance

        # cmd_doctor imports these from jarvis.api.health at call time.
        monkeypatch.setattr(
            "jarvis.api.health.check_ollama", lambda *a, **k: ollama_ok
        )
        monkeypatch.setattr(
            maintenance, "_check_ollama_model", lambda *a, **k: model_ok
        )

        class FakeSandbox:
            def __init__(self, image=None):
                pass

            def is_available(self):
                return sandbox_available

        monkeypatch.setattr(
            "jarvis.core.sandbox.DockerCodeSandbox", FakeSandbox
        )
        monkeypatch.setattr(maintenance.settings, "ENABLE_CODE_EXECUTION", code_exec)
        monkeypatch.setattr(maintenance.settings, "ollama_base_url", "http://x")
        monkeypatch.setattr(maintenance.settings, "ollama_model", "m")
        return maintenance.cmd_doctor(store)

    def test_doctor_healthy(self, store, monkeypatch, capsys):
        assert self._run_doctor(store, monkeypatch) == 0
        out = capsys.readouterr().out
        assert "all checks passed" in out

    def test_doctor_ollama_unavailable(self, store, monkeypatch, capsys):
        assert self._run_doctor(store, monkeypatch, ollama_ok=False) == 1
        out = capsys.readouterr().out
        assert "ollama" in out

    def test_doctor_model_unavailable(self, store, monkeypatch, capsys):
        assert self._run_doctor(store, monkeypatch, model_ok=False) == 1
        out = capsys.readouterr().out
        assert "not installed" in out

    def test_doctor_stale_lease_detected(self, store, monkeypatch, capsys):
        store.acquire_session_lease("s1", "host:1:runtime:aaaa", ttl_seconds=-1)
        assert self._run_doctor(store, monkeypatch) == 1
        out = capsys.readouterr().out
        assert "0 active, 1 stale" in out
        assert "sessions --expired" in out
        assert out.count("FAIL") >= 1

    def test_doctor_unknown_action_detected(self, store, monkeypatch, capsys):
        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        _to_unknown(store, aid)
        assert self._run_doctor(store, monkeypatch) == 1
        out = capsys.readouterr().out
        assert "1 UNKNOWN" in out
        assert "unknown-actions" in out

    def test_doctor_sandbox_unavailable(self, store, monkeypatch, capsys):
        assert self._run_doctor(
            store, monkeypatch, sandbox_available=False
        ) == 1
        out = capsys.readouterr().out
        assert "sandbox" in out

    def test_doctor_sandbox_disabled_is_ok(self, store, monkeypatch, capsys):
        assert self._run_doctor(store, monkeypatch, code_exec=False) == 0
        out = capsys.readouterr().out
        assert "disabled (safe default)" in out


class TestMaintenanceCLI:
    def test_actions_json(self, store, capsys):
        from jarvis import maintenance

        _park(store, "s1")
        rc = maintenance.cmd_actions(store, state=None, session=None,
                                     limit=50, as_json=True)
        assert rc == 0
        rows = _stdout_json(capsys)
        assert len(rows) == 1
        assert "tool_args" not in rows[0]

    def test_actions_human_filtered(self, store, capsys):
        from jarvis import maintenance

        c = _park(store, "s1")
        aid = _action_id_for(store, c)
        store.claim_action_execution(aid, "host:1:t:o")
        rc = maintenance.cmd_actions(store, state="RUNNING", session=None,
                                     limit=50, as_json=False)
        out = capsys.readouterr().out
        assert rc == 0
        assert aid in out
        assert "RUNNING" in out

    def test_actions_invalid_state_exit_1(self, store, capsys):
        from jarvis import maintenance

        rc = maintenance.cmd_actions(store, state="BOGUS", session=None,
                                     limit=50, as_json=False)
        assert rc == 1

    def test_unknown_actions_json(self, store, capsys):
        from jarvis import maintenance

        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        _to_unknown(store, aid)
        rc = maintenance.cmd_unknown_actions(store, limit=50, as_json=True)
        assert rc == 0
        rows = _stdout_json(capsys)
        assert rows[0]["action_id"] == aid
        assert rows[0]["reissue_depth"] == 0
        assert rows[0]["max_reissues"] == MAX_REISSUES_PER_ACTION

    def test_unknown_actions_human_has_guidance(self, store, capsys):
        from jarvis import maintenance

        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        _to_unknown(store, aid)
        maintenance.cmd_unknown_actions(store, limit=50, as_json=False)
        out = _report_after(capsys.readouterr().out, "UNKNOWN ACTIONS")
        assert "UNKNOWN ACTIONS" in out
        assert "NOT re-run these automatically" in out
        assert "reissue" in out
        assert "deadbeef" in out  # redacted owner keeps the rand part only

    def test_sessions_json(self, store, capsys):
        from jarvis import maintenance

        store.acquire_session_lease("s1", "host:1:runtime:aaaa")
        rc = maintenance.cmd_sessions(store, expired_only=False, limit=100,
                                      as_json=True)
        assert rc == 0
        rows = _stdout_json(capsys)
        assert rows[0]["active"] is True

    def test_sessions_expired_filter(self, store, capsys):
        from jarvis import maintenance

        store.acquire_session_lease("s1", "host:1:runtime:aaaa")
        store.acquire_session_lease("s2", "host:2:runtime:bbbb", ttl_seconds=-1)
        maintenance.cmd_sessions(store, expired_only=True, limit=100,
                                 as_json=False)
        out = _report_after(capsys.readouterr().out, "STALE SESSION LEASES")
        assert "s2" in out and "s1" not in out
        assert "STALE" in out

    def test_sessions_redacts_owner(self, store, capsys):
        from jarvis import maintenance

        store.acquire_session_lease("s1", "host:123:runtime:abcd1234")
        maintenance.cmd_sessions(store, expired_only=False, limit=100,
                                 as_json=False)
        out = _report_after(capsys.readouterr().out, "SESSION LEASES")
        assert "host:123" not in out
        assert "runtime:abcd1234" in out

    def test_reissue_yes_flag(self, store, capsys):
        from jarvis import maintenance

        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        _to_unknown(store, aid)
        rc = maintenance.cmd_reissue(store, aid, "req-cli-1", assume_yes=True)
        assert rc == 0
        assert store.get_reissue_chain(aid)
        assert "Reissued" in capsys.readouterr().out

    def test_reissue_interactive_abort(self, store, capsys, monkeypatch):
        from jarvis import maintenance

        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        _to_unknown(store, aid)
        monkeypatch.setattr("builtins.input", lambda *a: "no")
        rc = maintenance.cmd_reissue(store, aid, "req-cli-2", assume_yes=False)
        assert rc == 1
        assert store.get_reissue_chain(aid) == []

    def test_reissue_interactive_confirm(self, store, capsys, monkeypatch):
        from jarvis import maintenance

        cid = _park(store, s_ := "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation(s_)
        _to_unknown(store, aid)
        monkeypatch.setattr("builtins.input", lambda *a: "yes")
        rc = maintenance.cmd_reissue(store, aid, "req-cli-3", assume_yes=False)
        assert rc == 0
        assert len(store.get_reissue_chain(aid)) == 1

    def test_reissue_missing_action(self, store, capsys):
        from jarvis import maintenance

        rc = maintenance.cmd_reissue(store, "missing", "req", assume_yes=True)
        assert rc == 1

    def test_reissue_wrong_state(self, store, capsys):
        from jarvis import maintenance

        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        rc = maintenance.cmd_reissue(store, aid, "req", assume_yes=True)
        assert rc == 1

    def test_main_dispatch_actions(self, store, monkeypatch, capsys):
        from jarvis import maintenance

        _park(store, "s1")
        monkeypatch.setattr(maintenance, "_open_store", lambda: store)
        rc = maintenance.main(["actions", "--json", "--limit", "10"])
        assert rc == 0
        rows = _stdout_json(capsys)
        assert rows[0]["session_id"] == "s1"

    def test_main_dispatch_reissue(self, store, monkeypatch):
        from jarvis import maintenance

        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        _to_unknown(store, aid)

        class NoCloseStore:
            """main() closes the store it opens; our fixture owns this one."""

            def __getattr__(self, name):
                return getattr(store, name)

            def close(self):
                pass

        monkeypatch.setattr(
            maintenance, "_open_store", lambda: NoCloseStore()
        )
        rc = maintenance.main(
            ["reissue", "--action", aid, "--request-id", "rq-1234567", "--yes"]
        )
        assert rc == 0
        assert len(store.get_reissue_chain(aid)) == 1


# ── Regression guardrails for the confirmation-continuation path ──────────────


class TestConfirmationContinuationUnaffected:
    def test_normal_approval_flow_still_works(self, store):
        """A freshly-parked (non-reissue) action still resolves normally."""
        sid = "s-reg"
        reg = ToolRegistry()
        calc = CountingCalcTool()
        reg._tools["calculator"] = calc
        orch = Orchestrator(store, reg, PermissionGuard())
        _park(store, sid, args="{}")
        out = orch.handle_confirmation(sid, True)
        assert calc.calls == 1
        assert "42" in out or "Executed" in out

    def test_confirmation_ttl_constant_unchanged(self):
        assert CONFIRMATION_TTL_MINUTES == 10

    def test_reissue_confirmation_uses_standard_ttl(self, store):
        aid_owner = "host:1:t:o"
        cid = _park(store, "s1")
        aid = _action_id_for(store, cid)
        store.complete_pending_confirmation("s1")
        store.claim_action_execution(aid, aid_owner)
        store.mark_action_unknown(aid, "test")
        before = datetime.now(tz=timezone.utc)
        store.request_action_reissue(aid, "req-1")
        pending = store.load_pending_confirmation("s1")
        assert pending is not None
        expires = datetime.fromisoformat(pending["expires_at"])
        expected_lo = before + timedelta(minutes=CONFIRMATION_TTL_MINUTES - 1)
        expected_hi = before + timedelta(minutes=CONFIRMATION_TTL_MINUTES + 1)
        assert expected_lo <= expires <= expected_hi
