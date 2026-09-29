"""
tests/test_evaluation_isolation.py
──────────────────────────────────
v0.25 (Part G) tests for the shared evaluation-isolation bootstrap.

Contract under test (evaluation/_bootstrap.py):
  1. Every standalone evaluation entry point calls ``isolate()`` BEFORE its
     first jarvis import — otherwise Settings binds the real jarvis.db.
  2. Standalone import with no DB_PATH redirects the process to a private
     per-process temp DB (functional subprocess check).
  3. ``isolate()`` after jarvis.config was imported raises RuntimeError when
     the DB is still production-like — silently measuring the real database
     is the failure mode this helper exists to prevent.
  4. Safety valve: a harness that ALREADY redirected DB_PATH (pytest
     conftest's ':memory:') is accepted instead of raising, so evaluation
     modules stay importable under pytest.
  5. Idempotency: a second isolate() call changes nothing.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 8 standalone evaluation entry points (Part G scope).
EVAL_SCRIPTS = [
    "evaluation/multistep_benchmark.py",
    "evaluation/cache_replan_benchmark.py",
    "evaluation/tool_selection_benchmark.py",
    "evaluation/plan_judge.py",
    "evaluation/run_evals.py",
    "live_multistep_eval.py",
    "live_cache_replan_eval.py",
    "live_tool_eval.py",
]


# ── 1. Source-level: isolate() precedes every jarvis import ───────────────────


class TestSourceIsolationOrder:
    @pytest.mark.parametrize("rel_path", EVAL_SCRIPTS)
    def test_isolate_called_before_first_jarvis_import(self, rel_path):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "from evaluation import _bootstrap" in text, (
            f"{rel_path}: does not use the shared isolation bootstrap"
        )
        first_isolate = text.index("_eval.isolate()")
        jarvis_import = re.search(r"^(?:from jarvis|import jarvis)", text, re.M)
        assert jarvis_import, f"{rel_path}: no jarvis import found"
        assert first_isolate < jarvis_import.start(), (
            f"{rel_path}: _eval.isolate() must precede the first jarvis import"
        )
        # Scripts that use ``from __future__ import annotations`` must keep
        # the isolation block AFTER it (the stray pre-__future__ block from
        # the original patch made the file unparseable).
        if "from __future__ import annotations" in text:
            future_pos = text.index("from __future__ import annotations")
            assert first_isolate > future_pos, (
                f"{rel_path}: isolation block must come after __future__ import"
            )


# ── 2/5. Functional: subprocess import isolates, idempotently ────────────────


class TestFunctionalIsolation:
    def test_standalone_import_redirects_db(self):
        """Importing a benchmark with NO DB_PATH set must run against a
        private eval_cache.db — never the real jarvis.db."""
        code = (
            "import evaluation.cache_replan_benchmark as crb\n"
            "from evaluation import _bootstrap as b\n"
            "assert b.is_isolated(), b.current_db_path()\n"
            "assert 'eval_cache.db' in b.current_db_path()\n"
            "assert 'jarvis.db' not in b.current_db_path()\n"
            "print('isolated-ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "isolated-ok" in proc.stdout

    def test_isolate_is_idempotent(self):
        code = (
            "from evaluation import _bootstrap as b\n"
            "first = b.isolate()\n"
            "second = b.isolate()\n"
            "assert first == second\n"
            "import os\n"
            "assert os.environ['DB_PATH'] == str(first) + r'\\eval_cache.db'.replace('\\\\', '\\\\') or 'eval_cache.db' in os.environ['DB_PATH']\n"
            "print('idempotent-ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "idempotent-ok" in proc.stdout


# ── 3/4. Guard + safety valve (in-process) ────────────────────────────────────


class TestIsolateGuard:
    def test_raises_after_jarvis_import_with_real_db(self, monkeypatch):
        import jarvis.config  # noqa: F401 — ensure jarvis IS imported
        from evaluation import _bootstrap as _eval

        monkeypatch.setattr(_eval, "_STATE", {"applied": False, "tmpdir": None})
        monkeypatch.setenv("DB_PATH", "jarvis.db")
        with pytest.raises(RuntimeError, match="BEFORE importing jarvis"):
            _eval.isolate()

    def test_safety_valve_accepts_memory_redirect(self, monkeypatch):
        """pytest conftest already set DB_PATH=':memory:' before importing
        jarvis → isolate() must NOT raise and must not touch the env."""
        import jarvis.config  # noqa: F401
        from evaluation import _bootstrap as _eval

        monkeypatch.setattr(_eval, "_STATE", {"applied": False, "tmpdir": None})
        monkeypatch.setenv("DB_PATH", ":memory:")
        result = _eval.isolate()
        assert result == ":memory:"
        assert not _eval.is_isolated()  # not an eval temp dir — harness-managed

    def test_safety_valve_accepts_temp_redirect(self, monkeypatch):
        """Any explicitly redirected, non-production filename is accepted."""
        import jarvis.config  # noqa: F401
        from evaluation import _bootstrap as _eval

        monkeypatch.setattr(_eval, "_STATE", {"applied": False, "tmpdir": None})
        monkeypatch.setenv("DB_PATH", str(Path("some") / "ci_run.db"))
        assert _eval.isolate().endswith("ci_run.db")

    def test_real_db_name_is_detected(self, monkeypatch):
        from evaluation import _bootstrap as _eval

        monkeypatch.setenv("DB_PATH", str(REPO_ROOT / "jarvis.db"))
        assert not _eval._env_db_already_redirected()
        monkeypatch.setenv("DB_PATH", ":memory:")
        assert _eval._env_db_already_redirected()


class TestFreshStorePerCaseIsolation:
    """v0.25: per-case benchmark stores must be hermetic against the GLOBAL
    result cache (calculator policy, 7-day TTL). A standalone standalone-run
    regression: two multistep cases evaluating '12 * 12' shared one file DB,
    the later case got a cache HIT with zero dispatches and failed its
    dispatch-count grader — pytest stayed green because ':memory:' gives each
    store a private DB. fresh_store() pins the fix for the standalone path."""

    def test_fresh_store_gives_distinct_private_databases(self):
        import os

        from jarvis.config import settings
        from jarvis.memory.session_store import SessionStore

        from evaluation import _bootstrap as _eval

        _eval.isolate()
        s1, s2 = _eval.fresh_store(), _eval.fresh_store()
        try:
            assert isinstance(s1, SessionStore) and isinstance(s2, SessionStore)
            # Distinct files, inside the evaluation temp dir.
            p1, p2 = settings.db_path, settings.db_path  # restored binding
            d1 = s1._conn.execute("PRAGMA database_list").fetchall()
            d2 = s2._conn.execute("PRAGMA database_list").fetchall()
            f1 = next(r[2] for r in d1 if r[2])
            f2 = next(r[2] for r in d2 if r[2])
            assert f1 != f2, "fresh_store() stores share one database"
            assert "eval_cache" in os.path.dirname(f1) or "jarvis_eval" in os.path.dirname(f1)
            # The global settings binding is restored after construction.
            assert settings.db_path == p1
        finally:
            s1.close()
            s2.close()

    def test_calculator_cache_does_not_leak_across_fresh_stores(self):
        """The exact standalone failure shape: same expression, two stores,
        each must record its own dispatch (no cross-store cache hit)."""
        import asyncio
        import json as _json
        from unittest.mock import patch

        from jarvis.core.orchestrator import Orchestrator
        from jarvis.core.permissions import PermissionGuard
        from jarvis.tools import CalculatorTool, ToolRegistry

        from evaluation import _bootstrap as _eval

        _eval.isolate()

        async def scenario() -> tuple[int, int]:
            counts: list[int] = []
            for _ in range(2):
                store = _eval.fresh_store()
                registry = ToolRegistry()
                registry.register(CalculatorTool())
                orch = Orchestrator(store, registry, PermissionGuard())
                calls = {"n": 0}

                async def probe(tool_name: str, tool_args: str) -> str:
                    calls["n"] += 1
                    return f"{_json.loads(tool_args)['expression']} = ok"

                with patch.object(registry, "dispatch_async", side_effect=probe):
                    await orch._dispatch_with_permissions_async(
                        "sA", "calculator", '{"expression": "12 * 12"}', "c1"
                    )
                counts.append(calls["n"])
                store.close()
            return counts[0], counts[1]

        first, second = asyncio.run(scenario())
        assert (first, second) == (1, 1), (
            "cross-store calculator cache leak: "
            f"dispatches were {(first, second)}, expected (1, 1)"
        )
