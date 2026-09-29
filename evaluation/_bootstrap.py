"""
evaluation/_bootstrap.py
────────────────────────
v0.25 (Part G): shared evaluation-isolation bootstrap.

The v0.24 cross-turn result cache lives in SQLite (`jarvis.db`). Global-scope
entries (calculator, web) persist across processes, so a standalone benchmark
run that opens the REAL database will legitimately SERVE cached results
instead of dispatching — the code works, but the measurement is invalid
(observed live in v0.24: missing calculator dispatches in the multistep
benchmark after a prior live run seeded the cache).

Every evaluation entry point that measures dispatch counts, cache misses, or
replan behavior MUST therefore run against an isolated throwaway database.
This helper is the one place that does it:

    import evaluation._bootstrap as bootstrap
    bootstrap.isolate()          # call BEFORE any jarvis import

Requirements:
    - sets DB_PATH (and Chroma path) BEFORE jarvis modules import — Settings
      parses the environment at import time;
    - uses a per-process private temp directory (never the user's jarvis.db,
      never a shared path two runs could collide on);
    - optional cleanup of the temp dir at process exit;
    - idempotent: calling twice is a no-op (the second call cannot retro-
      actively re-bind already-imported settings);
    - FAILS LOUDLY (RuntimeError) if jarvis.config was already imported when
      isolate() is first called AND the DB is not already redirected —
      silently continuing would quietly measure the real database. The one
      safety valve: a harness that ALREADY redirected DB_PATH away from the
      production default before importing jarvis (tests/conftest.py sets
      ':memory:') is already isolated, so evaluation modules can be imported
      under pytest without the raise.

Production DB defaults are NOT modified; tests use tests/conftest.py (which
predates this helper); this module is evaluation-only.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_STATE = {"applied": False, "tmpdir": None}

_ENV_KEYS = ("DB_PATH", "VECTOR_DB_PATH")

# Filenames that mean "the production database". An explicitly-set DB_PATH
# pointing anywhere else is a deliberate redirection by the harness.
_REAL_DB_NAMES = {"jarvis.db"}


def _jarvis_config_imported() -> bool:
    import sys

    return "jarvis.config" in sys.modules


def isolate(*, cleanup_on_exit: bool = True) -> str:
    """
    Point DB_PATH / VECTOR_DB_PATH at a private temp dir BEFORE jarvis imports.

    Returns the temp directory path (also stored in ``_STATE``). Idempotent:
    a second call returns the original directory and changes nothing.

    Raises:
        RuntimeError: if a jarvis module was already imported by this process
            (too late to re-bind Settings) — the caller must fix its import
            order rather than silently measure the real database.
    """
    if _STATE["applied"]:
        return _STATE["tmpdir"]  # type: ignore[return-value]

    if _jarvis_config_imported():
        if _env_db_already_redirected():
            # Safety valve: a harness (tests/conftest.py) already pointed
            # DB_PATH away from the real database BEFORE importing jarvis, so
            # the real DB is out of reach. Keep that environment untouched —
            # failing here would only break legitimate pytest imports of
            # evaluation modules.
            _STATE["applied"] = True
            _STATE["tmpdir"] = None
            return os.environ.get("DB_PATH", "")
        raise RuntimeError(
            "evaluation._bootstrap.isolate() must run BEFORE importing jarvis "
            "modules: jarvis.config already imported, so DB_PATH can no longer "
            "be re-bound and the run would touch the real jarvis.db. Move the "
            "isolate() call above all jarvis imports."
        )

    tmpdir = tempfile.mkdtemp(prefix=f"jarvis_eval_{os.getpid()}_")
    os.environ["DB_PATH"] = os.path.join(tmpdir, "eval_cache.db")
    os.environ.setdefault("VECTOR_DB_PATH", os.path.join(tmpdir, "chroma"))
    # Evaluations never need to reach a real model by accident.
    os.environ.setdefault("OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))

    _STATE["applied"] = True
    _STATE["tmpdir"] = tmpdir

    if cleanup_on_exit:
        atexit.register(_cleanup)

    return tmpdir


def _env_db_already_redirected() -> bool:
    """
    True when DB_PATH is explicitly set AWAY from the production default
    (pytest conftest sets ':memory:'; CI harnesses set temp paths) — i.e. the
    real jarvis.db is already out of reach for this process.
    """
    db = os.environ.get("DB_PATH", "").strip()
    if not db:
        return False
    if db == ":memory:":
        return True
    return os.path.basename(db).lower() not in _REAL_DB_NAMES


def fresh_store():
    """
    A SessionStore on its OWN private database (v0.25 benchmark isolation).

    ``isolate()`` gives the whole PROCESS one throwaway database — enough for
    correctness, but not for per-case dispatch counting: v0.25 added the
    calculator to the cached set (global scope, 7-day TTL), so two benchmark
    cases in one standalone process that evaluate the same expression share a
    cache entry through the shared file DB and the later case records zero
    dispatches (observed: 'calculator repeat with changed numbers' failed
    standalone at 9/11 while pytest stayed green — under pytest each
    SessionStore already gets its own private ':memory:' database).

    fresh_store() re-binds settings.db_path to a per-store file inside the
    evaluation temp dir, constructs the store, and restores the previous
    binding — every case is hermetic no matter which global-scope policies
    exist. Files land inside ``isolate()``'s tmpdir when one exists (cleaned
    by its atexit hook); otherwise a dedicated temp dir is created and
    registered for cleanup itself.
    """
    import sys
    import uuid

    if "jarvis.memory.session_store" not in sys.modules and "jarvis" not in sys.modules:
        raise RuntimeError(
            "fresh_store() requires the jarvis package; call isolate() before "
            "importing jarvis, then construct stores via fresh_store()."
        )
    from jarvis.config import settings
    from jarvis.memory.session_store import SessionStore

    base = _STATE.get("tmpdir")
    if not base or not os.path.isdir(base):
        base = tempfile.mkdtemp(prefix="jarvis_eval_store_")
        atexit.register(_cleanup_dir, base)
    per_store_db = os.path.join(base, f"store_{uuid.uuid4().hex[:12]}.db")

    previous = settings.db_path
    settings.db_path = per_store_db
    try:
        store = SessionStore()
    finally:
        settings.db_path = previous
    return store


def _cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def current_db_path() -> str:
    """The isolated DB path (call after isolate()); '' when not isolated."""
    return os.environ.get("DB_PATH", "")


def is_isolated() -> bool:
    """True when this process runs against an isolated evaluation DB."""
    db = os.environ.get("DB_PATH", "")
    return bool(db) and _STATE["applied"] and "eval_cache.db" in db


def _cleanup() -> None:
    tmpdir = _STATE.get("tmpdir")
    if tmpdir and os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
