"""
tests/conftest.py
──────────────────
Shared pytest fixtures.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Override settings before any jarvis module imports them ───────────────────
# This prevents tests from accidentally touching the real jarvis.db or
# reading a local .env file.
os.environ.setdefault("OLLAMA_MODEL", "test-model")
os.environ.setdefault("DB_PATH", ":memory:")  # SQLite in-memory for tests
os.environ.setdefault("VECTOR_DB_PATH", tempfile.mkdtemp(prefix="jarvis_chroma_test_"))


@pytest.fixture(autouse=True)
def _offline_llm_guard(request):
    """Block real LLM calls in every test not marked ``live_llm``.

    Defense in depth for test isolation: the Planner captures its LLM client
    at Orchestrator construction (BEFORE a per-test patch of
    ``jarvis.core.orchestrator.chat_completion`` applies), so mocking that
    name does NOT mock the planner — a test can silently issue a real
    Ollama request and hang for minutes when the endpoint is slow or wedged
    (observed in practice). The true network boundary is
    ``litellm.completion``, so that is what this fixture refuses: any code
    path that still reaches the real model fails fast with a clear
    diagnosis instead of hanging on the network.

    Tests that genuinely need the local model are marked ``live_llm`` and
    bypass this guard.
    """
    if request.node.get_closest_marker("live_llm"):
        yield
        return
    import litellm

    def _refuse(*args, **kwargs):
        raise RuntimeError(
            "OFFLINE_TEST_VIOLATION: a real LLM call was attempted in an "
            "offline test. The Planner captures its client at construction, "
            "so patch 'jarvis.core.orchestrator.chat_completion' BEFORE "
            "building the runtime/orchestrator, or mark the test with "
            "@pytest.mark.live_llm."
        )

    with patch.object(litellm, "completion", _refuse):
        yield


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live_llm: this test intentionally calls the local Ollama model"
    )


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """A temporary directory that the read_file tool is allowed to access."""
    return tmp_path


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """A small text file inside the allowed temp directory."""
    f = temp_dir / "hello.txt"
    f.write_text("Hello from JARVIS test file!\nLine 2.\n", encoding="utf-8")
    return f
