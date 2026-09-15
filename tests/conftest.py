"""
tests/conftest.py
──────────────────
Shared pytest fixtures.
"""

import os
import tempfile
from pathlib import Path

import pytest

# ── Override settings before any jarvis module imports them ───────────────────
# This prevents tests from accidentally touching the real jarvis.db or
# reading a local .env file.
os.environ.setdefault("OLLAMA_MODEL", "test-model")
os.environ.setdefault("DB_PATH", ":memory:")  # SQLite in-memory for tests
os.environ.setdefault("VECTOR_DB_PATH", tempfile.mkdtemp(prefix="jarvis_chroma_test_"))


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
