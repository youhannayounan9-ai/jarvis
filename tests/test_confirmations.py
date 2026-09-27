"""
tests/test_confirmations.py
───────────────────────────
Tests for durable SQLite-based confirmation state.
"""

import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from jarvis.memory.session_store import SessionStore
from jarvis.core.orchestrator import Orchestrator
from jarvis.tools.registry import ToolRegistry
from jarvis.core.permissions import PermissionGuard
from jarvis.config import settings

@pytest.fixture
def store():
    # Use memory database but we'll share the connection if needed for "restarts"
    store = SessionStore()
    # Explicitly clear tables if reusing
    store._conn.execute("DELETE FROM pending_confirmations")
    store._conn.execute("DELETE FROM messages")
    store._conn.execute("DELETE FROM sessions")
    store._conn.commit()
    yield store
    store.close()

def test_confirmation_survives_restart(store):
    """Verify that a pending confirmation can be loaded even if the orchestrator/store is recreated."""
    session_id = "session_restart_test"
    store._conn.execute("INSERT INTO sessions (id, created_at) VALUES (?, ?)", (session_id, datetime.now(timezone.utc).isoformat()))
    store._conn.commit()
    
    # Save a confirmation
    store.save_pending_confirmation(
        session_id=session_id,
        tool_name="nuclear_launch",
        tool_args='{"target": "mars"}',
        tool_call_id="call_999",
        risk_level="SYSTEM"
    )
    
    # Simulate restart by creating a new store instance pointing to the same DB connection
    # Note: In-memory DBs are per-connection, so we'll just use the same connection to mock persistent file DB
    store2 = SessionStore()
    store2._conn = store._conn 
    
    loaded = store2.load_pending_confirmation(session_id)
    assert loaded is not None
    assert loaded["tool_name"] == "nuclear_launch"
    assert loaded["risk_level"] == "SYSTEM"

def test_confirmation_expires(store):
    """Verify that an expired confirmation returns None and is deleted."""
    session_id = "session_expiry_test"
    
    # Save a confirmation but force its expiry time to be in the past
    store.save_pending_confirmation(
        session_id=session_id,
        tool_name="expired_tool",
        tool_args="{}",
        tool_call_id="call_000",
        risk_level="HIGH"
    )
    
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store._conn.execute("UPDATE pending_confirmations SET expires_at = ? WHERE session_id = ?", (past, session_id))
    store._conn.commit()
    
    # Loading should now fail and clean it up
    loaded = store.load_pending_confirmation(session_id)
    assert loaded is None
    
    # Check it's gone from DB
    row = store._conn.execute("SELECT * FROM pending_confirmations WHERE session_id = ?", (session_id,)).fetchone()
    assert row is None

def test_complete_confirmation(store):
    """Verify that completing a confirmation works and prevents it from being loaded again."""
    session_id = "session_complete_test"
    
    store.save_pending_confirmation(
        session_id=session_id,
        tool_name="tool_a",
        tool_args="{}",
        tool_call_id="call_1",
        risk_level="HIGH"
    )
    
    # Complete it
    data = store.complete_pending_confirmation(session_id)
    assert data is not None
    assert data["tool_name"] == "tool_a"
    
    # Try loading again -> should be None
    loaded = store.load_pending_confirmation(session_id)
    assert loaded is None

def test_cleanup_expired(store):
    """Verify that the cleanup method removes expired and completed rows."""
    # 1. Active row
    store.save_pending_confirmation("sess_active", "tool_active", "{}", "c1", "SAFE")
    
    # 2. Expired row
    store.save_pending_confirmation("sess_expired", "tool_expired", "{}", "c2", "SAFE")
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store._conn.execute("UPDATE pending_confirmations SET expires_at = ? WHERE session_id = ?", (past, "sess_expired"))
    
    # 3. Completed row
    store.save_pending_confirmation("sess_completed", "tool_completed", "{}", "c3", "SAFE")
    store.complete_pending_confirmation("sess_completed")
    
    store._conn.commit()
    
    # Run cleanup
    removed = store.cleanup_expired_confirmations()
    assert removed == 2 # The expired and completed one
    
    # Active one is still there
    assert store.load_pending_confirmation("sess_active") is not None
