"""
tests/test_permissions.py
─────────────────────────
Tests for the PermissionGuard and risk level logic.
"""

from jarvis.core.permissions import PermissionGuard


class TestPermissionGuard:
    def setup_method(self):
        self.guard = PermissionGuard()
        self.guard.register_tool_risk("safe_tool", "SAFE")
        self.guard.register_tool_risk("read_tool", "FILE_READ")
        self.guard.register_tool_risk("system_tool", "SYSTEM")
        self.guard.register_tool_risk("destructive_tool", "DESTRUCTIVE")

    def test_get_tool_risk_level(self):
        assert self.guard.get_tool_risk_level("safe_tool") == "SAFE"
        assert self.guard.get_tool_risk_level("unknown_tool") == "UNKNOWN"

    def test_require_confirmation(self):
        # Only SYSTEM and DESTRUCTIVE require confirmation
        assert self.guard.require_confirmation("safe_tool", "SAFE") is False
        assert self.guard.require_confirmation("read_tool", "FILE_READ") is False
        assert self.guard.require_confirmation("system_tool", "SYSTEM") is True
        assert self.guard.require_confirmation("destructive_tool", "DESTRUCTIVE") is True
