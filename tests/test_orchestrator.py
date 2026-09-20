import pytest
from unittest.mock import patch, MagicMock

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.core.permissions import PermissionGuard
from jarvis.config import settings

def test_high_risk_tool_requires_confirmation():
    store = SessionStore()
    registry = ToolRegistry()
    guard = PermissionGuard()
    
    orchestrator = Orchestrator(store, registry, guard)
    
    # We mock the chat_completion to always request a specific tool
    def mock_chat_completion(messages, tools=None, **kwargs):
        class Function:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments
        class ToolCall:
            def __init__(self, id, function):
                self.id = id
                self.function = function
        class Message:
            def __init__(self, role, content, tool_calls):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
        class Choice:
            def __init__(self, message):
                self.message = message
        class Response:
            def __init__(self, choices):
                self.choices = choices
                
        tc = ToolCall("call_test123", Function("high_risk_tool", "{}"))
        return Response([Choice(Message("assistant", None, [tc]))])

    with patch("jarvis.core.orchestrator.chat_completion", side_effect=mock_chat_completion):
        with patch.object(registry, "get_tool_risk_level", return_value="SYSTEM"):
            with patch.object(guard, "require_confirmation", return_value=True):
                # Ensure confirmation is enabled in settings
                with patch.object(settings, "REQUIRE_CONFIRMATION_FOR_HIGH_RISK", True, create=True):
                    # Use a simple prompt to bypass the planner and test the permission check directly
                    with patch.object(orchestrator, "route_intent", return_value="simple"):
                        response = orchestrator.chat("session_test", "Run high risk tool")
                        
                        pending = orchestrator.get_pending_confirmation("session_test")
                        assert pending is not None
                        assert pending["tool_name"] == "high_risk_tool"
                        assert pending["risk_level"] == "SYSTEM"
