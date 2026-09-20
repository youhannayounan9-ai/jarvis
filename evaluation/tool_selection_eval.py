import sys
from unittest.mock import patch, MagicMock
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.core.permissions import PermissionGuard

def run_evaluation():
    queries = [
        ("What time is it?", "get_current_datetime", "{}"),
        ("Calculate 25 * 4", "calculator", '{"expression": "25 * 4"}'),
        ("Search for local AI news", "web_search", '{"query": "local AI news"}'),
        ("Remember my name is Alex", "remember_fact", '{"fact": "My name is Alex"}')
    ]
    
    print(f"{'Query':<30} | {'Expected Tool':<20} | {'Actual Tool':<20} | Status")
    print("-" * 85)
    
    all_passed = True
    
    for query, expected_tool, expected_args in queries:
        store = SessionStore()
        registry = ToolRegistry()
        guard = PermissionGuard()
        orchestrator = Orchestrator(store, registry, guard)
        
        # We need to mock chat_completion to return the expected tool
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
                    
            tc = ToolCall("call_1", Function(expected_tool, expected_args))
            return Response([Choice(Message("assistant", None, [tc]))])
            
        with patch("jarvis.core.orchestrator.chat_completion", side_effect=mock_chat_completion):
            with patch.object(registry, 'dispatch', return_value="success") as mock_dispatch:
                with patch.object(guard, 'require_confirmation', return_value=False):
                    with patch.object(guard, 'is_allowed', return_value=True):
                        try:
                            with patch.object(orchestrator, 'route_intent', return_value="simple"):
                                orchestrator.chat("session_1", query)
                            
                            if mock_dispatch.called:
                                actual_tool = mock_dispatch.call_args[0][0]
                                actual_args = mock_dispatch.call_args[0][1]
                            else:
                                actual_tool = "None"
                                
                            passed = (actual_tool == expected_tool)
                            status = "✅" if passed else "❌"
                            if not passed: all_passed = False
                            
                            print(f"{query[:30]:<30} | {expected_tool:<20} | {actual_tool:<20} | {status}")
                        except Exception as e:
                            print(f"{query[:30]:<30} | {expected_tool:<20} | ERROR                | ❌")
                            all_passed = False

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()
