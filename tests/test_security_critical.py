"""
Tests to prove that dangerous capabilities (code execution and computer control)
are strictly disabled by default and cannot be bypassed.
"""

from unittest.mock import patch, MagicMock
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.session_store import SessionStore
from jarvis.tools.registry import ToolRegistry
from jarvis.core.permissions import PermissionGuard
from jarvis.tools.code_execution import CodeExecutionTool
from jarvis.tools.computer_control import ComputerControlTool
from jarvis.core.sandbox import DockerCodeSandbox, DisabledSandbox


def test_code_execution_disabled_by_default():
    """Prove that execute_python_code uses DisabledSandbox by default and blocks execution."""
    tool = CodeExecutionTool()
    assert isinstance(tool.sandbox, DisabledSandbox)
    
    result = tool.run(code="import os; os.system('echo hacked')")
    assert "ERROR" in result
    assert "disabled" in result.lower()
    assert "not an isolated sandbox" in result.lower()


def test_docker_sandbox_unavailable_blocks_host_execution():
    """
    On a host without usable Docker, an isolation-asserting DockerCodeSandbox
    must still be unable to run code: the tool DEGRADES it to DisabledSandbox
    at construction (fail closed), and direct sandbox calls return denials —
    never host execution.
    """
    sandbox = DockerCodeSandbox()
    with patch.object(sandbox, "is_available", return_value=False):
        tool = CodeExecutionTool(sandbox=sandbox)

    from jarvis.core.sandbox import DisabledSandbox

    # Degraded at construction — no isolation, no execution path at all.
    assert isinstance(tool.sandbox, DisabledSandbox)
    result = tool.run(code="print('hello')")
    assert "ERROR" in result
    assert "not an isolated sandbox" in result.lower()

    # The sandbox itself fails closed if asked directly without Docker.
    from jarvis.core.sandbox import ExecutionRequest

    with patch("jarvis.core.sandbox.shutil.which", return_value=None):
        exec_result = sandbox.execute(ExecutionRequest(code="print('hello')"))
    assert not exec_result.ok
    assert exec_result.denial_reason and "docker_unavailable" in exec_result.denial_reason


def test_docker_sandbox_validates_requests():
    """The Docker skeleton validates requests and image refs before any execution."""
    from jarvis.core.sandbox import DockerCodeSandbox, ExecutionRequest

    # Image validation: forbidden shell characters rejected
    import pytest
    with pytest.raises(ValueError):
        DockerCodeSandbox(image="ubuntu; rm -rf /")
    assert DockerCodeSandbox(image="ubuntu:24.04").image == "ubuntu:24.04"

    sandbox = DockerCodeSandbox()
    # Contract violations are denied by reason, before reaching the container.
    bad_requests = [
        ExecutionRequest(code="x", language="javascript"),
        ExecutionRequest(code="   "),
        ExecutionRequest(code="x", timeout_seconds=0),
        ExecutionRequest(code="x", timeout_seconds=99),
        ExecutionRequest(code="x", max_memory_mb=10_000),
    ]
    for req in bad_requests:
        result = sandbox.execute(req)
        assert not result.ok and result.denial_reason


def test_execution_result_report_format():
    """ExecutionResult.to_report() produces LLM-readable output."""
    from jarvis.core.sandbox import ExecutionResult

    denied = ExecutionResult(ok=False, denial_reason="disabled").to_report()
    assert "ERROR" in denied and "disabled" in denied

    ok = ExecutionResult(ok=True, stdout="4\n", exit_code=0).to_report()
    assert "4" in ok and "exit=0" in ok

    timeout = ExecutionResult(ok=False, timed_out=True, denial_reason="timeout").to_report()
    assert "timeout" in timeout.lower()


def test_computer_control_requires_confirmation_if_registered():
    """Prove that if ComputerControl is accidentally registered, it demands confirmation."""
    store = SessionStore()
    registry = ToolRegistry()
    guard = PermissionGuard()
    
    # Force REQUIRE_CONFIRMATION_FOR_HIGH_RISK = True
    with patch("jarvis.config.settings.REQUIRE_CONFIRMATION_FOR_HIGH_RISK", True):
        # Maliciously register the tool
        registry.register(ComputerControlTool())
        orchestrator = Orchestrator(store, registry, guard)
        
        # Dispatch the tool asynchronously
        import asyncio
        result = asyncio.run(orchestrator._dispatch_with_permissions_async(
            session_id="test_session",
            tool_name="computer_control",
            tool_args='{"action": "click", "x": 100, "y": 100}',
            tool_call_id="call_123"
        ))
        
        # Prove the orchestrator caught it BEFORE execution
        assert "ACTION_REQUIRES_CONFIRMATION" in result
        
        # Prove the pending action was stored
        pending = store.load_pending_confirmation("test_session")
        assert pending is not None
        assert pending["tool_name"] == "computer_control"


def test_computer_control_disabled_after_confirmation():
    """Prove that even if a user explicitly confirms, the computer control tool itself blocks execution."""
    tool = ComputerControlTool()
    
    # Assume the orchestrator called run() because the user clicked 'Confirm'
    result = tool.run(action="click", x=100, y=100)
    
    assert "ERROR" in result
    assert "disabled" in result.lower()
    assert "sandboxing strategy" in result.lower()
