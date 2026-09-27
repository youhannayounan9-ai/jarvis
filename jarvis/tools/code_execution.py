"""
jarvis/tools/code_execution.py
──────────────────────────────
Tool: execute_python_code — gated, defense-in-depth code execution.

Registration model (see jarvis/runtime.py):
  - Not registered unless explicitly enabled via config (ENABLE_CODE_EXECUTION)
    AND a verified-isolation sandbox is available.
  -computer_control remains disabled unconditionally.

Layered gates before any code runs:
  1. Registry:      the tool is absent from the active surface by default.
  2. Sandbox gate:  only a sandbox whose ``provides_isolation`` is True may
                    execute — and it must ALSO prove availability at
                    construction (an unreachable Docker daemon degrades the
                    sandbox to DisabledSandbox, never to host execution).
  3. Sandbox:       DockerCodeSandbox re-validates everything per run and
                    fails closed on any uncertainty.
"""

from typing import Any

from jarvis.core.sandbox import CodeSandbox, DisabledSandbox
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class CodeExecutionTool(BaseTool):
    name = "execute_python_code"
    description = (
        "Execute Python code in an isolated, resource-limited container "
        "(no network, read-only filesystem). Disabled unless verified "
        "container isolation is available."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }
    risk_level = "SYSTEM"
    timeout_seconds = 5.0

    def __init__(self, sandbox: CodeSandbox | None = None):
        super().__init__()
        if (
            sandbox is not None
            and getattr(sandbox, "provides_isolation", False)
            and hasattr(sandbox, "is_available")
            and not sandbox.is_available()
        ):
            # An isolation-asserting sandbox that cannot actually run (no
            # docker, daemon down, image missing) must degrade to the disabled
            # sandbox — never to host execution. Fail closed.
            log.error(
                "isolation_sandbox_unavailable_degrading_to_disabled",
                sandbox=type(sandbox).__name__,
            )
            sandbox = DisabledSandbox()
        self.sandbox = sandbox or DisabledSandbox()

    def run(self, code: str, **kwargs: Any) -> str:
        log.warning("code_execution_attempted", sandbox=type(self.sandbox).__name__)
        if not getattr(self.sandbox, "provides_isolation", False):
            return (
                "ERROR: Code execution is currently disabled because the current "
                "implementation is not an isolated sandbox."
            )
        # Live path: only a verified, available, isolated sandbox reaches here.
        return self.sandbox.execute_code(code, timeout_seconds=self.timeout_seconds)
