"""
tests/test_docker_sandbox.py
────────────────────────────
Offline tests for the real Docker-isolated sandbox. The docker CLI and the
docker daemon are NEVER required: subprocess is mocked so these tests verify
the command construction, the validation ladder, and fail-closed behavior.
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from jarvis.core.sandbox import (
    DockerCodeSandbox,
    DockerUnavailableError,
    ExecutionRequest,
    ExecutionResult,
)
from jarvis.tools.code_execution import CodeExecutionTool


def _ok_proc(stdout: bytes = b"hello\n") -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = b""
    return proc


class TestRequestValidation:
    def test_valid_request_passes(self):
        assert DockerCodeSandbox._validate_request(ExecutionRequest(code="print(1)")) is None

    def test_non_python_denied(self):
        assert (
            DockerCodeSandbox._validate_request(ExecutionRequest(code="x", language="js"))
            == "unsupported_language"
        )

    @pytest.mark.parametrize(
        "req",
        [
            ExecutionRequest(code="   "),
            ExecutionRequest(code="x", timeout_seconds=0),
            ExecutionRequest(code="x", timeout_seconds=31),
            ExecutionRequest(code="x", max_output_bytes=2_000_000),
            ExecutionRequest(code="x", max_memory_mb=4096),
        ],
    )
    def test_out_of_bounds_denied(self, req):
        reason = DockerCodeSandbox._validate_request(req)
        assert reason

    @pytest.mark.parametrize(
        "image",
        ["ubuntu; rm -rf /", "img$(whoami)", "a b", "x`id`", "y|z", "", "a" * 201],
    )
    def test_dangerous_image_refs_rejected(self, image):
        with pytest.raises(ValueError):
            DockerCodeSandbox(image=image)

    def test_image_with_digest_accepted(self):
        ref = "ubuntu:24.04@sha256:" + "a" * 64
        assert DockerCodeSandbox(image=ref).image == ref


class TestRunContainerCommand:
    def test_isolation_flags_present(self, tmp_path):
        sandbox = DockerCodeSandbox()
        script = tmp_path / "script.py"
        script.write_text("print('hi')", encoding="utf-8")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _ok_proc()

        with (
            patch("jarvis.core.sandbox.subprocess.run", side_effect=fake_run),
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch.object(sandbox, "_image_available", return_value=True),
            patch.object(
                DockerCodeSandbox, "_materialize_script", return_value=str(script)
            ),
        ):
            result = sandbox.execute(ExecutionRequest(code="print('hi')", max_memory_mb=128))

        assert result.ok and result.stdout == "hello\n"
        cmd = captured["cmd"]
        flat = " ".join(cmd)
        # Every isolation flag must be in the constructed command.
        for flag in (
            "--rm",
            "--network none",
            "--read-only",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--pids-limit",
        ):
            assert flag in flat, flag
        assert cmd[cmd.index("--memory") + 1] == "128m"
        assert cmd[cmd.index("--memory-swap") + 1] == "128m"
        assert cmd[cmd.index("--user") + 1] == DockerCodeSandbox.CONTAINER_USER
        # Code enters via a read-only mount, never via shell interpolation.
        assert any(":ro" in part for part in cmd)
        assert not captured["kwargs"].get("shell", False)

    def test_output_capped(self, tmp_path):
        sandbox = DockerCodeSandbox()
        script = tmp_path / "script.py"

        big = b"x" * 5000
        with (
            patch("jarvis.core.sandbox.subprocess.run", return_value=_ok_proc(stdout=big)),
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch.object(sandbox, "_image_available", return_value=True),
            patch.object(
                DockerCodeSandbox, "_materialize_script", return_value=str(script)
            ),
        ):
            result = sandbox.execute(
                ExecutionRequest(code="print('hi')", max_output_bytes=1000)
            )
        assert result.truncated
        assert len(result.stdout) <= 1000


class TestFailClosed:
    def test_missing_docker_denied(self, tmp_path):
        sandbox = DockerCodeSandbox()
        script = tmp_path / "s.py"
        with (
            patch("jarvis.core.sandbox.shutil.which", return_value=None),
            patch.object(sandbox, "_image_available", return_value=True),
        ):
            result = sandbox.execute(ExecutionRequest(code="print(1)"))
        assert not result.ok
        assert result.denial_reason == "docker_unavailable: docker CLI not found"

    def test_missing_image_denied(self, tmp_path):
        sandbox = DockerCodeSandbox()
        script = tmp_path / "s.py"
        with (
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch.object(sandbox, "_image_available", return_value=False),
        ):
            result = sandbox.execute(ExecutionRequest(code="print(1)"))
        assert not result.ok
        assert "docker_unavailable" in result.denial_reason
        assert "not available locally" in result.denial_reason

    def test_subprocess_timeout_maps_to_timeout_denial(self, tmp_path):
        """Layer A (host) fires → container force-removed, layer=host_kill."""
        sandbox = DockerCodeSandbox()
        script = tmp_path / "s.py"
        with (
            patch(
                "jarvis.core.sandbox.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=1),
            ),
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch.object(sandbox, "_image_available", return_value=True),
            patch.object(
                DockerCodeSandbox, "_materialize_script", return_value=str(script)
            ),
            patch.object(sandbox, "_force_remove_container") as force_rm,
        ):
            result = sandbox.execute(ExecutionRequest(code="print(1)"))
        assert not result.ok
        assert result.timed_out
        assert result.denial_reason == "timeout"
        assert result.timeout_layer == "host_kill"
        force_rm.assert_called_once()

    def test_unexpected_error_fails_closed(self, tmp_path):
        sandbox = DockerCodeSandbox()
        script = tmp_path / "s.py"
        with (
            patch(
                "jarvis.core.sandbox.subprocess.run",
                side_effect=OSError("disk exploded"),
            ),
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch.object(sandbox, "_image_available", return_value=True),
            patch.object(
                DockerCodeSandbox, "_materialize_script", return_value=str(script)
            ),
        ):
            result = sandbox.execute(ExecutionRequest(code="print(1)"))
        assert not result.ok
        assert result.denial_reason.startswith("sandbox_error:")

    def test_execute_never_raises(self):
        """Contract: expected failures return a result, never raise."""
        sandbox = DockerCodeSandbox()
        with patch("jarvis.core.sandbox.shutil.which", return_value=None):
            for code in ("", "print(1)", "import os"):
                r = sandbox.execute(ExecutionRequest(code=code or " "))
                assert isinstance(r, ExecutionResult)


class TestAvailability:
    def test_unavailable_on_windows_host(self):
        sandbox = DockerCodeSandbox()
        with patch("jarvis.core.sandbox.sys") as fake_sys:
            fake_sys.platform = "win32"
            assert sandbox.is_available() is False

    def test_unavailable_without_cli(self):
        sandbox = DockerCodeSandbox()
        with (
            patch("jarvis.core.sandbox.sys") as fake_sys,
            patch("jarvis.core.sandbox.shutil.which", return_value=None),
        ):
            fake_sys.platform = "linux"
            assert sandbox.is_available() is False

    def test_unavailable_when_daemon_down(self):
        sandbox = DockerCodeSandbox()
        proc = MagicMock()
        proc.returncode = 1
        with (
            patch("jarvis.core.sandbox.sys") as fake_sys,
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch("jarvis.core.sandbox.subprocess.run", return_value=proc),
        ):
            fake_sys.platform = "linux"
            assert sandbox.is_available() is False

    def test_available_when_all_checks_pass(self):
        sandbox = DockerCodeSandbox()
        proc = MagicMock()
        proc.returncode = 0
        with (
            patch("jarvis.core.sandbox.sys") as fake_sys,
            patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch("jarvis.core.sandbox.subprocess.run", return_value=proc),
            patch.object(sandbox, "_image_available", return_value=True),
        ):
            fake_sys.platform = "linux"
            assert sandbox.is_available() is True


class TestToolGate:
    def test_unavailable_isolation_sandbox_degrades_to_disabled(self):
        """provides_isolation=True but docker absent → sandbox swapped to Disabled."""
        sandbox = DockerCodeSandbox()
        assert sandbox.provides_isolation is True
        with patch.object(sandbox, "is_available", return_value=False):
            tool = CodeExecutionTool(sandbox=sandbox)
        from jarvis.core.sandbox import DisabledSandbox

        assert isinstance(tool.sandbox, DisabledSandbox)
        assert "ERROR" in tool.run(code="print(1)")

    def test_available_isolation_sandbox_stays_live(self):
        sandbox = DockerCodeSandbox()
        with patch.object(sandbox, "is_available", return_value=True):
            tool = CodeExecutionTool(sandbox=sandbox)
        assert tool.sandbox is sandbox

    def test_default_tool_remains_disabled(self):
        tool = CodeExecutionTool()
        from jarvis.core.sandbox import DisabledSandbox

        assert isinstance(tool.sandbox, DisabledSandbox)

    def test_tool_report_shape_from_sandbox_result(self):
        """The live path returns the sandbox's to_report() string."""
        sandbox = DockerCodeSandbox()
        with patch.object(sandbox, "is_available", return_value=True):
            tool = CodeExecutionTool(sandbox=sandbox)
        fake_result = ExecutionResult(ok=True, stdout="4\n", exit_code=0)
        with patch.object(sandbox, "execute", return_value=fake_result):
            out = tool.run(code="print(2+2)")
        assert "exit=0" in out and "4" in out


class TestRuntimeWiring:
    def test_tool_absent_when_disabled_by_config(self):
        from unittest.mock import patch as p

        from jarvis.runtime import _build_code_execution_tool

        with p("jarvis.config.settings.ENABLE_CODE_EXECUTION", False):
            assert _build_code_execution_tool() is None

    def test_tool_absent_when_enabled_but_docker_unavailable(self):
        from unittest.mock import patch as p

        from jarvis.runtime import _build_code_execution_tool

        with (
            p("jarvis.config.settings.ENABLE_CODE_EXECUTION", True),
            p("jarvis.runtime.DockerCodeSandbox") as mock_cls,
        ):
            mock_cls.return_value.is_available.return_value = False
            assert _build_code_execution_tool() is None

    def test_tool_registered_when_enabled_and_available(self):
        from unittest.mock import patch as p

        from jarvis.runtime import _build_code_execution_tool

        with (
            p("jarvis.config.settings.ENABLE_CODE_EXECUTION", True),
            p("jarvis.runtime.DockerCodeSandbox") as mock_cls,
        ):
            mock_cls.return_value.is_available.return_value = True
            mock_cls.return_value.provides_isolation = True
            tool = _build_code_execution_tool()
        assert tool is not None
        assert tool.sandbox.provides_isolation is True

    def test_standard_runtime_never_has_the_tool(self):
        from unittest.mock import patch as p

        from jarvis.runtime import build_runtime

        with p("jarvis.runtime.get_vector_store"):
            rt = build_runtime()
        assert "execute_python_code" not in rt.registry.list_tools()
        assert "computer_control" not in rt.registry.list_tools()
        rt.close()
