"""
tests/test_sandbox_timeout_layers.py
────────────────────────────────────
v0.15: the sandbox distinguishes three timeout layers.

  A. host process timeout  — the docker CLI client stops being waited on
  B. container timeout     — coreutils `timeout` kills the WORKLOAD (exit 124)
  C. actual termination    — layer B kills the workload; layer A additionally
                             force-removes the container (docker rm -f)

All offline: subprocess is mocked, docker is never required.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from jarvis.core.sandbox import DockerCodeSandbox, ExecutionRequest


def _proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


@pytest.fixture()
def sandbox():
    sb = DockerCodeSandbox()
    with (
        patch("jarvis.core.sandbox.shutil.which", return_value="/usr/bin/docker"),
        patch.object(sb, "_image_available", return_value=True),
        patch.object(DockerCodeSandbox, "_materialize_script", return_value="/tmp/sbx.py"),
    ):
        yield sb


class TestContainerTimeoutLayer:
    def test_workload_wrapped_in_container_timeout(self, sandbox):
        """The image entrypoint must be `timeout <cap> python3 script.py`."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _proc(returncode=0, stdout=b"done\n")

        with patch("jarvis.core.sandbox.subprocess.run", side_effect=fake_run):
            result = sandbox.execute(ExecutionRequest(code="print('hi')", timeout_seconds=7))

        assert result.ok
        cmd = captured["cmd"]
        # Entry point shape: [ ..., "timeout", "7s", "python3", "/sandbox/script.py" ]
        assert cmd[-4] == DockerCodeSandbox.CONTAINER_TIMEOUT_CMD
        assert cmd[-3] == "7s"
        assert cmd[-2] == "python3"

    def test_exit_124_is_container_timeout(self, sandbox):
        """Exit 124 = the WORKLOAD was killed at the container boundary."""
        with patch(
            "jarvis.core.sandbox.subprocess.run",
            return_value=_proc(returncode=124, stderr=b"Terminated"),
        ):
            result = sandbox.execute(
                ExecutionRequest(code="while True: pass", timeout_seconds=5)
            )
        assert not result.ok
        assert result.timed_out is True
        assert result.timeout_layer == "container"
        assert result.denial_reason == "timeout"
        # A container-killed workload reports a timeout denial.
        assert result.to_report().startswith("ERROR: Code execution denied (timeout)")

    def test_normal_exit_is_not_a_timeout(self, sandbox):
        with patch(
            "jarvis.core.sandbox.subprocess.run",
            return_value=_proc(returncode=0, stdout=b"4\n"),
        ):
            result = sandbox.execute(ExecutionRequest(code="print(2+2)"))
        assert result.ok
        assert result.timed_out is False
        assert result.timeout_layer is None
        assert result.denial_reason is None

    def test_timeout_value_reaches_container_wrapper(self, sandbox):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _proc(returncode=0, stdout=b"x")

        with patch("jarvis.core.sandbox.subprocess.run", side_effect=fake_run):
            sandbox.execute(ExecutionRequest(code="print(1)", timeout_seconds=12.5))

        assert captured["cmd"][-3] == "12.5s"


class TestHostTimeoutLayer:
    def test_host_timeout_force_removes_container(self, sandbox):
        """Layer A must actively terminate (rm -f), not just stop waiting."""
        with patch(
            "jarvis.core.sandbox.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=3),
        ):
            with patch.object(sandbox, "_force_remove_container") as force_rm:
                result = sandbox.execute(ExecutionRequest(code="print(1)", timeout_seconds=1))
        assert not result.ok
        assert result.timed_out is True
        assert result.timeout_layer == "host_kill"
        assert result.denial_reason == "timeout"
        # v0.17: layer C must target THIS run's container by name.
        force_rm.assert_called_once_with(result.container_name)

    def test_force_remove_uses_docker_rm_f(self):
        """Layer C termination mechanism: `docker rm -f <per-run name>`."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _proc(returncode=0)

        with (
            patch("jarvis.core.sandbox.shutil.which", return_value="docker"),
            patch("jarvis.core.sandbox.subprocess.run", side_effect=fake_run),
        ):
            DockerCodeSandbox()._force_remove_container("jarvis-sbx-deadbeef1234")

        assert calls and calls[0][:3] == ["docker", "rm", "-f"]
        assert calls[0][3] == "jarvis-sbx-deadbeef1234"

    def test_force_remove_failure_does_not_raise(self):
        """Layer C is best-effort; its failure must not mask the denial."""
        with (
            patch("jarvis.core.sandbox.shutil.which", return_value="docker"),
            patch(
                "jarvis.core.sandbox.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=5),
            ),
        ):
            DockerCodeSandbox()._force_remove_container("jarvis-sbx-whatever")  # must not raise

    def test_host_deadline_exceeds_container_deadline(self):
        """Layer B should normally report first: host wait = cap + slack."""
        assert DockerCodeSandbox.HOST_TIMEOUT_SLACK_SECONDS > 0

    def test_layer_c_reachable_end_to_end(self):
        """Full flow: run raises host-timeout → rm -f attempted → denial."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            raise subprocess.TimeoutExpired(cmd=" ".join(cmd[:2]), timeout=2)

        with (
            patch("jarvis.core.sandbox.shutil.which", return_value="docker"),
            patch("jarvis.core.sandbox.subprocess.run", side_effect=fake_run),
            patch.object(DockerCodeSandbox, "_image_available", return_value=True),
            patch.object(DockerCodeSandbox, "_materialize_script", return_value="/tmp/s.py"),
        ):
            result = DockerCodeSandbox().execute(
                ExecutionRequest(code="print(1)", timeout_seconds=1)
            )

        assert result.timeout_layer == "host_kill"
        # First call = the run, second = the layer-C rm -f.
        assert any("rm" in c for c in calls[1:][:1])


class TestSecurityFlagsPreservedWithTimeout:
    def test_all_isolation_flags_survive_timeout_wrapper(self, sandbox):
        """Adding the timeout wrapper must not disturb any isolation flag."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _proc(returncode=0, stdout=b"ok")

        with patch("jarvis.core.sandbox.subprocess.run", side_effect=fake_run):
            sandbox.execute(ExecutionRequest(code="print(1)", max_memory_mb=128))

        flat = " ".join(captured["cmd"])
        for flag in (
            "--rm",
            "--network none",
            "--read-only",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--pids-limit",
        ):
            assert flag in flat, flag
        cmd = captured["cmd"]
        assert cmd[cmd.index("--memory") + 1] == "128m"
        assert cmd[cmd.index("--memory-swap") + 1] == "128m"
        assert cmd[cmd.index("--user") + 1] == DockerCodeSandbox.CONTAINER_USER
        assert any(":ro" in part for part in cmd)

    def test_timeout_cap_still_validated_before_container(self):
        """Contract validation is unchanged: out-of-range timeouts deny."""
        reason = DockerCodeSandbox._validate_request(
            ExecutionRequest(code="x", timeout_seconds=999)
        )
        assert reason == "timeout_out_of_range"
