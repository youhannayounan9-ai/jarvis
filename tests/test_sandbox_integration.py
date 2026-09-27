"""
tests/test_sandbox_integration.py
─────────────────────────────────
REAL Docker integration tests (v0.16): these tests assert OBSERVED container
behavior against a real Linux Docker engine — not that the CLI arguments
contain expected strings (that is the unit suite's job).

They are the "we observed the security boundary working" layer:

  - builds deploy/Dockerfile.sandbox (the production image, digest-pinned base)
  - verifies isolation boundaries from INSIDE the container (uid, caps, fs)
  - verifies network / rootfs / tmpfs / output-cap behavior with real payloads
  - verifies the workload-killing timeout (`timeout` wrapper, exit 124) and
    that no orphan container survives it
  - verifies resource ceilings (memory, pids) reject exhaustion attempts
  - verifies DockerCodeSandbox.execute() end-to-end incl. the host layer

Every test SKIPS (with a clear reason) when prerequisites are missing:
  - docker CLI absent
  - daemon unreachable
  - daemon is not a Linux engine (e.g. Windows containers)
  - deploy/Dockerfile.sandbox still pins the placeholder digest

These tests never run on Windows-container engines; the sandbox itself fails
closed there (see jarvis/core/sandbox.py is_available()).
"""

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from jarvis.core.sandbox import DockerCodeSandbox, ExecutionRequest

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
DOCKERFILE = DEPLOY_DIR / "Dockerfile.sandbox"
PLACEHOLDER = "REPLACE_WITH_VERIFIED_DIGEST"
IMAGE_TAG = "jarvis-sandbox:it"


def _docker_available() -> tuple[bool, str]:
    """Return (ok, reason) for the real-Docker integration prerequisites."""
    if shutil.which("docker") is None:
        return False, "docker CLI not found"
    try:
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"docker daemon unreachable: {e}"
    if probe.returncode != 0:
        return False, "docker daemon not reachable"
    ostype = probe.stdout.decode(errors="replace").strip()
    if ostype != "linux":
        return False, f"daemon is not a Linux engine (OSType={ostype!r})"
    return True, ""


def _digest_pinned() -> bool:
    try:
        return PLACEHOLDER not in DOCKERFILE.read_text(encoding="utf-8")
    except OSError:
        return False


_DOCKER_OK, _DOCKER_REASON = _docker_available()
_DIGEST_OK = _digest_pinned()
_SKIP_REASON = (
    f"integration prerequisites unavailable: {_DOCKER_REASON}"
    if not _DOCKER_OK
    else None
)
_SKIP_REASON = (
    _SKIP_REASON
    or (
        "deploy/Dockerfile.sandbox still pins the placeholder digest "
        "(fail-closed state); integration tests cannot build the image"
        if not _DIGEST_OK
        else None
    )
)

pytestmark = pytest.mark.skipif(
    not (_DOCKER_OK and _DIGEST_OK), reason=_SKIP_REASON or "integration prerequisites unavailable"
)

# Per-test cap used throughout; keeps the suite bounded on any machine.
_TIMEOUT = 20.0


def _run(container_args: list[str], code: str, timeout: float = _TIMEOUT) -> subprocess.CompletedProcess:
    """Run one snippet in a container with the production isolation flags."""
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:noexec,nosuid,nodev,size=16m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--memory", "256m",
        "--memory-swap", "256m",
        "--cpus", "0.5",
        "--pids-limit", "64",
        "-v", f"{DEPLOY_DIR}:/sandbox/deploy:ro",
        "-w", "/sandbox",
    ] + container_args + [IMAGE_TAG, "python3", "-c", code]
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def _sbx_run(code: str, timeout_seconds: float = 5.0, **kwargs) -> "object":
    """Run a snippet through the real DockerCodeSandbox (end-to-end)."""
    sandbox = DockerCodeSandbox(image=IMAGE_TAG)
    request = ExecutionRequest(code=code, timeout_seconds=timeout_seconds, **kwargs)
    return sandbox.execute(request)


@pytest.fixture(scope="module")
def built_image():
    """Build the production sandbox image once per session (module scope)."""
    result = subprocess.run(
        [
            "docker", "build", "-t", IMAGE_TAG,
            "-f", str(DOCKERFILE), str(DEPLOY_DIR),
        ],
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
    return IMAGE_TAG


# ── Identity / isolation observed from inside ─────────────────────────────────


class TestObservedIsolation:
    def test_container_starts_and_python_executes(self, built_image):
        r = _run([], "print('sandbox-alive')")
        assert r.returncode == 0
        assert "sandbox-alive" in r.stdout.decode()

    def test_runs_as_non_root_nobody_uid(self, built_image):
        r = _run([], "import os; print(os.getuid(), os.geteuid())")
        assert r.returncode == 0
        assert r.stdout.decode().strip() == "65534 65534"

    def test_all_capabilities_dropped(self, built_image):
        """With CAP_CHOWN dropped, chown(0,0) must fail with EPERM/EACCES."""
        r = _run([], "import os;\ntry:\n    os.chown('/tmp', 0, 0)\n    print('CHOWN-OK')\nexcept (PermissionError, OSError):\n    print('CHOWN-BLOCKED')\n")
        assert r.returncode == 0
        assert "CHOWN-BLOCKED" in r.stdout.decode()

    def test_no_privilege_escalation_possible(self, built_image):
        """The process cannot elevate itself: setuid(0) must be refused."""
        r = _run([], "import os;\ntry:\n    os.setuid(0)\n    print('SETUID-OK uid=%d' % os.getuid())\nexcept (PermissionError, OSError):\n    print('SETUID-BLOCKED uid=%d' % os.getuid())\n")
        assert r.returncode == 0
        out = r.stdout.decode()
        assert "SETUID-BLOCKED" in out
        assert "uid=65534" in out
        assert "SETUID-OK" not in out

    def test_root_filesystem_read_only(self, built_image):
        r = _run([], "open('/etc/passwd','a').write('x')")
        assert r.returncode != 0

    def test_tmp_writable_but_noexec(self, built_image):
        r = _run([], "open('/tmp/x','w').write('1'); print(open('/tmp/x').read())")
        assert r.returncode == 0 and "1" in r.stdout.decode()
        # noexec: executing a binary copied into /tmp must be refused. The
        # exec fails at the kernel level (EACCES → PermissionError in the
        # parent) or the copy cannot even materialize (16m tmpfs cap).
        code = (
            "import shutil, subprocess, sys\n"
            "try:\n"
            "    shutil.copy(sys.executable, '/tmp/py')\n"
            "except OSError:\n"
            "    print('COPY-BLOCKED')\n"
            "else:\n"
            "    try:\n"
            "        subprocess.run(['/tmp/py', '--version'], capture_output=True)\n"
            "        print('EXEC-SUCCEEDED')\n"
            "    except (PermissionError, OSError):\n"
            "        print('EXEC-BLOCKED')\n"
        )
        r2 = _run([], code)
        assert r2.returncode == 0
        out = r2.stdout.decode()
        assert "EXEC-SUCCEEDED" not in out, "/tmp must be noexec"
        assert ("EXEC-BLOCKED" in out) or ("COPY-BLOCKED" in out)

    def test_network_unavailable(self, built_image):
        # Sockets cannot even be created to a network namespace that has none.
        r = _run([], "import socket\ns=socket.socket(); s.settimeout(3)\ntry:\n    s.connect(('1.1.1.1', 80))\n    print('NETWORK-REACHABLE')\nexcept OSError as e:\n    print('BLOCKED')\n")
        assert r.returncode == 0
        assert "BLOCKED" in r.stdout.decode()
        assert "NETWORK-REACHABLE" not in r.stdout.decode()


# ── Escape attempts (realistic malicious payloads) ────────────────────────────


class TestEscapeAttempts:
    def test_cannot_read_host_proc(self, built_image):
        # /proc inside the container is the container's own, not the host's.
        r = _run([], "import pathlib; print(len(list(pathlib.Path('/proc').glob('*/cmdline'))))")
        assert r.returncode == 0
        n = int(r.stdout.decode().strip() or 0)
        assert n < 15, "container /proc should show only its own tiny PID space"

    def test_cannot_write_outside_tmp(self, built_image):
        for target in ("/etc", "/usr", "/sandbox", "/home"):
            r = _run([], f"open({target!r} + '/probe.txt', 'w')")
            assert r.returncode != 0, f"{target} must not be writable"

    def test_cannot_modify_mounted_code(self, built_image):
        # deploy/ is mounted :ro — writing to it must fail.
        r = _run([], "open('/sandbox/deploy/Dockerfile.sandbox','a').write('x')")
        assert r.returncode != 0

    def test_mounted_ro_directory_traversal_rejected(self, built_image):
        # A traversal out of the mount point must not grant write access.
        r = _run([], "open('/sandbox/deploy/../etc/probe.txt','w')")
        assert r.returncode != 0


# ── Timeout: the workload is actually killed ─────────────────────────────────


class TestObservedTimeout:
    def test_container_timeout_kills_infinite_loop(self, built_image):
        started = __import__("time").time()
        result = _sbx_run("while True: pass", timeout_seconds=4.0)
        elapsed = __import__("time").time() - started
        assert not result.ok
        assert result.timed_out
        assert result.timeout_layer == "container"
        assert result.exit_code == 124
        # The workload died close to the cap, not at the host slack (cap+5s).
        assert 4.0 <= elapsed < 10.0

    def test_no_orphan_container_after_timeout(self, built_image):
        result = _sbx_run("while True: pass", timeout_seconds=3.0)
        # v0.17: the run reports which container served it — check exactly that
        # one (the old fixed-name check cannot work with per-run identities).
        check = subprocess.run(
            ["docker", "ps", "--filter", f"name={result.container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            timeout=15,
        )
        assert result.container_name not in check.stdout.decode()

    def test_force_remove_container_works_against_real_daemon(self, built_image):
        """Layer C mechanism verified for real: rm -f kills a live container."""
        # v0.17: names are per-run and server-generated; this test mints its
        # own unique name in the same namespace for the same purpose.
        name = f"{DockerCodeSandbox.CONTAINER_NAME_PREFIX}-forcetest-{uuid.uuid4().hex[:8]}"
        # Start a long-lived container under the fixed name (as layer A would
        # find after a host-side timeout).
        start = subprocess.run(
            [
                "docker", "run", "-d", "--rm", "--name", name,
                "--network", "none", IMAGE_TAG,
                "python3", "-c", "import time; time.sleep(120)",
            ],
            capture_output=True,
            timeout=30,
        )
        assert start.returncode == 0, start.stderr.decode()[-500:]
        try:
            running = subprocess.run(
                ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True, timeout=15,
            )
            assert name in running.stdout.decode()
            # The layer-C helper must actually terminate it.
            DockerCodeSandbox(image=IMAGE_TAG)._force_remove_container(name)
            after = subprocess.run(
                ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True, timeout=15,
            )
            assert name not in after.stdout.decode()
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)

    def test_normal_code_terminates_normally(self, built_image):
        result = _sbx_run("print('hello')", timeout_seconds=10.0)
        assert result.ok
        assert result.stdout.strip() == "hello"
        assert result.timeout_layer is None

    def test_output_cap_enforced(self, built_image):
        result = _sbx_run(
            "print('x' * 200000)", timeout_seconds=10.0, max_output_bytes=1000
        )
        assert result.ok
        assert result.truncated
        assert len(result.stdout) <= 1000


# ── Resource exhaustion is bounded ────────────────────────────────────────────


class TestObservedResourceLimits:
    def test_memory_hog_killed_not_ooming_host(self, built_image):
        # Allocate 300MB repeatedly against a 128MB cap: the kernel OOM-killer
        # must stop the container, and the host must be unaffected.
        code = "a=[]\nfor i in range(1000000):\n    a.append(bytearray(1024*1024))\n"
        r = _run(
            ["--memory", "128m", "--memory-swap", "128m"],
            code,
            timeout=40,
        )
        assert r.returncode != 0, "memory exhaustion must not succeed"

    def test_process_explosion_bounded(self, built_image):
        """pids-limit caps CONCURRENT processes: forking must hit EAGAIN."""
        code = (
            "import os, time\n"
            "count = 0\n"
            "kids = []\n"
            "try:\n"
            "    while True:\n"
            "        pid = os.fork()\n"
            "        if pid == 0:\n"
            "            time.sleep(60)\n"
            "            os._exit(0)\n"
            "        kids.append(pid)\n"
            "        count += 1\n"
            "        if count > 500:\n"
            "            print('FORK-UNBOUNDED', flush=True)\n"
            "            break\n"
            "except OSError:\n"
            "    print('FORK-LIMITED', count, flush=True)\n"
            "for k in kids:\n"
            "    try:\n"
            "        os.kill(k, 9)\n"
            "    except OSError:\n"
            "        pass\n"
        )
        r = _run([], code, timeout=60)
        out = r.stdout.decode()
        assert "FORK-LIMITED" in out, out or "fork loop never reported"
        assert "FORK-UNBOUNDED" not in out
        # Observed ceiling must be at or below the configured pids limit.
        reported = int(out.split("FORK-LIMITED")[1].split()[0])
        assert reported <= 64

    def test_host_survives_all_exhaustion_tests(self, built_image):
        """Meta-assertion: after the exhaustion tests, the daemon is healthy."""
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=15,
        )
        assert r.returncode == 0


# ── End-to-end through the real sandbox implementation ────────────────────────


class TestSandboxEndToEnd:
    def test_execute_returns_stdout_and_stderr(self, built_image):
        result = _sbx_run("import sys; print('out'); print('err', file=sys.stderr)")
        assert result.ok
        assert "out" in result.stdout
        assert "err" in result.stderr

    def test_python_exception_reported_not_hidden(self, built_image):
        result = _sbx_run("raise ValueError('boom')")
        assert not result.ok
        assert "ValueError" in result.stderr
        assert result.exit_code == 1

    def test_tmp_cleanup_between_runs(self, built_image):
        # /tmp is tmpfs per-run (--rm + fresh container): files must not leak.
        _sbx_run("open('/tmp/leak.txt','w').write('leak')")
        result = _sbx_run(
            "import os; print(os.path.exists('/tmp/leak.txt'))"
        )
        assert result.stdout.strip() == "False"

    def test_container_removed_after_run(self, built_image):
        result = _sbx_run("print(1)")
        check = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={result.container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            timeout=15,
        )
        assert result.container_name not in check.stdout.decode()

    def test_rejection_of_unknown_language_never_touches_docker(self, built_image):
        sandbox = DockerCodeSandbox(image=IMAGE_TAG)
        result = sandbox.execute(
            ExecutionRequest(code="x", language="javascript", timeout_seconds=5)
        )
        assert not result.ok
        assert result.denial_reason == "unsupported_language"


# ── Availability gate against the real daemon ─────────────────────────────────


class TestAvailabilityGate:
    def test_is_available_true_on_linux_engine_with_image(self, built_image):
        import sys as _sys

        sandbox = DockerCodeSandbox(image=IMAGE_TAG)
        if _sys.platform.startswith("win32"):
            pytest.skip("sandbox deliberately refuses Windows hosts (fail-closed)")
        assert sandbox.is_available() is True

    def test_is_available_refuses_windows_host_even_with_engine(self, built_image):
        """On a Windows host the sandbox refuses regardless of daemon state."""
        import sys as _sys

        if not _sys.platform.startswith("win32"):
            pytest.skip("only meaningful on a Windows host")
        sandbox = DockerCodeSandbox(image=IMAGE_TAG)
        assert sandbox.is_available() is False


# ── v0.17: concurrent executions get isolated per-run identities ─────────────


class TestConcurrentExecutions:
    """v0.17 Track C: per-run container names under REAL concurrency.

    The v0.16 sandbox used one FIXED container name, so two concurrent
    executions either collided (docker error) or serialized. Per-run names
    must make concurrent executions independent: unique identity, separate
    lifecycle, execution-specific cleanup — observed against a real daemon.
    """

    def test_container_names_unique_across_runs(self, built_image):
        seen = {_sbx_run("print(1)").container_name for _ in range(5)}
        assert len(seen) == 5, "every execution must get a unique container name"
        assert all(n.startswith(DockerCodeSandbox.CONTAINER_NAME_PREFIX) for n in seen)

    def test_concurrent_executions_all_succeed_with_unique_names(self, built_image):
        """The decisive v0.17 behavior: parallel runs do not collide."""
        from concurrent.futures import ThreadPoolExecutor

        def one(i: int):
            r = _sbx_run(f"print('worker-{i}')", timeout_seconds=15)
            return r

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, range(4)))

        names = [r.container_name for r in results]
        assert len(set(names)) == 4, "concurrent runs must not share a name"
        for r in results:
            assert r.ok, f"run {r.container_name} failed: {r.stderr[-300:]}"
        assert {r.stdout.strip() for r in results} == {
            f"worker-{i}" for i in range(4)
        }

    def test_timeout_cleanup_is_execution_specific(self, built_image):
        """A timeout rm -f must remove ONLY its own container."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(_sbx_run, "print('slow-ok')", 15)  # long-lived
            timeouted = pool.submit(_sbx_run, "while True: pass", 3)
            ok_result = slow.result(timeout=60)
            dead_result = timeouted.result(timeout=60)

        assert ok_result.ok and ok_result.stdout.strip() == "slow-ok"
        assert dead_result.timed_out and dead_result.timeout_layer == "container"
        # Distinct identities throughout — no cross-talk was possible.
        assert dead_result.container_name != ok_result.container_name
        # Neither container survives its own completion.
        for name in (ok_result.container_name, dead_result.container_name):
            ps = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                capture_output=True, timeout=15,
            )
            assert name not in ps.stdout.decode()
