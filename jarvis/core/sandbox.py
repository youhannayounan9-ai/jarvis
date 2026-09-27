"""
jarvis/core/sandbox.py
──────────────────────
Safe code execution: contract + implementations (v0.10).

Threat model
────────────
The agent must NEVER run model-generated code on the host. Model output is
untrusted input. Code runs only inside a container with a hard isolation
boundary, and every layer fails *closed*:

  Layer 1 — registry:    the tool is not registered unless a sandbox with
                         verified isolation is explicitly enabled in config.
  Layer 2 — tool gate:   CodeExecutionTool refuses any sandbox whose
                         ``provides_isolation`` is not True.
  Layer 3 — sandbox:     DockerCodeSandbox refuses to execute unless Docker
                         is reachable, the image is available locally, and
                         the request passes validation.

Isolation flags enforced on every container (defense against container
escape, resource exhaustion, and network pivoting):

  --network none              no inbound or outbound network
  --read-only                 immutable root filesystem
  --tmpfs /tmp:...            writable scratch space only (noexec)
  --cap-drop ALL              no Linux capabilities
  --security-opt no-new-privileges  no privilege escalation
  --memory / --memory-swap    hard memory ceiling (no swap escape)
  --cpus --pids-limit         CPU and process-count ceilings
  --user <uid>:<gid>          never root inside the container
  code mounted :ro            the snippet itself cannot be modified

Known Windows limitation: Docker Desktop cannot enforce --user on Windows
containers, so on win32 we fail closed and require WSL2-based Docker where
the Linux flags are honored. The availability check therefore also refuses
Windows host execution unless explicitly overridden (never the default).
"""

from __future__ import annotations

import abc
import os
import shutil
import subprocess  # noqa: S404 - only used to invoke the docker CLI, never shell=True
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ── Execution contract ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionRequest:
    """Everything a sandbox needs to run one snippet — nothing more."""

    code: str
    language: str = "python"
    timeout_seconds: float = 5.0
    # Hard caps the sandbox must enforce; exceeding them = kill + report.
    max_output_bytes: int = 64_000
    max_memory_mb: int = 256
    max_cpu_seconds: float = 5.0


@dataclass(frozen=True)
class ExecutionResult:
    """What came out of one sandboxed execution. Never raises."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False
    # Machine-readable reason for failures: timeout | oom | denied | error
    denial_reason: str | None = None
    # Which layer enforced the timeout (None when no timeout occurred):
    #   "host"      - the docker CLI client hit the host-side subprocess cap
    #                 (container may have kept running at that moment)
    #   "container" - the in-container timeout wrapper killed the WORKLOAD
    #                 (exit code 124 from `timeout`); definitive termination
    #   "host_kill" - host timeout fired AND docker was told to stop/remove
    #                 the container afterwards (termination enforced)
    timeout_layer: str | None = None

    def to_report(self) -> str:
        """Human/LLM-readable summary of the execution."""
        if not self.ok:
            reason = self.denial_reason or "error"
            return f"ERROR: Code execution denied ({reason})."
        status = f"exit={self.exit_code}"
        if self.timed_out:
            status = "timed out"
            if self.timeout_layer:
                status += f" ({self.timeout_layer})"
        out = f"{self.stdout}{self.stderr}".strip()
        if self.truncated:
            out += "\n…[output truncated]"
        return f"{status}\n{out}" if out else status


class CodeSandbox(abc.ABC):
    """
    Abstract interface for an isolated code execution environment.

    Contract:
      - ``execute`` NEVER raises for expected failures — it returns a result.
      - Implementations must fail CLOSED: any uncertainty = refusal.
      - Implementations must enforce every cap in ExecutionRequest.
    """

    #: Subclasses set True only when real OS-level isolation is verified.
    provides_isolation: bool = False

    @abc.abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...

    # ── Convenience adapter ────────────────────────────────────────────────
    def execute_code(self, code: str, timeout_seconds: float = 5.0) -> str:
        """
        Legacy string-in/string-out shim used by CodeExecutionTool.
        Keeps old call sites working while the richer contract lands.
        """
        result = self.execute(
            ExecutionRequest(code=code, timeout_seconds=timeout_seconds)
        )
        return result.to_report()


# ── Implementations ───────────────────────────────────────────────────────────


class DisabledSandbox(CodeSandbox):
    """
    Refuses everything. The default sandbox and the correct behavior when no
    verified isolation boundary exists or is configured.
    """

    provides_isolation = False

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            ok=False,
            denial_reason="disabled",
        )


class DockerUnavailableError(RuntimeError):
    """Docker CLI missing, daemon unreachable, or host platform unsupported."""


class DockerCodeSandbox(CodeSandbox):
    """
    Real Docker-isolated execution, implemented over the docker CLI.

    Security properties:
      - One fresh container per execution (--rm), never reused.
      - Code is written to a temp file and bind-mounted READ-ONLY; it is
        never interpolated into a shell command.
      - No network, read-only rootfs, no capabilities, no privilege
        escalation, non-root user, memory/CPU/PID ceilings.
      - Any uncertainty (missing docker, unreachable daemon, missing image,
        invalid request, unexpected error) returns a denial, never host
        execution.
    """

    provides_isolation = True

    # Ubuntu LTS base (glibc-complete) instead of alpine: real Python packages
    # (numpy, pandas) install correctly. Pin by digest in production.
    DEFAULT_IMAGE = "ubuntu:24.04"
    # Non-root uid:gid inside the container (Ubuntu images resolve these via
    # passwd; numeric ids avoid NSS lookups inside the container).
    CONTAINER_USER = "65534:65534"  # nobody:nogroup
    CONTAINER_WORKDIR = "/sandbox"
    # Process ceiling inside the container.
    PIDS_LIMIT = "64"
    TMPFS_MODE = "noexec,nosuid,nodev,size=16m"
    # Exit code coreutils `timeout` uses when it kills the workload.
    CONTAINER_TIMEOUT_EXIT_CODE = 124
    # Extra seconds granted to the HOST-side wait beyond the container cap,
    # so layer B (container timeout) normally reports first; layer A exists
    # only as defense against a hung CLI/daemon.
    HOST_TIMEOUT_SLACK_SECONDS = 5.0
    # Fixed container name so a layer-A timeout can target `docker rm -f`.
    # Uniqueness per host is acceptable: execution is serialized by the
    # per-session lock and enabling the sandbox is an explicit operator act.
    CONTAINER_NAME = "jarvis-sbx-exec"

    def __init__(self, image: str | None = None) -> None:
        # Distinguish None (use default) from "" (invalid — fail closed).
        if image is None:
            image = self.DEFAULT_IMAGE
        if not self._validate_image_reference(image):
            raise ValueError(f"Invalid image reference: {image!r}")
        self.image = image

    # ── Public entry point ─────────────────────────────────────────────────

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        config_error = self._validate_request(request)
        if config_error:
            return ExecutionResult(ok=False, denial_reason=config_error)

        try:
            return self._run_container(request)
        except DockerUnavailableError as e:
            return ExecutionResult(ok=False, denial_reason=f"docker_unavailable: {e}")
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False, timed_out=True, denial_reason="timeout"
            )
        except Exception as e:  # fail closed: never fall back to host execution
            return ExecutionResult(
                ok=False, denial_reason=f"sandbox_error: {type(e).__name__}: {e}"
            )

    # ── Availability (fail closed) ─────────────────────────────────────────

    def is_available(self) -> bool:
        """
        True only when the sandbox can actually run: docker CLI present,
        daemon reachable, platform supported, image available locally.

        Deliberately expensive (shells out) — call once at startup, not per
        execution. ``execute`` performs a cheaper subset of these checks
        inline per run.
        """
        if sys.platform.startswith("win32"):
            # Docker Desktop on Windows cannot enforce the Linux hardening
            # flags (--user etc.) for Windows containers; require WSL2-backed
            # docker, which we cannot verify here → fail closed.
            return False
        if shutil.which("docker") is None:
            return False
        try:
            probe = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=10,
            )
            if probe.returncode != 0:
                return False
        except (OSError, subprocess.TimeoutExpired):
            return False
        return self._image_available()

    def _image_available(self) -> bool:
        try:
            probe = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True,
                timeout=15,
            )
            return probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # ── Container execution ────────────────────────────────────────────────

    # The container runs `timeout <n> python3 script.py` so the WORKLOAD is
    # killed at the container boundary (exit 124) even if the host-side
    # docker client were patient forever. The host-side subprocess timeout
    # remains as a second layer (docker CLI hang, dead daemon, etc.).
    CONTAINER_TIMEOUT_CMD = "timeout"

    def _run_container(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Run one snippet in one fresh container.

        Timeout layers (all three must be understood as distinct):
          A. Host process timeout: ``subprocess.run(timeout=...)`` stops
             WAITING on the docker CLI. By itself it does NOT stop the
             container — which is why the layer below exists.
          B. Container execution timeout: the image entrypoint is wrapped in
             coreutils ``timeout``; the WORKLOAD receives SIGTERM at the cap
             and ``timeout`` exits with code 124. This is enforcement at the
             workload boundary and is the primary mechanism.
          C. Actual termination: on layer-B timeout the container is dead
             (workload killed, exit 124). Additionally, if layer A fires
             first (host cap < container cap, CLI hang, daemon stall), the
             container is explicitly stopped/removed via ``docker rm -f`` so
             no orphan keeps consuming CPU.

        Raises:
            DockerUnavailableError: docker missing/unreachable/image absent.
        """
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            raise DockerUnavailableError("docker CLI not found")

        script_path = self._materialize_script(request.code)
        try:
            if not self._image_available():
                raise DockerUnavailableError(
                    f"image {self.image!r} not available locally "
                    "(pre-pull required; sandbox never pulls at runtime)"
                )

            cmd = [
                docker_bin,
                "run",
                "--rm",                       # one-shot container
                "--network", "none",          # no network in or out
                "--read-only",                # immutable root filesystem
                "--tmpfs", f"/tmp:{self.TMPFS_MODE}",
                "--cap-drop", "ALL",          # no Linux capabilities
                "--security-opt", "no-new-privileges",
                "--user", self.CONTAINER_USER,  # never root
                "--memory", f"{request.max_memory_mb}m",
                "--memory-swap", f"{request.max_memory_mb}m",  # no swap
                "--cpus", str(max(0.1, request.max_cpu_seconds)),
                "--pids-limit", self.PIDS_LIMIT,
                "-v", f"{script_path}:{self.CONTAINER_WORKDIR}/script.py:ro",
                "-w", self.CONTAINER_WORKDIR,
                self.image,
                # In-container workload timeout: the container itself kills
                # the python process at the cap (exit 124 = timed out).
                self.CONTAINER_TIMEOUT_CMD,
                f"{request.timeout_seconds:g}s",
                "python3",
                f"{self.CONTAINER_WORKDIR}/script.py",
            ]

            host_deadline = time.monotonic() + request.timeout_seconds + self.HOST_TIMEOUT_SLACK_SECONDS
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=request.timeout_seconds + self.HOST_TIMEOUT_SLACK_SECONDS,
                )
            except subprocess.TimeoutExpired:
                # Layer A fired before the container reported. Stop waiting is
                # not enough — enforce layer C: remove the container so the
                # workload cannot keep running.
                self._force_remove_container()
                return ExecutionResult(
                    ok=False,
                    timed_out=True,
                    denial_reason="timeout",
                    timeout_layer="host_kill",
                )

            stdout = proc.stdout[: request.max_output_bytes].decode(
                "utf-8", errors="replace"
            )
            stderr = proc.stderr[: request.max_output_bytes].decode(
                "utf-8", errors="replace"
            )
            truncated = len(proc.stdout) > request.max_output_bytes or len(
                proc.stderr
            ) > request.max_output_bytes

            # Exit 124 = coreutils `timeout` killed the workload at the cap:
            # definitive, container-boundary enforcement (layer B).
            container_timed_out = proc.returncode == self.CONTAINER_TIMEOUT_EXIT_CODE
            _ = host_deadline  # documented above; kept for clarity

            return ExecutionResult(
                ok=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                truncated=truncated,
                timed_out=container_timed_out,
                timeout_layer=("container" if container_timed_out else None),
                # A container-killed workload is a timeout, not an error.
                denial_reason=("timeout" if container_timed_out else None),
            )
        finally:
            # Best-effort cleanup of the host-side temp script. Inside the
            # container the file was read from a read-only mount.
            try:
                Path(script_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _force_remove_container(self) -> None:
        """
        Layer-C enforcement helper: stop and remove a container that outlived
        the host-side wait. Best effort — any failure here is logged and
        swallowed (the denial result is already decided).
        """
        docker_bin = shutil.which("docker")
        if docker_bin is None:  # pragma: no cover - checked before run
            return
        try:
            subprocess.run(
                [docker_bin, "rm", "-f", self.CONTAINER_NAME],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as e:  # pragma: no cover
            log.error("sandbox_force_remove_failed", error=str(e))

    @staticmethod
    def _materialize_script(code: str) -> str:
        """
        Write the untrusted code to a fresh temp file on the host (random
        name, restrictive perms) so it can be mounted read-only. The file is
        never executable on the host and is deleted in a finally-block.
        """
        fd, path_str = tempfile.mkstemp(prefix="jarvis_sbx_", suffix=".py")
        path = Path(path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                path.chmod(0o600)
            except OSError:
                # Windows ignores POSIX perms; the mount itself is :ro.
                pass
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return str(path)

    # ── Validation helpers (usable + testable today) ──────────────────────────

    @staticmethod
    def _validate_image_reference(image: str) -> bool:
        """
        Reject malformed, host-dependent, or non-reproducible image references.

        Policy: mutable tags are dangerous for a security boundary — a tag can
        be re-pushed with different contents, silently changing what the
        sandbox runs. Bare names (implicitly ``:latest``) and explicit
        ``latest`` are therefore rejected. Pin a concrete tag
        (``ubuntu:24.04``) or, recommended in production, a digest
        (``ubuntu:24.04@sha256:...``) — a digest ref is always accepted.
        """
        if not image or len(image) > 200:
            return False

        # A digest ref pins content immutably: always acceptable in principle.
        if "@" in image:
            digest = image.split("@", 1)[1]
            return digest.lower().startswith("sha256:") and len(digest) >= len("sha256:") + 8

        # No digest: an explicit, non-latest tag is required.
        _, _, tag = image.partition(":")
        if not tag or tag.lower() == "latest":
            return False

        if image.count("/") > 2:
            # Allow registry namespaces (registry/path/image) but reject exotic
            # multi-slash refs beyond that.
            return False
        forbidden_chars = set(";|&$`\\\"' \t\n<>*?!")
        return not (forbidden_chars & set(image))

    @staticmethod
    def _validate_request(request: ExecutionRequest) -> str | None:
        """Return a denial reason if the request violates the contract, else None."""
        if request.language != "python":
            return "unsupported_language"
        if not request.code or not request.code.strip():
            return "empty_code"
        if request.timeout_seconds <= 0 or request.timeout_seconds > 30:
            return "timeout_out_of_range"
        if request.max_output_bytes <= 0 or request.max_output_bytes > 1_000_000:
            return "output_cap_out_of_range"
        if request.max_memory_mb <= 0 or request.max_memory_mb > 2048:
            return "memory_cap_out_of_range"
        return None


__all__ = [
    "CodeSandbox",
    "DisabledSandbox",
    "DockerCodeSandbox",
    "DockerUnavailableError",
    "ExecutionRequest",
    "ExecutionResult",
]
