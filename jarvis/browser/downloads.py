"""
jarvis/browser/downloads.py
───────────────────────────
v0.28 Part 18 — conservative download safety.

Contract:
  - Downloads go ONLY to a controlled temp directory created per browser
    session (``tempfile.mkdtemp``), never the user's Downloads folder,
    never anywhere executable is served from.
  - Size is BOUNDED: a download exceeding ``max_download_mb`` is aborted
    (the driver enforces via Playwright's expect_download + size check;
    this module also re-checks the final file size — fail closed).
  - Every accepted download is recorded as bounded METADATA (name hash,
    size, mime, timestamp) — never file content — and reported to the
    model as a NON-EXECUTED artifact.
  - Downloaded files are NEVER executed, opened, or registered with the
    OS. If execution is ever supported, it goes through the existing
    Docker sandbox (which itself is fail-closed on Windows).
  - The whole directory is wiped when the browser session closes.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MAX_DOWNLOAD_MB = 50.0
# Extensions that must NEVER land on disk even inside temp (defense in
# depth against "saved then opened" accidents): script/executable shapes.
_FORBIDDEN_SUFFIXES: frozenset[str] = frozenset(
    {".exe", ".msi", ".bat", ".cmd", ".ps1", ".sh", ".com", ".scr", ".vbs",
     ".js", ".jse", ".wsf", ".hta", ".dll"}
)


class DownloadRejected(RuntimeError):
    """Raised when a download violates policy. Message is safe."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Download rejected: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class DownloadRecord:
    """Bounded metadata of one accepted download (never content)."""

    suggested_name: str      # sanitized display name (NOT the on-disk name)
    stored_name: str         # random on-disk name (path-traversal-proof)
    size_bytes: int
    mime_type: str
    sha256_16: str           # first 16 hex chars of content hash
    created_at: float


@dataclass
class DownloadArea:
    """Owns the session's controlled download directory."""

    max_download_mb: float = _DEFAULT_MAX_DOWNLOAD_MB
    records: list[DownloadRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="jarvis_dl_"))

    # ── Intake ─────────────────────────────────────────────────────────────

    @property
    def dir(self) -> Path:
        return self._dir

    def accept(
        self,
        source_path: Path,
        *,
        suggested_name: str,
        mime_type: str = "application/octet-stream",
    ) -> DownloadRecord:
        """
        Move a completed browser download into the controlled area.
        Validates size and extension; stores under a RANDOM name.
        """
        suggested = _sanitize_name(suggested_name)
        suffix = Path(suggested).suffix.lower()
        if suffix in _FORBIDDEN_SUFFIXES:
            raise DownloadRejected(f"executable/script content ('{suffix}') is never stored")

        try:
            size = source_path.stat().st_size
        except OSError as e:
            raise DownloadRejected(f"download file unreadable ({e.__class__.__name__})") from e
        if size > self.max_download_mb * 1024 * 1024:
            raise DownloadRejected(
                f"download exceeds {self.max_download_mb:g} MB limit ({size} bytes)"
            )

        stored_name = f"dl_{int(time.time() * 1000):x}_{hashlib.sha1(suggested.encode('utf-8', 'replace')).hexdigest()[:8]}"
        target = self._dir / stored_name
        try:
            shutil.move(str(source_path), str(target))
        except (shutil.Error, OSError) as e:
            raise DownloadRejected(f"download move failed ({e.__class__.__name__})") from e

        digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        record = DownloadRecord(
            suggested_name=suggested,
            stored_name=stored_name,
            size_bytes=size,
            mime_type=(mime_type or "application/octet-stream")[:100],
            sha256_16=digest,
            created_at=time.time(),
        )
        self.records.append(record)
        log.info(
            "download_accepted",
            suggested_name=suggested,
            size_bytes=size,
            mime=record.mime_type,
            hash16=digest,
        )
        return record

    # ── Reporting / cleanup ────────────────────────────────────────────────

    def report(self) -> str:
        """Model-safe report of accepted downloads (metadata only)."""
        if not self.records:
            return "(no downloads this session)"
        lines = ["downloads (stored, NEVER executed):"]
        for r in self.records:
            lines.append(
                f"- {r.suggested_name} | {r.size_bytes} bytes | {r.mime_type} "
                f"| sha256:{r.sha256_16}… | stored as {r.stored_name}"
            )
        return "\n".join(lines)

    def cleanup(self) -> None:
        """Wipe the whole controlled directory (session end / e-stop)."""
        try:
            shutil.rmtree(self._dir, ignore_errors=True)
        except Exception as e:  # pragma: no cover - best effort
            log.warning("download_cleanup_failed", error=str(e))


def _sanitize_name(name: str) -> str:
    """
    Display-name sanitizer: strip path components and control chars.
    Path separators AND dot-segments are replaced (never merely stripped),
    so ``../../etc/passwd`` cannot survive as ``_.._.._etc_passwd``.
    """
    cleaned = "".join(ch for ch in str(name or "download.bin") if ord(ch) >= 0x20)
    cleaned = cleaned.replace("\\", "_").replace("/", "_")
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip().lstrip(".")
    cleaned = cleaned[:120] or "download.bin"
    return cleaned
