"""
jarvis/utils/ffmpeg_utils.py
─────────────────────────────
Shared utility for resolving the FFmpeg/FFplay binary paths.

Problem: On Windows, newly installed programs (like FFmpeg) are often not
visible in `shutil.which()` until the terminal/IDE is fully restarted, even
if the PATH is correctly set. This helper falls back through well-known
install locations to find the binary directly on disk.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


_FFMPEG_FALLBACK_DIRS = [
    Path("C:/Program Files/FFmpeg/bin"),
    Path("C:/Program Files (x86)/FFmpeg/bin"),
    Path(os.environ.get("USERPROFILE", ""), "scoop", "apps", "ffmpeg", "current", "bin"),
    Path("C:/ProgramData/chocolatey/bin"),
    Path("C:/tools/ffmpeg/bin"),
]


def _get_ffmpeg_path() -> str:
    """
    Resolve the absolute path to the ffmpeg executable.

    Resolution order:
      1. shutil.which("ffmpeg")  — respects the current PATH (fastest)
      2. Common Windows install locations (fallback for stale PATH environments)
      3. Return "ffmpeg" as a bare string so callers still get a clear
         FileNotFoundError (which surfaces a user-friendly error message).
    """
    # 1. Standard PATH lookup
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Common fallback directories
    for directory in _FFMPEG_FALLBACK_DIRS:
        candidate = directory / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)

    # 3. Nothing found — return bare name; callers handle FileNotFoundError
    return "ffmpeg"


def _get_ffplay_path() -> str:
    """
    Resolve the absolute path to the ffplay executable.

    Derived from the ffmpeg path — they always live in the same directory.
    """
    ffmpeg_path = _get_ffmpeg_path()

    # If we got an absolute path, derive ffplay from the same directory
    p = Path(ffmpeg_path)
    if p.is_absolute():
        ffplay = p.parent / "ffplay.exe"
        if ffplay.exists():
            return str(ffplay)
        # Non-.exe fallback (Linux/macOS)
        ffplay_no_ext = p.parent / "ffplay"
        if ffplay_no_ext.exists():
            return str(ffplay_no_ext)

    # Fall back to PATH lookup
    found = shutil.which("ffplay")
    if found:
        return found

    return "ffplay"
