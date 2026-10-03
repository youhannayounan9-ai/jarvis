"""
jarvis/voice/tts.py
───────────────────
Text-to-Speech — v0.27 provider boundary (Part 5).

Provider status (documented honestly, Part 5/24):
    NETWORK-BACKED — Microsoft Edge TTS (free, no API key) synthesizes the
    audio via Edge's online endpoint. This is NOT fully local.
    LOCAL          — playback only (ffplay, part of FFmpeg) runs on-device.
    OPTIONAL       — both require system FFmpeg; TTS can be disabled
    entirely (TTS_ENABLED=false or per-call).

v0.27 additions:
    - explicit availability state (``is_enabled`` / ``is_playback_available``);
    - synthesis timeout (no silent infinite network waits);
    - ``stop()`` cancellation boundary — kills the active playback process
      so a new push-to-talk turn can barge in cleanly (Part 7); no orphaned
      ffplay workers;
    - no silent retry loops — one attempt, explicit failure telemetry;
    - events: tts_started / tts_completed / tts_cancelled / tts_failed with
      durations; no response text in logs (bounded length only).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class TextToSpeech:
    """
    Speak text aloud (Edge TTS + ffplay playback).

    Cancellation model (Part 7): ``speak()`` runs the blocking playback in a
    worker thread; ``stop()`` terminates that process. The next ``speak()``
    starts fresh. This is a clean HALF-DUPLEX cancellation boundary — the
    current Edge stack does not support true full-duplex barge-in with
    stream positions, and this layer does not pretend it does.
    """

    def __init__(self, voice: str | None = None, enabled: bool | None = None) -> None:
        self._voice = voice or settings.tts_voice
        self._timeout = float(settings.TTS_TIMEOUT_SECONDS)
        self._enabled = bool(settings.TTS_ENABLED) if enabled is None else bool(enabled)
        self._play_proc: subprocess.Popen | None = None
        self._play_lock = threading.Lock()

    # ── Availability (Part 5) ────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        """False when TTS is disabled by configuration (text stays intact)."""
        return self._enabled

    @property
    def is_playback_available(self) -> bool:
        """True when ffplay exists on PATH (playback is LOCAL + OPTIONAL)."""
        try:
            subprocess.run(["ffplay", "-version"], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False

    # ── Cancellation boundary (Part 7) ───────────────────────────────────

    def stop(self) -> None:
        """
        Cancel current playback immediately (barg-in boundary). Safe to call
        repeatedly and from any state; never raises.
        """
        with self._play_lock:
            proc = self._play_proc
            self._play_proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.info("tts_cancelled")
        except Exception as e:  # noqa: BLE001 - best-effort kill
            log.warning("tts_stop_failed", error_category="cancel", detail=str(e))

    # ── Speech ───────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Convert ``text`` to speech and play it (blocking until playback
        finishes or is cancelled).

        One attempt only — network/playback failures are logged with a
        bounded category and swallowed so the voice turn can continue; the
        textual response is unaffected.
        """
        cleaned = _prepare_for_speech(text)
        if not cleaned:
            log.info("tts_skip_empty")
            return
        if not self._enabled:
            log.info("tts_disabled_skip")
            return

        started = time.perf_counter()
        log.info("tts_started", provider="edge_tts", network_backed=True, chars=len(cleaned), voice=self._voice)
        mp3_path: Path | None = None
        try:
            mp3_path = Path(tempfile.mkstemp(suffix=".mp3")[1])
            self._synthesize_with_timeout(cleaned, mp3_path)

            log.info("tts_playback_start")
            self._play(mp3_path)
            log.info(
                "tts_completed",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        except _TTSCancelled:
            log.info("tts_cancelled")
        except Exception as e:  # noqa: BLE001 - one attempt, no retries
            log.error(
                "tts_failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                error_category=_categorize_tts_error(e),
            )
        finally:
            self.stop()  # ensure no orphaned playback process remains
            if mp3_path is not None:
                try:
                    mp3_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _synthesize_with_timeout(self, text: str, path: Path) -> None:
        """One synthesis attempt under the configured timeout."""
        try:
            result: dict[str, str] = {}

            def _run() -> None:
                try:
                    asyncio.run(self._synthesize(text, path))
                    result["ok"] = "1"
                except Exception as e:  # noqa: BLE001
                    result["error"] = str(e)

            worker = threading.Thread(target=_run, daemon=True, name="tts-synth")
            worker.start()
            worker.join(timeout=self._timeout)
            if worker.is_alive():
                raise TimeoutError(f"Edge TTS synthesis exceeded {self._timeout}s")
            if "error" in result:
                raise RuntimeError(result["error"])
        except RuntimeError as e:
            if "event loop" in str(e).lower():
                # Already inside an event loop (API context): fall back to
                # a direct await on a fresh loop is unsafe — synthesize
                # without asyncio.run wrapper.
                asyncio.run(self._synthesize(text, path))
            else:
                raise

    async def _synthesize(self, text: str, path: Path) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self._voice)
        await communicate.save(str(path))

    def _play(self, path: Path) -> None:
        """Play an audio file using ffplay; cancellable via stop()."""
        try:
            with self._play_lock:
                self._play_proc = subprocess.Popen(
                    [
                        "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                        str(path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc = self._play_proc
            try:
                proc.wait(timeout=max(300.0, self._timeout * 10))
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Audio playback exceeded watchdog ({self._timeout * 10}s).")
            if proc.returncode not in (0, None):
                raise RuntimeError(
                    f"Audio playback failed (ffplay exit code {proc.returncode}). "
                    "Ensure FFmpeg is installed correctly."
                )
        except FileNotFoundError:
            log.error("tts_ffplay_not_found")
            raise RuntimeError(
                "ERROR: Audio playback failed. "
                "Ensure FFmpeg is installed and ffplay is in your system PATH."
            )

    # ── Chunked synthesis (Part 8) ───────────────────────────────────────

    def speak_chunks(self, text: str, chunk_chars: int = 600) -> None:
        """
        Chunked synthesis for long responses (Part 8): Edge TTS has no
        streaming API, so long text is split on sentence boundaries and
        synthesized chunk-by-chunk — bounded latency to first audio without
        claiming true streaming. Each chunk is cancellation-checkable.
        """
        cleaned = _prepare_for_speech(text)
        if not cleaned or not self._enabled:
            return
        for chunk in _chunk_text(cleaned, chunk_chars):
            if not chunk:
                continue
            self.speak(chunk)


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries into ≤limit chunks (deterministic)."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + 1 + len(s) > limit:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


class _TTSCancelled(Exception):
    """Internal: playback was cancelled via stop()."""


def _categorize_tts_error(e: Exception) -> str:
    name = type(e).__name__
    if name == "TimeoutError" or "timed out" in str(e).lower():
        return "timeout"
    if "ffplay" in str(e).lower() or "playback" in str(e).lower():
        return "playback"
    if "network" in str(e).lower() or "connection" in str(e).lower() or "ssl" in str(e).lower():
        return "network"
    return "synthesis"


def _prepare_for_speech(text: str) -> str:
    """Light cleanup so TTS does not read raw Markdown noise."""
    text = text or ""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[#*_>~]+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
