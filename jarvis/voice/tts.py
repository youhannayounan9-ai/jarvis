"""
jarvis/voice/tts.py
───────────────────
Text-to-Speech via Microsoft Edge TTS (free, no API key).

Requires network access to Edge's TTS endpoint. Audio is played locally
with ``ffplay`` (part of FFmpeg).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class TextToSpeech:
    """
    Speak text aloud using Edge TTS.

    Args:
        voice: Edge neural voice name (default from ``settings.tts_voice``).
    """

    def __init__(self, voice: str | None = None) -> None:
        self._voice = voice or settings.tts_voice

    def speak(self, text: str) -> None:
        """
        Convert ``text`` to speech and play it.

        Errors (network, playback) are logged and swallowed so the voice
        session can continue.
        """
        cleaned = _prepare_for_speech(text)
        if not cleaned:
            log.info("tts_skip_empty")
            return

        mp3_path: Path | None = None
        try:
            mp3_path = Path(tempfile.mkstemp(suffix=".mp3")[1])
            log.info(
                "tts_generate_start",
                voice=self._voice,
                chars=len(cleaned),
            )
            asyncio.run(self._synthesize(cleaned, mp3_path))
            log.info("tts_generate_done", path=str(mp3_path))

            log.info("tts_playback_start")
            self._play(mp3_path)
            log.info("tts_playback_done")
        except Exception as e:
            log.error("tts_speak_failed", error=str(e))
        finally:
            if mp3_path is not None:
                try:
                    mp3_path.unlink(missing_ok=True)
                except OSError:
                    pass

    async def _synthesize(self, text: str, path: Path) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self._voice)
        await communicate.save(str(path))

    def _play(self, path: Path) -> None:
        """Play an audio file using ffplay (bundled with FFmpeg)."""
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            log.error("tts_ffplay_not_found")
            raise RuntimeError(
                "ERROR: Audio playback failed. "
                "Ensure FFmpeg is installed and ffplay is in your system PATH."
            )
        except subprocess.CalledProcessError as e:
            log.error("tts_ffplay_failed", returncode=e.returncode)
            raise RuntimeError(
                f"ERROR: Audio playback failed (ffplay exit code {e.returncode}). "
                "Ensure FFmpeg is installed correctly."
            )


def _prepare_for_speech(text: str) -> str:
    """Light cleanup so TTS does not read raw Markdown noise."""
    text = text or ""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[#*_>~]+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
