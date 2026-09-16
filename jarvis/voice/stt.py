"""
jarvis/voice/stt.py
───────────────────
Speech-to-Text via OpenAI Whisper (local, free).

Requires system ``ffmpeg`` on PATH for some audio formats; we write WAV
directly so recording works even when ffmpeg is only used by Whisper internals.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000


class SpeechToText:
    """
    Capture microphone audio and transcribe it with a local Whisper model.

    Args:
        model_name: Whisper model size (default from ``settings.whisper_model``).
        record_seconds: Capture duration per ``listen()`` call.
    """

    def __init__(
        self,
        model_name: str | None = None,
        record_seconds: float | None = None,
    ) -> None:
        self._model_name = model_name or settings.whisper_model
        self._record_seconds = float(
            record_seconds if record_seconds is not None else settings.voice_record_seconds
        )
        self._model: object | None = None
        self._load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper weights once; failures are stored, not raised."""
        try:
            import whisper  # lazy: heavy import / torch

            log.info("whisper_loading", model=self._model_name)
            self._model = whisper.load_model(self._model_name)
            log.info("whisper_ready", model=self._model_name)
        except Exception as e:
            self._model = None
            self._load_error = str(e)
            log.error("whisper_load_failed", model=self._model_name, error=str(e))

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def listen(self) -> str:
        """
        Record from the default microphone and return transcribed text.

        Returns:
            Transcribed utterance, or ``""`` on failure / silence.
        """
        if self._model is None:
            log.error(
                "stt_unavailable",
                reason=self._load_error or "Whisper model not loaded",
            )
            return ""

        wav_path: Path | None = None
        try:
            audio = self._record_microphone()
            wav_path = self._write_wav(audio)
            log.info("stt_transcribe_start", path=str(wav_path))
            result = self._model.transcribe(  # type: ignore[union-attr]
                str(wav_path),
                fp16=False,
            )
            text = (result.get("text") or "").strip()
            log.info("stt_transcribe_done", chars=len(text), text_preview=text[:120])
            return text
        except Exception as e:
            log.error("stt_listen_failed", error=str(e))
            return ""
        finally:
            if wav_path is not None:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _record_microphone(self) -> np.ndarray:
        """Record mono float32 audio from the default input device."""
        import sounddevice as sd

        frames = int(self._record_seconds * _SAMPLE_RATE)
        log.info(
            "stt_recording_start",
            seconds=self._record_seconds,
            sample_rate=_SAMPLE_RATE,
        )
        try:
            audio = sd.rec(
                frames,
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as e:
            log.error("stt_microphone_error", error=str(e))
            raise RuntimeError(
                f"Microphone recording failed: {e}. "
                "Check that a mic is connected and OS permissions allow access."
            ) from e

        log.info("stt_recording_end", samples=len(audio))
        return np.squeeze(audio)

    def _write_wav(self, audio: np.ndarray) -> Path:
        """Persist float32 mono audio as a 16-bit PCM WAV tempfile."""
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        return tmp_path
