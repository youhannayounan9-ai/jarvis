"""
jarvis/voice/stt.py
───────────────────
Speech-to-Text via OpenAI Whisper (LOCAL, free) — v0.27 provider interface.

Provider boundary (Part 4):
    LOCAL    — Whisper runs on-device; audio never leaves the machine.
    OPTIONAL — system ``ffmpeg`` must be on PATH (used by Whisper internals).

v0.27 additions:
    - ``record()`` / ``transcribe()`` split so push-to-talk and API-audio
      paths share one provider without a microphone dependency;
    - ``transcribe_bytes()`` for uploaded audio (MIME/sniffing validated
      upstream in jarvis/multimodal.models);
    - transcription timeout (no silent infinite waits);
    - language configuration (``whisper_language``, None = auto);
    - explicit failure results (empty string on silence, ERROR: string on
      hard failures — same convention as before);
    - telemetry events with durations; NO raw audio or full transcripts in
      logs (bounded previews only).
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16_000


class SpeechToText:
    """
    Capture microphone audio and transcribe it with a LOCAL Whisper model.

    Provider: LOCAL (openai-whisper). Availability is explicit via
    ``is_ready``; failures are logged with bounded categories, never raised
    through the voice loop.
    """

    def __init__(
        self,
        model_name: str | None = None,
        record_seconds: float | None = None,
    ) -> None:
        self._model_name = model_name or settings.whisper_model
        self._record_seconds = float(
            min(
                record_seconds if record_seconds is not None else settings.voice_record_seconds,
                float(settings.MAX_RECORD_SECONDS),
            )
        )
        self._timeout = float(settings.STT_TIMEOUT_SECONDS)
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

    @property
    def record_seconds(self) -> float:
        return self._record_seconds

    # ── Push-to-talk split (Part 3) ──────────────────────────────────────

    def record(self) -> np.ndarray | None:
        """
        One explicit capture window from the default microphone.

        Returns float32 mono audio, or None when the model is unavailable or
        capture fails (caller decides how to surface). Never blocks beyond
        ``record_seconds``.
        """
        if self._model is None:
            log.error("stt_unavailable", reason=self._load_error or "model not loaded")
            return None
        started = time.perf_counter()
        log.info(
            "voice_started",
            provider="whisper",
            model=self._model_name,
            seconds=self._record_seconds,
        )
        try:
            audio = self._record_microphone()
            log.info(
                "voice_captured",
                samples=len(audio),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return audio
        except sd.PortAudioError as e:
            log.error("stt_portaudio_error", error=str(e))
            return None
        except Exception as e:  # noqa: BLE001
            log.error("stt_microphone_error", error_category="capture", detail=str(e))
            return None

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe in-memory audio. Returns the utterance text (``""`` on
        silence/unavailable; ``ERROR: …`` on hard failures — the historical
        convention, preserved).
        """
        if self._model is None:
            log.error("stt_unavailable", reason=self._load_error or "model not loaded")
            return ""
        wav_path: Path | None = None
        started = time.perf_counter()
        try:
            wav_path = self._write_wav(audio)
            log.info("stt_transcribe_start", chars=int(audio.size))
            try:
                result = self._model.transcribe(  # type: ignore[union-attr]
                    str(wav_path),
                    fp16=False,
                    **({"language": settings.whisper_language} if getattr(settings, "whisper_language", None) else {}),
                )
            except TimeoutError:
                log.error("stt_transcribe_timeout", timeout_s=self._timeout)
                return "ERROR: Transcription timed out. Try a shorter utterance."
            text = (result.get("text") or "").strip()
            log.info(
                "voice_transcription_completed",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                chars=len(text),
                text_preview=text[:120],
            )
            return text
        except Exception as e:  # noqa: BLE001
            log.error("stt_listen_failed", error_category="transcription", detail=str(e))
            return (
                "ERROR: Transcription failed. Check that ffmpeg is installed "
                "and the Whisper model loaded (see logs)."
            )
        finally:
            if wav_path is not None:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # ── Legacy combined call (used by the always-listening CLI loop) ─────

    def listen(self) -> str:
        """record() + transcribe() in one call (historical entry point)."""
        audio = self.record()
        if audio is None:
            # Keep the historical user-facing hint for mic failures.
            return (
                "ERROR: Microphone access failed. Please check your microphone "
                "and ensure PortAudio/sounddevice is installed."
            )
        return self.transcribe(audio)

    # ── API audio uploads (Part 13) ──────────────────────────────────────

    def transcribe_bytes(self, data: bytes, mime: str) -> str:
        """
        Transcribe uploaded audio bytes (already content-validated by the
        multimodal layer). Written to a private temp file; ffmpeg handles
        container conversion for Whisper. Temp file is always removed.
        """
        if self._model is None:
            log.error("stt_unavailable", reason=self._load_error or "model not loaded")
            return ""
        suffix = ".mp3" if "mpeg" in mime or "mp3" in mime else (".webm" if "webm" in mime else ".wav")
        started = time.perf_counter()
        tmp_path: Path | None = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = Path(tmp.name)
            tmp.write(data)
            tmp.close()
            log.info("voice_transcribed", source="upload", mime=mime, bytes=len(data))
            result = self._model.transcribe(  # type: ignore[union-attr]
                str(tmp_path),
                fp16=False,
                **({"language": settings.whisper_language} if getattr(settings, "whisper_language", None) else {}),
            )
            text = (result.get("text") or "").strip()
            log.info(
                "voice_transcription_completed",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                chars=len(text),
                text_preview=text[:120],
            )
            return text
        except Exception as e:  # noqa: BLE001
            log.error("stt_upload_transcribe_failed", error_category="transcription", detail=str(e))
            return "ERROR: Uploaded audio could not be transcribed."
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # ── Internals ────────────────────────────────────────────────────────

    def _check_ffmpeg(self) -> str | None:
        """Return a user-facing ERROR string when ffmpeg is missing."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
            return None
        except FileNotFoundError:
            log.error("stt_ffmpeg_not_found")
            return (
                "ERROR: FFmpeg is not found in your system PATH. "
                "Please fully restart your terminal/IDE, or add FFmpeg to "
                "your Windows Environment Variables."
            )

    def _record_microphone(self) -> np.ndarray:
        """Record mono float32 audio from the default input device."""
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
        except sd.PortAudioError as e:
            log.error("stt_microphone_portaudio_error", error=str(e))
            raise
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
