"""
jarvis/multimodal/models.py
───────────────────────────
v0.27 normalized multimodal request model (Part 2) and the voice-turn
lifecycle (Parts 6/18).

Architecture invariant: TEXT, VOICE and IMAGE requests converge on the SAME
core runtime. The orchestrator signature is unchanged — a multimodal request
is normalized into (text, image_path?) and dispatched through
``Orchestrator.chat`` exactly like any typed request. There is no separate
voice agent, no separate vision agent, and no second security model.

Modality-specific behavior lives at the EDGES (STT, TTS, image handling),
never in the planning/permission/evidence core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── Voice turn lifecycle (Part 6) ─────────────────────────────────────────────

class VoiceTurnState(str, Enum):
    """Observable states of one push-to-talk voice turn."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


# ── Normalized multimodal request (Part 2) ────────────────────────────────────

class MultimodalValidationError(ValueError):
    """Raised for unsupported MIME types, oversized or malformed uploads."""


# ── Image input validation (Parts 12/13/20) ──────────────────────────────────

# The upload contract is exactly these four types. Image validation is
# content-sniffed (magic bytes), NOT taken from client names/headers.
SUPPORTED_IMAGE_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# Magic-byte signatures (content sniffing; declared MIME alone is untrusted).
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),   # + 4 bytes + "WEBP" — validated below
)


def validate_image_bytes(data: bytes, declared_mime: str | None = None) -> str:
    """
    Validate an uploaded image by CONTENT (Part 13/20).

    Returns the detected MIME type; raises ``MultimodalValidationError`` for
    empty, oversized, or non-image payloads. The client's declared MIME or
    filename is advisory only — sniffing decides. Decoded pixel dimensions
    are additionally bounded by ``validate_image_pixels`` (decompression-bomb
    defense) where Pillow is available.
    """
    if not data:
        raise MultimodalValidationError("Empty image upload.")
    max_bytes = _max_upload_bytes()
    if len(data) > max_bytes:
        raise MultimodalValidationError(
            f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
        )
    detected = _sniff_image_mime(data)
    if detected is None:
        raise MultimodalValidationError(
            "Unsupported or unrecognized image content. Supported: JPEG, PNG, WebP, GIF."
        )
    if declared_mime and declared_mime not in SUPPORTED_IMAGE_TYPES and declared_mime != detected:
        # A *contradictory* declaration is suspicious but sniffing wins.
        log.warning(
            "multimodal_mime_mismatch",
            declared=declared_mime,
            detected=detected,
        )
    return detected


def _max_upload_bytes() -> int:
    from jarvis.config import settings

    return int(settings.MAX_IMAGE_UPLOAD_MB) * 1024 * 1024


def _sniff_image_mime(data: bytes) -> str | None:
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            if mime == "image/webp":
                # RIFF....WEBP
                return "image/webp" if data[8:12] == b"WEBP" else None
            return mime
    return None


def validate_image_pixels(data: bytes) -> None:
    """
    Decompression-bomb defense: bound decoded pixel dimensions (Part 20).
    Raises ``MultimodalValidationError`` when Pillow is available and the
    image decodes larger than the configured ceiling. Missing Pillow (a
    headless install) degrades to byte-size-only bounding.
    """
    try:
        from PIL import Image
    except ImportError:
        log.debug("multimodal_pixel_check_unavailable")
        return
    import io

    max_pixels = int(_max_upload_pixels())
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception as e:  # noqa: BLE001 - malformed image
        raise MultimodalValidationError(f"Malformed image: {e}") from e
    if width * height > max_pixels:
        raise MultimodalValidationError(
            f"Image decodes to {width}x{height} pixels, above the "
            f"{max_pixels:,}-pixel safety limit."
        )


def _max_upload_pixels() -> int:
    from jarvis.config import settings

    return int(settings.MAX_IMAGE_PIXELS)


@dataclass
class Attachment:
    """
    One file attached to a multimodal request. Bytes are validated at
    construction (``from_upload``); persisted to a private, random-named
    file under the vision sandbox (never the client's name) so the existing
    ``vision_analyze`` path-sandbox applies unchanged.
    """

    kind: str                    # "image" (extensible)
    mime: str
    data: bytes
    stored_path: Path | None = None

    @classmethod
    def from_upload(
        cls,
        data: bytes,
        *,
        declared_mime: str | None = None,
        client_name: str | None = None,
    ) -> "Attachment":
        """Validate and build an image attachment from raw upload bytes."""
        del client_name  # NEVER trusted for naming or type decisions
        mime = validate_image_bytes(data, declared_mime)
        validate_image_pixels(data)
        return cls(kind="image", mime=mime, data=data)

    def persist(self) -> Path:
        """
        Write bytes to a random, sanitized name under the configured upload
        directory (inside the vision sandbox). Returns the path; idempotent.
        """
        if self.stored_path is not None:
            return self.stored_path
        import secrets

        from jarvis.config import settings

        upload_dir = Path(settings.multimodal_upload_dir).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)
        ext = SUPPORTED_IMAGE_TYPES.get(self.mime, ".bin")
        # Random name — client filenames and paths are never used (Part 20).
        path = upload_dir / f"img_{secrets.token_hex(8)}{ext}"
        path.write_bytes(self.data)
        self.stored_path = path
        log.info(
            "multimodal_image_stored",
            mime=self.mime,
            bytes=len(self.data),
            dir=str(upload_dir),
        )
        return path

    def cleanup(self) -> None:
        """Delete the persisted file (best-effort) — temp hygiene (Part 19)."""
        if self.stored_path is None:
            return
        try:
            self.stored_path.unlink(missing_ok=True)
        except OSError:
            log.warning("multimodal_image_cleanup_failed", path=str(self.stored_path))
        self.stored_path = None


@dataclass
class MultimodalRequest:
    """
    The normalized request every client type converges on (Part 2).

    Construction rules:
      - ``text`` is required after normalization (voice → STT text; image-only
        requests get a canonical prompt);
      - ``image`` (optional) must already be a validated ``Attachment``;
      - session/request IDs and response mode ride along unchanged.
    """

    text: str
    session_id: str | None = None
    request_id: str | None = None
    modality: str = "text"                    # text | voice | image | image+text | audio
    image: Attachment | None = None
    refresh: bool = False
    response_mode: str = "text"               # text | speech
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        if not self.text and self.image is not None:
            self.text = "Describe this image in detail."
        if not self.text:
            raise MultimodalValidationError(
                "Multimodal request has no text after normalization."
            )
        if self.image is not None:
            self.modality = "image+text" if self.text != "Describe this image in detail." else "image"

    def prompt_for_runtime(self) -> str:
        """
        The text handed to the EXISTING orchestrator. An attached image is
        referenced by its persisted sandbox path using the same phrasing the
        tool-policy contract already teaches the model to route to
        ``vision_analyze`` — no orchestrator logic changes.
        """
        if self.image is None:
            return self.text
        path = self.image.persist()
        return f"{self.text}\n\n[Analyzed image: {path}]"


# ── Audio upload validation (voice via API, Part 13) ─────────────────────────

SUPPORTED_AUDIO_TYPES = ("audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm")
_AUDIO_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", "audio/wav"),          # RIFF....WAVE
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),     # MPEG frame sync
    (b"\xff\xf3", "audio/mpeg"),
    (b"\x1aE\xdf\xa3", "audio/webm"),  # EBML
)


def validate_audio_bytes(data: bytes, declared_mime: str | None = None) -> str:
    """Content-sniff an uploaded audio payload (Part 13/20)."""
    if not data:
        raise MultimodalValidationError("Empty audio upload.")
    max_bytes = _max_audio_bytes()
    if len(data) > max_bytes:
        raise MultimodalValidationError(
            f"Audio exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
        )
    detected = _sniff_audio_mime(data)
    if detected is None:
        raise MultimodalValidationError(
            "Unsupported or unrecognized audio content. Supported: WAV, MP3, WebM."
        )
    return detected


def _max_audio_bytes() -> int:
    from jarvis.config import settings

    return int(settings.MAX_AUDIO_UPLOAD_MB) * 1024 * 1024


def _sniff_audio_mime(data: bytes) -> str | None:
    for magic, mime in _AUDIO_MAGIC:
        if data.startswith(magic):
            if mime == "audio/wav":
                return "audio/wav" if data[8:12] == b"WAVE" else None
            return mime
    return None
