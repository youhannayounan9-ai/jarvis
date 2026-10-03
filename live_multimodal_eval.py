"""
live_multimodal_eval.py
───────────────────────
v0.27 LIVE multimodal validation against the REAL local stack — manual-only,
NOT part of pytest. Requires: Ollama (qwen2.5:7b + llava), optional mic +
ffmpeg for voice cases (skipped gracefully when absent).

Cases (Part 22):
  1. text request                         (baseline sanity)
  2. image understanding (llava)          "What's in this image?"
  3. image + text question                "Read this screenshot."
  4. image observation → tool call        image + calculator follow-up
  5. voice: audio file → STT → runtime    (mic case handled interactively)
  6. voice turn with calculator (STT)
  7. TTS spoken response (network-backed Edge TTS)
  8. multimodal session follow-up         (image turn then "what did you see?")
  9. grounding on multimodal/tool response

Metrics are reported in three SEPARATE bands (Part 22 requirement):
  infrastructure — server, validation, session plumbing
  provider/model — Whisper, llava, Edge TTS, qwen tool choice
  reasoning      — model answer quality (small sample; never universal)

Run:
    uv run python live_multimodal_eval.py --json live_multimodal_report.json
    uv run python live_multimodal_eval.py --with-mic     # + microphone cases
    uv run python live_multimodal_eval.py --with-tts     # + spoken response
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) if (os := __import__("os")) else "."
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ISOLATION (v0.25 Part G): private temp DB BEFORE any jarvis import.
from evaluation import _bootstrap as _eval

_eval.isolate()

from jarvis.config import settings
from jarvis.multimodal.models import Attachment, MultimodalRequest
from jarvis.multimodal.service import MultimodalService
from jarvis.runtime import build_runtime


def _png_bytes(color=(200, 40, 40), size=(320, 200)) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, color=color)
    d = ImageDraw.Draw(img)
    d.rectangle([40, 60, 280, 140], fill=(240, 240, 240), outline=(10, 10, 10))
    d.text((60, 90), "ERROR: disk full", fill=(180, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _wav_bytes(seconds=2.0, rate=16000) -> bytes:
    import math
    import struct
    import wave as _wave

    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(int(rate * seconds))
        )
        wf.writeframes(frames)
    return buf.getvalue()


class _Recorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, case, band, ok, detail="", duration=None):
        row = {"case": case, "band": band, "ok": bool(ok), "detail": detail[:200]}
        if duration is not None:
            row["duration_s"] = round(duration, 2)
        self.rows.append(row)
        print(json.dumps(row, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="live_multimodal_eval")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--with-mic", action="store_true",
                        help="Include live microphone capture cases")
    parser.add_argument("--with-tts", action="store_true",
                        help="Include Edge TTS playback case (network-backed)")
    args = parser.parse_args(argv)

    rec = _Recorder()
    runtime = build_runtime()
    service = MultimodalService(runtime.orchestrator)
    session_id = runtime.start_session()

    def run_turn(text, image_bytes=None, sid=None, label=""):
        t0 = time.perf_counter()
        image = Attachment.from_upload(image_bytes) if image_bytes else None
        req = MultimodalRequest(
            text=text,
            session_id=sid or session_id,
            image=image,
        )
        try:
            out = service.run(req)
            return out, time.perf_counter() - t0
        finally:
            if image is not None:
                image.cleanup()

    # ── 1. text baseline ─────────────────────────────────────────────────
    try:
        out, dt = run_turn("In one sentence, what is a palindrome?", label="text")
        rec.add("1_text_request", "provider/model", bool(out), out, dt)
    except Exception as e:
        rec.add("1_text_request", "infrastructure", False, f"EXC {e}")

    # ── 2. image understanding ───────────────────────────────────────────
    try:
        out, dt = run_turn("What's in this image?", _png_bytes())
        ok = "error" not in out.lower()[:60]
        rec.add("2_image_understanding", "provider/model", ok, out, dt)
    except Exception as e:
        rec.add("2_image_understanding", "infrastructure", False, f"EXC {e}")

    # ── 3. image + text question ─────────────────────────────────────────
    try:
        out, dt = run_turn("Read this screenshot and tell me what the error text says.",
                           _png_bytes((30, 40, 200)))
        rec.add("3_image_plus_text", "provider/model", bool(out), out, dt)
    except Exception as e:
        rec.add("3_image_plus_text", "infrastructure", False, f"EXC {e}")

    # ── 4. image observation → tool call (calculator) ────────────────────
    try:
        out, dt = run_turn(
            "Look at this image, then use the calculator to compute 893 * 47 "
            "and give me the product.",
            _png_bytes((20, 120, 60)),
        )
        from jarvis.core.grounding import canonical_numbers

        ok = any(c.value == 41971 for c in canonical_numbers(out))
        rec.add("4_image_then_tool", "provider/model", ok, out, dt)
    except Exception as e:
        rec.add("4_image_then_tool", "infrastructure", False, f"EXC {e}")

    # ── 5/6. voice via audio file → STT → runtime ────────────────────────
    try:
        from jarvis.voice.stt import SpeechToText

        stt = SpeechToText()
        if not stt.is_ready:
            rec.add("5_audio_file_stt", "infrastructure", False, "Whisper not loaded (expected in CI)")
        else:
            # A synthetic tone transcribes to nothing meaningful; the point
            # is the PIPELINE (bytes → whisper → text → runtime). We feed a
            # real speech-like case only when a mic is available (--with-mic).
            wav = _wav_bytes(1.0)
            t0 = time.perf_counter()
            text = stt.transcribe_bytes(wav, "audio/wav")
            rec.add("5_audio_file_stt_pipeline", "provider/model", not text.startswith("ERROR"),
                    f"transcribed: {text!r} (tone → likely empty)", time.perf_counter() - t0)
    except Exception as e:
        rec.add("5_audio_file_stt", "infrastructure", False, f"EXC {e}")

    if args.with_mic:
        try:
            from jarvis.voice.stt import SpeechToText

            stt = SpeechToText()
            input("Press ENTER, then say: 'What is 893 times 47?' ...")
            t0 = time.perf_counter()
            heard = stt.listen()
            dt = time.perf_counter() - t0
            if not heard or heard.startswith("ERROR"):
                rec.add("6_mic_voice_turn", "provider/model", False, heard or "(silence)", dt)
            else:
                out, dt2 = run_turn(heard)
                from jarvis.core.grounding import canonical_numbers

                ok = any(c.value == 41971 for c in canonical_numbers(out))
                rec.add("6_mic_voice_turn", "provider/model", ok,
                        f"heard={heard!r} → {out}", dt + dt2)
        except Exception as e:
            rec.add("6_mic_voice_turn", "infrastructure", False, f"EXC {e}")

    # ── 7. TTS spoken response ───────────────────────────────────────────
    if args.with_tts:
        try:
            from jarvis.voice.tts import TextToSpeech

            tts = TextToSpeech()
            t0 = time.perf_counter()
            tts.speak("Multimodal text to speech is working.")
            rec.add("7_tts_playback", "provider/model", True,
                    "(network-backed Edge TTS; judge by ear)", time.perf_counter() - t0)
        except Exception as e:
            rec.add("7_tts_playback", "infrastructure", False, f"EXC {e}")

    # ── 8. multimodal session follow-up ──────────────────────────────────
    try:
        out1, dt1 = run_turn("What's in this image?", _png_bytes((90, 20, 90)))
        out2, dt2 = run_turn("Based on the image you just saw, what color dominated it?", sid=session_id)
        rec.add("8_session_followup", "reasoning", bool(out2),
                f"follow-up answer: {out2}", dt1 + dt2)
    except Exception as e:
        rec.add("8_session_followup", "infrastructure", False, f"EXC {e}")

    # ── 9. grounding on a tool-bearing multimodal turn ───────────────────
    try:
        out, dt = run_turn(
            "Use the calculator for 7284 * 931 and tell me the product.",
            _png_bytes((120, 120, 20)),
        )
        from jarvis.core.grounding import canonical_numbers

        ok = any(c.value == 6781404 for c in canonical_numbers(out))
        rec.add("9_grounding_multimodal", "reasoning", ok, out, dt)
    except Exception as e:
        rec.add("9_grounding_multimodal", "infrastructure", False, f"EXC {e}")

    runtime.close()

    # ── Summary: bands kept SEPARATE ─────────────────────────────────────
    by_band: dict[str, list[dict]] = {}
    for r in rec.rows:
        by_band.setdefault(r["band"], []).append(r)
    durations = [r["duration_s"] for r in rec.rows if "duration_s" in r]
    summary = {
        "cases": len(rec.rows),
        "infrastructure_ok": f"{sum(r['ok'] for r in by_band.get('infrastructure', []))}"
        f"/{len(by_band.get('infrastructure', []))}",
        "provider_model_ok": f"{sum(r['ok'] for r in by_band.get('provider/model', []))}"
        f"/{len(by_band.get('provider/model', []))}",
        "reasoning_ok": f"{sum(r['ok'] for r in by_band.get('reasoning', []))}"
        f"/{len(by_band.get('reasoning', []))}",
        "latency_mean_s": round(statistics.mean(durations), 2) if durations else None,
        "latency_median_s": round(statistics.median(durations), 2) if durations else None,
        "note": (
            "Small sample. Infrastructure = plumbing/validation; "
            "provider/model = Whisper/llava/Edge/qwen behavior (varies); "
            "reasoning = answer quality (never universal). "
            "TTS is NETWORK-BACKED (Edge); STT is LOCAL (Whisper); "
            "vision is LOCAL (llava via Ollama)."
        ),
    }
    report = {"summary": summary, "cases": rec.rows}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
