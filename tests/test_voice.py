"""
tests/test_voice.py
───────────────────
Unit tests for STT / TTS (fully mocked — no mic, Whisper, or network).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from jarvis.voice.interface import VoiceInterface
from jarvis.voice.stt import SpeechToText
from jarvis.voice.tts import TextToSpeech, _prepare_for_speech


class TestPrepareForSpeech:
    def test_strips_markdown(self):
        raw = "**Hello** `world`\n\n# Title\n[link](http://x.com)"
        cleaned = _prepare_for_speech(raw)
        assert "**" not in cleaned
        assert "`" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned
        assert "link" in cleaned


class TestSpeechToText:
    def test_listen_returns_transcription(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello JARVIS  "}
        fake_whisper = SimpleNamespace(load_model=MagicMock(return_value=mock_model))
        audio = np.zeros(16000, dtype=np.float32)

        with patch.dict("sys.modules", {"whisper": fake_whisper}):
            with patch(
                "jarvis.voice.stt.SpeechToText._record_microphone",
                return_value=audio,
            ):
                stt = SpeechToText(model_name="base", record_seconds=1.0)
                assert stt.is_ready
                text = stt.listen()

        assert text == "Hello JARVIS"
        mock_model.transcribe.assert_called_once()

    def test_listen_returns_empty_when_model_missing(self):
        fake_whisper = SimpleNamespace(
            load_model=MagicMock(side_effect=RuntimeError("no ffmpeg"))
        )
        with patch.dict("sys.modules", {"whisper": fake_whisper}):
            stt = SpeechToText(model_name="base")
        assert not stt.is_ready
        assert stt.listen() == ""


class TestTextToSpeech:
    def test_speak_generates_and_plays(self):
        tts = TextToSpeech(voice="en-US-GuyNeural")
        with patch("asyncio.run") as run_mock, patch.object(tts, "_play") as play_mock:
            tts.speak("Hello there")
            run_mock.assert_called_once()
            play_mock.assert_called_once()

    def test_speak_skips_empty(self):
        tts = TextToSpeech()
        with patch("asyncio.run") as run_mock:
            tts.speak("   ")
            run_mock.assert_not_called()

    def test_speak_swallows_errors(self):
        tts = TextToSpeech()
        with patch("asyncio.run", side_effect=RuntimeError("offline")):
            tts.speak("Hello")  # must not raise


class TestVoiceInterface:
    def test_exit_phrase_stops_loop(self):
        orch = MagicMock()
        stt = MagicMock()
        stt.is_ready = True
        stt.listen.side_effect = ["quit"]
        tts = MagicMock()

        VoiceInterface(orch, stt, tts).run_voice_session("sess-1")

        orch.chat.assert_not_called()
        assert tts.speak.call_count >= 1

    def test_routes_speech_to_orchestrator(self):
        orch = MagicMock()
        orch.chat.return_value = "Hi there"
        stt = MagicMock()
        stt.is_ready = True
        stt.listen.side_effect = ["What time is it?", "exit"]
        tts = MagicMock()

        VoiceInterface(orch, stt, tts).run_voice_session("sess-1")

        orch.chat.assert_called_once_with("sess-1", "What time is it?")
        assert any("Hi there" in str(c) for c in tts.speak.call_args_list)

    def test_aborts_when_stt_not_ready(self):
        orch = MagicMock()
        stt = MagicMock()
        stt.is_ready = False
        tts = MagicMock()

        VoiceInterface(orch, stt, tts).run_voice_session("sess-1")
        stt.listen.assert_not_called()
        orch.chat.assert_not_called()
