"""
jarvis/voice/interface.py
─────────────────────────
Voice-mode session loop: listen → orchestrator → speak.
"""

from __future__ import annotations

import time

from jarvis.core.orchestrator import Orchestrator
from jarvis.utils.logging import get_logger
from jarvis.voice.stt import SpeechToText
from jarvis.voice.tts import TextToSpeech

log = get_logger(__name__)

_EXIT_PHRASES = frozenset({
    "exit",
    "quit",
    "stop",
    "goodbye",
    "good bye",
    "stop listening",
})


class VoiceInterface:
    """
    Hands-free conversation loop.

    Args:
        orchestrator: The Plan-and-Execute orchestrator.
        stt:          Speech-to-text engine.
        tts:          Text-to-speech engine.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        stt: SpeechToText,
        tts: TextToSpeech,
    ) -> None:
        self._orchestrator = orchestrator
        self._stt = stt
        self._tts = tts

    def run_voice_session(self, session_id: str) -> None:
        """
        Continuously listen and respond until the user says exit/quit
        or presses Ctrl+C.
        """
        log.info("voice_session_start", session_id=session_id)

        if not self._stt.is_ready:
            print(
                "Voice input is unavailable (Whisper failed to load). "
                "Returning to text mode."
            )
            log.error("voice_session_aborted_stt_not_ready")
            return

        print("Voice mode active. Say 'exit' or 'quit' to stop. (Ctrl+C also works.)")
        self._tts.speak("Voice mode ready. How can I help?")

        try:
            while True:
                print("\nListening...")
                log.info("voice_listening", session_id=session_id)

                try:
                    user_input = self._stt.listen()
                except Exception as e:
                    log.error("voice_listen_error", error=str(e))
                    print(f"Could not hear you: {e}")
                    time.sleep(3)
                    continue

                # Back off if microphone returned an error or silence
                if not user_input:
                    print("(No speech detected — try again.)")
                    time.sleep(3)
                    continue

                if user_input.startswith("ERROR:"):
                    print(f"⚠ {user_input}")
                    log.warning("voice_stt_error", message=user_input)
                    time.sleep(3)
                    continue

                print(f"You: {user_input}")
                normalised = user_input.strip().lower().rstrip(".!")
                if normalised in _EXIT_PHRASES:
                    print("Ending voice mode.")
                    self._tts.speak("Goodbye.")
                    break

                try:
                    print("Thinking...")
                    response = self._orchestrator.chat(session_id, user_input)
                except Exception as e:
                    log.error("voice_orchestrator_error", error=str(e))
                    print(f"Error: {e}")
                    self._tts.speak(
                        "Sorry, I ran into a problem. Is Ollama running?"
                    )
                    continue

                print(f"JARVIS: {response}\n")
                try:
                    self._tts.speak(response)
                except Exception as e:
                    log.error("voice_tts_error", error=str(e))
                    print(f"(Could not speak response: {e})")

        except KeyboardInterrupt:
            print("\nVoice mode interrupted.")
            log.info("voice_session_interrupted", session_id=session_id)

        log.info("voice_session_end", session_id=session_id)
