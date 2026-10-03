"""
jarvis/voice/interface.py
─────────────────────────
Voice-mode session loops (v0.27).

Two modes:
  - run_voice_session()  — the v0.4 always-listening loop (legacy, kept).
  - run_push_to_talk()   — v0.27 push-to-talk: one explicit turn per key
    press; an observable lifecycle (IDLE/LISTENING/TRANSCRIBING/THINKING/
    SPEAKING/ERROR); speaks ONLY the final grounded response (Part 17);
    clean cancellation boundary on Ctrl+C (Part 7); a failed turn never
    corrupts the session (Part 6).
"""

from __future__ import annotations

import time

from jarvis.core.orchestrator import Orchestrator
from jarvis.multimodal.service import VoiceTurn
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
    Voice conversation loops over the SAME orchestrator (no separate voice
    agent): STT → normalized text → orchestrator.chat → grounded text → TTS.
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

    # ── Legacy always-listening loop (v0.4 behavior, preserved) ──────────

    def run_voice_session(self, session_id: str) -> None:
        """
        Continuously listen and respond until the user says exit/quit
        or presses Ctrl+C.
        """
        log.info("voice_session_start", session_id=session_id, mode="always_listening")

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

    # ── v0.27 push-to-talk loop (Part 3) ─────────────────────────────────

    def run_push_to_talk(self, session_id: str, max_turns: int | None = None) -> None:
        """
        Push-to-talk conversation: each turn starts with an explicit Enter
        press, captures one utterance, routes it through the normal runtime
        (grounding included) and speaks the FINAL validated response.

        No background listening, no wake word, no persistent capture
        (Part 3). Ctrl+C cancels current playback/turn cleanly.
        """
        log.info("voice_session_start", session_id=session_id, mode="push_to_talk")

        if not self._stt.is_ready:
            print(
                "Voice input is unavailable (Whisper failed to load). "
                "Returning to text mode."
            )
            log.error("voice_session_aborted_stt_not_ready")
            return
        if not self._tts.is_playback_available:
            print(
                "Audio playback unavailable (ffplay missing). Responses will "
                "be printed as text only."
            )

        print(
            "Push-to-talk voice mode. Press ENTER and speak "
            f"(up to {self._stt.record_seconds:.0f}s per turn). "
            "Type 'q' + ENTER to quit. Ctrl+C cancels playback."
        )
        self._tts.speak("Push to talk ready.")

        turns = 0
        try:
            while max_turns is None or turns < max_turns:
                try:
                    cmd = input("\n[ENTER] = speak, 'q' = quit > ")
                except EOFError:
                    break
                if cmd.strip().lower() in ("q", "quit", "exit"):
                    break

                turn = VoiceTurn(self._stt, self._tts, self._orchestrator, session_id)
                response = turn.run()
                turns += 1

                if turn.state.value == "error":
                    print("(Turn failed — you can start a new one; the session is intact.)")
                    continue
                if response is None:
                    print("(No speech detected — press ENTER and try again.)")
                    continue
                print(f"JARVIS: {response}")
        except KeyboardInterrupt:
            # Barge-in boundary: stop playback immediately, end cleanly.
            self._tts.stop()
            print("\nVoice mode cancelled (playback stopped).")
            log.info("voice_session_interrupted", session_id=session_id)

        log.info("voice_session_end", session_id=session_id, turns=turns)
