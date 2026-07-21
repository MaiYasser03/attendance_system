import os
import threading
import logging

DISABLE_TTS = os.getenv("DISABLE_TTS", "false").lower() == "true"

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 150)
        _engine.setProperty("volume", 1.0)
    return _engine


def speak(text):
    if DISABLE_TTS:
        logging.info("TTS (disabled): %s", text)
        return

    def _speak():
        try:
            engine = _get_engine()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error("Text-to-speech error: %s", e)

    threading.Thread(target=_speak, daemon=True).start()
