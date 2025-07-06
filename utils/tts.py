import pyttsx3
import threading
import logging

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Text-to-speech error: {str(e)}")
    threading.Thread(target=_speak, daemon=True).start()
