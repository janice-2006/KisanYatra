"""Text-to-speech playback for translated interface results."""

from io import BytesIO
import json

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

def text_to_speech(text, language_code):
    if not text:
        return None
    if gTTS is None:
        return None
    audio = BytesIO()
    gTTS(text=text, lang=language_code).write_to_fp(audio)
    return audio.getvalue()