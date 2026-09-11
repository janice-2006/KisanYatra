"""Text-to-speech playback for translated interface results."""

from io import BytesIO

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

def text_to_speech(text, language_code):
    """Convert text string to audio bytes matching the given language code."""
    if not text:
        return None
    if gTTS is None:
        return None
    
    # Map any custom language codes securely if needed
    lang_map = {
        "en": "en",
        "hi": "hi",
        "ta": "ta",
        "te": "te",
        "ml": "ml",
        "pa": "pa",
        "bn": "bn"
    }
    target_lang = lang_map.get(language_code, "en")
    
    try:
        audio = BytesIO()
        tts = gTTS(text=str(text), lang=target_lang, slow=False)
        tts.write_to_fp(audio)
        return audio.getvalue()
    except Exception:
        return None