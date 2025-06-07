
"""Audio and TTS services for ScriptVoice."""

import tempfile
from gtts import gTTS
from typing import Tuple, Optional


def generate_tts(text: str, speed: float = 1.0) -> Tuple[Optional[str], str]:
    """Generate TTS audio from text."""
    if not text.strip():
        return None, '<div class="status-error">❌ Please enter some text to convert to speech</div>'
    
    try:
        # Create a temporary file for the audio
        tts = gTTS(text=text, lang='en', slow=(speed < 1.0))
        
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name, '<div class="status-success">✅ Audio generated successfully</div>'
    
    except Exception as e:
        return None, f'<div class="status-error">❌ Error generating audio: {str(e)}</div>'
