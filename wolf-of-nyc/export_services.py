
"""Export services for ScriptVoice."""

import tempfile
from gtts import gTTS
from models import load_projects
from typing import Tuple, Optional


def export_project(project_id: str, export_type: str) -> Tuple[Optional[str], str]:
    """Export project content."""
    if not project_id:
        return None, '<div class="status-error">❌ No project selected</div>'
    
    data = load_projects()
    if project_id not in data["projects"]:
        return None, '<div class="status-error">❌ Project not found</div>'
    
    project = data["projects"][project_id]
    
    if export_type == "text":
        # Create text file
        content = f"Project: {project['name']}\n"
        content += f"Created: {project['created_at']}\n"
        content += f"Word Count: {project['word_count']}\n\n"
        content += "SCRIPT:\n" + "="*50 + "\n"
        content += project['content'] + "\n\n"
        content += "NOTES:\n" + "="*50 + "\n"
        content += project['notes']
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            return tmp_file.name, '<div class="status-success">✅ Text file exported</div>'
    
    elif export_type == "audio":
        # Generate TTS audio
        if not project['content'].strip():
            return None, '<div class="status-error">❌ No content to convert to audio</div>'
        
        try:
            tts = gTTS(text=project['content'], lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name, '<div class="status-success">✅ Audio file exported</div>'
        except Exception as e:
            return None, f'<div class="status-error">❌ Error generating audio: {str(e)}</div>'
    
    return None, '<div class="status-error">❌ Invalid export type</div>'
