
import gradio as gr
import json
import os
from datetime import datetime
import tempfile
import io
from gtts import gTTS
import pytesseract
from PIL import Image
import zipfile

# Global state for projects
PROJECTS_FILE = "projects.json"
current_project_id = None

def load_projects():
    """Load projects from JSON file"""
    if os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    return {
        "projects": {
            "1": {
                "id": "1",
                "name": "Sample Script",
                "content": "Welcome to ScriptVoice! This is your first script. Start editing to create amazing voice content.",
                "notes": "This is a sample note for your script.",
                "created_at": datetime.now().isoformat(),
                "word_count": 0
            }
        },
        "settings": {
            "dyslexic_mode": False,
            "voice_speed": 1.0,
            "voice_volume": 1.0
        }
    }

def save_projects(data):
    """Save projects to JSON file"""
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_word_count(text):
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())

def update_word_count(text):
    """Update word count display"""
    count = get_word_count(text)
    return f"**Word Count:** {count}"

def create_new_project(name):
    """Create a new project"""
    if not name.strip():
        return "Please enter a project name", None
    
    data = load_projects()
    new_id = str(len(data["projects"]) + 1)
    
    data["projects"][new_id] = {
        "id": new_id,
        "name": name.strip(),
        "content": "",
        "notes": "",
        "created_at": datetime.now().isoformat(),
        "word_count": 0
    }
    
    save_projects(data)
    
    # Return updated project choices and select the new project
    choices = [(proj["name"], proj_id) for proj_id, proj in data["projects"].items()]
    return f"Project '{name}' created successfully!", gr.update(choices=choices, value=new_id)

def load_project(project_id):
    """Load a specific project"""
    if not project_id:
        return "", "", "**Word Count:** 0"
    
    global current_project_id
    current_project_id = project_id
    
    data = load_projects()
    if project_id in data["projects"]:
        project = data["projects"][project_id]
        word_count = get_word_count(project["content"])
        return project["content"], project["notes"], f"**Word Count:** {word_count}"
    
    return "", "", "**Word Count:** 0"

def save_script_content(project_id, content, notes):
    """Save script content and notes"""
    if not project_id:
        return "No project selected"
    
    data = load_projects()
    if project_id in data["projects"]:
        data["projects"][project_id]["content"] = content
        data["projects"][project_id]["notes"] = notes
        data["projects"][project_id]["word_count"] = get_word_count(content)
        save_projects(data)
        return "✅ Saved successfully"
    
    return "❌ Error saving"

def generate_tts(text, speed=1.0):
    """Generate TTS audio from text"""
    if not text.strip():
        return None, "Please enter some text to convert to speech"
    
    try:
        # Create a temporary file for the audio
        tts = gTTS(text=text, lang='en', slow=(speed < 1.0))
        
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name, "✅ Audio generated successfully"
    
    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"

def extract_text_from_image(image):
    """Extract text from uploaded image using OCR"""
    if image is None:
        return "", "Please upload an image"
    
    try:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(Image.open(image))
        if text.strip():
            return text.strip(), "✅ Text extracted successfully"
        else:
            return "", "No text found in the image"
    
    except Exception as e:
        return "", f"❌ Error extracting text: {str(e)}"

def enhance_script_placeholder(text, enhancement_type):
    """Placeholder for AI script enhancement"""
    enhancements = {
        "dramatic": f"[DRAMATIC VERSION]\n{text}\n\n(Note: AI enhancement feature coming soon!)",
        "romantic": f"[ROMANTIC VERSION]\n{text}\n\n(Note: AI enhancement feature coming soon!)",
        "professional": f"[PROFESSIONAL VERSION]\n{text}\n\n(Note: AI enhancement feature coming soon!)",
        "casual": f"[CASUAL VERSION]\n{text}\n\n(Note: AI enhancement feature coming soon!)"
    }
    
    return enhancements.get(enhancement_type, text), "✅ Enhancement applied (demo mode)"

def export_project(project_id, export_type):
    """Export project content"""
    if not project_id:
        return None, "No project selected"
    
    data = load_projects()
    if project_id not in data["projects"]:
        return None, "Project not found"
    
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
            return tmp_file.name, "✅ Text file exported"
    
    elif export_type == "audio":
        # Generate TTS audio
        if not project['content'].strip():
            return None, "No content to convert to audio"
        
        try:
            tts = gTTS(text=project['content'], lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name, "✅ Audio file exported"
        except Exception as e:
            return None, f"❌ Error generating audio: {str(e)}"
    
    return None, "Invalid export type"

# Initialize the Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    # Load initial projects
    data = load_projects()
    project_choices = [(proj["name"], proj_id) for proj_id, proj in data["projects"].items()]
    
    with gr.Blocks(title="ScriptVoice - TTS Script Editor", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">
                <span style="color: #ff6b6b;">Script</span><span style="color: #ffd93d;">Voice</span>
            </h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">AI-Powered TTS Script Editor</p>
        </div>
        """)
        
        with gr.Row():
            # Left Sidebar
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📁 Projects")
                
                # New Project Section
                with gr.Group():
                    new_project_name = gr.Textbox(label="New Project Name", placeholder="Enter project name...")
                    create_btn = gr.Button("➕ Create Project", variant="primary")
                    create_status = gr.Textbox(label="Status", interactive=False, visible=False)
                
                # Project Selection
                project_dropdown = gr.Dropdown(
                    choices=project_choices,
                    label="Select Project",
                    value=list(data["projects"].keys())[0] if data["projects"] else None
                )
                
                # Notes Section
                gr.Markdown("### 📝 Notes")
                notes_textbox = gr.Textbox(
                    label="Project Notes",
                    placeholder="Add your notes here...",
                    lines=5,
                    max_lines=10
                )
                
                # Settings Section
                gr.Markdown("### ⚙️ Settings")
                with gr.Group():
                    dyslexic_mode = gr.Checkbox(label="Dyslexic-friendly font", value=False)
                    voice_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Voice Speed")
                    voice_volume = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Voice Volume")
            
            # Main Editor Panel
            with gr.Column(scale=2):
                # Word Count Display
                word_count_display = gr.Markdown("**Word Count:** 0")
                
                # Script Editor
                script_textbox = gr.Textbox(
                    label="Script Editor",
                    placeholder="Start writing your script here...",
                    lines=15,
                    max_lines=25
                )
                
                # Control Buttons Row
                with gr.Row():
                    save_btn = gr.Button("💾 Save", variant="secondary")
                    tts_btn = gr.Button("🔊 Play TTS", variant="primary")
                    save_status = gr.Textbox(label="Save Status", interactive=False, visible=False)
                
                # TTS Audio Output
                audio_output = gr.Audio(label="Generated Audio")
                tts_status = gr.Textbox(label="TTS Status", interactive=False, visible=False)
                
                # OCR Section
                with gr.Group():
                    gr.Markdown("### 📷 Extract Text from Image")
                    with gr.Row():
                        image_input = gr.Image(type="filepath", label="Upload Image")
                        ocr_btn = gr.Button("Extract Text")
                    ocr_status = gr.Textbox(label="OCR Status", interactive=False, visible=False)
                
                # AI Enhancement Section
                with gr.Group():
                    gr.Markdown("### 🤖 AI Script Enhancement")
                    with gr.Row():
                        enhancement_type = gr.Dropdown(
                            choices=["dramatic", "romantic", "professional", "casual"],
                            label="Enhancement Style",
                            value="dramatic"
                        )
                        enhance_btn = gr.Button("✨ Enhance Script")
                    enhance_status = gr.Textbox(label="Enhancement Status", interactive=False, visible=False)
                
                # Export Section
                with gr.Group():
                    gr.Markdown("### 📤 Export")
                    with gr.Row():
                        export_type = gr.Dropdown(
                            choices=["text", "audio"],
                            label="Export Type",
                            value="text"
                        )
                        export_btn = gr.Button("📥 Export")
                    export_file = gr.File(label="Download")
                    export_status = gr.Textbox(label="Export Status", interactive=False, visible=False)
        
        # Event Handlers
        
        # Create new project
        create_btn.click(
            fn=create_new_project,
            inputs=[new_project_name],
            outputs=[create_status, project_dropdown]
        ).then(
            lambda: ("", gr.update(visible=True)),
            outputs=[new_project_name, create_status]
        )
        
        # Load project when selected
        project_dropdown.change(
            fn=load_project,
            inputs=[project_dropdown],
            outputs=[script_textbox, notes_textbox, word_count_display]
        )
        
        # Update word count as user types
        script_textbox.change(
            fn=update_word_count,
            inputs=[script_textbox],
            outputs=[word_count_display]
        )
        
        # Save script content
        save_btn.click(
            fn=save_script_content,
            inputs=[project_dropdown, script_textbox, notes_textbox],
            outputs=[save_status]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[save_status]
        )
        
        # Generate TTS
        tts_btn.click(
            fn=generate_tts,
            inputs=[script_textbox, voice_speed],
            outputs=[audio_output, tts_status]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[tts_status]
        )
        
        # OCR text extraction
        ocr_btn.click(
            fn=extract_text_from_image,
            inputs=[image_input],
            outputs=[script_textbox, ocr_status]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[ocr_status]
        )
        
        # AI Enhancement
        enhance_btn.click(
            fn=enhance_script_placeholder,
            inputs=[script_textbox, enhancement_type],
            outputs=[script_textbox, enhance_status]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[enhance_status]
        )
        
        # Export functionality
        export_btn.click(
            fn=export_project,
            inputs=[project_dropdown, export_type],
            outputs=[export_file, export_status]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[export_status]
        )
    
    return app

if __name__ == "__main__":
    # Create and launch the app
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
