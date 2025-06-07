
#!/usr/bin/env python3
"""
Simple launcher for ScriptVoice Gradio Application

Install dependencies with:
    pip install -r requirements.txt

Then run:
    python run.py
"""

if __name__ == "__main__":
    from main import create_interface
    
    print("🚀 Starting ScriptVoice - AI-Powered Story Intelligence Platform")
    print("📦 Make sure you have installed dependencies: pip install -r requirements.txt")
    print("🌐 The app will be available at: http://localhost:7860")
    
    # Create and launch the app
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
