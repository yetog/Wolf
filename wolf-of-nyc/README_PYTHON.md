
# ScriptVoice - Pure Python Gradio Application

## 🎯 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python main.py
   ```
   or
   ```bash
   python run.py
   ```

3. **Access the Application**
   Open your browser to: http://localhost:7860

## 📦 Dependencies

This application uses the following Python packages:
- `gradio>=4.0.0` - Web interface framework
- `gtts>=2.3.0` - Text-to-speech
- `pytesseract>=0.3.10` - OCR text extraction
- `Pillow>=10.0.0` - Image processing
- `langchain>=0.1.0` - LLM framework
- `sentence-transformers>=2.2.0` - Text embeddings
- `faiss-cpu>=1.7.0` - Vector database
- `langchain-openai>=0.1.0` - OpenAI integration
- `tiktoken>=0.5.0` - Text tokenization
- pip install gtts gradio faiss-cpu numpy pandas requests pytesseract

## 🌟 Features

### Scripts Tab
- Project management and script editing
- Text-to-speech generation
- OCR text extraction from images
- AI script enhancement
- Export functionality

### Story Intelligence Tab
- Knowledge Assistant with command processing
- Context-aware AI tools
- Character and story management
- World building elements
- RAG-powered search and analysis

## 🔧 Configuration

- All data is stored in local JSON files
- No external database required
- Gradio handles the web interface automatically

## 🚀 Deployment

The application is ready for deployment on platforms that support Python:
- Hugging Face Spaces
- Railway
- Render
- Heroku
- Any Python hosting service

For production deployment, make sure to:
1. Set appropriate server configurations
2. Configure environment variables if needed
3. Ensure all dependencies are installed
