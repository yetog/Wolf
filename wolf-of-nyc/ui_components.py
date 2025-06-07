
"""UI components and styling for ScriptVoice application."""

CUSTOM_CSS = """
/* ScriptVoice Custom CSS - Black/White/Red/Gold Theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    background: #000000 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #FFFFFF !important;
}

.word-count-highlight {
    background: #E63946 !important;
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 1.1em;
    padding: 12px 16px;
    border-radius: 8px;
    border: 2px solid #E63946;
    box-shadow: 0 2px 8px rgba(230, 57, 70, 0.3);
}

.primary-button {
    background: #E63946 !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    border-radius: 6px !important;
    padding: 12px 20px !important;
}

.primary-button:hover {
    background: #d12a36 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(230, 57, 70, 0.4) !important;
}

.secondary-button {
    background: #1a1a1a !important;
    border: 2px solid #FFD700 !important;
    color: #FFD700 !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    border-radius: 6px !important;
    padding: 12px 20px !important;
}

.secondary-button:hover {
    background: #FFD700 !important;
    color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4) !important;
}

.sidebar-column {
    background: #1a1a1a !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-right: 20px !important;
    border: 1px solid #333333 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
}

.status-success {
    color: #FFD700 !important;
    background-color: rgba(255, 215, 0, 0.1) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: 500;
}

.status-error {
    color: #E63946 !important;
    background-color: rgba(230, 57, 70, 0.1) !important;
    border: 1px solid rgba(230, 57, 70, 0.3) !important;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: 500;
}

/* Story Intelligence Styling */
.stories-grid, .characters-grid, .world-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.story-card, .character-card, .world-card {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.story-card:hover, .character-card:hover, .world-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(230, 57, 70, 0.2);
    border-color: #E63946;
}

.story-card h3, .character-card h3, .world-card h3 {
    color: #E63946 !important;
    margin: 0 0 12px 0;
    font-size: 1.2em;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

.story-card p, .character-card p, .world-card p {
    color: #CCCCCC !important;
    margin: 10px 0;
    line-height: 1.5;
    font-weight: 400;
}

.story-card small, .character-card small, .world-card small {
    color: #888888 !important;
    font-size: 0.85em;
    font-weight: 300;
}

.type-badge {
    background: #FFD700 !important;
    color: #000000 !important;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
    margin-left: 10px;
}

/* Tab styling */
.gradio-tabs .tab-nav {
    background: #1a1a1a !important;
    border-radius: 12px !important;
    margin-bottom: 24px !important;
    border: 1px solid #333333 !important;
}

.gradio-tabs .tab-nav button {
    background: transparent !important;
    color: #CCCCCC !important;
    border: none !important;
    padding: 14px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

.gradio-tabs .tab-nav button.selected {
    background: #E63946 !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(230, 57, 70, 0.3) !important;
}

.gradio-tabs .tab-nav button:hover:not(.selected) {
    background: rgba(255, 215, 0, 0.1) !important;
    color: #FFD700 !important;
}

/* Search results styling */
.search-results {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.search-results h4 {
    color: #E63946 !important;
    margin: 0 0 16px 0;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

.search-results h5 {
    color: #FFD700 !important;
    margin: 16px 0 12px 0;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

.search-results ul {
    margin: 0 0 16px 20px;
}

.search-results li {
    color: #CCCCCC !important;
    margin: 8px 0;
    line-height: 1.5;
}

.search-results pre {
    background: #0a0a0a !important;
    border: 1px solid #333333 !important;
    border-radius: 6px;
    padding: 16px;
    color: #FFFFFF !important;
    font-family: 'Inter', monospace;
    overflow-x: auto;
}

/* Section headers */
.section-header {
    background: linear-gradient(45deg, #E63946, #FFD700) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700;
    font-size: 1.3em;
    margin: 24px 0 16px 0;
    padding: 12px 0;
    border-bottom: 2px solid #E63946;
    font-family: 'Inter', sans-serif;
}

/* Form groups */
.gradio-group {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 16px 0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

/* Input styling */
.gradio-textbox input, .gradio-textbox textarea {
    background: #0a0a0a !important;
    border: 2px solid #333333 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    transition: all 0.3s ease !important;
}

.gradio-textbox input:focus, .gradio-textbox textarea:focus {
    border-color: #E63946 !important;
    box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.2) !important;
    outline: none !important;
}

.gradio-textbox input::placeholder, .gradio-textbox textarea::placeholder {
    color: #666666 !important;
}

.gradio-dropdown .wrap {
    background: #0a0a0a !important;
    border: 2px solid #333333 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}

.gradio-dropdown .wrap:focus-within {
    border-color: #E63946 !important;
    box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.2) !important;
}

.gradio-dropdown .wrap .wrap-inner {
    background: #0a0a0a !important;
    color: #FFFFFF !important;
}

.gradio-dropdown .wrap .wrap-inner .token {
    background: #E63946 !important;
    color: #FFFFFF !important;
}

/* Label styling */
.gradio-group label, .gradio-textbox label, .gradio-dropdown label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    margin-bottom: 8px !important;
}

/* Audio player styling */
.gradio-audio {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

/* File upload styling */
.gradio-file {
    background: #1a1a1a !important;
    border: 2px dashed #333333 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}

.gradio-file:hover {
    border-color: #E63946 !important;
    background: rgba(230, 57, 70, 0.05) !important;
}

/* Progress bars */
.gradio-progress {
    background: #333333 !important;
    border-radius: 4px !important;
}

.gradio-progress .progress-bar {
    background: linear-gradient(45deg, #E63946, #FFD700) !important;
    border-radius: 4px !important;
}

/* Slider styling */
.gradio-slider input[type="range"] {
    background: #333333 !important;
}

.gradio-slider input[type="range"]::-webkit-slider-thumb {
    background: #E63946 !important;
    border: 2px solid #FFFFFF !important;
}

.gradio-slider input[type="range"]::-moz-range-thumb {
    background: #E63946 !important;
    border: 2px solid #FFFFFF !important;
}

/* Checkbox styling */
.gradio-checkbox input[type="checkbox"]:checked {
    background: #E63946 !important;
    border-color: #E63946 !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #E63946;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #d12a36;
}

/* Overall container styling */
body, .gradio-container, .gradio-container > div {
    background: #000000 !important;
    color: #FFFFFF !important;
}

/* Ensure proper contrast for all text elements */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: inherit !important;
}
"""

def get_header_html():
    """Generate the application header HTML."""
    return """
    <div style="text-align: center; padding: 40px 0; background: #000000; border-radius: 15px; margin-bottom: 30px; border: 2px solid #E63946; box-shadow: 0 4px 16px rgba(230, 57, 70, 0.3);">
        <h1 style="margin: 0; font-size: 3.2em; background: linear-gradient(45deg, #E63946, #FFD700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-family: 'Inter', sans-serif;">
            🎬 ScriptVoice
        </h1>
        <p style="margin: 15px 0 0 0; color: #FFFFFF; font-size: 1.3em; font-weight: 400; font-family: 'Inter', sans-serif;">
            AI-Powered Story Intelligence Platform
        </p>
        <p style="margin: 8px 0 0 0; color: #CCCCCC; font-size: 1em; font-weight: 300; font-family: 'Inter', sans-serif;">
            Transform your stories with intelligent script writing, voice synthesis, and creative knowledge management
        </p>
    </div>
    """

def get_section_header(title):
    """Generate a section header with consistent styling."""
    return f'<div class="section-header">{title}</div>'
