
"""AI enhancement services for ScriptVoice with RAG integration."""

from typing import Tuple
from langchain_tools import context_enhancer
from rag_services import rag_service


def enhance_script_placeholder(text: str, enhancement_type: str) -> Tuple[str, str]:
    """Enhanced script enhancement with context awareness."""
    if not text.strip():
        return text, '<div class="status-error">❌ Please provide text to enhance</div>'
    
    # Use context-aware enhancement
    enhanced_text, status = context_enhancer.enhance_script_with_context(text, enhancement_type)
    
    return enhanced_text, status


def enhance_script_with_context(text: str, enhancement_type: str) -> Tuple[str, str]:
    """Context-aware script enhancement using RAG."""
    return context_enhancer.enhance_script_with_context(text, enhancement_type)


def analyze_character_consistency(text: str) -> Tuple[str, str]:
    """Analyze character consistency in the provided text."""
    if not text.strip():
        return "", '<div class="status-error">❌ Please provide text to analyze</div>'
    
    analysis = context_enhancer.analyze_character_consistency(text)
    
    return analysis, '<div class="status-success">✅ Character consistency analysis complete</div>'


def suggest_story_elements(text: str) -> Tuple[str, str]:
    """Suggest relevant story elements for the text."""
    if not text.strip():
        return "", '<div class="status-error">❌ Please provide text for suggestions</div>'
    
    suggestions = context_enhancer.suggest_story_elements(text)
    
    return suggestions, '<div class="status-success">✅ Story element suggestions generated</div>'


def update_knowledge_base(content_type: str, content_id: str, title: str, content: str) -> str:
    """Update the knowledge base with new or modified content."""
    try:
        rag_service.add_content(content, content_type, content_id, title)
        return '<div class="status-success">✅ Knowledge base updated</div>'
    except Exception as e:
        return f'<div class="status-error">❌ Error updating knowledge base: {str(e)}</div>'


def remove_from_knowledge_base(content_id: str) -> str:
    """Remove content from the knowledge base."""
    try:
        rag_service.remove_content(content_id)
        return '<div class="status-success">✅ Content removed from knowledge base</div>'
    except Exception as e:
        return f'<div class="status-error">❌ Error removing from knowledge base: {str(e)}</div>'
