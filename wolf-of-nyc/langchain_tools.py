
"""LangChain tools and chains for context-aware AI enhancement."""

from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from rag_services import rag_service


class ContextAwareEnhancer:
    """Handles context-aware script enhancement using RAG."""
    
    def __init__(self):
        self.enhancement_prompts = {
            "dramatic": """
You are enhancing a script with dramatic flair. Use the following context about characters and story elements to make the enhancement more consistent and engaging.

CONTEXT:
{context}

ORIGINAL SCRIPT:
{script}

Please enhance this script with dramatic elements while maintaining consistency with the established characters and world. Focus on:
- Heightened emotional stakes
- Compelling character motivations
- Dramatic tension and conflict
- Rich, evocative language

ENHANCED SCRIPT:
""",
            "romantic": """
You are enhancing a script with romantic elements. Use the following context about characters and relationships to create authentic romantic moments.

CONTEXT:
{context}

ORIGINAL SCRIPT:
{script}

Please enhance this script with romantic elements while staying true to the established characters and their relationships. Focus on:
- Emotional intimacy and connection
- Character chemistry and dynamics
- Tender, heartfelt dialogue
- Romantic atmosphere and mood

ENHANCED SCRIPT:
""",
            "professional": """
You are enhancing a script for professional presentation. Use the following context to ensure accuracy and consistency.

CONTEXT:
{context}

ORIGINAL SCRIPT:
{script}

Please enhance this script for a professional context while maintaining consistency with established facts and characters. Focus on:
- Clear, authoritative language
- Proper structure and flow
- Professional tone and delivery
- Accurate information and details

ENHANCED SCRIPT:
""",
            "casual": """
You are making a script more casual and conversational. Use the following context about characters to match their established personalities.

CONTEXT:
{context}

ORIGINAL SCRIPT:
{script}

Please enhance this script with a casual, conversational tone while keeping character voices consistent. Focus on:
- Natural, everyday language
- Relaxed, friendly tone
- Character-appropriate dialogue
- Conversational flow and rhythm

ENHANCED SCRIPT:
""",
            "character_consistent": """
You are enhancing a script to be more consistent with established characters. Use the character information below to guide your enhancement.

CHARACTER CONTEXT:
{context}

ORIGINAL SCRIPT:
{script}

Please enhance this script to be more consistent with the established characters. Ensure:
- Dialogue matches character personalities and speech patterns
- Actions align with character motivations
- Character relationships are honored
- Character development feels authentic

ENHANCED SCRIPT:
""",
            "plot_coherent": """
You are enhancing a script to improve plot coherence. Use the story context below to ensure consistency.

STORY CONTEXT:
{context}

ORIGINAL SCRIPT:
{script}

Please enhance this script to improve plot coherence and consistency. Focus on:
- Logical story progression
- Consistent world-building elements
- Proper setup and payoff
- Clear cause and effect relationships

ENHANCED SCRIPT:
"""
        }
    
    def get_relevant_context(self, script: str, enhancement_type: str, max_context_length: int = 1000) -> str:
        """Get relevant context for script enhancement."""
        if not script.strip():
            return "No relevant context found."
        
        # Search for relevant content
        results = rag_service.search(script, k=5)
        
        if not results:
            return "No relevant context found."
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for result in results:
            metadata = result['metadata']
            content = result['content']
            
            # Create context entry
            context_entry = f"\n--- {metadata.get('content_type', 'Content').title()}: {metadata.get('title', 'Unknown')} ---\n{content}\n"
            
            if current_length + len(context_entry) > max_context_length:
                break
            
            context_parts.append(context_entry)
            current_length += len(context_entry)
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."
    
    def enhance_script_with_context(self, script: str, enhancement_type: str) -> tuple[str, str]:
        """Enhance script using relevant context from the knowledge base."""
        if enhancement_type not in self.enhancement_prompts:
            return script, f'<div class="status-error">❌ Unknown enhancement type: {enhancement_type}</div>'
        
        # Get relevant context
        context = self.get_relevant_context(script, enhancement_type)
        
        # For now, return a placeholder with context info
        enhanced_script = f"""[CONTEXT-AWARE {enhancement_type.upper()} ENHANCEMENT]

RELEVANT CONTEXT FOUND:
{context[:300]}{'...' if len(context) > 300 else ''}

ENHANCED SCRIPT:
{script}

(Note: Full LLM integration will be added when API keys are configured)
"""
        
        status_message = f'<div class="status-success">✅ Enhanced with {enhancement_type} style using relevant context from knowledge base</div>'
        return enhanced_script, status_message
    
    def analyze_character_consistency(self, script: str) -> str:
        """Analyze script for character consistency."""
        # Search for character-related content
        results = rag_service.search(script, k=3, content_type="character")
        
        if not results:
            return "No character information found in knowledge base."
        
        analysis = "CHARACTER CONSISTENCY ANALYSIS:\n\n"
        for result in results:
            char_name = result['metadata'].get('title', 'Unknown Character')
            analysis += f"• {char_name}: Found in knowledge base\n"
            analysis += f"  Context: {result['content'][:100]}...\n\n"
        
        return analysis
    
    def suggest_story_elements(self, script: str) -> str:
        """Suggest relevant story elements for the script."""
        # Search across all content types
        story_results = rag_service.search(script, k=2, content_type="story")
        world_results = rag_service.search(script, k=2, content_type="world_element")
        
        suggestions = "STORY ELEMENT SUGGESTIONS:\n\n"
        
        if story_results:
            suggestions += "Related Stories:\n"
            for result in story_results:
                title = result['metadata'].get('title', 'Unknown')
                suggestions += f"• {title}\n"
        
        if world_results:
            suggestions += "\nRelevant World Elements:\n"
            for result in world_results:
                title = result['metadata'].get('title', 'Unknown')
                elem_type = result['metadata'].get('content_type', 'element')
                suggestions += f"• {title} ({elem_type})\n"
        
        if not story_results and not world_results:
            suggestions += "No relevant story elements found."
        
        return suggestions


# Global enhancer instance
context_enhancer = ContextAwareEnhancer()
