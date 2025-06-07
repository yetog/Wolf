
"""Knowledge assistant for intelligent content queries."""

from typing import List, Dict, Any, Tuple
from rag_services import rag_service


class KnowledgeAssistant:
    """Provides intelligent assistance based on the knowledge base."""
    
    def __init__(self):
        self.commands = {
            "!search": self.search_knowledge,
            "!characters": self.list_characters,
            "!stories": self.list_stories,
            "!world": self.list_world_elements,
            "!analyze": self.analyze_content,
            "!suggest": self.suggest_related,
            "!consistency": self.check_consistency,
            "!rebuild": self.rebuild_index
        }
    
    def process_query(self, query: str) -> str:
        """Process user queries and provide intelligent responses."""
        query = query.strip()
        
        # Check for commands
        if query.startswith("!"):
            command_parts = query.split(" ", 1)
            command = command_parts[0]
            args = command_parts[1] if len(command_parts) > 1 else ""
            
            if command in self.commands:
                return self.commands[command](args)
            else:
                return f"Unknown command: {command}\nAvailable commands: {', '.join(self.commands.keys())}"
        
        # Regular search query
        return self.search_knowledge(query)
    
    def search_knowledge(self, query: str) -> str:
        """Search across all knowledge base content."""
        if not query.strip():
            return "Please provide a search query."
        
        results = rag_service.search(query, k=5)
        
        if not results:
            return f"No results found for: {query}"
        
        response = f"🔍 SEARCH RESULTS FOR: '{query}'\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            content_type = metadata.get('content_type', 'content')
            title = metadata.get('title', 'Unknown')
            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            score = result['score']
            
            response += f"{i}. {content_type.title()}: {title} (Relevance: {score:.2f})\n"
            response += f"   {content}\n\n"
        
        return response
    
    def list_characters(self, query: str = "") -> str:
        """List characters in the knowledge base."""
        results = rag_service.search(query if query else "character", k=10, content_type="character")
        
        if not results:
            return "No characters found in knowledge base."
        
        response = "👥 CHARACTERS IN KNOWLEDGE BASE:\n\n"
        for result in results:
            title = result['metadata'].get('title', 'Unknown')
            content = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
            response += f"• {title}\n  {content}\n\n"
        
        return response
    
    def list_stories(self, query: str = "") -> str:
        """List stories in the knowledge base."""
        results = rag_service.search(query if query else "story", k=10, content_type="story")
        
        if not results:
            return "No stories found in knowledge base."
        
        response = "📚 STORIES IN KNOWLEDGE BASE:\n\n"
        for result in results:
            title = result['metadata'].get('title', 'Unknown')
            content = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
            response += f"• {title}\n  {content}\n\n"
        
        return response
    
    def list_world_elements(self, query: str = "") -> str:
        """List world elements in the knowledge base."""
        results = rag_service.search(query if query else "world", k=10, content_type="world_element")
        
        if not results:
            return "No world elements found in knowledge base."
        
        response = "🌍 WORLD ELEMENTS IN KNOWLEDGE BASE:\n\n"
        for result in results:
            title = result['metadata'].get('title', 'Unknown')
            content = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
            response += f"• {title}\n  {content}\n\n"
        
        return response
    
    def analyze_content(self, content: str) -> str:
        """Analyze provided content against the knowledge base."""
        if not content.strip():
            return "Please provide content to analyze."
        
        # Find related content
        results = rag_service.search(content, k=5)
        
        response = "📊 CONTENT ANALYSIS:\n\n"
        
        if results:
            response += "Related content found:\n"
            for result in results:
                metadata = result['metadata']
                content_type = metadata.get('content_type', 'content')
                title = metadata.get('title', 'Unknown')
                score = result['score']
                response += f"• {content_type.title()}: {title} (Similarity: {score:.2f})\n"
        else:
            response += "No related content found in knowledge base."
        
        return response
    
    def suggest_related(self, content: str) -> str:
        """Suggest related content based on input."""
        if not content.strip():
            return "Please provide content for suggestions."
        
        # Get diverse suggestions
        char_results = rag_service.search(content, k=2, content_type="character")
        story_results = rag_service.search(content, k=2, content_type="story")
        world_results = rag_service.search(content, k=2, content_type="world_element")
        
        response = "💡 SUGGESTIONS BASED ON YOUR CONTENT:\n\n"
        
        if char_results:
            response += "Relevant Characters:\n"
            for result in char_results:
                title = result['metadata'].get('title', 'Unknown')
                response += f"• {title}\n"
            response += "\n"
        
        if story_results:
            response += "Related Stories:\n"
            for result in story_results:
                title = result['metadata'].get('title', 'Unknown')
                response += f"• {title}\n"
            response += "\n"
        
        if world_results:
            response += "Relevant World Elements:\n"
            for result in world_results:
                title = result['metadata'].get('title', 'Unknown')
                response += f"• {title}\n"
            response += "\n"
        
        if not any([char_results, story_results, world_results]):
            response += "No related content found in knowledge base."
        
        return response
    
    def check_consistency(self, content: str) -> str:
        """Check content consistency against knowledge base."""
        if not content.strip():
            return "Please provide content to check for consistency."
        
        from langchain_tools import context_enhancer
        analysis = context_enhancer.analyze_character_consistency(content)
        
        response = "✅ CONSISTENCY CHECK:\n\n"
        response += analysis
        
        return response
    
    def rebuild_index(self, args: str = "") -> str:
        """Rebuild the vector index from current data."""
        try:
            rag_service.rebuild_index_from_projects()
            return "✅ Knowledge base index rebuilt successfully!"
        except Exception as e:
            return f"❌ Error rebuilding index: {str(e)}"


# Global assistant instance
knowledge_assistant = KnowledgeAssistant()
