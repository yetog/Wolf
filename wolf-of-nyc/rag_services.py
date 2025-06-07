
"""RAG (Retrieval Augmented Generation) services for ScriptVoice."""

import os
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import PROJECTS_FILE


class RAGService:
    """Handles vector database operations and content retrieval."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        self.index = None
        self.documents = []
        self.metadata = []
        self.index_file = "vector_index.faiss"
        self.metadata_file = "vector_metadata.pkl"
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                print(f"Loaded vector index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Error loading index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create empty FAISS index."""
        dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.metadata = []
    
    def chunk_content(self, content: str, content_type: str, content_id: str, title: str) -> List[Document]:
        """Split content into chunks for embedding."""
        chunks = self.text_splitter.split_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'content_type': content_type,
                    'content_id': content_id,
                    'title': title,
                    'chunk_id': i,
                    'chunk_count': len(chunks)
                }
            )
            documents.append(doc)
        
        return documents
    
    def add_content(self, content: str, content_type: str, content_id: str, title: str):
        """Add content to the vector database."""
        if not content.strip():
            return
        
        # Remove existing content for this ID
        self.remove_content(content_id)
        
        # Chunk the content
        documents = self.chunk_content(content, content_type, content_id, title)
        
        if not documents:
            return
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        for doc in documents:
            self.metadata.append(doc.metadata)
        
        # Save index
        self._save_index()
    
    def remove_content(self, content_id: str):
        """Remove content from vector database."""
        indices_to_remove = []
        for i, metadata in enumerate(self.metadata):
            if metadata.get('content_id') == content_id:
                indices_to_remove.append(i)
        
        if indices_to_remove:
            # Rebuild index without removed items
            new_documents = []
            new_metadata = []
            new_embeddings = []
            
            for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
                if i not in indices_to_remove:
                    new_documents.append(doc)
                    new_metadata.append(meta)
                    embedding = self.model.encode([doc.page_content])
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                    new_embeddings.append(embedding[0])
            
            # Recreate index
            self._create_empty_index()
            if new_embeddings:
                embeddings_array = np.array(new_embeddings).astype('float32')
                self.index.add(embeddings_array)
                self.documents = new_documents
                self.metadata = new_metadata
            
            self._save_index()
    
    def search(self, query: str, k: int = 5, content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content."""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k * 2, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                metadata = self.metadata[idx]
                
                # Filter by content type if specified
                if content_type and metadata.get('content_type') != content_type:
                    continue
                
                result = {
                    'content': self.documents[idx].page_content,
                    'metadata': metadata,
                    'score': float(score)
                }
                results.append(result)
                
                if len(results) >= k:
                    break
        
        return results
    
    def get_context_for_content(self, content_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant context from other content for a specific item."""
        results = self.search(query, k=k)
        # Filter out results from the same content
        filtered_results = [r for r in results if r['metadata'].get('content_id') != content_id]
        return filtered_results[:k]
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def rebuild_index_from_projects(self):
        """Rebuild the entire vector index from current projects data."""
        from models import load_projects
        
        # Clear existing index
        self._create_empty_index()
        
        # Load all projects data
        data = load_projects()
        
        # Add stories
        for story_id, story in data.get("stories", {}).items():
            content = f"{story['title']}\n\n{story['description']}\n\n{story['content']}"
            self.add_content(content, "story", story_id, story['title'])
        
        # Add characters
        for char_id, char in data.get("characters", {}).items():
            content = f"{char['name']}\n\n{char['description']}\n\nTraits: {', '.join(char.get('traits', []))}\n\n{char.get('notes', '')}"
            self.add_content(content, "character", char_id, char['name'])
        
        # Add world elements
        for elem_id, elem in data.get("world_elements", {}).items():
            content = f"{elem['name']} ({elem['type']})\n\n{elem['description']}\n\nTags: {', '.join(elem.get('tags', []))}\n\n{elem.get('notes', '')}"
            self.add_content(content, "world_element", elem_id, elem['name'])
        
        # Add scripts
        for proj_id, proj in data.get("projects", {}).items():
            if proj.get('content'):
                content = f"{proj['name']}\n\n{proj['content']}\n\nNotes: {proj.get('notes', '')}"
                self.add_content(content, "script", proj_id, proj['name'])


# Global RAG service instance
rag_service = RAGService()
