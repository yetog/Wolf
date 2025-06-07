"""RAG (Retrieval Augmented Generation) services for ScriptVoice with ChromaDB."""

import os
import pickle
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import PROJECTS_FILE


class RAGService:
    """Handles vector database operations and content retrieval using Chroma."""

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        self.documents = []
        self.metadata = []

        # Set up Chroma client and collection
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="scriptvoice_vectors")

    def chunk_content(self, content: str, content_type: str, content_id: str, title: str) -> List[Document]:
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
        if not content.strip():
            return

        self.remove_content(content_id)
        documents = self.chunk_content(content, content_type, content_id, title)
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        ids = [f"{content_id}_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(documents=texts, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)
        self.documents.extend(documents)
        self.metadata.extend(metadatas)

    def remove_content(self, content_id: str):
        ids_to_delete = [md['content_id'] for md in self.metadata if md.get('content_id') == content_id]
        if ids_to_delete:
            self.collection.delete(where={"content_id": content_id})
            self.documents = [doc for doc in self.documents if doc.metadata.get('content_id') != content_id]
            self.metadata = [md for md in self.metadata if md.get('content_id') != content_id]

    def search(self, query: str, k: int = 5, content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.collection.count():
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k*2)

        final_results = []
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            if content_type and meta.get('content_type') != content_type:
                continue
            final_results.append({
                "content": results['documents'][0][i],
                "metadata": meta,
                "score": float(results['distances'][0][i])
            })
            if len(final_results) >= k:
                break
        return final_results

    def get_context_for_content(self, content_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        results = self.search(query, k=k)
        return [r for r in results if r['metadata'].get('content_id') != content_id][:k]

    def rebuild_index_from_projects(self):
        from models import load_projects
        self.collection.delete()  # Clear the whole collection
        self.documents.clear()
        self.metadata.clear()

        data = load_projects()
        for story_id, story in data.get("stories", {}).items():
            content = f"{story['title']}\n\n{story['description']}\n\n{story['content']}"
            self.add_content(content, "story", story_id, story['title'])

        for char_id, char in data.get("characters", {}).items():
            content = f"{char['name']}\n\n{char['description']}\n\nTraits: {', '.join(char.get('traits', []))}\n\n{char.get('notes', '')}"
            self.add_content(content, "character", char_id, char['name'])

        for elem_id, elem in data.get("world_elements", {}).items():
            content = f"{elem['name']} ({elem['type']})\n\n{elem['description']}\n\nTags: {', '.join(elem.get('tags', []))}\n\n{elem.get('notes', '')}"
            self.add_content(content, "world_element", elem_id, elem['name'])

        for proj_id, proj in data.get("projects", {}).items():
            if proj.get('content'):
                content = f"{proj['name']}\n\n{proj['content']}\n\nNotes: {proj.get('notes', '')}"
                self.add_content(content, "script", proj_id, proj['name'])


# Global instance
rag_service = RAGService()
