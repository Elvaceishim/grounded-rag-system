"""
Vector retriever for semantic search.

Retrieves relevant document chunks based on query embedding similarity.
"""

from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embedder import Embedder
from ..config import settings


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with similarity score."""
    
    chunk_id: str
    content: str
    document_id: str
    source: str
    section: Optional[str]
    similarity_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "document_id": self.document_id,
            "source": self.source,
            "section": self.section,
            "similarity_score": self.similarity_score,
        }


class Retriever:
    """
    Vector-based document retriever.
    
    Uses ChromaDB for vector storage and similarity search.
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        collection_name: str = "documents",
    ):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance for query embedding
            collection_name: Name of the ChromaDB collection
        """
        self.embedder = embedder or Embedder()
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None
        
    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._client.get_collection(self.collection_name)
        return self._collection
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve (default: from settings)
            
        Returns:
            Tuple of (list of RetrievedChunks, query_embedding_hash)
        """
        top_k = top_k or settings.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        embedding_hash = self.embedder.get_embedding_hash(query_embedding)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to RetrievedChunk objects
        chunks = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i]
                similarity = 1 - distance
                
                metadata = results["metadatas"][0][i]
                
                chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i],
                    document_id=metadata.get("document_id", ""),
                    source=metadata.get("source", ""),
                    section=metadata.get("section") or None,
                    similarity_score=round(similarity, 4),
                ))
        
        return chunks, embedding_hash
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievedChunk]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            RetrievedChunk or None if not found
        """
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        
        if results["ids"]:
            metadata = results["metadatas"][0]
            return RetrievedChunk(
                chunk_id=chunk_id,
                content=results["documents"][0],
                document_id=metadata.get("document_id", ""),
                source=metadata.get("source", ""),
                section=metadata.get("section") or None,
                similarity_score=1.0,  # Direct lookup
            )
        
        return None
