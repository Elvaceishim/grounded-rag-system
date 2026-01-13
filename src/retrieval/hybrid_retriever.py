"""
Hybrid retriever combining vector search with keyword search.

Hybrid search improves retrieval by:
- Vector search: Finds semantically similar content
- Keyword search: Finds exact matches for specific terms

The results are merged using Reciprocal Rank Fusion (RRF).
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .retriever import Retriever, RetrievedChunk
from .embedder import Embedder
from ..ingestion.ingest import get_collection


@dataclass
class HybridRetrievedChunk(RetrievedChunk):
    """Extended chunk with hybrid search scores."""
    
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    rrf_score: float = 0.0


class HybridRetriever:
    """
    Retriever that combines vector search with keyword search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results:
    RRF(d) = sum(1 / (k + rank(d))) for each result list
    
    Where k is a constant (default 60) that reduces the impact
    of high rankings from individual lists.
    """
    
    # RRF constant - higher values give more weight to lower-ranked items
    RRF_K = 60
    
    # Weight for vector vs keyword (1.0 = equal)
    VECTOR_WEIGHT = 1.0
    KEYWORD_WEIGHT = 0.5
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        collection=None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedder: Embedder instance for vector search
            collection: ChromaDB collection (optional, will load if not provided)
        """
        self.embedder = embedder or Embedder()
        self._collection = collection
        self._vector_retriever = Retriever(embedder=self.embedder)
    
    @property
    def collection(self):
        """Lazy load the collection."""
        if self._collection is None:
            self._collection = get_collection()
        return self._collection
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 1.0,
        keyword_weight: float = 0.5,
    ) -> tuple[list[HybridRetrievedChunk], dict]:
        """
        Retrieve chunks using hybrid search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            
        Returns:
            Tuple of (chunks, metadata)
        """
        # Get more results from each method, then merge
        fetch_k = min(top_k * 2, 20)
        
        # 1. Vector search
        vector_chunks, embedding_hash = self._vector_retriever.retrieve(query, fetch_k)
        
        # 2. Keyword search
        keyword_chunks = self._keyword_search(query, fetch_k)
        
        # 3. Merge with RRF
        merged = self._rrf_merge(
            vector_chunks,
            keyword_chunks,
            vector_weight,
            keyword_weight,
        )
        
        # Take top K
        results = merged[:top_k]
        
        metadata = {
            "embedding_hash": embedding_hash,
            "vector_results": len(vector_chunks),
            "keyword_results": len(keyword_chunks),
            "search_type": "hybrid",
        }
        
        return results, metadata
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Perform keyword search using ChromaDB's where_document filter.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of matching chunks
        """
        # Extract keywords (remove common words)
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        # Search for documents containing keywords
        results = []
        
        # Get all documents and filter (ChromaDB doesn't have great keyword support)
        all_docs = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        if not all_docs["documents"]:
            return []
        
        # Score documents by keyword matches
        scored = []
        for i, (doc, metadata, chunk_id) in enumerate(zip(
            all_docs["documents"],
            all_docs["metadatas"],
            all_docs["ids"],
        )):
            doc_lower = doc.lower()
            score = 0
            matches = 0
            
            for keyword in keywords:
                # Count occurrences
                count = doc_lower.count(keyword.lower())
                if count > 0:
                    matches += 1
                    score += count
            
            if matches > 0:
                # Normalize by document length
                normalized_score = score / (len(doc.split()) + 1)
                scored.append((chunk_id, doc, metadata, normalized_score, matches))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: (x[4], x[3]), reverse=True)
        
        # Convert to RetrievedChunk objects
        for chunk_id, content, metadata, score, _ in scored[:top_k]:
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=content,
                document_id=metadata.get("document_id", "unknown"),
                source=metadata.get("source", "unknown"),
                section=metadata.get("section"),
                similarity_score=score,  # Using keyword score
            ))
        
        return results
    
    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract meaningful keywords from a query.
        
        Removes common stop words and short words.
        """
        # Common stop words to filter out
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "whom", "whose",
            "where", "when", "why", "how", "and", "or", "but", "if",
            "then", "else", "for", "of", "to", "in", "on", "at", "by",
            "with", "about", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down",
            "out", "off", "over", "under", "again", "further", "once",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [
            w for w in words 
            if w not in stop_words and len(w) > 2
        ]
        
        return keywords
    
    def _rrf_merge(
        self,
        vector_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
        vector_weight: float,
        keyword_weight: float,
    ) -> list[HybridRetrievedChunk]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        RRF(d) = sum(weight / (k + rank(d))) for each result list
        """
        scores = defaultdict(float)
        chunk_data = {}
        vector_ranks = {}
        keyword_ranks = {}
        
        # Score vector results
        for rank, chunk in enumerate(vector_results, 1):
            rrf_score = vector_weight / (self.RRF_K + rank)
            scores[chunk.chunk_id] += rrf_score
            chunk_data[chunk.chunk_id] = chunk
            vector_ranks[chunk.chunk_id] = rank
        
        # Score keyword results
        for rank, chunk in enumerate(keyword_results, 1):
            rrf_score = keyword_weight / (self.RRF_K + rank)
            scores[chunk.chunk_id] += rrf_score
            if chunk.chunk_id not in chunk_data:
                chunk_data[chunk.chunk_id] = chunk
            keyword_ranks[chunk.chunk_id] = rank
        
        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build hybrid chunks
        results = []
        for chunk_id in sorted_ids:
            chunk = chunk_data[chunk_id]
            hybrid_chunk = HybridRetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                document_id=chunk.document_id,
                source=chunk.source,
                section=chunk.section,
                similarity_score=chunk.similarity_score,
                vector_rank=vector_ranks.get(chunk_id),
                keyword_rank=keyword_ranks.get(chunk_id),
                rrf_score=scores[chunk_id],
            )
            results.append(hybrid_chunk)
        
        return results
