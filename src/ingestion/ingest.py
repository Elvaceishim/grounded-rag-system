"""
Corpus ingestion orchestration.

Loads, chunks, embeds, and stores documents in the vector database.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .loader import load_corpus, Document
from .chunker import chunk_document, Chunk
from ..config import settings


def ingest_corpus(
    corpus_dir: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedder = None,
) -> dict:
    """
    Ingest all documents from the corpus directory into the vector store.
    
    Args:
        corpus_dir: Directory containing documents (default: from settings)
        chunk_size: Characters per chunk (default: from settings)
        chunk_overlap: Overlap between chunks (default: from settings)
        embedder: Embedder instance for generating embeddings
        
    Returns:
        Dictionary with ingestion statistics
    """
    corpus_dir = corpus_dir or settings.corpus_dir
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    # Load all documents
    print(f"Loading documents from {corpus_dir}...")
    documents = load_corpus(corpus_dir)
    print(f"Loaded {len(documents)} documents")
    
    if not documents:
        return {"documents": 0, "chunks": 0, "error": "No documents found"}
    
    # Chunk all documents
    print(f"Chunking with size={chunk_size}, overlap={chunk_overlap}...")
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks")
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {settings.chroma_persist_dir}...")
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    
    # Delete existing collection if present
    try:
        chroma_client.delete_collection("documents")
    except Exception:
        pass  # Collection doesn't exist
    
    # Create collection
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
    
    # Generate embeddings if embedder provided
    if embedder:
        print("Generating embeddings...")
        contents = [chunk.content for chunk in all_chunks]
        embeddings = embedder.embed_batch(contents)
        
        # Add to collection with embeddings
        collection.add(
            ids=[chunk.chunk_id for chunk in all_chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in all_chunks],
            metadatas=[chunk.to_dict() for chunk in all_chunks],
        )
    else:
        # Add without embeddings (ChromaDB will use default)
        print("Adding chunks to collection (using default embeddings)...")
        collection.add(
            ids=[chunk.chunk_id for chunk in all_chunks],
            documents=[chunk.content for chunk in all_chunks],
            metadatas=[chunk.to_dict() for chunk in all_chunks],
        )
    
    print(f"Ingestion complete!")
    
    return {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "collection": "documents",
    }


def get_collection():
    """Get the ChromaDB collection for querying."""
    chroma_client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return chroma_client.get_collection("documents")
