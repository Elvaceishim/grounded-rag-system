#!/usr/bin/env python3
"""
Corpus ingestion script.

Usage:
    python scripts/ingest.py [--corpus-dir PATH] [--chunk-size N] [--chunk-overlap N]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ingest_corpus
from src.retrieval import Embedder
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=settings.corpus_dir,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Maximum characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Character overlap between chunks",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip custom embeddings (use ChromaDB default)",
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Corpus Ingestion")
    print("=" * 50)
    print(f"Corpus directory: {args.corpus_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print()
    
    # Create embedder unless skipped
    embedder = None
    if not args.no_embeddings:
        print("Initializing embedder...")
        embedder = Embedder()
    
    # Run ingestion
    result = ingest_corpus(
        corpus_dir=args.corpus_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder=embedder,
    )
    
    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Documents processed: {result['documents']}")
    print(f"Chunks created: {result['chunks']}")
    
    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print()
    print("Ingestion complete! Start the API with:")
    print("  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
