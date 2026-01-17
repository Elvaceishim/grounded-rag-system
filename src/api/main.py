"""
FastAPI application setup.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .routes import router


# Get static files directory
STATIC_DIR = Path(__file__).parent.parent.parent / "static"


def run_ingestion_if_needed():
    """Run ingestion if ChromaDB is empty."""
    from ..config import settings
    from ..ingestion.ingest import ingest_corpus, get_collection
    
    chroma_dir = Path(settings.chroma_persist_dir)
    
    # Check if ChromaDB has data
    try:
        collection = get_collection()
        count = collection.count()
        if count > 0:
            print(f"=== ChromaDB already has {count} chunks, skipping ingestion ===")
            return
    except Exception:
        pass  # Collection doesn't exist yet
    
    print("=== ChromaDB is empty, running ingestion ===")
    try:
        ingest_corpus(
            corpus_dir=settings.corpus_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        print("=== Ingestion complete! ===")
    except Exception as e:
        print(f"=== Ingestion failed: {e} ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: run ingestion if needed
    run_ingestion_if_needed()
    yield
    # Shutdown: nothing to do


app = FastAPI(
    title="Grounded RAG System",
    description="Retrieval-Augmented Generation with Evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Serve static files (must be after routes)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Redirect to the web UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Grounded RAG System",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /query - Process a query",
            "agent_query": "POST /agent/query - Process with intelligent retry",
            "evaluate": "POST /evaluate - Evaluate with ground truth",
            "metrics": "GET /metrics - Get aggregated metrics",
            "health": "GET /health - Health check",
        },
    }

