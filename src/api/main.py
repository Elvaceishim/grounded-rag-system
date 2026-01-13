"""
FastAPI application setup.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .routes import router


# Get static files directory
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

app = FastAPI(
    title="Grounded RAG System",
    description="Retrieval-Augmented Generation with Evaluation",
    version="1.0.0",
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

