#!/usr/bin/env python3
"""
Startup script for Render deployment.
Runs ingestion if ChromaDB is empty, then starts the server.
"""

import os
import subprocess
import sys

def main():
    # Check if ChromaDB already has data
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_has_data = os.path.exists(chroma_dir) and os.listdir(chroma_dir)
    
    if not chroma_has_data:
        print("=== ChromaDB is empty, running ingestion ===")
        result = subprocess.run(
            [sys.executable, "scripts/ingest.py"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("Warning: Ingestion failed, but continuing anyway")
    else:
        print(f"=== ChromaDB already has data at {chroma_dir} ===")
    
    # Start the server
    port = os.environ.get("PORT", "8000")
    print(f"=== Starting server on port {port} ===")
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", port]
    )

if __name__ == "__main__":
    main()
