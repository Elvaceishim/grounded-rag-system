"""
Embedding interface using OpenRouter API.

Provides text embedding capabilities for queries and documents.
"""

from typing import Optional
import httpx

from ..config import settings


class Embedder:
    """
    Text embedder using OpenRouter API.
    
    Uses OpenAI's text-embedding-3-small model via OpenRouter.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            api_key: OpenRouter API key (default: from settings)
            model: Embedding model name (default: from settings)
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.embedding_model
        self.base_url = settings.openrouter_base_url
        
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        embeddings = self.embed_batch([text])
        return embeddings[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        # Sort by index to ensure correct order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]
    
    def get_embedding_hash(self, embedding: list[float]) -> str:
        """
        Generate a hash of an embedding for logging/reproducibility.
        
        Args:
            embedding: The embedding vector
            
        Returns:
            A short hash string
        """
        import hashlib
        # Use first 10 values for hash (sufficient for identification)
        embed_str = ",".join(f"{v:.6f}" for v in embedding[:10])
        return hashlib.sha256(embed_str.encode()).hexdigest()[:12]
