"""
Configuration management using Pydantic Settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    
    # Model Configuration
    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model for generation and evaluation"
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model for vector search"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=1000,
        description="Maximum characters per chunk"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Character overlap between chunks"
    )
    
    # Retrieval Configuration
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve"
    )
    
    # Paths
    chroma_persist_dir: Path = Field(
        default=Path("./chroma_db"),
        description="Directory for ChromaDB persistence"
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Directory for query logs"
    )
    corpus_dir: Path = Field(
        default=Path("./data/corpus"),
        description="Directory containing source documents"
    )
    
    # OpenRouter API
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
