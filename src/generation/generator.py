"""
LLM-based answer generation.

Generates grounded answers using retrieved context.
"""

from dataclasses import dataclass
from typing import Optional
import re
import httpx

from .prompts import build_generation_prompt
from ..retrieval.retriever import RetrievedChunk
from ..config import settings


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with citations."""
    
    answer: str
    citations: list[str]
    raw_response: str
    is_refusal: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "raw_response": self.raw_response,
            "is_refusal": self.is_refusal,
        }


class Generator:
    """
    LLM-based answer generator.
    
    Uses OpenRouter API to generate grounded answers.
    """
    
    REFUSAL_PHRASES = [
        "cannot answer based on the provided",
        "don't have enough information",
        "context does not contain",
        "not mentioned in the",
        "no information about",
        "cannot find information",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            api_key: OpenRouter API key (default: from settings)
            model: LLM model name (default: from settings)
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.llm_model
        self.base_url = settings.openrouter_base_url
    
    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float = 0.1,
    ) -> GeneratedAnswer:
        """
        Generate an answer using retrieved chunks.
        
        Args:
            query: The user's question
            chunks: Retrieved context chunks
            temperature: LLM temperature (lower = more deterministic)
            
        Returns:
            GeneratedAnswer with answer, citations, and metadata
        """
        # Format chunks for prompt
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        prompt = build_generation_prompt(query, chunk_dicts)
        
        # Call LLM
        raw_response = self._call_llm(prompt, temperature)
        
        # Extract citations
        citations = self._extract_citations(raw_response, chunks)
        
        # Check for refusal
        is_refusal = self._is_refusal(raw_response)
        
        return GeneratedAnswer(
            answer=raw_response.strip(),
            citations=citations,
            raw_response=raw_response,
            is_refusal=is_refusal,
        )
    
    def _call_llm(self, prompt: str, temperature: float) -> str:
        """Make API call to LLM."""
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        return data["choices"][0]["message"]["content"]
    
    def _extract_citations(
        self,
        response: str,
        chunks: list[RetrievedChunk],
    ) -> list[str]:
        """Extract chunk IDs cited in the response."""
        # Look for [chunk_id] patterns
        pattern = r"\[([a-zA-Z0-9_]+)\]"
        matches = re.findall(pattern, response)
        
        # Filter to only valid chunk IDs
        valid_ids = {chunk.chunk_id for chunk in chunks}
        citations = [m for m in matches if m in valid_ids]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)
        
        return unique_citations
    
    def _is_refusal(self, response: str) -> bool:
        """Check if the response is a refusal to answer."""
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.REFUSAL_PHRASES)
    
    def call_llm_for_eval(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Make LLM call for evaluation purposes.
        
        Uses lower temperature for consistent evaluation.
        
        Args:
            prompt: The evaluation prompt
            temperature: LLM temperature
            
        Returns:
            Raw LLM response
        """
        return self._call_llm(prompt, temperature)
