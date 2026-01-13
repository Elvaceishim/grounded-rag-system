"""
Faithfulness evaluation.

Measures whether generated answers are grounded in retrieved context.
"""

from dataclasses import dataclass
from typing import Optional
import json
import re

from ..generation.prompts import FAITHFULNESS_PROMPT, format_context
from ..retrieval.retriever import RetrievedChunk


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    
    score: float
    reasoning: str
    citation_coverage: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "citation_coverage": self.citation_coverage,
        }


class FaithfulnessEvaluator:
    """
    Evaluates answer faithfulness using LLM-as-judge and heuristics.
    """
    
    def __init__(self, generator):
        """
        Initialize the evaluator.
        
        Args:
            generator: Generator instance for LLM calls
        """
        self.generator = generator
    
    def evaluate(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        citations: list[str],
    ) -> FaithfulnessResult:
        """
        Evaluate faithfulness of an answer.
        
        Args:
            answer: The generated answer
            chunks: Retrieved context chunks
            citations: Chunk IDs cited in the answer
            
        Returns:
            FaithfulnessResult with score and reasoning
        """
        # Calculate citation coverage (programmatic check)
        citation_coverage = self._calculate_citation_coverage(citations, chunks)
        
        # Get LLM-based faithfulness score
        llm_score, reasoning = self._llm_faithfulness_check(answer, chunks)
        
        # Combine scores (weighted average)
        # LLM judgment is primary, citation coverage is secondary
        combined_score = (llm_score * 0.7) + (citation_coverage * 0.3)
        
        return FaithfulnessResult(
            score=round(combined_score, 2),
            reasoning=reasoning,
            citation_coverage=round(citation_coverage, 2),
        )
    
    def _calculate_citation_coverage(
        self,
        citations: list[str],
        chunks: list[RetrievedChunk],
    ) -> float:
        """
        Calculate what fraction of cited chunks are in the context.
        
        A form of "did the model cite real sources?"
        """
        if not citations:
            return 0.0
        
        valid_ids = {chunk.chunk_id for chunk in chunks}
        valid_citations = sum(1 for c in citations if c in valid_ids)
        
        return valid_citations / len(citations)
    
    def _llm_faithfulness_check(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[float, str]:
        """Use LLM to evaluate faithfulness."""
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        context = format_context(chunk_dicts)
        
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            answer=answer,
        )
        
        try:
            response = self.generator.call_llm_for_eval(prompt)
            result = self._parse_json_response(response)
            
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except Exception as e:
            return 0.5, f"Evaluation failed: {str(e)}"
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to find JSON in response
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            return json.loads(json_match.group())
        
        # Try parsing entire response
        return json.loads(response)
