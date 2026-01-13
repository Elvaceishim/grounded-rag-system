"""
Re-query Agent for intelligent query retry.

When initial query confidence is low, the agent tries different strategies:
1. Rephrase the query
2. Increase Top-K
3. Try a more specific query

This is the bridge from simple RAG to agentic AI.
"""

from dataclasses import dataclass
from typing import Optional
import re

from ..retrieval import Retriever, RetrievedChunk
from ..generation import Generator, GeneratedAnswer
from ..evaluation import FaithfulnessEvaluator, UsefulnessEvaluator
from ..config import settings


@dataclass
class AgentResult:
    """Result from the agent's query processing."""
    
    final_answer: str
    citations: list[str]
    attempts: int
    strategy_used: str
    confidence_score: float
    all_attempts: list[dict]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "final_answer": self.final_answer,
            "citations": self.citations,
            "attempts": self.attempts,
            "strategy_used": self.strategy_used,
            "confidence_score": self.confidence_score,
            "all_attempts": self.all_attempts,
        }


REPHRASE_PROMPT = """Rephrase this question to be more specific and searchable.
Keep the same meaning but use different words.

Original question: {question}

Respond with ONLY the rephrased question, nothing else."""


class ReQueryAgent:
    """
    Agent that retries queries when confidence is low.
    
    Strategies:
    1. Initial query with default settings
    2. Rephrase: LLM rephrases the question
    3. Expand: Increase Top-K to get more context
    4. Combine: Use both original and rephrased queries
    """
    
    # Thresholds for retry decisions
    CONFIDENCE_THRESHOLD = 0.7  # Below this, try again
    MAX_ATTEMPTS = 3
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            retriever: Retriever instance
            generator: Generator instance
        """
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()
        self.faithfulness_eval = FaithfulnessEvaluator(self.generator)
        self.usefulness_eval = UsefulnessEvaluator(self.generator)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        auto_retry: bool = True,
    ) -> AgentResult:
        """
        Process a query with automatic retry if confidence is low.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            auto_retry: Whether to automatically retry on low confidence
            
        Returns:
            AgentResult with final answer and attempt history
        """
        attempts = []
        best_result = None
        best_score = 0.0
        
        # Strategy 1: Initial query
        result = self._try_query(question, top_k, "initial")
        attempts.append(result)
        
        if result["confidence"] >= self.CONFIDENCE_THRESHOLD or not auto_retry:
            return self._build_result(result, attempts, "initial")
        
        best_result = result
        best_score = result["confidence"]
        
        # Strategy 2: Rephrase the question
        rephrased = self._rephrase_question(question)
        if rephrased and rephrased != question:
            result = self._try_query(rephrased, top_k, "rephrase")
            attempts.append(result)
            
            if result["confidence"] > best_score:
                best_result = result
                best_score = result["confidence"]
            
            if result["confidence"] >= self.CONFIDENCE_THRESHOLD:
                return self._build_result(result, attempts, "rephrase")
        
        # Strategy 3: Expand context (increase Top-K)
        expanded_k = min(top_k * 2, 10)
        result = self._try_query(question, expanded_k, "expand")
        attempts.append(result)
        
        if result["confidence"] > best_score:
            best_result = result
            best_score = result["confidence"]
        
        if result["confidence"] >= self.CONFIDENCE_THRESHOLD:
            return self._build_result(result, attempts, "expand")
        
        # Return best result we found
        return self._build_result(best_result, attempts, f"best_of_{len(attempts)}")
    
    def _try_query(
        self,
        question: str,
        top_k: int,
        strategy: str,
    ) -> dict:
        """Execute a single query attempt."""
        # Retrieve
        chunks, embedding_hash = self.retriever.retrieve(question, top_k)
        
        # Generate
        answer = self.generator.generate(question, chunks)
        
        # Evaluate (quick check)
        faith_result = self.faithfulness_eval.evaluate(
            answer.answer,
            chunks,
            answer.citations,
        )
        
        # Calculate confidence (combination of faithfulness and citation coverage)
        confidence = faith_result.score
        
        # Penalize refusals slightly less
        if answer.is_refusal:
            confidence = max(confidence, 0.5)
        
        return {
            "question": question,
            "strategy": strategy,
            "top_k": top_k,
            "answer": answer.answer,
            "citations": answer.citations,
            "is_refusal": answer.is_refusal,
            "confidence": confidence,
            "faithfulness": faith_result.score,
            "chunk_ids": [c.chunk_id for c in chunks],
        }
    
    def _rephrase_question(self, question: str) -> Optional[str]:
        """Use LLM to rephrase the question."""
        prompt = REPHRASE_PROMPT.format(question=question)
        
        try:
            rephrased = self.generator.call_llm_for_eval(prompt, temperature=0.7)
            # Clean up response
            rephrased = rephrased.strip().strip('"').strip("'")
            # Validate it's actually a question
            if len(rephrased) > 10 and len(rephrased) < 500:
                return rephrased
        except Exception:
            pass
        
        return None
    
    def _build_result(
        self,
        best: dict,
        attempts: list[dict],
        strategy: str,
    ) -> AgentResult:
        """Build the final result object."""
        return AgentResult(
            final_answer=best["answer"],
            citations=best["citations"],
            attempts=len(attempts),
            strategy_used=strategy,
            confidence_score=best["confidence"],
            all_attempts=[
                {
                    "strategy": a["strategy"],
                    "question": a["question"],
                    "confidence": a["confidence"],
                    "is_refusal": a["is_refusal"],
                }
                for a in attempts
            ],
        )
