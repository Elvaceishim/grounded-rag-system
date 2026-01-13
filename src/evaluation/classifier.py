"""
Failure classification.

Categorizes query failures for debugging and systematic improvement.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class FailureType(str, Enum):
    """Types of failures that can occur in the RAG pipeline."""
    
    SUCCESS = "success"
    RETRIEVAL_MISS = "retrieval_miss"        # No relevant chunks found
    PARTIAL_RETRIEVAL = "partial_retrieval"  # Some relevant chunks missed
    CONTEXT_OVERLOAD = "context_overload"    # Too much context confused the LLM
    HALLUCINATION = "hallucination"          # Answer not grounded in context
    OVER_REFUSAL = "over_refusal"            # Refused when answer was possible


@dataclass
class FailureClassification:
    """Result of failure classification."""
    
    failure_type: FailureType
    confidence: float
    reasoning: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "failure_type": self.failure_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class FailureClassifier:
    """
    Classifies failures in the RAG pipeline.
    
    Uses a combination of metrics to determine failure type.
    """
    
    # Thresholds for classification
    RECALL_MISS_THRESHOLD = 0.0      # No relevant chunks = retrieval miss
    RECALL_PARTIAL_THRESHOLD = 0.5   # Less than half = partial retrieval
    FAITHFULNESS_THRESHOLD = 0.5     # Below this = hallucination
    CONTEXT_MAX_CHARS = 10000        # Context overload threshold
    
    def classify(
        self,
        retrieval_recall: Optional[float],
        faithfulness_score: float,
        is_refusal: bool,
        context_chars: int,
    ) -> FailureClassification:
        """
        Classify the failure type for a query.
        
        Args:
            retrieval_recall: Recall@K score (None if no ground truth)
            faithfulness_score: Faithfulness evaluation score
            is_refusal: Whether the answer was a refusal
            context_chars: Total characters in retrieved context
            
        Returns:
            FailureClassification with type, confidence, and reasoning
        """
        # Check for retrieval issues (if we have ground truth)
        if retrieval_recall is not None:
            if retrieval_recall == self.RECALL_MISS_THRESHOLD:
                return FailureClassification(
                    failure_type=FailureType.RETRIEVAL_MISS,
                    confidence=0.95,
                    reasoning="No relevant chunks were retrieved",
                )
            
            if retrieval_recall < self.RECALL_PARTIAL_THRESHOLD:
                return FailureClassification(
                    failure_type=FailureType.PARTIAL_RETRIEVAL,
                    confidence=0.85,
                    reasoning=f"Only {retrieval_recall:.0%} of relevant chunks retrieved",
                )
        
        # Check for context overload
        if context_chars > self.CONTEXT_MAX_CHARS:
            return FailureClassification(
                failure_type=FailureType.CONTEXT_OVERLOAD,
                confidence=0.7,
                reasoning=f"Context size ({context_chars} chars) exceeds threshold",
            )
        
        # Check for hallucination
        if faithfulness_score < self.FAITHFULNESS_THRESHOLD and not is_refusal:
            return FailureClassification(
                failure_type=FailureType.HALLUCINATION,
                confidence=0.8,
                reasoning=f"Low faithfulness score ({faithfulness_score:.2f})",
            )
        
        # Check for over-refusal
        if is_refusal:
            # If recall is good but still refused, it's over-refusal
            if retrieval_recall is not None and retrieval_recall >= 0.7:
                return FailureClassification(
                    failure_type=FailureType.OVER_REFUSAL,
                    confidence=0.75,
                    reasoning="Refused despite good retrieval coverage",
                )
            # Otherwise, refusal might be appropriate
            return FailureClassification(
                failure_type=FailureType.SUCCESS,
                confidence=0.6,
                reasoning="Appropriate refusal due to insufficient information",
            )
        
        # No failure detected
        return FailureClassification(
            failure_type=FailureType.SUCCESS,
            confidence=0.9,
            reasoning="All metrics within acceptable thresholds",
        )
    
    def classify_without_ground_truth(
        self,
        faithfulness_score: float,
        is_refusal: bool,
        context_chars: int,
    ) -> FailureClassification:
        """
        Classify failure when no ground truth is available.
        
        Args:
            faithfulness_score: Faithfulness evaluation score
            is_refusal: Whether the answer was a refusal
            context_chars: Total characters in retrieved context
            
        Returns:
            FailureClassification (retrieval metrics not evaluated)
        """
        return self.classify(
            retrieval_recall=None,
            faithfulness_score=faithfulness_score,
            is_refusal=is_refusal,
            context_chars=context_chars,
        )
