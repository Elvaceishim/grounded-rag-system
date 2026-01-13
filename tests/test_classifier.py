"""
Tests for failure classification.
"""

import pytest
from src.evaluation.classifier import FailureClassifier, FailureType


@pytest.fixture
def classifier():
    """Create a classifier instance."""
    return FailureClassifier()


class TestFailureClassifier:
    """Tests for failure classification."""
    
    def test_success(self, classifier):
        """High scores result in success."""
        result = classifier.classify(
            retrieval_recall=0.9,
            faithfulness_score=0.8,
            is_refusal=False,
            context_chars=5000,
        )
        assert result.failure_type == FailureType.SUCCESS
    
    def test_retrieval_miss(self, classifier):
        """Zero recall is retrieval miss."""
        result = classifier.classify(
            retrieval_recall=0.0,
            faithfulness_score=0.8,
            is_refusal=False,
            context_chars=5000,
        )
        assert result.failure_type == FailureType.RETRIEVAL_MISS
    
    def test_partial_retrieval(self, classifier):
        """Low recall is partial retrieval."""
        result = classifier.classify(
            retrieval_recall=0.3,
            faithfulness_score=0.8,
            is_refusal=False,
            context_chars=5000,
        )
        assert result.failure_type == FailureType.PARTIAL_RETRIEVAL
    
    def test_hallucination(self, classifier):
        """Low faithfulness without refusal is hallucination."""
        result = classifier.classify(
            retrieval_recall=0.9,
            faithfulness_score=0.2,
            is_refusal=False,
            context_chars=5000,
        )
        assert result.failure_type == FailureType.HALLUCINATION
    
    def test_over_refusal(self, classifier):
        """Refusal with good retrieval is over-refusal."""
        result = classifier.classify(
            retrieval_recall=0.9,
            faithfulness_score=0.8,
            is_refusal=True,
            context_chars=5000,
        )
        assert result.failure_type == FailureType.OVER_REFUSAL
    
    def test_appropriate_refusal(self, classifier):
        """Refusal with low retrieval is appropriate."""
        result = classifier.classify(
            retrieval_recall=0.2,
            faithfulness_score=0.8,
            is_refusal=True,
            context_chars=5000,
        )
        # Should be success (appropriate refusal) or partial retrieval
        assert result.failure_type in [FailureType.SUCCESS, FailureType.PARTIAL_RETRIEVAL]
    
    def test_context_overload(self, classifier):
        """Large context triggers context overload."""
        result = classifier.classify(
            retrieval_recall=0.9,
            faithfulness_score=0.6,
            is_refusal=False,
            context_chars=15000,  # Above threshold
        )
        assert result.failure_type == FailureType.CONTEXT_OVERLOAD
    
    def test_without_ground_truth(self, classifier):
        """Classification works without ground truth."""
        result = classifier.classify_without_ground_truth(
            faithfulness_score=0.2,
            is_refusal=False,
            context_chars=5000,
        )
        assert result.failure_type == FailureType.HALLUCINATION


class TestFailureTypeEnum:
    """Tests for FailureType enum."""
    
    def test_enum_values(self):
        """All expected failure types exist."""
        assert FailureType.SUCCESS.value == "success"
        assert FailureType.RETRIEVAL_MISS.value == "retrieval_miss"
        assert FailureType.PARTIAL_RETRIEVAL.value == "partial_retrieval"
        assert FailureType.CONTEXT_OVERLOAD.value == "context_overload"
        assert FailureType.HALLUCINATION.value == "hallucination"
        assert FailureType.OVER_REFUSAL.value == "over_refusal"
