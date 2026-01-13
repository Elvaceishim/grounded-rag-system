"""
Tests for retrieval metrics.
"""

import pytest
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)


class TestRecallAtK:
    """Tests for Recall@K metric."""
    
    def test_perfect_recall(self):
        """All relevant items retrieved."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=5) == 1.0
    
    def test_partial_recall(self):
        """Some relevant items retrieved."""
        retrieved = ["a", "x", "y", "z", "w"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1/3)
    
    def test_zero_recall(self):
        """No relevant items retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0
    
    def test_empty_relevant(self):
        """No relevant items (vacuously true)."""
        retrieved = ["a", "b", "c"]
        relevant = []
        assert recall_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_k_limit(self):
        """Only considers top K."""
        retrieved = ["x", "y", "z", "a", "b"]  # relevant at positions 4, 5
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2/3)


class TestPrecisionAtK:
    """Tests for Precision@K metric."""
    
    def test_perfect_precision(self):
        """All retrieved items are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c", "d"]
        assert precision_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_partial_precision(self):
        """Some retrieved items are relevant."""
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2/5)
    
    def test_zero_precision(self):
        """No retrieved items are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, k=3) == 0.0
    
    def test_k_zero(self):
        """K=0 returns 0."""
        assert precision_at_k(["a"], ["a"], k=0) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""
    
    def test_first_position(self):
        """Relevant item at first position."""
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    
    def test_second_position(self):
        """Relevant item at second position."""
        retrieved = ["x", "a", "c"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    
    def test_third_position(self):
        """Relevant item at third position."""
        retrieved = ["x", "y", "a"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1/3)
    
    def test_no_relevant(self):
        """No relevant items found."""
        retrieved = ["x", "y", "z"]
        relevant = ["a"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0


class TestNDCG:
    """Tests for NDCG@K metric."""
    
    def test_perfect_ndcg(self):
        """All relevant at top positions."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = ["a", "b", "c"]
        assert ndcg_at_k(retrieved, relevant, k=5) == 1.0
    
    def test_zero_ndcg(self):
        """No relevant items."""
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0
