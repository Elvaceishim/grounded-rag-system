"""
Structured query logging.

Logs every query with full context for debugging and analysis.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import uuid

from ..config import settings
from ..evaluation.classifier import FailureType


@dataclass
class QueryLog:
    """Structured log entry for a single query."""
    
    query_id: str
    timestamp: str
    query_text: str
    top_k: int
    
    # Retrieval
    retrieved_chunk_ids: list[str]
    similarity_scores: list[float]
    embedding_hash: str
    
    # Generation
    generated_answer: str
    citations: list[str]
    is_refusal: bool
    
    # Evaluation scores
    faithfulness_score: Optional[float] = None
    usefulness_score: Optional[float] = None
    retrieval_recall: Optional[float] = None
    retrieval_precision: Optional[float] = None
    
    # Failure classification
    failure_type: str = "unknown"
    failure_reasoning: str = ""
    
    # Timing
    retrieval_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class QueryLogger:
    """
    Logger for query processing.
    
    Writes structured logs to a JSONL file for analysis.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory for log files (default: from settings)
        """
        self.log_dir = log_dir or settings.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "queries.jsonl"
    
    @staticmethod
    def generate_query_id() -> str:
        """Generate a unique query ID."""
        return str(uuid.uuid4())[:8]
    
    @staticmethod
    def now() -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def log(self, query_log: QueryLog) -> None:
        """
        Write a query log entry.
        
        Args:
            query_log: The log entry to write
        """
        with open(self.log_file, "a") as f:
            f.write(query_log.to_json() + "\n")
    
    def get_logs(self, limit: int = 100) -> list[QueryLog]:
        """
        Read recent log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of QueryLog objects (most recent first)
        """
        if not self.log_file.exists():
            return []
        
        logs = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    logs.append(QueryLog(**data))
        
        return logs[-limit:][::-1]  # Most recent first
    
    def get_metrics_summary(self) -> dict:
        """
        Calculate aggregate metrics from logs.
        
        Returns:
            Dictionary with metric summaries
        """
        logs = self.get_logs(limit=1000)
        
        if not logs:
            return {
                "total_queries": 0,
                "avg_faithfulness": None,
                "avg_usefulness": None,
                "failure_breakdown": {},
            }
        
        # Calculate averages
        faithfulness_scores = [l.faithfulness_score for l in logs if l.faithfulness_score is not None]
        usefulness_scores = [l.usefulness_score for l in logs if l.usefulness_score is not None]
        
        # Count failures
        failure_counts = {}
        for log in logs:
            ft = log.failure_type
            failure_counts[ft] = failure_counts.get(ft, 0) + 1
        
        return {
            "total_queries": len(logs),
            "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else None,
            "avg_usefulness": round(sum(usefulness_scores) / len(usefulness_scores), 3) if usefulness_scores else None,
            "failure_breakdown": failure_counts,
            "success_rate": round(failure_counts.get("success", 0) / len(logs), 3) if logs else 0,
        }
