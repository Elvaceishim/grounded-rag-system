"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ============== Query Endpoint ==============

class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    
    query: str = Field(..., description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    include_evaluation: bool = Field(default=True, description="Whether to run evaluation")


class ChunkResponse(BaseModel):
    """A retrieved chunk in the response."""
    
    chunk_id: str
    content: str
    source: str
    section: Optional[str]
    similarity_score: float


class EvaluationScores(BaseModel):
    """Evaluation scores for a query."""
    
    faithfulness_score: Optional[float] = None
    faithfulness_reasoning: Optional[str] = None
    usefulness_score: Optional[float] = None
    usefulness_reasoning: Optional[str] = None
    failure_type: str = "unknown"
    failure_reasoning: Optional[str] = None


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    
    query_id: str
    query: str
    retrieved_chunks: list[ChunkResponse]
    generated_answer: str
    citations: list[str]
    is_refusal: bool
    evaluation: Optional[EvaluationScores] = None


# ============== Evaluate Endpoint ==============

class EvaluateRequest(BaseModel):
    """Request body for /evaluate endpoint."""
    
    query_id: str = Field(..., description="Query ID to evaluate")
    ground_truth_chunk_ids: Optional[list[str]] = Field(
        default=None,
        description="List of relevant chunk IDs for retrieval evaluation"
    )


class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics."""
    
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    mrr: Optional[float] = None


class EvaluateResponse(BaseModel):
    """Response body for /evaluate endpoint."""
    
    query_id: str
    retrieval_metrics: Optional[RetrievalMetrics] = None
    faithfulness_score: Optional[float] = None
    usefulness_score: Optional[float] = None
    failure_type: str
    failure_reasoning: str


# ============== Metrics Endpoint ==============

class MetricsResponse(BaseModel):
    """Response body for /metrics endpoint."""
    
    total_queries: int
    avg_faithfulness: Optional[float]
    avg_usefulness: Optional[float]
    success_rate: float
    failure_breakdown: dict[str, int]


# ============== Health Endpoint ==============

class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    
    status: str
    version: str = "1.0.0"
    corpus_size: Optional[int] = None


# ============== Agent Endpoint ==============

class AgentQueryRequest(BaseModel):
    """Request body for /agent/query endpoint."""
    
    query: str = Field(..., description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    auto_retry: bool = Field(default=True, description="Automatically retry on low confidence")


class AttemptInfo(BaseModel):
    """Information about a single query attempt."""
    
    strategy: str
    question: str
    confidence: float
    is_refusal: bool


class AgentQueryResponse(BaseModel):
    """Response body for /agent/query endpoint."""
    
    final_answer: str
    citations: list[str]
    attempts: int
    strategy_used: str
    confidence_score: float
    all_attempts: list[AttemptInfo]

