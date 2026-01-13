"""
API route handlers.

Implements the core RAG system endpoints.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from .models import (
    QueryRequest,
    QueryResponse,
    ChunkResponse,
    EvaluationScores,
    EvaluateRequest,
    EvaluateResponse,
    RetrievalMetrics,
    MetricsResponse,
    HealthResponse,
)
from ..retrieval import Retriever, Embedder
from ..generation import Generator
from ..evaluation import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    FaithfulnessEvaluator,
    UsefulnessEvaluator,
    FailureClassifier,
)
from ..logging import QueryLogger, QueryLog


router = APIRouter()

# Initialize components (lazy loading)
_retriever: Optional[Retriever] = None
_generator: Optional[Generator] = None
_logger: Optional[QueryLogger] = None
_faithfulness_evaluator: Optional[FaithfulnessEvaluator] = None
_usefulness_evaluator: Optional[UsefulnessEvaluator] = None
_failure_classifier: Optional[FailureClassifier] = None


def get_retriever() -> Retriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        embedder = Embedder()
        _retriever = Retriever(embedder=embedder)
    return _retriever


def get_generator() -> Generator:
    """Get or create generator instance."""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def get_logger() -> QueryLogger:
    """Get or create logger instance."""
    global _logger
    if _logger is None:
        _logger = QueryLogger()
    return _logger


def get_faithfulness_evaluator() -> FaithfulnessEvaluator:
    """Get or create faithfulness evaluator."""
    global _faithfulness_evaluator
    if _faithfulness_evaluator is None:
        _faithfulness_evaluator = FaithfulnessEvaluator(get_generator())
    return _faithfulness_evaluator


def get_usefulness_evaluator() -> UsefulnessEvaluator:
    """Get or create usefulness evaluator."""
    global _usefulness_evaluator
    if _usefulness_evaluator is None:
        _usefulness_evaluator = UsefulnessEvaluator(get_generator())
    return _usefulness_evaluator


def get_failure_classifier() -> FailureClassifier:
    """Get or create failure classifier."""
    global _failure_classifier
    if _failure_classifier is None:
        _failure_classifier = FailureClassifier()
    return _failure_classifier


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and corpus status."""
    try:
        retriever = get_retriever()
        count = retriever.collection.count()
        return HealthResponse(status="healthy", corpus_size=count)
    except Exception as e:
        return HealthResponse(status="unhealthy", corpus_size=None)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query through the RAG pipeline.
    
    1. Retrieve relevant chunks
    2. Generate grounded answer
    3. Optionally evaluate quality
    4. Log everything
    """
    start_time = time.time()
    
    retriever = get_retriever()
    generator = get_generator()
    logger = get_logger()
    
    query_id = logger.generate_query_id()
    
    # Step 1: Retrieve
    retrieval_start = time.time()
    try:
        chunks, embedding_hash = retriever.retrieve(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}. Have you ingested documents?",
        )
    retrieval_latency = (time.time() - retrieval_start) * 1000
    
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please ingest a corpus first.",
        )
    
    # Step 2: Generate
    generation_start = time.time()
    answer = generator.generate(request.query, chunks)
    generation_latency = (time.time() - generation_start) * 1000
    
    # Step 3: Evaluate (optional)
    evaluation = None
    faithfulness_score = None
    usefulness_score = None
    failure_type = "unknown"
    failure_reasoning = ""
    
    if request.include_evaluation:
        faith_eval = get_faithfulness_evaluator()
        use_eval = get_usefulness_evaluator()
        classifier = get_failure_classifier()
        
        # Faithfulness
        faith_result = faith_eval.evaluate(
            answer.answer,
            chunks,
            answer.citations,
        )
        faithfulness_score = faith_result.score
        
        # Usefulness
        use_result = use_eval.evaluate(request.query, answer.answer)
        usefulness_score = use_result.score
        
        # Failure classification
        context_chars = sum(len(c.content) for c in chunks)
        failure = classifier.classify_without_ground_truth(
            faithfulness_score=faithfulness_score,
            is_refusal=answer.is_refusal,
            context_chars=context_chars,
        )
        failure_type = failure.failure_type.value
        failure_reasoning = failure.reasoning
        
        evaluation = EvaluationScores(
            faithfulness_score=faithfulness_score,
            faithfulness_reasoning=faith_result.reasoning,
            usefulness_score=usefulness_score,
            usefulness_reasoning=use_result.reasoning,
            failure_type=failure_type,
            failure_reasoning=failure_reasoning,
        )
    
    total_latency = (time.time() - start_time) * 1000
    
    # Step 4: Log
    log_entry = QueryLog(
        query_id=query_id,
        timestamp=logger.now(),
        query_text=request.query,
        top_k=request.top_k,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        similarity_scores=[c.similarity_score for c in chunks],
        embedding_hash=embedding_hash,
        generated_answer=answer.answer,
        citations=answer.citations,
        is_refusal=answer.is_refusal,
        faithfulness_score=faithfulness_score,
        usefulness_score=usefulness_score,
        failure_type=failure_type,
        failure_reasoning=failure_reasoning,
        retrieval_latency_ms=round(retrieval_latency, 2),
        generation_latency_ms=round(generation_latency, 2),
        total_latency_ms=round(total_latency, 2),
    )
    logger.log(log_entry)
    
    # Build response
    return QueryResponse(
        query_id=query_id,
        query=request.query,
        retrieved_chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                content=c.content,
                source=c.source,
                section=c.section,
                similarity_score=c.similarity_score,
            )
            for c in chunks
        ],
        generated_answer=answer.answer,
        citations=answer.citations,
        is_refusal=answer.is_refusal,
        evaluation=evaluation,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate a previous query with optional ground truth.
    
    Requires ground_truth_chunk_ids for retrieval metrics.
    """
    logger = get_logger()
    
    # Find the query in logs
    logs = logger.get_logs(limit=1000)
    query_log = next((l for l in logs if l.query_id == request.query_id), None)
    
    if not query_log:
        raise HTTPException(status_code=404, detail="Query not found")
    
    # Calculate retrieval metrics if ground truth provided
    retrieval_metrics = None
    retrieval_recall = None
    
    if request.ground_truth_chunk_ids:
        retrieved = query_log.retrieved_chunk_ids
        relevant = request.ground_truth_chunk_ids
        k = len(retrieved)
        
        retrieval_recall = recall_at_k(retrieved, relevant, k)
        
        retrieval_metrics = RetrievalMetrics(
            recall_at_k=round(retrieval_recall, 3),
            precision_at_k=round(precision_at_k(retrieved, relevant, k), 3),
            mrr=round(mean_reciprocal_rank(retrieved, relevant), 3),
        )
    
    # Re-classify with ground truth
    classifier = get_failure_classifier()
    failure = classifier.classify(
        retrieval_recall=retrieval_recall,
        faithfulness_score=query_log.faithfulness_score or 0.5,
        is_refusal=query_log.is_refusal,
        context_chars=sum(len(c) for c in query_log.retrieved_chunk_ids),  # Approximation
    )
    
    return EvaluateResponse(
        query_id=request.query_id,
        retrieval_metrics=retrieval_metrics,
        faithfulness_score=query_log.faithfulness_score,
        usefulness_score=query_log.usefulness_score,
        failure_type=failure.failure_type.value,
        failure_reasoning=failure.reasoning,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get aggregated metrics from query logs."""
    logger = get_logger()
    summary = logger.get_metrics_summary()
    
    return MetricsResponse(
        total_queries=summary["total_queries"],
        avg_faithfulness=summary["avg_faithfulness"],
        avg_usefulness=summary["avg_usefulness"],
        success_rate=summary["success_rate"],
        failure_breakdown=summary["failure_breakdown"],
    )


# ============== Agent Endpoint ==============

from .models import AgentQueryRequest, AgentQueryResponse, AttemptInfo
from ..agent import ReQueryAgent

_agent: Optional[ReQueryAgent] = None


def get_agent() -> ReQueryAgent:
    """Get or create agent instance."""
    global _agent
    if _agent is None:
        _agent = ReQueryAgent(
            retriever=get_retriever(),
            generator=get_generator(),
        )
    return _agent


@router.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Process a query with intelligent retry.
    
    When confidence is low, the agent automatically tries:
    1. Rephrasing the question
    2. Expanding the search (more chunks)
    3. Combining strategies
    
    Returns the best answer found across all attempts.
    """
    agent = get_agent()
    
    result = agent.query(
        question=request.query,
        top_k=request.top_k,
        auto_retry=request.auto_retry,
    )
    
    return AgentQueryResponse(
        final_answer=result.final_answer,
        citations=result.citations,
        attempts=result.attempts,
        strategy_used=result.strategy_used,
        confidence_score=result.confidence_score,
        all_attempts=[
            AttemptInfo(
                strategy=a["strategy"],
                question=a["question"],
                confidence=a["confidence"],
                is_refusal=a["is_refusal"],
            )
            for a in result.all_attempts
        ],
    )

