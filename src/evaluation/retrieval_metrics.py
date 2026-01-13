"""
Retrieval quality metrics.

Measures how well the retrieval system finds relevant documents.
"""


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (relevant items in top K) / (total relevant items)
    
    Measures: Of all relevant chunks, what fraction did we retrieve?
    
    Args:
        retrieved_ids: List of chunk IDs returned by retrieval (ordered by rank)
        relevant_ids: List of chunk IDs that are actually relevant
        k: Number of top results to consider
        
    Returns:
        Recall score between 0 and 1
    """
    if not relevant_ids:
        return 1.0  # No relevant items means perfect recall (vacuously true)
    
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (relevant items in top K) / K
    
    Measures: Of the chunks we retrieved, what fraction are relevant?
    
    Args:
        retrieved_ids: List of chunk IDs returned by retrieval (ordered by rank)
        relevant_ids: List of chunk IDs that are actually relevant
        k: Number of top results to consider
        
    Returns:
        Precision score between 0 and 1
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / k


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / (rank of first relevant item)
    
    Measures: How high up is the first relevant result?
    
    Args:
        retrieved_ids: List of chunk IDs returned by retrieval (ordered by rank)
        relevant_ids: List of chunk IDs that are actually relevant
        
    Returns:
        MRR score between 0 and 1
    """
    relevant_set = set(relevant_ids)
    
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    
    return 0.0  # No relevant item found


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    NDCG weighs relevant items higher when they appear earlier in results.
    
    Args:
        retrieved_ids: List of chunk IDs returned by retrieval
        relevant_ids: List of chunk IDs that are actually relevant
        k: Number of top results to consider
        
    Returns:
        NDCG score between 0 and 1
    """
    import math
    
    relevant_set = set(relevant_ids)
    
    # Calculate DCG
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids[:k]):
        rel = 1 if chunk_id in relevant_set else 0
        dcg += rel / math.log2(i + 2)  # +2 because rank starts at 1 and log2(1)=0
    
    # Calculate ideal DCG (all relevant items at top)
    ideal_rels = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_rels))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
