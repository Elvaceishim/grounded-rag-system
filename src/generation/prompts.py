"""
Prompt templates for grounded answer generation.

These prompts enforce strict grounding in retrieved context.
"""

GROUNDING_RULES = """
RULES - You MUST follow these exactly:
1. Use ONLY the information provided in the CONTEXT below
2. Do NOT use any external knowledge or make assumptions
3. For EVERY claim, cite the chunk ID in brackets like [chunk_abc123]
4. If the context does not contain enough information, respond: "I cannot answer based on the provided documents."
5. Be concise and factual - no unnecessary elaboration
6. Do NOT use markdown formatting - no asterisks, no bullet points, no numbered lists. Write in plain prose.
"""

GENERATION_PROMPT = """You are a precise research assistant. Your task is to answer questions using ONLY the provided context.

{grounding_rules}

CONTEXT:
{context}

---

QUESTION: {question}

ANSWER:"""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as context for the prompt.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_id' and 'content'
        
    Returns:
        Formatted context string
    """
    context_parts = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        content = chunk.get("content", "")
        source = chunk.get("source", "")
        section = chunk.get("section", "")
        
        header = f"[{chunk_id}]"
        if source:
            header += f" (Source: {source}"
            if section:
                header += f", Section: {section}"
            header += ")"
        
        context_parts.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(context_parts)


def build_generation_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the complete prompt for answer generation.
    
    Args:
        question: The user's question
        chunks: List of retrieved chunk dictionaries
        
    Returns:
        Complete prompt string
    """
    context = format_context(chunks)
    
    return GENERATION_PROMPT.format(
        grounding_rules=GROUNDING_RULES,
        context=context,
        question=question,
    )


# Prompt for faithfulness evaluation
FAITHFULNESS_PROMPT = """You are an evaluator checking if an answer is grounded in the provided context.

CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

TASK: Evaluate the faithfulness of the answer.

Score the answer:
- 1.0 = All claims in the answer are directly supported by the context
- 0.5 = Some claims are supported, but some lack support or are unclear  
- 0.0 = The answer contains claims that contradict the context or are hallucinated

Respond in this exact JSON format:
{{"score": <float>, "reasoning": "<brief explanation>"}}"""


# Prompt for usefulness evaluation
USEFULNESS_PROMPT = """You are an evaluator assessing the usefulness of an answer.

QUESTION:
{question}

ANSWER:
{answer}

TASK: Evaluate how useful this answer is.

Score the answer:
- 1.0 = The answer is clear, complete, and directly addresses the question
- 0.5 = The answer partially addresses the question or lacks clarity
- 0.0 = The answer does not address the question or is unhelpful

Respond in this exact JSON format:
{{"score": <float>, "reasoning": "<brief explanation>"}}"""
