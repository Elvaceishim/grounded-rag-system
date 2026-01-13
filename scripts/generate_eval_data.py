#!/usr/bin/env python3
"""
Evaluation dataset generator.

Generates query-chunk pairs by asking LLM to create questions
that each chunk can answer.

Usage:
    python scripts/generate_eval_data.py [--num-per-chunk 2] [--output PATH]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest import get_collection
from src.generation import Generator
from src.config import settings


QUESTION_GENERATION_PROMPT = """Given this document excerpt, generate {num_questions} factual questions that this text can answer.

Rules:
1. Questions should be specific and answerable from THIS text alone
2. Include a mix of simple lookup and reasoning questions
3. Avoid yes/no questions

Text:
{chunk_content}

Respond in JSON format:
{{"questions": ["question 1", "question 2", ...]}}"""


def generate_questions(generator: Generator, chunk_content: str, num_questions: int = 2) -> list[str]:
    """Generate questions that a chunk can answer."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=num_questions,
        chunk_content=chunk_content[:2000],  # Limit length
    )
    
    try:
        response = generator.call_llm_for_eval(prompt, temperature=0.7)
        # Extract JSON
        import re
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("questions", [])
    except Exception as e:
        print(f"Warning: Failed to generate questions: {e}")
    
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation dataset")
    parser.add_argument(
        "--num-per-chunk",
        type=int,
        default=2,
        help="Number of questions to generate per chunk",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        help="Maximum number of chunks to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/generated_queries.json"),
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Evaluation Dataset Generator")
    print("=" * 50)
    
    # Get collection
    print("Loading chunks from collection...")
    try:
        collection = get_collection()
    except Exception as e:
        print(f"Error: Could not load collection. Have you ingested documents? {e}")
        sys.exit(1)
    
    # Get all chunks
    results = collection.get(
        limit=args.max_chunks,
        include=["documents", "metadatas"],
    )
    
    if not results["ids"]:
        print("Error: No chunks found in collection.")
        sys.exit(1)
    
    print(f"Found {len(results['ids'])} chunks")
    print(f"Generating {args.num_per_chunk} questions per chunk...")
    
    # Initialize generator
    generator = Generator()
    
    # Generate questions
    eval_data = []
    
    for i, (chunk_id, content, metadata) in enumerate(zip(
        results["ids"], results["documents"], results["metadatas"]
    )):
        print(f"Processing chunk {i+1}/{len(results['ids'])}...", end=" ")
        
        questions = generate_questions(generator, content, args.num_per_chunk)
        
        for question in questions:
            eval_data.append({
                "query": question,
                "relevant_chunk_ids": [chunk_id],
                "source": metadata.get("source", ""),
                "section": metadata.get("section", ""),
            })
        
        print(f"Generated {len(questions)} questions")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Total questions generated: {len(eval_data)}")
    print(f"Output file: {args.output}")
    print()
    print("Review the generated questions and remove any low-quality ones.")


if __name__ == "__main__":
    main()
