# Grounded Knowledge Retrieval & Answering System

An evaluated RAG system that produces factually grounded answers from a document corpus and provides quantitative evaluation of retrieval quality and answer faithfulness.

## Features

- **Document Ingestion**: Configurable chunking with metadata extraction
- **Vector Retrieval**: Semantic search using embeddings
- **Grounded Generation**: Strict prompting with citation requirements
- **Evaluation Layer**: Retrieval metrics (Recall@K, Precision@K), faithfulness scoring, failure classification
- **Observability**: Structured logging for every query

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OpenRouter API key

# Ingest documents
python scripts/ingest.py

# Run the API server
uvicorn src.api.main:app --reload

# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a transformer model?", "top_k": 5}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Retrieve chunks and generate answer |
| `/evaluate` | POST | Evaluate a query's retrieval and answer quality |
| `/metrics` | GET | Get aggregated evaluation metrics |

## Project Structure

```
src/
├── config.py          # Configuration management
├── ingestion/         # Document loading and chunking
├── retrieval/         # Embedding and vector search
├── generation/        # LLM answer generation
├── evaluation/        # Metrics and scoring
├── logging/           # Structured query logging
└── api/               # FastAPI endpoints
```

## Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `CHUNK_SIZE` | Characters per chunk | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `TOP_K` | Number of chunks to retrieve | 5 |
| `LLM_MODEL` | Model for generation | `openai/gpt-4o-mini` |

## Evaluation

The system evaluates every query on:

1. **Retrieval Quality**: Recall@K, Precision@K
2. **Answer Faithfulness**: Grounded vs hallucinated
3. **Answer Usefulness**: Clarity and completeness
4. **Failure Classification**: Categorizes failures for debugging

## License

MIT
