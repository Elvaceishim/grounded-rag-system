# Grounded RAG System with Evaluation

A production-ready Retrieval-Augmented Generation system with **built-in evaluation**, **agentic retry capabilities**, and **hybrid search**. Unlike typical RAG demos, this system prioritizes measurability, debuggability, and answer correctness.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)

## Key Features

| Feature | Description |
|---------|-------------|
| **Grounded Answers** | LLM only uses retrieved documents, with citations |
| **Built-in Evaluation** | Faithfulness, usefulness, and failure classification |
| **Re-query Agent** | Automatic retry with rephrasing when confidence is low |
| **Hybrid Search** | Vector + keyword search with RRF merging |
| **Web UI** | Dark-themed interface with citation highlighting |
| **Docker Ready** | One-command deployment |

## Architecture

```
User Query → Embedder → Vector Search (Top-K) → Context Assembly
                              ↓
                      Keyword Search (Hybrid)
                              ↓
                     RRF Merge → LLM Generation → Evaluation
                                                      ↓
                                              Failure Classification
                                                      ↓
                                                Structured Logs
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Elvaceishim/grounded-rag-system.git
cd grounded-rag-system
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your OpenRouter API key to .env
```

### 3. Ingest Corpus

```bash
python scripts/ingest.py
```

### 4. Run

```bash
uvicorn src.api.main:app --reload
# Open http://127.0.0.1:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/query` | POST | Standard RAG query with evaluation |
| `/agent/query` | POST | Agent query with auto-retry |
| `/evaluate` | POST | Evaluate with ground truth labels |
| `/metrics` | GET | Aggregated performance metrics |
| `/health` | GET | Health check |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is BERT?", "top_k": 5}'
```

### Example Response

```json
{
  "query_id": "5371a5bc",
  "generated_answer": "BERT is a pretrained language model...",
  "citations": ["f70f08b5ae1b"],
  "evaluation": {
    "faithfulness_score": 1.0,
    "usefulness_score": 1.0,
    "failure_type": "success"
  }
}
```

## Evaluation System

### Metrics

- **Faithfulness**: Is the answer grounded in retrieved documents?
- **Usefulness**: Is the answer clear, complete, and helpful?
- **Recall@K**: Did we find all relevant documents?
- **Precision@K**: Were retrieved documents relevant?

### Failure Classification

| Type | Meaning |
|------|---------|
| `success` | All good |
| `retrieval_miss` | Wrong documents retrieved |
| `partial_retrieval` | Some relevant docs missing |
| `hallucination` | Answer made up facts |
| `over_refusal` | Refused when answer was available |
| `context_overload` | Too much context confused LLM |

## Configuration

Environment variables (`.env`):

```env
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

## Project Structure

```
grounded-rag-system/
├── src/
│   ├── ingestion/      # Document loading & chunking
│   ├── retrieval/      # Vector & hybrid search
│   ├── generation/     # LLM prompts & generation
│   ├── evaluation/     # Metrics & failure classification
│   ├── agent/          # Re-query agent
│   ├── logging/        # Structured query logs
│   └── api/            # FastAPI endpoints
├── static/             # Web UI
├── data/corpus/        # Sample HuggingFace docs (15 files)
├── scripts/            # CLI tools
└── tests/              # Unit tests
```

## Running Tests

```bash
pytest tests/ -v
```

## Deployed Render UI

https://grounded-rag-system.onrender.com/static/index.html

## Sample Corpus

Includes 15 HuggingFace Transformers documentation files:
- Transformers overview, BERT, GPT
- Tokenizers, Pipeline API, Trainer
- Fine-tuning, LoRA/PEFT, Quantization
- Vision Transformers, Datasets library

## What Makes This Different

1. **Evaluation-first** - Every query is scored for quality
2. **Failure taxonomy** - Know WHY something failed
3. **Observable** - Full structured logging for debugging
4. **Agentic** - Auto-retry with different strategies
5. **Hybrid search** - Best of vector + keyword

## License

MIT License - feel free to use for learning and projects.

## Contributing

Contributions welcome! Please open an issue first to discuss changes.
