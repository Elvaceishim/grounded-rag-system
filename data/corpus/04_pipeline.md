# Pipeline API

The Pipeline API is the easiest way to use pretrained models for inference.

## What is a Pipeline?

A pipeline wraps together a model and its preprocessing/postprocessing steps into a single, easy-to-use API.

## Basic Usage

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Use it
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Available Pipelines

| Task | Pipeline Name |
|------|---------------|
| Text Classification | `sentiment-analysis`, `text-classification` |
| Named Entity Recognition | `ner`, `token-classification` |
| Question Answering | `question-answering` |
| Text Generation | `text-generation` |
| Fill Mask | `fill-mask` |
| Summarization | `summarization` |
| Translation | `translation` |
| Zero-Shot Classification | `zero-shot-classification` |
| Feature Extraction | `feature-extraction` |
| Image Classification | `image-classification` |

## Using Specific Models

```python
# Use a specific model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
```

## Batch Processing

Process multiple inputs at once:

```python
texts = [
    "I love this!",
    "This is terrible.",
    "Not bad, not great."
]
results = classifier(texts)
```

## Question Answering Pipeline

```python
qa = pipeline("question-answering")

context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
question = "Where is the Eiffel Tower?"

result = qa(question=question, context=context)
# {'answer': 'Paris, France', 'score': 0.98, 'start': 28, 'end': 41}
```

## Text Generation Pipeline

```python
generator = pipeline("text-generation", model="gpt2")

result = generator(
    "The future of AI is",
    max_length=50,
    num_return_sequences=3
)
```

## Zero-Shot Classification

Classify text without training examples:

```python
classifier = pipeline("zero-shot-classification")

text = "This is a tutorial about machine learning"
labels = ["education", "politics", "sports"]

result = classifier(text, candidate_labels=labels)
# {'labels': ['education', 'politics', 'sports'], 'scores': [0.95, 0.03, 0.02]}
```

## GPU Acceleration

Use GPU if available:

```python
classifier = pipeline("sentiment-analysis", device=0)  # GPU 0
# or
classifier = pipeline("sentiment-analysis", device="cuda:0")
```
