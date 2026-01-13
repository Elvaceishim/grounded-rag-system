# GPT: Generative Pre-trained Transformer

GPT models are decoder-only Transformers designed for text generation.

## Architecture

GPT uses only the **decoder** part of the Transformer:

- Causal (unidirectional) attention - only looks at previous tokens
- Trained to predict the next token
- No encoder component

## GPT Model Sizes

| Model | Layers | Hidden | Heads | Parameters |
|-------|--------|--------|-------|------------|
| GPT | 12 | 768 | 12 | 117M |
| GPT-2 Small | 12 | 768 | 12 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 355M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |

## Pre-training Objective

GPT uses **Causal Language Modeling (CLM)**:

- Predict the next token given all previous tokens
- Autoregressive: generate one token at a time

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

## Using GPT-2 for Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

## Generation Parameters

| Parameter | Description |
|-----------|-------------|
| `max_length` | Maximum tokens to generate |
| `temperature` | Higher = more random (0.1-1.0) |
| `top_k` | Only sample from top K tokens |
| `top_p` | Nucleus sampling threshold |
| `do_sample` | Enable sampling vs greedy |
| `num_beams` | Beam search width |
| `repetition_penalty` | Penalize repeated tokens |

## Decoding Strategies

### Greedy Decoding
Always pick the highest probability token. Fast but repetitive.

### Beam Search
Keep top N sequences at each step. More diverse but still deterministic.

### Sampling with Temperature
Sample from probability distribution. Temperature controls randomness.

### Top-K Sampling
Only consider the K most likely tokens.

### Nucleus (Top-P) Sampling
Sample from smallest set of tokens whose cumulative probability exceeds P.

## GPT vs BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (left-to-right) | Bidirectional |
| Pre-training | Next token prediction | Masked + NSP |
| Best for | Generation | Understanding |

## Modern GPT Models

- **GPT-3**: 175B parameters, few-shot learning
- **GPT-3.5**: Powers ChatGPT
- **GPT-4**: Multimodal, improved reasoning
- **GPT-4o**: Optimized for speed
