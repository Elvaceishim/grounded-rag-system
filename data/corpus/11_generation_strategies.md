# Text Generation Strategies

Guide to different text generation strategies with Transformers.

## Greedy Search

The simplest strategy - always pick the highest probability token.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=10)
```

**Pros:**
- Fast
- Deterministic
- Good for factual queries

**Cons:**
- Often repetitive
- May get stuck in loops
- Not creative

## Beam Search

Keeps track of top N sequences at each step.

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    num_beams=5,
    early_stopping=True,
)
```

**Parameters:**
- `num_beams`: Number of beams (sequences) to keep
- `early_stopping`: Stop when all beams reach EOS
- `no_repeat_ngram_size`: Prevent n-gram repetition

## Sampling

Sample from the probability distribution for diversity.

### Temperature Sampling

Temperature controls randomness:
- Low temperature (0.1): More focused, conservative
- High temperature (1.5): More random, creative

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
)
```

### Top-K Sampling

Only consider the K most likely tokens:

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
)
```

### Top-P (Nucleus) Sampling

Sample from smallest set of tokens with cumulative probability >= P:

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.92,
)
```

## Contrastive Search

New method that balances relevance and diversity:

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    penalty_alpha=0.6,
    top_k=4,
)
```

## Repetition Control

Prevent repetitive text:

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
)
```

| Parameter | Effect |
|-----------|--------|
| `repetition_penalty` | Penalize already-used tokens |
| `no_repeat_ngram_size` | Prevent repeated n-grams |
| `encoder_repetition_penalty` | For encoder-decoder models |
