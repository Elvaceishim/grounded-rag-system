# Tokenizers

Tokenizers convert text into numerical representations that models can process.

## What is Tokenization?

Tokenization is the process of splitting text into smaller units called tokens. These tokens are then converted to numerical IDs.

## Types of Tokenization

### Word-level Tokenization
Splits text by spaces and punctuation. Simple but creates large vocabularies and struggles with unknown words.

### Character-level Tokenization
Uses individual characters as tokens. Small vocabulary but loses word meaning and creates long sequences.

### Subword Tokenization
The most common approach in modern NLP. Splits words into smaller meaningful units.

**Examples of subword tokenizers:**
- **BPE (Byte Pair Encoding)**: Used by GPT-2, RoBERTa
- **WordPiece**: Used by BERT
- **SentencePiece**: Used by T5, ALBERT

## Using Tokenizers

Basic usage:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
# ['hello', ',', 'how', 'are', 'you', '?']

# Get token IDs
ids = tokenizer.encode(text)
# [101, 7592, 1010, 2129, 2024, 2017, 102]
```

## Special Tokens

Tokenizers add special tokens for model-specific purposes:

- `[CLS]` / `<s>`: Start of sequence (classification token)
- `[SEP]` / `</s>`: End of sequence / separator
- `[PAD]`: Padding token for batching
- `[UNK]`: Unknown token for out-of-vocabulary words
- `[MASK]`: Mask token for masked language modeling

## Encoding for Models

The recommended way to prepare inputs:

```python
# Single text
encoded = tokenizer("Hello, world!", return_tensors="pt")
# Returns: input_ids, attention_mask

# Batch of texts
encoded = tokenizer(["Hello!", "World!"], padding=True, return_tensors="pt")
```

## Truncation and Padding

Handle variable-length sequences:

```python
encoded = tokenizer(
    texts,
    max_length=512,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)
```

## Decoding

Convert IDs back to text:

```python
ids = [101, 7592, 2088, 102]
text = tokenizer.decode(ids)
# "[CLS] hello world [SEP]"

text = tokenizer.decode(ids, skip_special_tokens=True)
# "hello world"
```
