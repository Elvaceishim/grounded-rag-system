# BERT: Bidirectional Encoder Representations from Transformers

BERT is a pretrained language model that revolutionized NLP when released by Google in 2018.

## Key Innovation

BERT uses **bidirectional context** - it looks at both left and right context when understanding each word, unlike previous models that only looked left-to-right or right-to-left.

## Architecture

BERT is an **encoder-only** Transformer:

| Model | Layers | Hidden Size | Heads | Parameters |
|-------|--------|-------------|-------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

## Pre-training Objectives

BERT is pre-trained on two tasks:

### 1. Masked Language Modeling (MLM)

- Randomly mask 15% of tokens
- Model predicts the masked tokens
- Enables bidirectional context learning

```
Input:  "The [MASK] sat on the mat"
Output: "The cat sat on the mat"
```

### 2. Next Sentence Prediction (NSP)

- Given two sentences, predict if B follows A
- Helps with tasks requiring sentence relationships

## Input Format

BERT expects special tokens:

```
[CLS] First sentence [SEP] Second sentence [SEP]
```

- `[CLS]`: Classification token, used for classification tasks
- `[SEP]`: Separator between sentences

## Using BERT

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

# outputs.last_hidden_state: [batch, seq_len, hidden_size]
# outputs.pooler_output: [batch, hidden_size] - CLS token representation
```

## Common Use Cases

1. **Text Classification**: Use [CLS] token representation
2. **Named Entity Recognition**: Use token-level representations
3. **Question Answering**: Predict start and end positions
4. **Sentence Similarity**: Compare [CLS] embeddings

## BERT Variants

| Model | Description |
|-------|-------------|
| RoBERTa | Improved pre-training, no NSP |
| ALBERT | Parameter-efficient BERT |
| DistilBERT | Smaller, faster (6 layers) |
| SciBERT | Pre-trained on scientific text |
| BioBERT | Pre-trained on biomedical text |
| ClinicalBERT | Pre-trained on clinical notes |

## Limitations

- Maximum sequence length of 512 tokens
- Computationally expensive
- Not designed for generation tasks
