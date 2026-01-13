# Attention Mechanism

Attention is the core innovation that makes Transformer models so effective.

## What is Attention?

Attention allows the model to focus on relevant parts of the input when producing each part of the output. Instead of compressing the entire input into a fixed vector, attention creates dynamic connections between input and output positions.

## Self-Attention

In self-attention, each position in a sequence attends to all positions in the same sequence. This allows the model to capture dependencies regardless of distance.

### How Self-Attention Works

1. **Create Q, K, V vectors**: For each token, create Query, Key, and Value vectors through linear projections.

2. **Compute attention scores**: Dot product of Query with all Keys, then scale by sqrt(d_k).

3. **Apply softmax**: Convert scores to attention weights (probabilities that sum to 1).

4. **Weighted sum**: Multiply Values by attention weights and sum.

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

## Multi-Head Attention

Instead of single attention, run multiple attention operations in parallel:

```python
# Conceptual multi-head attention
heads = []
for i in range(num_heads):
    Q_i = linear_q[i](input)
    K_i = linear_k[i](input)
    V_i = linear_v[i](input)
    heads.append(attention(Q_i, K_i, V_i))

output = linear_out(concat(heads))
```

Benefits:
- Different heads can focus on different relationships
- One head might focus on syntax, another on semantics
- Improves model's representational capacity

## Attention Patterns

Different layers learn different attention patterns:

- **Early layers**: Focus on adjacent tokens, local patterns
- **Middle layers**: Syntactic relationships, sentence structure
- **Later layers**: Semantic relationships, task-specific patterns

## Cross-Attention

In encoder-decoder models, the decoder attends to the encoder outputs:

- **Query**: From decoder hidden state
- **Key, Value**: From encoder outputs

This allows the decoder to focus on relevant parts of the input when generating each output token.

## Attention Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| Scaled Dot-Product | Standard Transformer attention | Most models |
| Additive | Uses a feed-forward layer | Bahdanau attention |
| Multi-Query | Shared K,V across heads | Efficient inference |
| Grouped-Query | Groups share K,V | Balance of efficiency and quality |
| Flash Attention | Memory-efficient implementation | Large models |

## Attention Masks

Masks prevent attention to certain positions:

- **Padding mask**: Ignore padding tokens
- **Causal mask**: Only attend to previous tokens (for generation)
- **Custom masks**: Task-specific attention patterns
