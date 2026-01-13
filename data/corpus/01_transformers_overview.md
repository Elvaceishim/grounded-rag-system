# Transformers Overview

Transformers is a library of pretrained models for Natural Language Processing (NLP), Computer Vision, Audio, and Multimodal tasks.

## What are Transformers?

The Transformer architecture was introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017. It revolutionized deep learning by replacing recurrent layers with self-attention mechanisms.

### Key Components

1. **Self-Attention**: Allows the model to weigh the importance of different parts of the input when processing each element.

2. **Multi-Head Attention**: Runs multiple attention operations in parallel, allowing the model to focus on different aspects of the input simultaneously.

3. **Feed-Forward Networks**: Applied to each position separately and identically, consisting of two linear transformations with a ReLU activation.

4. **Positional Encoding**: Since transformers don't have recurrence, positional encodings are added to give the model information about the position of tokens.

## Encoder vs Decoder

The original Transformer has two main parts:

- **Encoder**: Processes the input sequence and creates a representation. Used for tasks like classification and named entity recognition. Examples: BERT, RoBERTa.

- **Decoder**: Generates output sequences, typically used for text generation. Examples: GPT, GPT-2.

- **Encoder-Decoder**: Combines both for sequence-to-sequence tasks like translation. Examples: T5, BART.

## Pre-training Objectives

Different models use different pre-training objectives:

- **Masked Language Modeling (MLM)**: Randomly mask tokens and predict them. Used by BERT.
- **Causal Language Modeling (CLM)**: Predict the next token given previous tokens. Used by GPT.
- **Sequence-to-Sequence**: Map input sequences to output sequences. Used by T5.
