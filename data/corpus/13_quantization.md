# Quantization

Reduce model size and speed up inference with quantization.

## What is Quantization?

Quantization converts model weights from higher precision (FP32) to lower precision (INT8, FP16).

**Benefits:**
- Smaller model size (2-4x reduction)
- Faster inference
- Lower memory usage
- Runs on smaller GPUs

**Trade-offs:**
- Slight accuracy loss
- Some models quantize better than others

## Precision Types

| Type | Bits | Range | Use Case |
|------|------|-------|----------|
| FP32 | 32 | Full precision | Training |
| FP16 | 16 | Half precision | Inference, mixed training |
| BF16 | 16 | Brain float | Training on TPU/Ampere+ |
| INT8 | 8 | Integer | Quantized inference |
| INT4 | 4 | Integer | Aggressive quantization |

## Basic Quantization

### Loading in Half Precision

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### BitsAndBytes (8-bit)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

### BitsAndBytes (4-bit)

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
)
```

## GPTQ Quantization

Post-training quantization method:

```python
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=tokenizer,
)

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config=gptq_config,
)
```

## AWQ Quantization

Activation-aware weight quantization:

```python
from transformers import AwqConfig

awq_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
)

model = AutoModelForCausalLM.from_pretrained(
    "model-awq",
    quantization_config=awq_config,
)
```

## Quantization Best Practices

1. **Start with 8-bit** - Best balance of speed and quality
2. **Test accuracy** - Run benchmarks before/after
3. **Use mixed precision training** - FP16/BF16 for training
4. **Monitor memory** - Quantization reduces memory needs
5. **Consider GPTQ/AWQ for 4-bit** - Better quality than naive INT4
