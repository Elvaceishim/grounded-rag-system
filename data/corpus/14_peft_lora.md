# PEFT: Parameter-Efficient Fine-Tuning

Train large models efficiently by updating only a small subset of parameters.

## What is PEFT?

Parameter-Efficient Fine-Tuning (PEFT) methods freeze most model weights and only train a small number of additional parameters.

**Benefits:**
- Train 10B+ parameter models on consumer GPUs
- Reduce training time by 10-100x
- Store fine-tuned models in MBs instead of GBs
- Avoid catastrophic forgetting

## Installation

```bash
pip install peft
```

## LoRA (Low-Rank Adaptation)

The most popular PEFT method. Adds trainable low-rank matrices to existing weights.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                      # Rank of the update matrices
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

### LoRA Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `r` | Rank of matrices | 4, 8, 16, 32 |
| `lora_alpha` | Scaling factor | 16, 32, 64 |
| `target_modules` | Layers to adapt | q_proj, v_proj, k_proj |
| `lora_dropout` | Dropout rate | 0.05, 0.1 |

## QLoRA

Combines LoRA with 4-bit quantization for even more efficiency:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
)

model = get_peft_model(model, lora_config)
```

## Prefix Tuning

Add trainable tokens to the input:

```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
)
```

## Prompt Tuning

Learn soft prompts that are prepended to inputs:

```python
from peft import PromptTuningConfig

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the sentiment:",
)
```

## Saving and Loading PEFT Models

```python
# Save adapter weights only (few MBs)
model.save_pretrained("my-lora-adapter")

# Load adapter on top of base model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "my-lora-adapter")

# Merge adapter into base model
model = model.merge_and_unload()
```

## Combining Multiple Adapters

```python
# Load multiple LoRA adapters
model.load_adapter("adapter1", adapter_name="task1")
model.load_adapter("adapter2", adapter_name="task2")

# Switch between adapters
model.set_adapter("task1")
```
