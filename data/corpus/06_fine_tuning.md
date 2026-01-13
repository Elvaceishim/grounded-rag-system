# Fine-Tuning Guide

Fine-tuning adapts a pretrained model to your specific task and data.

## Why Fine-Tune?

Pretrained models learn general language understanding from large corpora. Fine-tuning adapts this knowledge to:
- Your specific domain (legal, medical, technical)
- Your specific task (classification, NER, QA)
- Your specific data distribution

## Fine-Tuning vs Feature Extraction

**Fine-Tuning**: Update all model weights during training. Usually gives better results.

**Feature Extraction**: Freeze pretrained weights, only train a new classification head. Faster but may be less accurate.

## Steps for Fine-Tuning

### 1. Prepare Your Data

```python
from datasets import Dataset

# Your data as lists
texts = ["example 1", "example 2", ...]
labels = [0, 1, ...]

# Create dataset
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2)
```

### 2. Tokenize

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized = dataset.map(preprocess, batched=True)
```

### 3. Initialize Model

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_classes
)
```

### 4. Set Up Training

```python
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)
```

### 5. Train and Evaluate

```python
trainer.train()
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']}")
```

## Common Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Learning Rate | 1e-5 to 5e-5 | Lower for larger models |
| Batch Size | 8-32 | Limited by GPU memory |
| Epochs | 2-5 | More for smaller datasets |
| Warmup | 0-10% of steps | Helps stability |

## Tips for Better Results

1. **Start with a good pretrained model** - Domain-specific models often work better
2. **Use a lower learning rate** - Pretrained weights shouldn't change too drastically
3. **Freeze layers initially** - Especially with small datasets
4. **Use early stopping** - Prevent overfitting
5. **Data augmentation** - Create more training examples

## Freezing Layers

To freeze certain layers:

```python
# Freeze all but the last 2 layers
for param in model.base_model.parameters():
    param.requires_grad = False

# Unfreeze last 2 encoder layers
for layer in model.base_model.encoder.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
```

## Low-Rank Adaptation (LoRA)

For efficient fine-tuning of large models:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
)

model = get_peft_model(model, config)
```
