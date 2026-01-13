# Training with Trainer

The `Trainer` class provides a complete training loop for PyTorch models.

## Basic Training Setup

```python
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load and tokenize dataset
dataset = load_dataset("imdb")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

## TrainingArguments

Configure training parameters:

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

## Important Training Arguments

| Argument | Description |
|----------|-------------|
| `output_dir` | Where to save model checkpoints |
| `num_train_epochs` | Total number of training epochs |
| `per_device_train_batch_size` | Batch size per GPU/CPU |
| `learning_rate` | Initial learning rate (default: 5e-5) |
| `weight_decay` | Weight decay for regularization |
| `warmup_steps` | Number of warmup steps |
| `evaluation_strategy` | When to evaluate: "no", "steps", "epoch" |
| `save_strategy` | When to save: "no", "steps", "epoch" |
| `fp16` | Use mixed precision training |
| `gradient_accumulation_steps` | Accumulate gradients over N steps |

## Creating and Using the Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start training
trainer.train()
```

## Custom Metrics

Define custom evaluation metrics:

```python
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

## Saving and Loading

```python
# Save the model
trainer.save_model("./my_model")

# Load for inference
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
```

## Hyperparameter Search

Built-in hyperparameter optimization:

```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Run hyperparameter search
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=10,
)
```
