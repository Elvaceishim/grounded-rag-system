# AutoModel and AutoTokenizer

The `Auto` classes in Transformers provide a simple way to load any pretrained model with its associated tokenizer.

## Why Use Auto Classes?

Instead of remembering which specific class to use for each model architecture, you can use:

- `AutoModelForSequenceClassification` for classification tasks
- `AutoModelForTokenClassification` for NER tasks
- `AutoModelForCausalLM` for text generation
- `AutoTokenizer` for the corresponding tokenizer

## Loading a Model

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

## Loading from Hub

Models can be loaded directly from the Hugging Face Hub:

```python
model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
```

## Loading Local Models

You can also load models from a local directory:

```python
model = AutoModel.from_pretrained("./my_local_model")
```

## Model Configuration

Each model has an associated configuration class:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-uncased")
print(config.hidden_size)  # 768
print(config.num_attention_heads)  # 12
print(config.num_hidden_layers)  # 12
```

## Saving Models

After fine-tuning, save your model:

```python
model.save_pretrained("./my_fine_tuned_model")
tokenizer.save_pretrained("./my_fine_tuned_model")
```

## Task-Specific Models

For specific tasks, use the appropriate Auto class:

| Task | Auto Class |
|------|------------|
| Text Classification | `AutoModelForSequenceClassification` |
| Token Classification | `AutoModelForTokenClassification` |
| Question Answering | `AutoModelForQuestionAnswering` |
| Text Generation | `AutoModelForCausalLM` |
| Translation/Summarization | `AutoModelForSeq2SeqLM` |
| Fill Mask | `AutoModelForMaskedLM` |
