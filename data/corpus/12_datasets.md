# Datasets Library

The Datasets library provides easy access to datasets for ML.

## Installation

```bash
pip install datasets
```

## Loading Datasets

### From the Hub

```python
from datasets import load_dataset

# Load a popular dataset
dataset = load_dataset("imdb")

# Load specific split
train = load_dataset("imdb", split="train")

# Load as streaming
dataset = load_dataset("imdb", streaming=True)
```

### Custom Data

```python
from datasets import Dataset

# From dictionary
data = {"text": ["hello", "world"], "label": [0, 1]}
dataset = Dataset.from_dict(data)

# From pandas
import pandas as pd
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# From CSV/JSON
dataset = load_dataset("csv", data_files="data.csv")
dataset = load_dataset("json", data_files="data.json")
```

## Data Processing

### Mapping Functions

```python
def preprocess(example):
    example["tokens"] = len(example["text"].split())
    return example

dataset = dataset.map(preprocess)

# Batched for efficiency
dataset = dataset.map(preprocess, batched=True)
```

### Filtering

```python
# Keep only long texts
dataset = dataset.filter(lambda x: len(x["text"]) > 100)
```

### Selecting Columns

```python
dataset = dataset.select_columns(["text", "label"])
dataset = dataset.remove_columns(["metadata"])
```

## Train/Test Split

```python
# Create splits
dataset = dataset.train_test_split(test_size=0.2)

# Access splits
train_data = dataset["train"]
test_data = dataset["test"]
```

## Saving and Loading

```python
# Save to disk
dataset.save_to_disk("my_dataset")

# Load from disk
from datasets import load_from_disk
dataset = load_from_disk("my_dataset")

# Push to Hub
dataset.push_to_hub("username/my-dataset")
```

## Dataset Features

Check dataset structure:

```python
print(dataset.features)
# {'text': Value(dtype='string'), 'label': ClassLabel(num_classes=2)}

print(dataset.column_names)
# ['text', 'label']

print(len(dataset))
# 25000
```

## DatasetDict

Multiple splits in one object:

```python
from datasets import DatasetDict

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})
```
