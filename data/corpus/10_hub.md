# Model Hub and Sharing

The Hugging Face Hub is a platform for sharing and discovering models, datasets, and demos.

## The Hugging Face Hub

The Hub hosts over 200,000 models covering:
- NLP (text classification, translation, summarization)
- Computer Vision (image classification, object detection)
- Audio (speech recognition, audio classification)
- Multimodal (vision-language models)

## Downloading Models

```python
from transformers import AutoModel

# Download from Hub
model = AutoModel.from_pretrained("bert-base-uncased")

# Download specific revision
model = AutoModel.from_pretrained("bert-base-uncased", revision="main")

# Download with authentication (for gated models)
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b", token="your_token")
```

## Model Card

Every model should have a model card describing:
- Model description and intended use
- Training data and procedure
- Evaluation results
- Limitations and biases
- How to use the model

## Uploading Models

### Using push_to_hub

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Fine-tune your model...

# Push to Hub
model.push_to_hub("my-fine-tuned-bert")
tokenizer.push_to_hub("my-fine-tuned-bert")
```

### Using the CLI

```bash
# Login first
huggingface-cli login

# Upload a directory
huggingface-cli upload my-username/my-model ./model_directory
```

## Repository Structure

A typical model repository contains:

```
my-model/
├── config.json           # Model configuration
├── pytorch_model.bin     # Model weights (PyTorch)
├── model.safetensors     # Model weights (SafeTensors)
├── tokenizer.json        # Tokenizer
├── vocab.txt            # Vocabulary
├── special_tokens_map.json
└── README.md            # Model card
```

## Private Models

Make a model private:

```python
from huggingface_hub import HfApi

api = HfApi()
api.update_repo_visibility("username/model-name", private=True)
```

## Gated Models

Some models require accepting license terms:

1. Visit the model page on Hub
2. Accept the license agreement
3. Use your access token when loading

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    token="hf_your_token_here"
)
```

## Organizations

Create an organization to:
- Share models with team members
- Use organization namespaces (org-name/model-name)
- Manage access permissions

## Spaces

Spaces let you host ML demos:
- Gradio apps
- Streamlit apps
- Static HTML
- Docker containers
