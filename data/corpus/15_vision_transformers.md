# Vision Transformers (ViT)

Applying Transformer architecture to computer vision tasks.

## What is ViT?

Vision Transformer (ViT) treats images as sequences of patches, applying the same attention mechanisms used in NLP.

**How it works:**
1. Split image into fixed-size patches (e.g., 16x16)
2. Flatten each patch into a vector
3. Add positional embeddings
4. Process through Transformer encoder
5. Use [CLS] token for classification

## Using ViT for Image Classification

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print(model.config.id2label[predicted_class])
```

## Pre-trained ViT Models

| Model | Patches | Image Size | Parameters |
|-------|---------|------------|------------|
| ViT-Base | 16x16 | 224 | 86M |
| ViT-Large | 16x16 | 224 | 307M |
| ViT-Huge | 14x14 | 224 | 632M |

## Feature Extraction

Get image embeddings for downstream tasks:

```python
from transformers import ViTModel

model = ViTModel.from_pretrained("google/vit-base-patch16-224")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state  # [batch, patches+1, hidden]
cls_embedding = outputs.last_hidden_state[:, 0]  # [batch, hidden]
```

## Image Classification Pipeline

Easy inference with pipeline API:

```python
from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("cat.jpg")
# [{'label': 'tabby cat', 'score': 0.95}]
```

## Fine-Tuning ViT

```python
from transformers import ViTForImageClassification, TrainingArguments, Trainer

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="./vit-finetuned",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Other Vision Models

| Model | Best For |
|-------|----------|
| **CLIP** | Zero-shot classification, image-text matching |
| **DINO** | Self-supervised learning, segmentation |
| **Swin** | Hierarchical vision, detection |
| **DeiT** | Data-efficient training |
| **BEiT** | Pre-training with masked image modeling |

## CLIP: Connecting Vision and Language

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(
    text=["a cat", "a dog", "a bird"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
similarity = outputs.logits_per_image.softmax(dim=1)
```
