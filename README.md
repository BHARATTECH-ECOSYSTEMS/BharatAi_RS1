

# BharatAI RS_1: Transformer-Based Language Model

## Overview
BharatAI RS_1 is a transformer-based language model designed for text generation. This repository contains the necessary components to train, fine-tune, and perform inference with BharatAI.

## Installation
Before running the model, install the required dependencies:
```sh
pip install torch transformers datasets sentencepiece evaluate accelerate zstandard
```

## File Structure
- **tokenizer.py** - Defines the SentencePiece tokenizer.
- **model.py** - Contains the BharatAI transformer architecture.
- **train.py** - Script for training the model.
- **inference.py** - Script for generating text using the trained model.
- **model.pkl** - Pre-generated model file.
- **tokenizer.model** - Pre-generated tokenizer file.

## Tokenizer
The tokenizer is based on SentencePiece and has been pre-generated. If you wish to train a new tokenizer, use:
```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(input='data.txt', model_prefix='tokenizer', vocab_size=1000)
```

## Model Architecture
The BharatAI RS_1 model consists of multiple transformer blocks with self-attention mechanisms. It includes:
- Multi-head self-attention
- Feedforward layers
- Layer normalization
- Positional embeddings

### Model Hyperparameters
The model uses the following default hyperparameters:
```python
batch_size = 64
block_size = 256
max_iters = 250
learning_rate = 3e-4
eval_iters = 150
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.2
```
These can be adjusted in `train.py` or `model.py` as needed.

## Training the Model
### Important: The model is untrained by default
Users must train the model before using it for text generation. To train the model, run:
```sh
python train.py
```
This script loads the dataset, tokenizes text, and trains the transformer model from scratch.

## Pre-Generated Model & Tokenizer
- A pre-generated model (`model.bin`) and tokenizer (`tokenizer.model`) are included in the repository.
- If you wish to use them, simply load them without retraining:
```python
import torch
model = torch.load("model.bin") 
```

## Generating Text
After training, or using the pre-generated model, you can generate text using:
```sh
python inference.py --input "Your prompt here"
```

## Notes
- The model is **untrained by default**, so users must train it first before inference.
- Modify hyperparameters in `train.py` to optimize performance.
