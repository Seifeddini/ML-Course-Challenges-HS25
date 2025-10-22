from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset

# Do not change function signature
def preprocess_function(examples: dict) -> dict:
    # Input:  {"text": List[str], "label": List[int]} 
    # Output: {"input_ids": [B, L], "attention_mask": [B, L]} 
    return tokenized_sample


# Do not change function signature
def init_model() -> nn.Module:
    model = ...
    # Input dimension:  [B, L]   (B = batch size, L = sequence length of tokenized review)
    # Output dimension: [B, 2]   (logits for sentiment classes: negative / positive)
    return model


# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
    # dev_dataset: tokenized IMDB dataset
    # Each sample: 
    #   input_ids: [L]        (token IDs for a review)
    #   attention_mask: [L]   (mask for padding)
    #   labels: int (0 = negative, 1 = positive)
    return model

