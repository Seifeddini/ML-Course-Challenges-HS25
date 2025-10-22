from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import random
import wandb
import os
from dotenv import load_dotenv
from torch.utils.data import random_split
import requests
import tarfile
import re
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Check GPU availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

env = {
    "max_length": 128,
    "freeze_layers": 11,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "lr": 1e-3,
    "dropout": 0.3,
    "batch_size": 32,
    "num_epochs": 5,
    "device": device
}

# Do not change function signature
def preprocess_function(examples: dict) -> dict:
    # Input:  {"text": List[str], "label": List[int]} 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    input_ids = []
    attention_masks = []

    for i, text in enumerate(examples["text"]):
        encoded = tokenizer(text, padding='max_length', truncation=True, max_length=env["max_length"])
        input_id = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        input_ids.append(input_id)
        attention_masks.append(attention_mask)

    tokenized_sample = {
        "input_ids": input_ids,
        "attention_mask": attention_masks
    }
    # Output: {"input_ids": [B, L], "attention_mask": [B, L]} 
    return tokenized_sample

class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for i, param in enumerate(self.bert.parameters()):
            if i < env["freeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=env["lstm_hidden_size"],
            num_layers=env["lstm_num_layers"],
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(env["lstm_hidden_size"]*2, 2)
        self.dropout = nn.Dropout(env["dropout"])

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        lstm_out, _ = self.bilstm(sequence_output)

        pooled_out = torch.mean(lstm_out, dim=1)
        out = self.dropout(pooled_out)
        logits = self.fc(out)
        return logits       

# Do not change function signature
def init_model() -> nn.Module:
    model = Model()
    # Input dimension:  [B, L]   (B = batch size, L = sequence length of tokenized review)
    # Output dimension: [B, 2]   (logits for sentiment classes: negative / positive)
    return model


# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset, run=None) -> nn.Module:
    # dev_dataset: tokenized IMDB dataset
    # Each sample: 
    #   input_ids: [L]        (token IDs for a review)
    #   attention_mask: [L]   (mask for padding)
    #   labels: int (0 = negative, 1 = positive)
    

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = env["lr"]
    )

    criterion = nn.CrossEntropyLoss()
    
    dataloader = DataLoader(
        dev_dataset, 
        batch_size=env["batch_size"], 
        shuffle=True,
        num_workers=0
    )
    
    model.to(env["device"])
    
    model.train()

    for epoch in range(env["num_epochs"]):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{env['num_epochs']}")

        for batch_idx, batch in enumerate(loop):
            optimizer.zero_grad()
            
            # Move batch data to device
            input_ids = batch["input_ids"].to(env["device"], non_blocking=True)
            attention_mask = batch["attention_mask"].to(env["device"], non_blocking=True)
            labels = batch["labels"].to(env["device"], non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())
                
            if run is not None:
                run.log({"loss": loss.item(), "batch": batch})

        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        

    return model

    # REMOVE ALL FROM HERE FOR SUBMISSION
def load_imdb_dataset():
    """
    Load IMDB dataset for movie review classification.
    Returns train and test data as dictionaries with 'text' and 'label' keys.
    """
    # Force the use of real IMDB dataset - no fallback to mock data
    try:
        from datasets import load_dataset
        print("Loading real IMDB dataset...")
        dataset = load_dataset("imdb")
        
        train_data = {
            "text": dataset["train"]["text"],
            "label": dataset["train"]["label"]
        }
        
        test_data = {
            "text": dataset["test"]["text"], 
            "label": dataset["test"]["label"]
        }
        
        print(f"✅ SUCCESS: Loaded real IMDB dataset")
        print(f"   - Training samples: {len(train_data['text'])}")
        print(f"   - Test samples: {len(test_data['text'])}")
        print(f"   - Labels: 0 = negative, 1 = positive")
        return train_data, test_data
        
    except ImportError as e:
        print(f"❌ ERROR: datasets library not available ({e})")
        print("❌ CRITICAL: Cannot load real IMDB dataset!")
        print("📦 Please install the datasets library:")
        print("   pip install datasets")
        print("   or")
        print("   conda install -c huggingface datasets")
        raise ImportError("datasets library is required to load IMDB dataset. Please install it with: pip install datasets")

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    """
    Main function to train and test the model on IMDB movie review dataset.
    """
    # Force correct wandb configuration (override any incorrect environment variables)
    os.environ["WANDB_ENTITY"] = "seifeddini-team"
    
    # Get wandb configuration from environment variables
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    API_KEY = os.environ.get("WANDB_API_KEY")
    print(f"WANDB_ENTITY: {wandb_entity}")
    print(f"WANDB_PROJECT: {wandb_project}")
    
    # Initialize wandb with error handling
    run = None
    try:
        if wandb_entity and wandb_project and API_KEY:
            wandb.login(key=API_KEY, relogin=True, verify=True)
            run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=env
            )
            print("Successfully initialized wandb")
        else:
            print("Warning: WANDB_ENTITY or WANDB_PROJECT not set. Skipping wandb logging.")
            print("To enable wandb logging, set the following environment variables:")
            print("  WANDB_ENTITY=your_username")
            print("  WANDB_PROJECT=your_project_name")
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        print("Continuing without wandb logging...")
        run = None

    # Load IMDB dataset
    print("Loading IMDB dataset...")
    train_data, test_data = load_imdb_dataset()
    
    # For training, we'll use a subset of the training data to make it manageable
    # You can adjust this based on your computational resources
    train_subset_size = min(5000, len(train_data["text"]))  # Use first 5000 samples for training
    train_subset = {
        "text": train_data["text"][:train_subset_size],
        "label": train_data["label"][:train_subset_size]
    }
    
    print(f"Using {len(train_subset['text'])} samples for training")
    print(f"Label mapping: 0 = negative sentiment, 1 = positive sentiment")
    
    # Preprocess training data
    print("Preprocessing training data...")
    tokenized_train_data = preprocess_function(train_subset)

    # Create training dataset from tokenized data
    train_dataset = TextDataset(
        input_ids=tokenized_train_data["input_ids"],
        attention_masks=tokenized_train_data["attention_mask"],
        labels=train_subset["label"]
    )

    # Instantiate model
    print("Initializing model...")
    model = init_model()

    # Train model
    print("Starting training...")
    trained_model = train_model(model, train_dataset, run=run)
    
    # Test the model on a subset of test data
    print("Testing model...")
    test_subset_size = min(10000, len(test_data["text"]))  # Use first 1000 samples for testing
    test_subset = {
        "text": test_data["text"][:test_subset_size],
        "label": test_data["label"][:test_subset_size]
    }
    
    # Preprocess test data
    tokenized_test_data = preprocess_function(test_subset)
    
    # Create test dataset
    test_dataset = TextDataset(
        input_ids=tokenized_test_data["input_ids"],
        attention_masks=tokenized_test_data["attention_mask"],
        labels=test_subset["label"]
    )
    
    # Evaluate model
    print(f"🔍 Starting evaluation on {env['device']}")
    trained_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(env["device"], non_blocking=True)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(env["device"], non_blocking=True)
            label = sample["labels"].to(env["device"], non_blocking=True)
            
            # Verify tensors are on correct device (only for first sample)
            if i == 0:
                print(f"✅ Evaluation tensors on device: {input_ids.device}")
            
            outputs = trained_model(input_ids, attention_mask)
            predicted = torch.argmax(outputs, dim=1)
            
            total += 1
            correct += (predicted == label).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    if run is not None:
        run.log({"test_accuracy": accuracy})
        run.finish()
        print("Wandb run finished successfully")
    else:
        print("No wandb run to finish")

if __name__ == "__main__":
    main()
