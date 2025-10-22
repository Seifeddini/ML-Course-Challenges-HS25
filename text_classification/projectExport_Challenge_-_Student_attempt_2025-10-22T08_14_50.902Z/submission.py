from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

env = {
    "max_length": 128,
    "freeze_layers": 10,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1,
    "lr": 1e-4,
    "dropout": 0.3,
    "batch_size": 16,
    "num_epochs": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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
            param.requires_grad = False
            if env["freeze_layers"] >= i:
                break
        
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
        return torch.softmax(logits, dim=1)
        

# Do not change function signature
def init_model() -> nn.Module:
    model = Model()
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
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = env["lr"]
    )

    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dev_dataset, batch_size=env["batch_size"], shuffle=True)
    
    model.to(env["device"])
    model.train()

    for epoch in range(env["num_epochs"]):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{env['num_epochs']}")

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(env["device"])
            attention_mask = batch["attention_mask"].to(env["device"])
            labels = batch["labels"].to(env["device"])

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")


    return model

    # REMOVE ALL FROM HERE FOR SUBMISSION

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
    # Example workflow: This is placeholder logic and should be 
    # replaced/adapted as appropriate for your actual data and project.
    # Here we just show how you might call your functions.

    # Mock example data
    example_data = {
        "text": ["This movie was great!", "This movie was terrible!"],
        "label": [1, 0]
    }

    # Preprocess data
    tokenized_data = preprocess_function(example_data)

    # Create dataset from tokenized data
    dev_dataset = TextDataset(
        input_ids=tokenized_data["input_ids"],
        attention_masks=tokenized_data["attention_mask"],
        labels=example_data["label"]
    )

    # Instantiate model
    model = init_model()

    # Train model
    trained_model = train_model(model, dev_dataset)

if __name__ == "__main__":
    main()
