from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

env = {
    "max_length": 128,
    "freeze_layers": 10,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1
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

class model(nn.Module):
    def __init__(self, num_classes=2):
        super(model, self).__init__()
        self.bert = BertModel.from_pretrained('bert_base_uncased')
        for i, param in enumerate(bert.parameters()):
            param.requires_grad = False
            if env["freeze_layers"] >= i:
                break
        
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=env["lstm_hidden_size"],
            num_layers=env["lstm_num_layers"]
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
    model = model()
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

    # REMOVE ALL FROM HERE FOR SUBMISSION
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

    # Instantiate model
    model = init_model()

    # Create a dummy Dataset object if required for train_model
    # Here we just use dev_dataset=None for illustrative purposes,
    # as actual implementation would require constructing a torch.utils.data.Dataset
    dev_dataset = None

    # Train model
    trained_model = train_model(model, dev_dataset)

if __name__ == "__main__":
    main()
