import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def init_model() -> nn.Module:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
            self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
            # Bottleneck
            self.bottleneck = nn.Conv2d(64, 128, 3, padding=1)
            # Decoder
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
            # Final classifier
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.classifier = nn.Linear(32, 10)
            
        def forward(self, x):
            # Encoder
            x1 = F.relu(self.enc1(x))          # [B,32,28,28]
            p1 = F.max_pool2d(x1, 2)           # [B,32,14,14]
            x2 = F.relu(self.enc2(p1))         # [B,64,14,14]
            p2 = F.max_pool2d(x2, 2)           # [B,64,7,7]
            # Bottleneck
            bn = F.relu(self.bottleneck(p2))   # [B,128,7,7]
            # Decoder
            u1 = self.up1(bn)                  # [B,64,14,14]
            cat1 = torch.cat([u1, x2], dim=1)  # [B,128,14,14]
            d1 = F.relu(self.dec1(cat1))       # [B,64,14,14]
            u2 = self.up2(d1)                  # [B,32,28,28]
            cat2 = torch.cat([u2, x1], dim=1)  # [B,64,28,28]
            d2 = F.relu(self.dec2(cat2))       # [B,32,28,28]
            # Final global pooling and classifier
            pooled = self.pool(d2).view(x.size(0), -1)  # [B,32]
            out = self.classifier(pooled)               # [B,10]
            return out

    return Model()

# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
  # Your code
  epochs = 10
  batch_size = 32
  learning_rate = 1e-3
  
  device = next(model.parameters()).device
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  
  model.train()
  for epoch in range(epochs):
    for batch_data, batch_label in dataloader:
      batch_data = batch_data.to(device)
      batch_label = batch_label.to(device)
      optimizer.zero_grad()
      outputs = model(batch_data)
      loss = criterion(outputs, batch_label)
      loss.backward()
      optimizer.step()
      
  return model
      
      
  
  # Uncomment to modify the dataset (optional)
  # class MyDataset(Dataset):
  #     def __init__(self, base_dataset: Dataset):
  #         self.base_dataset = base_dataset
  #
  #     def __len__(self) -> int:
  #         return len(self.base_dataset)
  #
  #     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
  #         data, target = self.base_dataset[idx]
  #         return data, target
  #
  # train_dataset = MyDataset(dev_dataset)

  return model