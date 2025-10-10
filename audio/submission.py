import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, kernel_size=32, stride=2):
        super(Encoder, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)

    def forward(self, x):
        return nn.functional.relu(self.conv(x))


def DepthwiseSeparableConv(in_ch, out_ch, kernel_size, dilation):
    return nn.Sequential(
        nn.Conv1d(in_ch, in_ch, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation, groups=in_ch,
                  bias=False),  # Depthwise
        nn.Conv1d(in_ch, out_ch, 1)  # Pointwise
    )


class TCNBlock(nn.Module):
    def __init__(self, in_channels=128, hidden_channels = 256, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size, dilation)
        self.PReLU1 = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(hidden_channels)

        self.skip_fin = nn.Conv1d(hidden_channels, in_channels, 1)
        self.res_fin = nn.Conv1d(hidden_channels, in_channels, 1)
    
    def forward(self, x):
        inter = self.norm1(self.PReLU1(self.conv1(x)))

        return self.skip_fin(inter), self.res_fin(inter)


class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, kernel_size=32, stride=2):
        super(Decoder, self).__init__()

        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)

    def forward(self, x):
        return self.deconv(x)


class ConvTasNet(nn.Module):
    def __init__(self, N=64, L=16, stride=8, B=32, stacks=2):
        super().__init__()
        self.encoder = Encoder(1, N, L, stride)

        self.tcn_blocks = nn.ModuleList([
            TCNBlock(N, B, kernel_size=3, dilation=2 ** i) for i in range(stacks)
        ])

        self.mask = nn.Conv1d(N, N, 1)
        self.decoder = Decoder(N, 1, L, stride)
    
    def forward(self, x):
        # x: [B, 1, L]
        original_length = x.size(-1)
        
        enc = self.encoder(x)
        res = enc
        skip_sum = 0
        
        for block in self.tcn_blocks:
            res, skip = block(res)
            skip_sum = skip_sum + skip
        
        mask = torch.sigmoid(self.mask(skip_sum))
        masked = enc * mask
        out = self.decoder(masked)
        
        # pad or trim output to match input length
        if out.size(-1) != original_length:
            if out.size(-1) < original_length:
                padding = original_length - out.size(-1)
                out = torch.nn.functional.pad(out, (0, padding))
            else:
                out = out[..., :original_length]
        
        return out


def init_model() -> nn.Module:
    # Your code
    model = ConvTasNet()
    # Input dimension: [B, 1, L]   (B = batch size, mono audio with variable length L)
    # Output dimension: [B, 1, L]  (denoised waveform, same length as input)
    
    return model


# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    epochs = 8
    batch_size = 8
    lr = 1e-3

    dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Assumes each batch is (input, target)
            if isinstance(batch, dict):
                inputs, targets = batch["input"], batch["target"]
            else:
                inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # [B, 1, L]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model
