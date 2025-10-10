"""
Local test script for audio denoising model
Simulates the evaluation framework
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from submission import init_model, train_model
import random

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AudioDenoisingDataset(Dataset):
    """Dataset for audio denoising task"""
    def __init__(self, num_samples=100, audio_length=16000*2, noise_level=0.1):
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.noise_level = noise_level
        
        # Try to load real speech data if available
        try:
            print("Attempting to load CMU Arctic dataset...")
            cmu_arctic = torchaudio.datasets.CMUARCTIC('../P2/data_scratch', download=False)
            self.real_data = True
            self.cmu_data = cmu_arctic
            print(f"Loaded {len(cmu_arctic)} audio samples from CMU Arctic")
        except:
            print("CMU Arctic not found, using synthetic audio")
            self.real_data = False
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.real_data and idx < len(self.cmu_data):
            # Use real speech data
            waveform, sample_rate, *_ = self.cmu_data[idx % len(self.cmu_data)]
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Adjust length
            if waveform.size(1) < self.audio_length:
                # Pad if too short
                padding = self.audio_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.size(1) > self.audio_length:
                # Random crop if too long
                start_idx = torch.randint(0, waveform.size(1) - self.audio_length + 1, (1,)).item()
                waveform = waveform[:, start_idx:start_idx + self.audio_length]
            
            clean = waveform.float()
        else:
            # Generate synthetic clean audio (sine waves + harmonics)
            t = torch.linspace(0, 2, self.audio_length)
            freq = 200 + torch.rand(1).item() * 300  # Random frequency 200-500 Hz
            clean = torch.sin(2 * np.pi * freq * t)
            # Add harmonics
            clean += 0.3 * torch.sin(2 * np.pi * 2 * freq * t)
            clean += 0.1 * torch.sin(2 * np.pi * 3 * freq * t)
            clean = clean.unsqueeze(0)  # Add channel dimension
        
        # Normalize
        clean = clean / (torch.abs(clean).max() + 1e-8)
        
        # Add noise
        noise = torch.randn_like(clean) * self.noise_level
        noisy = clean + noise
        
        # Normalize noisy signal
        noisy = noisy / (torch.abs(noisy).max() + 1e-8)
        
        return noisy, clean

def get_data():
    """Create train and test datasets"""
    print("Creating datasets...")
    train_dataset = AudioDenoisingDataset(num_samples=200, audio_length=16000*2, noise_level=0.15)
    test_dataset = AudioDenoisingDataset(num_samples=50, audio_length=16000*2, noise_level=0.15)
    return train_dataset, test_dataset

def evaluate_model(model, test_dataset, device=None):
    """
    Evaluate model on test dataset
    Returns a perceptual quality score (higher is better)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    total_mse = 0.0
    total_snr = 0.0
    count = 0
    
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            denoised = model(noisy)
            
            # Handle length mismatch
            if denoised.size(-1) != clean.size(-1):
                min_len = min(denoised.size(-1), clean.size(-1))
                denoised = denoised[..., :min_len]
                clean = clean[..., :min_len]
            
            # Calculate MSE
            mse = torch.mean((denoised - clean) ** 2, dim=-1)
            total_mse += mse.sum().item()
            
            # Calculate SNR improvement
            noise_power = torch.mean((noisy - clean) ** 2, dim=-1)
            signal_power = torch.mean(clean ** 2, dim=-1)
            snr_improvement = 10 * torch.log10(signal_power / (mse + 1e-8))
            total_snr += snr_improvement.sum().item()
            
            count += noisy.size(0)
    
    avg_mse = total_mse / count
    avg_snr = total_snr / count
    
    # Convert to perceptual quality score
    # Higher SNR and lower MSE = better quality
    # Scale to approximate the scoring system
    score = avg_snr / 10.0  # Normalize
    
    print(f"\nEvaluation Results:")
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average SNR Improvement: {avg_snr:.2f} dB")
    print(f"  Perceptual Quality Score: {score:.4f}")
    
    return score

def run():
    """Main evaluation function matching the framework"""
    set_seed(42)
    
    # Get datasets for training and testing
    train_dataset, test_dataset = get_data()
    
    # Initialize the model using student's init_model function
    print("\nInitializing model...")
    model = init_model()
    
    # Print model info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dummy_input = torch.randn(1, 1, 16000*2).to(device)
    model = model.to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Model input shape: {dummy_input.shape}")
    print(f"Model output shape: {dummy_output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model using student's train_model function
    print("\nTraining model...")
    model = train_model(model, train_dataset)
    
    # Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    model.eval()
    score = evaluate_model(model, test_dataset, device)
    
    return score

if __name__ == "__main__":
    result = run()
    print(f"\n{'='*50}")
    print(f"Final Test Score: {result:.4f}")
    print(f"{'='*50}")
    
    # Provide feedback based on score
    if result >= 1.3:
        print("🎉 Excellent! Score >= 1.3 (6 points)")
    elif result >= 1.2:
        print("🎯 Great! Score >= 1.2 (5 points)")
    elif result >= 1.15:
        print("👍 Good! Score >= 1.15 (4 points)")
    elif result >= 1.11:
        print("✓ Decent! Score >= 1.11 (3 points)")
    elif result >= 1.08:
        print("→ Passing! Score >= 1.08 (2 points)")
    elif result >= 1.0:
        print("· Basic! Score >= 1.0 (1 point)")
    else:
        print("⚠ Below threshold. Score < 1.0 (0 points)")

