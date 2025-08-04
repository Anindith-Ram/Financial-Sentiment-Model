"""
Test script to verify GPU usage
"""
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config.config import DEVICE
from src.models.cnn_model import EnhancedCandleCNN

def test_gpu_usage():
    """Test if GPU is being used properly"""
    print("[SEARCH] GPU Usage Test")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Check device from config
    print(f"Config DEVICE: {DEVICE}")
    
    # Create a simple model
    model = EnhancedCandleCNN(
        features_per_day=35,
        num_classes=5,
        hidden_size=128,
        use_attention=True
    )
    
    print(f"Model device before .to(): {next(model.parameters()).device}")
    
    # Move model to device
    model = model.to(DEVICE)
    
    print(f"Model device after .to(): {next(model.parameters()).device}")
    
    # Create dummy data
    batch_size = 32
    seq_len = 20
    features_per_day = 35
    
    dummy_data = torch.randn(batch_size, seq_len, features_per_day)
    dummy_target = torch.randint(0, 5, (batch_size,))
    
    print(f"Dummy data device: {dummy_data.device}")
    print(f"Dummy target device: {dummy_target.device}")
    
    # Move data to device
    dummy_data = dummy_data.to(DEVICE)
    dummy_target = dummy_target.to(DEVICE)
    
    print(f"Dummy data device after .to(): {dummy_data.device}")
    print(f"Dummy target device after .to(): {dummy_target.device}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_data)
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
            print("[SUCCESS] GPU test completed!")

if __name__ == "__main__":
    test_gpu_usage() 