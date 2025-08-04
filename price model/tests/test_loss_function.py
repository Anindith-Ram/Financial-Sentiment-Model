"""
Test script to check if the loss function is causing NaN values
"""
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.models.cnn_model import EnhancedCandleCNN
from src.training.advanced_utils import FocalLoss

def test_loss_functions():
    """Test different loss functions to identify NaN issues"""
    print("[SEARCH] Testing Loss Functions")
    print("=" * 50)
    
    # Model parameters
    features_per_day = 82
    num_classes = 5
    hidden_size = 128
    batch_size = 32
    seq_len = 20
    
    # Create model
    model = EnhancedCandleCNN(
        features_per_day=features_per_day,
        num_classes=num_classes,
        hidden_size=hidden_size,
        use_attention=True
    )
    
    # Create dummy data
    data = torch.randn(batch_size, seq_len, features_per_day)
    target = torch.randint(0, num_classes, (batch_size,))
    
            print(f"[CHART] Data shape: {data.shape}")
        print(f"[CHART] Target shape: {target.shape}")
        print(f"[CHART] Target distribution: {torch.bincount(target)}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(data)
        print(f"[SUCCESS] Model output shape: {output.shape}")
        print(f"[SUCCESS] Model output range: {output.min().item():.4f} to {output.max().item():.4f}")
        
        if torch.isnan(output).any():
            print("[ERROR] NaN in model output!")
            return False
    
    # Test different loss functions
    print(f"\n[SEARCH] Testing Loss Functions")
    print("-" * 40)
    
    # 1. CrossEntropyLoss
    print("1. Testing CrossEntropyLoss...")
    criterion1 = nn.CrossEntropyLoss()
    loss1 = criterion1(output, target)
    print(f"   Loss: {loss1.item():.4f}")
    if torch.isnan(loss1):
        print("   [ERROR] NaN in CrossEntropyLoss!")
        return False
    else:
        print("   [SUCCESS] CrossEntropyLoss OK")
    
    # 2. CrossEntropyLoss with class weights
    print("2. Testing CrossEntropyLoss with class weights...")
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    criterion2 = nn.CrossEntropyLoss(weight=class_weights)
    loss2 = criterion2(output, target)
    print(f"   Loss: {loss2.item():.4f}")
    if torch.isnan(loss2):
        print("   [ERROR] NaN in CrossEntropyLoss with weights!")
        return False
    else:
        print("   [SUCCESS] CrossEntropyLoss with weights OK")
    
    # 3. FocalLoss
    print("3. Testing FocalLoss...")
    criterion3 = FocalLoss(alpha=0.2, gamma=2.0)
    loss3 = criterion3(output, target)
    print(f"   Loss: {loss3.item():.4f}")
    if torch.isnan(loss3):
        print("   [ERROR] NaN in FocalLoss!")
        return False
    else:
        print("   [SUCCESS] FocalLoss OK")
    
    # 4. Test with extreme values
    print("4. Testing with extreme output values...")
    extreme_output = torch.tensor([[1000.0, -1000.0, 0.0, 0.0, 0.0]] * batch_size)
    loss4 = criterion1(extreme_output, target)
    print(f"   Loss: {loss4.item():.4f}")
    if torch.isnan(loss4):
        print("   [ERROR] NaN with extreme values!")
        return False
    else:
        print("   [SUCCESS] Extreme values OK")
    
    # 5. Test with zero values
    print("5. Testing with zero output values...")
    zero_output = torch.zeros(batch_size, num_classes)
    loss5 = criterion1(zero_output, target)
    print(f"   Loss: {loss5.item():.4f}")
    if torch.isnan(loss5):
        print("   [ERROR] NaN with zero values!")
        return False
    else:
        print("   [SUCCESS] Zero values OK")
    
    print("\n[SUCCESS] All loss function tests passed!")
    return True

if __name__ == "__main__":
    success = test_loss_functions()
    if success:
        print("\n[SUCCESS] Loss functions are working correctly!")
    else:
        print("\n[ERROR] Loss function issue detected!") 