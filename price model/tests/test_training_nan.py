"""
Test script to identify where NaN values are introduced during training
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.models.cnn_model import EnhancedCandleCNN

def test_training_step():
    """Test a single training step to identify NaN introduction"""
    print("🔍 Testing Training Step")
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
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data
    data = torch.randn(batch_size, seq_len, features_per_day)
    target = torch.randint(0, num_classes, (batch_size,))
    
    print(f"📊 Data shape: {data.shape}")
    print(f"📊 Target shape: {target.shape}")
    print(f"📊 Data range: {data.min().item():.4f} to {data.max().item():.4f}")
    
    # Test forward pass
    print("\n🔍 Step 1: Forward Pass")
    model.train()
    output = model(data)
    print(f"✅ Forward pass successful")
    print(f"📊 Output shape: {output.shape}")
    print(f"📊 Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    if torch.isnan(output).any():
        print("❌ NaN in output after forward pass!")
        return False
    
    # Test loss computation
    print("\n🔍 Step 2: Loss Computation")
    loss = criterion(output, target)
    print(f"✅ Loss computation successful")
    print(f"📊 Loss value: {loss.item():.4f}")
    
    if torch.isnan(loss):
        print("❌ NaN in loss!")
        return False
    
    # Test backward pass
    print("\n🔍 Step 3: Backward Pass")
    optimizer.zero_grad()
    loss.backward()
    print(f"✅ Backward pass successful")
    
    # Check gradients
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"❌ NaN gradient in {name}")
            has_nan_grad = True
    
    if has_nan_grad:
        print("❌ NaN gradients detected!")
        return False
    else:
        print("✅ No NaN gradients")
    
    # Test optimizer step
    print("\n🔍 Step 4: Optimizer Step")
    optimizer.step()
    print(f"✅ Optimizer step successful")
    
    # Check parameters after update
    has_nan_param = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN parameter in {name}")
            has_nan_param = True
    
    if has_nan_param:
        print("❌ NaN parameters after optimizer step!")
        return False
    else:
        print("✅ No NaN parameters after optimizer step")
    
    return True

if __name__ == "__main__":
    success = test_training_step()
    if success:
        print("\n✅ Training step test passed!")
    else:
        print("\n❌ Training step test failed!") 