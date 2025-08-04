"""
Test script to check model initialization and identify NaN issues
"""
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.models.cnn_model import EnhancedCandleCNN

def test_model_initialization():
    """Test if model initialization produces NaN values"""
    print("🔍 Testing Model Initialization")
    print("=" * 50)
    
    # Test parameters
    features_per_day = 82  # From the training output
    num_classes = 5
    hidden_size = 128
    batch_size = 32
    seq_len = 20
    
    print(f"📊 Test Parameters:")
    print(f"   Features per day: {features_per_day}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    
    # Create model
    model = EnhancedCandleCNN(
        features_per_day=features_per_day,
        num_classes=num_classes,
        hidden_size=hidden_size,
        use_attention=True
    )
    
    print(f"✅ Model created successfully")
    
    # Check for NaN in model parameters
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN found in {name}")
            has_nan = True
        else:
            print(f"✅ {name}: OK")
    
    if not has_nan:
        print("✅ No NaN values in model parameters")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, seq_len, features_per_day)
    print(f"📊 Input shape: {dummy_input.shape}")
    print(f"📊 Input range: {dummy_input.min().item():.4f} to {dummy_input.max().item():.4f}")
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful")
            print(f"📊 Output shape: {output.shape}")
            print(f"📊 Output range: {output.min().item():.4f} to {output.max().item():.4f}")
            
            if torch.isnan(output).any():
                print("❌ NaN values in output!")
                return False
            else:
                print("✅ No NaN values in output")
                return True
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_initialization()
    if success:
        print("\n✅ Model initialization test passed!")
    else:
        print("\n❌ Model initialization test failed!") 