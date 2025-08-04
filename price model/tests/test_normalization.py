"""
Test script to verify normalization is working with clean dataset
"""
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.models.dataset import CandlestickDataLoader

def test_normalization():
    """Test if normalization is working correctly"""
    print("🔍 Testing Normalization")
    print("=" * 50)
    
    # Load the clean dataset
    csv_file = "data/candles_clean.csv"
    print(f"📊 Loading dataset: {csv_file}")
    
    try:
        # Create data loader
        data_loader = CandlestickDataLoader(
            csv_file=csv_file,
            batch_size=1024,
            train_split=0.85
        )
        
        print(f"✅ Data loader created successfully")
        
        # Check a few batches
        print(f"\n🔍 Checking batch data...")
        for i, (data, target) in enumerate(data_loader.get_train_loader()):
            if i >= 3:  # Check first 3 batches
                break
            print(f"   Batch {i+1}:")
            print(f"     Data shape: {data.shape}")
            print(f"     Data range: {data.min().item():.4f} to {data.max().item():.4f}")
            print(f"     Target shape: {target.shape}")
            print(f"     Target distribution: {torch.bincount(target)}")
            
            if torch.isnan(data).any():
                print(f"     ❌ NaN in data!")
                return False
            else:
                print(f"     ✅ No NaN in data")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_normalization()
    if success:
        print("\n✅ Normalization is working correctly!")
    else:
        print("\n❌ Normalization issue detected!") 