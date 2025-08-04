"""
Test script to check if class weights are causing NaN values
"""
import torch
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.models.dataset import CandlestickDataLoader

def test_class_weights():
    """Test if class weights calculation is causing NaN values"""
    print("[SEARCH] Testing Class Weights")
    print("=" * 50)
    
    # Load the clean dataset
    csv_file = "data/candles.csv"
    print(f"[CHART] Loading dataset: {csv_file}")
    
    try:
        # Create data loader
        data_loader = CandlestickDataLoader(
            csv_file=csv_file,
            batch_size=1024,
            train_split=0.85
        )
        
        print(f"[SUCCESS] Data loader created successfully")
        
        # Get class weights
        class_weights = data_loader.get_class_weights()
        print(f"[CHART] Class weights shape: {class_weights.shape}")
        print(f"[CHART] Class weights: {class_weights}")
        
        if torch.isnan(class_weights).any():
            print("[ERROR] NaN values in class weights!")
            return False
        else:
            print("[SUCCESS] No NaN values in class weights")
        
        # Check class distribution
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        
        print(f"\n[CHART] Dataset Statistics:")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        print(f"   Total samples: {len(train_loader.dataset) + len(val_loader.dataset)}")
        
        # Check a few batches
        print(f"\n[SEARCH] Checking batch data...")
        for i, (data, target) in enumerate(train_loader):
            if i >= 3:  # Check first 3 batches
                break
            print(f"   Batch {i+1}:")
            print(f"     Data shape: {data.shape}")
            print(f"     Data range: {data.min().item():.4f} to {data.max().item():.4f}")
            print(f"     Target shape: {target.shape}")
            print(f"     Target distribution: {torch.bincount(target)}")
            
            if torch.isnan(data).any():
                print(f"     [ERROR] NaN in data!")
                return False
            else:
                print(f"     [SUCCESS] No NaN in data")
        
        print("\n[SUCCESS] All tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    success = test_class_weights()
    if success:
        print("\n[SUCCESS] Class weights are working correctly!")
    else:
        print("\n[ERROR] Class weights issue detected!") 