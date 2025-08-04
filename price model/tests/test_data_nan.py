"""
Test script to check if the dataset contains NaN values
"""
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_dataset_nan():
    """Test if the dataset contains NaN values"""
    print("[SEARCH] Testing Dataset for NaN Values")
    print("=" * 50)
    
    # Load the dataset
    csv_file = "data/candles.csv"
    print(f"[CHART] Loading dataset: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"[SUCCESS] Dataset loaded successfully")
        print(f"[CHART] Dataset shape: {df.shape}")
        print(f"[CHART] Columns: {len(df.columns)}")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        total_nan = nan_counts.sum()
        
        print(f"\n[SEARCH] NaN Analysis:")
        print(f"   Total NaN values: {total_nan}")
        
        if total_nan > 0:
            print(f"[ERROR] Found {total_nan} NaN values in dataset!")
            print(f"   NaN by column:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"     {col}: {count} NaN values")
            return False
        else:
            print(f"[SUCCESS] No NaN values found in dataset")
        
        # Check data ranges
        print(f"\n[CHART] Data Range Analysis:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:  # Show first 10 columns
            col_data = df[col].dropna()
            if len(col_data) > 0:
                print(f"   {col}: {col_data.min():.4f} to {col_data.max():.4f}")
        
        # Check for infinite values
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        total_inf = inf_counts.sum()
        
        print(f"\n[SEARCH] Infinite Values Analysis:")
        print(f"   Total infinite values: {total_inf}")
        
        if total_inf > 0:
            print(f"[ERROR] Found {total_inf} infinite values in dataset!")
            return False
        else:
            print(f"[SUCCESS] No infinite values found in dataset")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_nan()
    if success:
        print("\n[SUCCESS] Dataset test passed!")
    else:
        print("\n[ERROR] Dataset test failed!") 