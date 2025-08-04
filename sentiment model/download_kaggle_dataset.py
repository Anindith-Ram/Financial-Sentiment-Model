"""
Download Kaggle Twitter sentiment dataset using kagglehub and place it in the project structure.

This script downloads the dataset and copies it to data/external/ for easy access.
"""

import kagglehub
import os
import shutil
from pathlib import Path

def download_kaggle_dataset():
    """Download the Kaggle dataset and place it in the project structure."""
    
    print("🔄 Downloading Kaggle Twitter sentiment dataset...")
    print("Dataset: thedevastator/tweet-sentiment-s-impact-on-stock-returns")
    print()
    
    try:
        # Download latest version using kagglehub
        download_path = kagglehub.dataset_download("thedevastator/tweet-sentiment-s-impact-on-stock-returns")
        print(f"[SUCCESS] Dataset downloaded to: {download_path}")
        
        # Create our target directory if it doesn't exist
        target_dir = Path("data/external")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the CSV file in the downloaded directory
        download_dir = Path(download_path)
        csv_files = list(download_dir.glob("*.csv"))
        
        if not csv_files:
            print("[ERROR] No CSV files found in the downloaded dataset!")
            return False
        
        # Copy the main CSV file to our project structure
        source_file = csv_files[0]  # Take the first CSV file
        target_file = target_dir / "kaggle_twitter_sentiment.csv"
        
        print(f"📁 Copying {source_file.name} to {target_file}")
        shutil.copy2(source_file, target_file)
        
        print()
        print("🎉 SUCCESS! Dataset ready for use.")
        print(f"📍 Dataset location: {target_file}")
                    print(f"[CHART] File size: {target_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Show a preview of the file structure
        if csv_files:
            print(f"📝 Found {len(csv_files)} CSV file(s):")
            for csv_file in csv_files:
                print(f"   - {csv_file.name}")
        
        print()
        print("🚀 Ready to run the pipeline! Use this command:")
        print("python src/data/backfill_historical_new.py \\")
        print("    --kaggle_path data/external/kaggle_twitter_sentiment.csv \\")
        print("    --tickers AAPL MSFT \\")
        print("    --since 2023-12-01 \\")
        print("    --until 2023-12-31")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error downloading dataset: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("1. Make sure you have kagglehub installed: pip install kagglehub")
        print("2. Ensure you're logged into Kaggle (kaggle configure)")
        print("3. Check your internet connection")
        return False

if __name__ == "__main__":
    download_kaggle_dataset() 