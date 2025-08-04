#!/usr/bin/env python3
"""
Dataset Switching Script
Easily switch between original and cleaned datasets for training
"""

import json
import os
import shutil
from datetime import datetime

def switch_to_cleaned_dataset():
    """Switch configuration to use cleaned dataset"""
    config_file = "config/config.json"
    backup_file = f"config/config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create backup of current config
    if os.path.exists(config_file):
        shutil.copy2(config_file, backup_file)
        print(f"[SAVE] Backup created: {backup_file}")
    
    # Load current config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Update to use cleaned dataset
    config['DATA']['DATA_OUTPUT_PATH'] = "data/candles_cleaned.csv"
    config['DATA']['MODEL_OUTPUT_PATH'] = "models/cnn5cls_cleaned.pth"
    
    # Save updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("[SUCCESS] Switched to cleaned dataset")
    print("[INFO] Training will now use: data/candles_cleaned.csv")
    print("[INFO] Model will be saved as: models/cnn5cls_cleaned.pth")


def switch_to_original_dataset():
    """Switch configuration to use original dataset"""
    config_file = "config/config.json"
    backup_file = f"config/config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create backup of current config
    if os.path.exists(config_file):
        shutil.copy2(config_file, backup_file)
        print(f"[SAVE] Backup created: {backup_file}")
    
    # Load current config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Update to use original dataset
    config['DATA']['DATA_OUTPUT_PATH'] = "data/candles.csv"
    config['DATA']['MODEL_OUTPUT_PATH'] = "models/cnn5cls_original.pth"
    
    # Save updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("[SUCCESS] Switched to original dataset")
    print("[INFO] Training will now use: data/candles.csv")
    print("[INFO] Model will be saved as: models/cnn5cls_original.pth")


def show_current_dataset():
    """Show which dataset is currently configured"""
    config_file = "config/config.json"
    
    if not os.path.exists(config_file):
        print("[ERROR] Config file not found")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    data_path = config['DATA']['DATA_OUTPUT_PATH']
    model_path = config['DATA']['MODEL_OUTPUT_PATH']
    
    print("[CHART] Current Configuration:")
    print(f"  Dataset: {data_path}")
    print(f"  Model output: {model_path}")
    
    if "cleaned" in data_path:
        print("[INFO] Using CLEANED dataset (recommended)")
    else:
        print("[INFO] Using ORIGINAL dataset")


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Switch between original and cleaned datasets')
    parser.add_argument('--to', choices=['cleaned', 'original'], 
                       help='Switch to specified dataset')
    parser.add_argument('--show', action='store_true',
                       help='Show current dataset configuration')
    
    args = parser.parse_args()
    
    if args.show:
        show_current_dataset()
    elif args.to == 'cleaned':
        switch_to_cleaned_dataset()
    elif args.to == 'original':
        switch_to_original_dataset()
    else:
        print("[INFO] Usage:")
        print("  python src/utils/switch_dataset.py --show")
        print("  python src/utils/switch_dataset.py --to cleaned")
        print("  python src/utils/switch_dataset.py --to original")


if __name__ == "__main__":
    main() 