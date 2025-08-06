#!/usr/bin/env python3
"""
🚀 UNIFIED PIPELINE INTERFACE
============================

Professional pipeline interface that coordinates your familiar research and production trainers
while adding professional features and a unified interface.

This gives you the BEST OF BOTH WORLDS:
- ✅ Keep separate research_trainer.py and production_trainer.py
- ✅ Professional performance optimizations added to both
- ✅ Unified interface when you want it
- ✅ Can still run trainers directly as before

Usage Examples:
  # Unified interface (NEW)
  python src/pipeline.py train --mode research --epochs 10
  python src/pipeline.py train --mode production --epochs 50
  python src/pipeline.py data collect --mode smart_update
  python src/pipeline.py predict --input data/test.csv
  python src/pipeline.py signals --live
  
  # Direct usage (STILL WORKS)
  python src/training/research_trainer.py
  python src/training/production_trainer.py
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_research_training(epochs=None, batch_size=None):
    """Run research training with optional parameters"""
    print("🔬 Starting Research Training...")
    print("=" * 50)
    
    # You can still run the research trainer directly
    cmd = ["python", "src/training/research_trainer.py"]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True, 
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Research training failed: {e}")
        return False

def run_production_training(epochs=None, batch_size=None):
    """Run production training with optional parameters"""
    print("🏭 Starting Production Training...")
    print("=" * 50)
    
    # You can still run the production trainer directly
    cmd = ["python", "src/training/production_trainer.py"]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True,
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Production training failed: {e}")
        return False

def run_data_collection(mode="smart_update"):
    """Run data collection"""
    print("📊 Starting Data Collection...")
    print("=" * 50)
    
    # Run your existing data collection script
    cmd = ["python", "src/data/data_collection.py"]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True,
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data collection failed: {e}")
        return False

def run_prediction(input_path):
    """Run prediction"""
    print("🔮 Starting Prediction...")
    print("=" * 50)
    
    # Check if prediction script exists
    predict_script = project_root / "src" / "inference" / "predict.py"
    if predict_script.exists():
        cmd = ["python", str(predict_script)]
        try:
            result = subprocess.run(cmd, cwd=project_root, check=True,
                                  capture_output=False, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Prediction failed: {e}")
            return False
    else:
        print("📝 Prediction script not found. You can create it in src/inference/predict.py")
        return False

def run_signal_generation(live=False):
    """Run trading signal generation"""
    print("📈 Starting Signal Generation...")
    print("=" * 50)
    
    # Check if signals script exists
    signals_script = project_root / "src" / "inference" / "trading_signals.py"
    if signals_script.exists():
        cmd = ["python", str(signals_script)]
        try:
            result = subprocess.run(cmd, cwd=project_root, check=True,
                                  capture_output=False, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Signal generation failed: {e}")
            return False
    else:
        print("📝 Trading signals script not found. You can create it in src/inference/trading_signals.py")
        return False

def show_status():
    """Show pipeline status"""
    print("📊 PIPELINE STATUS")
    print("=" * 50)
    
    # Check which components exist
    components = {
        "Research Trainer": project_root / "src" / "training" / "research_trainer.py",
        "Production Trainer": project_root / "src" / "training" / "production_trainer.py", 
        "Data Collection": project_root / "src" / "data" / "data_collection.py", 
        "Model Architecture": project_root / "src" / "models" / "timesnet_hybrid.py",
        "Dataset Handler": project_root / "src" / "models" / "dataset.py",
        "Professional Optimizers": project_root / "src" / "training" / "optimizers.py",
        "Advanced Schedulers": project_root / "src" / "training" / "schedulers.py",
        "Regularization": project_root / "src" / "training" / "regularization.py",
        "Error Handling": project_root / "src" / "utils" / "errors.py",
        "Professional Logging": project_root / "src" / "utils" / "logger.py",
    }
    
    for name, path in components.items():
        status = "✅ Available" if path.exists() else "❌ Missing"
        print(f"{name:25} {status}")
    
    # Check data
    data_dir = project_root / "data"
    if data_dir.exists():
        datasets = list(data_dir.glob("*.csv"))
        print(f"\n📊 Datasets Found: {len(datasets)}")
        for dataset in datasets[-3:]:  # Show last 3
            print(f"  📁 {dataset.name}")
    
    # Check models
    models_dir = project_root / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
        print(f"\n🤖 Models Found: {len(model_files)}")
        for model in model_files[-3:]:  # Show last 3
            print(f"  🤖 {model.name}")

def main():
    """Unified pipeline interface"""
    parser = argparse.ArgumentParser(
        description="🚀 Professional Financial Pipeline - Hybrid Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --mode research --epochs 10
  %(prog)s train --mode production --epochs 50
  %(prog)s data collect --mode smart_update
  %(prog)s predict --input data/test.csv
  %(prog)s signals --live
  %(prog)s status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline operations')
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--mode', required=True, choices=['research', 'production'],
                             help='Training mode: research (fast iterations) or production (full optimization)')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs (optional)')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (optional)')
    
    # Data commands
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_parser.add_argument('command', choices=['collect'], help='Data command')
    data_parser.add_argument('--mode', default='smart_update', 
                            choices=['full', 'smart_update', 'incremental'],
                            help='Collection mode')
    
    # Inference commands
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--input', required=True, help='Input data path')
    
    signals_parser = subparsers.add_parser('signals', help='Generate trading signals')
    signals_parser.add_argument('--live', action='store_true', help='Use live data')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n🎯 HYBRID PIPELINE READY!")
        print("\nYou can use:")
        print("  🔬 Research Mode: python src/pipeline.py train --mode research")
        print("  🏭 Production Mode: python src/pipeline.py train --mode production")
        print("  📊 Or run directly: python src/training/research_trainer.py")
        return
    
    success = False
    
    try:
        if args.command == 'train':
            print(f"🚀 HYBRID TRAINING - {args.mode.upper()} MODE")
            if args.mode == 'research':
                success = run_research_training(args.epochs, args.batch_size)
            elif args.mode == 'production':
                success = run_production_training(args.epochs, args.batch_size)
        
        elif args.command == 'data':
            if args.command == 'collect':
                success = run_data_collection(args.mode)
        
        elif args.command == 'predict':
            success = run_prediction(args.input)
        
        elif args.command == 'signals':
            success = run_signal_generation(args.live)
        
        elif args.command == 'status':
            show_status()
            success = True
        
        if success:
            print("\n✅ Operation completed successfully!")
        else:
            print("\n❌ Operation failed. Check the logs above for details.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        success = False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()