#!/usr/bin/env python3
"""
Candlestick Pattern Price Model - Streamlined Pipeline
Optimized for 17.1 GB RAM with progressive training and memory management
"""
import argparse
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_collection import build_dataset
# Standard training removed - using progressive training only
from src.training.progressive_trainer import run_progressive_training
from config.config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG

def print_banner():
    """Print application banner"""
    print("="*80)
    print("🚀 CANDLESTICK PATTERN PRICE MODEL")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Optimized for: 17.1 GB RAM")
    print(f"🎯 Batch Sizes: Stage 1:512, Stage 2:256, Stage 3:128, Stage 4:64")
    print("="*80)

def check_system_requirements():
    """Check if system meets requirements"""
    print("🔍 Checking system requirements...")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device}")
        else:
            print("⚠️  CUDA not available, using CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check if data directory exists
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        print("📁 Created data directory")
    
    # Check if models directory exists
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        print("📁 Created models directory")
    
    # Check if logs directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
        print("📁 Created logs directory")
    
    print("✅ System requirements met")
    return True

def run_data_pipeline():
    """Run data collection and preprocessing pipeline"""
    print("\n📊 DATA PIPELINE")
    print("="*50)
    
    try:
        print("🔄 Building dataset...")
        build_dataset()
        print("✅ Dataset built successfully")
        return True
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        return False

# Standard training function removed - using progressive training only

def run_progressive_training_optimized(experiment_name=None):
    """Run progressive training with optimized settings"""
    print("\n🚀 PROGRESSIVE TRAINING (OPTIMIZED)")
    print("="*50)
    
    if experiment_name is None:
        experiment_name = f"progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Import already available at top level
        results = run_progressive_training(experiment_name=experiment_name)
        print(f"✅ Progressive training completed for experiment: {experiment_name}")
        return results
    except Exception as e:
        print(f"❌ Progressive training failed: {e}")
        return None

def run_prediction(ticker=None, model_path=None):
    """Run pattern prediction"""
    print("\n🔮 PATTERN PREDICTION")
    print("="*50)
    
    # Import only when needed
    from src.inference.predict import predict_patterns
    
    if ticker is None:
        ticker = "AAPL"  # Default ticker
    
    if model_path is None:
        # Find the best model
        if os.path.exists("models/stage4_best.pth"):
            model_path = "models/stage4_best.pth"
        elif os.path.exists("models/stage3_best.pth"):
            model_path = "models/stage3_best.pth"
        elif os.path.exists("models/stage2_best.pth"):
            model_path = "models/stage2_best.pth"
        elif os.path.exists("models/stage1_best.pth"):
            model_path = "models/stage1_best.pth"
        else:
            print("❌ No trained model found")
            return None
    
    try:
        print(f"📈 Predicting patterns for {ticker}")
        print(f"🤖 Using model: {model_path}")
        
        predictions = predict_patterns(ticker, model_path)
        
        if predictions and 'error' not in predictions:
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Ticker: {predictions['ticker']}")
            print(f"   Signal: {predictions['signal_description']}")
            print(f"   Confidence: {predictions['confidence']:.2f}")
            print(f"   Prediction: {predictions['prediction']}")
            if 'note' in predictions:
                print(f"   Note: {predictions['note']}")
        else:
            print(f"❌ Prediction failed: {predictions.get('error', 'Unknown error')}")
        
        print("✅ Prediction completed")
        return predictions
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def run_trading_signals(portfolio=None):
    """Generate trading signals"""
    print("\n💰 TRADING SIGNALS")
    print("="*50)
    
    # Import only when needed
    from src.inference.trading_signals import generate_trading_signals
    
    if portfolio is None:
        portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    try:
        print(f"📊 Generating signals for {len(portfolio)} stocks")
        signals = generate_trading_signals(portfolio)
        
        if signals and 'error' not in signals[0]:
            print(f"\n🎯 TRADING SIGNALS RESULTS:")
            for signal in signals:
                print(f"   {signal['ticker']}: {signal['signal']} (Confidence: {signal['confidence']:.2f})")
                print(f"      Reason: {signal['reason']}")
                if 'note' in signal:
                    print(f"      Note: {signal['note']}")
                print()
        else:
            print(f"❌ Trading signals failed: {signals[0].get('error', 'Unknown error')}")
        
        print("✅ Trading signals generated")
        return signals
    except Exception as e:
        print(f"❌ Trading signals failed: {e}")
        return None

def run_demo():
    """Run a complete demo pipeline"""
    print("\n🎬 DEMO PIPELINE")
    print("="*50)
    
    print("1️⃣ Building dataset...")
    if not run_data_pipeline():
        return False
    
    print("\n2️⃣ Running progressive training...")
    results = run_progressive_training_optimized("demo_training")
    if results is None:
        return False
    
    print("\n3️⃣ Running prediction...")
    predictions = run_prediction("AAPL")
    if predictions is None:
        return False
    
    print("\n4️⃣ Generating trading signals...")
    signals = run_trading_signals()
    if signals is None:
        return False
    
    print("\n✅ Demo completed successfully!")
    return True

def show_help():
    """Show detailed help information"""
    print("\n📖 USAGE GUIDE")
    print("="*50)
    print("Available modes:")
    print("  data      - Build and prepare dataset")
    print("  progressive - Optimized progressive training (4 stages)")
    print("  predict   - Predict patterns for a ticker")
    print("  signals   - Generate trading signals")
    print("  demo      - Complete pipeline demo")
    print("  all       - Run everything")
    print("\nExamples:")
    print("  python main.py --mode data")
    print("  python main.py --mode progressive --experiment-name my_experiment")
    print("  python main.py --mode predict --ticker AAPL")
    print("  python main.py --mode signals --portfolio AAPL MSFT GOOGL")
    print("  python main.py --mode demo")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Candlestick Pattern Price Model")
    parser.add_argument("--mode", choices=["data", "progressive", "predict", "signals", "demo", "all", "help"],
                       default="demo", help="Pipeline mode")
    parser.add_argument("--ticker", type=str, help="Stock ticker for prediction")
    parser.add_argument("--portfolio", nargs="+", help="Portfolio tickers for signals")
    parser.add_argument("--experiment-name", type=str, help="Experiment name for training")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--enable-logging", action="store_true", help="Enable enhanced logging")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.mode == "help":
        show_help()
        return
    
    # Print banner
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        return
    
    # Run based on mode
    if args.mode == "data":
        run_data_pipeline()
    
    elif args.mode == "train":
        print("❌ Standard training removed. Use 'progressive' mode instead.")
        print("   Example: python main.py --mode progressive --experiment-name my_experiment")
    
    elif args.mode == "progressive":
        run_progressive_training_optimized(args.experiment_name)
    
    elif args.mode == "predict":
        run_prediction(args.ticker, args.model_path)
    
    elif args.mode == "signals":
        run_trading_signals(args.portfolio)
    
    elif args.mode == "demo":
        run_demo()
    
    elif args.mode == "all":
        print("\n🔄 RUNNING COMPLETE PIPELINE")
        print("="*50)
        
        # Data pipeline
        if not run_data_pipeline():
            return
        
        # Progressive training
        results = run_progressive_training_optimized(args.experiment_name)
        if results is None:
            return
        
        # Prediction
        predictions = run_prediction(args.ticker)
        if predictions is None:
            return
        
        # Trading signals
        signals = run_trading_signals(args.portfolio)
        if signals is None:
            return
        
        print("\n✅ Complete pipeline finished!")
    
    print(f"\n🎯 Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 