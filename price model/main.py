"""
Main execution script for the Candlestick Pattern Price Model
This script demonstrates the complete pipeline: data collection, training, and prediction
"""
import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import DATA_OUTPUT_PATH, MODEL_OUTPUT_PATH
from src.data.data_collection import build_dataset, build_dataset_incremental, build_dataset_smart_update, compare_adjusted_vs_raw_analysis
from src.training.train import Trainer
from src.inference.predict import CandlestickPredictor, demo_prediction
from src.inference.trading_signals import TradingSignalGenerator


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Candlestick Pattern Price Model')
    parser.add_argument('--mode', choices=['data', 'train', 'predict', 'demo', 'all', 'signals'], 
                       default='demo', help='Mode to run')
    parser.add_argument('--ticker', type=str, default='AAPL', 
                       help='Ticker to predict (for predict mode)')
    parser.add_argument('--data-file', type=str, default=DATA_OUTPUT_PATH,
                       help='Path to dataset CSV file')
    parser.add_argument('--model-file', type=str, default=MODEL_OUTPUT_PATH,
                       help='Path to model file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    # Note: Removed --no-adjusted flag since professional pipeline always uses both raw and adjusted
    parser.add_argument('--compare-adjustment', action='store_true',
                       help='Compare raw vs adjusted close analysis')
    parser.add_argument('--incremental', action='store_true',
                       help='Use incremental data collection (saves every 100 tickers)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save data every N tickers (default: 100)')
    parser.add_argument('--smart-update', action='store_true',
                       help='Smart update: only collect recent data for new tickers and updates')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days of recent data to collect for updates (default: 30)')
    parser.add_argument('--portfolio', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                       help='List of tickers for portfolio signals')
    parser.add_argument('--progressive', action='store_true',
                       help='Use progressive training mode')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable enhanced logging')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--balance-classes', action='store_true',
                       help='Balance class distribution during training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Candlestick Pattern Price Model")
    print("=" * 60)
    
    # Handle adjusted close comparison
    if args.compare_adjustment:
        print("\n[CHART] Comparing Raw vs Adjusted Close Analysis...")
        print("-" * 40)
        try:
            analysis = compare_adjusted_vs_raw_analysis("AAPL", days=252)
            if analysis:
                print("[SUCCESS] Comparison completed successfully!")
            return
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
            return

    if args.mode == 'data' or args.mode == 'all':
        print("\n[CHART] Building professional dataset...")
        print("-" * 40)
        print("Using explicit raw/adjusted column families for maximum accuracy")
        
        try:
            if args.incremental:
                print("🔄 Using incremental data collection mode")
                print(f"💾 Saving data every {args.save_interval} tickers")
                print("⚠️  Press Ctrl+C to stop gracefully and save collected data")
                dataset_path = build_dataset_incremental(args.data_file, args.save_interval)
            elif args.smart_update:
                print("🧠 Using smart update mode")
                print(f"📅 Collecting {args.days_back} days of recent data for updates")
                dataset_path = build_dataset_smart_update(args.data_file, args.days_back)
            else:
                dataset_path = build_dataset(args.data_file)
            print(f"[SUCCESS] Professional dataset created successfully: {dataset_path}")
        except Exception as e:
            print(f"❌ Error building dataset: {e}")
            if args.mode != 'all':
                return
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n🧠 Training model...")
        print("-" * 40)
        try:
            if not os.path.exists(args.data_file):
                print(f"❌ Dataset not found: {args.data_file}")
                print("Please run with --mode data first")
                return
            
            # Handle progressive training and enhanced logging
            enable_logging = args.enable_logging
            experiment_name = args.experiment_name
            balance_classes = args.balance_classes
            
            if args.progressive:
                print("[PROGRESSIVE] Using progressive training mode")
                if enable_logging:
                    print("[LOGGING] Enhanced logging enabled")
                if experiment_name:
                    print(f"[EXPERIMENT] Experiment name: {experiment_name}")
                if balance_classes:
                    print("[BALANCE] Class balancing enabled")
                
                # Use progressive trainer
                from src.training.progressive_trainer import run_progressive_training
                print("[INFO] Progressive training function called without parameters")
                results = run_progressive_training()
                best_accuracy = results.get('stage2', {}).get('best_val_acc', 0.0)
            else:
                # Use standard trainer
                trainer = Trainer(csv_file=args.data_file, model_save_path=args.model_file)
                best_accuracy = trainer.train(epochs=args.epochs)
            
            print(f"[SUCCESS] Training completed! Best accuracy: {best_accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            if args.mode != 'all':
                return
    
    if args.mode == 'predict':
        print(f"\n🔮 Making prediction for {args.ticker}...")
        print("-" * 40)
        try:
            if not os.path.exists(args.model_file):
                print(f"❌ Model not found: {args.model_file}")
                print("Please run with --mode train first")
                return
            
            predictor = CandlestickPredictor(args.model_file)
            result = predictor.predict_single(args.ticker)
            
            if 'error' in result:
                print(f"❌ Error predicting {args.ticker}: {result['error']}")
            else:
                print(f"[PREDICT] Prediction for {args.ticker}:")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.2%}")
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            if args.mode != 'all':
                return
    
    if args.mode == 'signals':
        print(f"\n[SIGNAL] Generating trading signals for portfolio...")
        print("-" * 40)
        try:
            if not os.path.exists(args.model_file):
                print(f"❌ Model not found: {args.model_file}")
                print("Please run with --mode train first")
                return
            
            generator = TradingSignalGenerator(args.model_file)
            
            print(f"[CHART] Analyzing {len(args.portfolio)} stocks: {', '.join(args.portfolio)}")
            signals = generator.get_portfolio_signals(args.portfolio)
            
            # Print summary
            print(f"\n🏆 TOP TRADING OPPORTUNITIES:")
            for i, signal in enumerate(signals[:5], 1):
                if 'error' not in signal:
                    print(f"{i}. {signal['ticker']}: {signal['recommendation']} "
                          f"(Strength: {signal['signal_strength']:.1%})")
            
            # Detailed report for best signal
            if signals and 'error' not in signals[0]:
                generator.print_signal_report(signals[0])
                
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            if args.mode != 'all':
                return
    
    if args.mode == 'demo' or args.mode == 'all':
        print("\n[DEMO] Running prediction demo...")
        print("-" * 40)
        try:
            if not os.path.exists(args.model_file):
                print(f"❌ Model not found: {args.model_file}")
                print("Run the full pipeline first with --mode all")
                return
            
            demo_prediction()
            print("[SUCCESS] Demo completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during demo: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Process completed!")
    print("=" * 60)


def quick_start():
    """Quick start function for first-time users"""
    print("🚀 Quick Start - Building everything from scratch...")
    print("This will take several minutes to complete.\n")
    
    # Step 1: Build dataset with professional pipeline
    print("Step 1/3: Building professional dataset...")
    try:
        print("Using professional data pipeline with explicit raw/adjusted columns...")
        dataset_file = build_dataset()
        print("✅ Professional dataset built successfully!\n")
    except Exception as e:
        print(f"❌ Failed to build dataset: {e}")
        return
    
    # Step 2: Train model
    print("Step 2/3: Training model...")
    try:
        trainer = Trainer(csv_file=dataset_file)
        trainer.train(epochs=3)  # Quick training for demo
        print("✅ Model trained successfully!\n")
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return
    
    # Step 3: Demo predictions
    print("Step 3/3: Running demo predictions...")
    try:
        demo_prediction()
        print("✅ All done! Your price model is ready to use.")
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")


if __name__ == "__main__":
    # Check if user wants quick start
    if len(sys.argv) > 1 and sys.argv[1] == 'quickstart':
        quick_start()
    else:
        main() 