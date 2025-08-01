"""Orchestrator for combined sentiment pipeline.

This script orchestrates:
1. Process Kaggle Twitter dataset (historical 2017-2018 data)
2. Fetch current NewsAPI headlines 
3. Fetch current Google Trends data
4. Combine all datasets preserving Kaggle structure
5. Apply FinancialBERT for consistent sentiment analysis

Usage:
python src/data/orchestrate_combined_pipeline.py \
    --kaggle_path data/external/kaggle_twitter_sentiment.csv \
    --tickers AAPL MSFT GOOGL TSLA \
    --news_since 2024-01-01 \
    --news_until 2024-12-31
"""

import argparse
import subprocess
import sys
import os
from datetime import date, timedelta, datetime
from pathlib import Path

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Combined sentiment pipeline orchestrator")
    parser.add_argument("--kaggle_path", type=str, required=True,
                       help="Path to Kaggle Twitter sentiment dataset")
    parser.add_argument("--tickers", nargs="+", required=True,
                       help="Stock tickers to process")
    parser.add_argument("--news_since", type=str, default="2024-01-01",
                       help="Start date for NewsAPI data (YYYY-MM-DD)")
    parser.add_argument("--news_until", type=str, default=date.today().strftime("%Y-%m-%d"),
                       help="End date for NewsAPI data (YYYY-MM-DD)")
    parser.add_argument("--trends_since", type=str, default="2024-01-01", 
                       help="Start date for Google Trends data (YYYY-MM-DD)")
    parser.add_argument("--trends_until", type=str, default=date.today().strftime("%Y-%m-%d"),
                       help="End date for Google Trends data (YYYY-MM-DD)")
    parser.add_argument("--skip_kaggle", action="store_true",
                       help="Skip Kaggle processing (if already done)")
    parser.add_argument("--skip_news", action="store_true",
                       help="Skip news collection")
    parser.add_argument("--skip_trends", action="store_true", 
                       help="Skip trends collection")
    parser.add_argument("--skip_financialbert", action="store_true",
                       help="Skip FinancialBERT analysis")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show commands without executing")
    parser.add_argument("--continue_on_error", action="store_true",
                       help="Continue if one step fails")
    return parser.parse_args()


def run_command(cmd, dry_run=False, continue_on_error=False):
    """Run a command with error handling."""
    print(f"\n{'='*80}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*80}")
    
    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("[SUCCESS] Command completed successfully")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        
        if continue_on_error:
            print("[CONTINUE] Continuing due to --continue_on_error flag")
            return False
        else:
            print("[ABORT] Stopping execution")
            sys.exit(1)


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/processed/kaggle",
        "data/raw/news", 
        "data/raw/trends",
        "data/processed/combined"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    args = parse_args()
    
    print("Combined Sentiment Pipeline Orchestrator")
    print("=" * 60)
    print(f"Kaggle dataset: {args.kaggle_path}")
    print(f"Tickers: {args.tickers}")
    print(f"News date range: {args.news_since} to {args.news_until}")
    print(f"Trends date range: {args.trends_since} to {args.trends_until}")
    print(f"Dry run: {args.dry_run}")
    
    # Create directories
    create_directories()
    
    tickers_str = " ".join(args.tickers)
    successful_steps = []
    failed_steps = []
    
    # Step 1: Process Kaggle dataset
    if not args.skip_kaggle:
        print(f"\n{'#'*100}")
        print("STEP 1: PROCESSING KAGGLE TWITTER DATASET")
        print(f"{'#'*100}")
        
        cmd = f"python src/data/ingest_kaggle_twitter.py --kaggle_path {args.kaggle_path} --tickers {tickers_str} --out_dir data/processed/kaggle"
        
        success = run_command(cmd, args.dry_run, args.continue_on_error)
        if success:
            successful_steps.append("kaggle_processing")
        else:
            failed_steps.append("kaggle_processing")
    else:
        print("\n[SKIP] Skipping Kaggle processing")
    
    # Step 2: Collect News data (current/recent)
    if not args.skip_news:
        print(f"\n{'#'*100}")
        print("STEP 2: COLLECTING CURRENT NEWS DATA")
        print(f"{'#'*100}")
        
        # Use the original news ingestion (without FinancialBERT) for raw collection
        cmd = f"python src/data/ingest_news.py --sources newsapi --tickers {tickers_str} --start_date {args.news_since} --end_date {args.news_until} --out_dir data/raw/news"
        
        success = run_command(cmd, args.dry_run, args.continue_on_error)
        if success:
            successful_steps.append("news_collection")
        else:
            failed_steps.append("news_collection")
    else:
        print("\n[SKIP] Skipping news collection")
    
    # Step 3: Collect Google Trends data (current/recent)  
    if not args.skip_trends:
        print(f"\n{'#'*100}")
        print("STEP 3: COLLECTING CURRENT GOOGLE TRENDS DATA")
        print(f"{'#'*100}")
        
        # Use the original trends ingestion
        keywords = ' '.join([f'"{t} stock"' for t in args.tickers])
        cmd = f"python src/data/ingest_trends.py --keywords {keywords} --start_date {args.trends_since} --end_date {args.trends_until} --out_dir data/raw/trends"
        
        success = run_command(cmd, args.dry_run, args.continue_on_error)
        if success:
            successful_steps.append("trends_collection")
        else:
            failed_steps.append("trends_collection")
    else:
        print("\n[SKIP] Skipping trends collection")
    
    # Step 4: Combine all datasets and apply FinancialBERT
    if not args.skip_financialbert:
        print(f"\n{'#'*100}")
        print("STEP 4: COMBINING DATASETS & APPLYING FINANCIALBERT")
        print(f"{'#'*100}")
        
        cmd = f"python src/data/build_combined_dataset.py --kaggle_data data/processed/kaggle/kaggle_twitter_processed.csv --news_data_dir data/raw/news --trends_data_dir data/raw/trends --out_dir data/processed/combined --tickers {tickers_str} --apply_financialbert"
        
        success = run_command(cmd, args.dry_run, args.continue_on_error)
        if success:
            successful_steps.append("combination_and_financialbert")
        else:
            failed_steps.append("combination_and_financialbert")
    else:
        print("\n[SKIP] Skipping dataset combination and FinancialBERT")
    
    # Summary
    print(f"\n{'='*100}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*100}")
    print(f"✅ Successful steps: {len(successful_steps)}")
    print(f"❌ Failed steps: {len(failed_steps)}")
    
    if successful_steps:
        print("\n✅ Successful steps:")
        for step in successful_steps:
            print(f"  • {step}")
    
    if failed_steps:
        print("\n❌ Failed steps:")
        for step in failed_steps:
            print(f"  • {step}")
    
    # Final output information
    if "combination_and_financialbert" in successful_steps:
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"📊 Final combined dataset with FinancialBERT sentiment analysis:")
        print(f"   📁 Location: data/processed/combined/")
        print(f"   📋 Contains:")
        print(f"      • Historical Kaggle Twitter sentiment (2017-2018)")
        print(f"      • Current NewsAPI headlines ({args.news_since} to {args.news_until})")
        print(f"      • Current Google Trends data ({args.trends_since} to {args.trends_until})")
        print(f"      • FinancialBERT sentiment scores for all text data")
        print(f"      • Unified structure preserving Kaggle dataset schema")
    
    elif successful_steps:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"Some steps completed successfully. You can:")
        print(f"1. Check the outputs in their respective directories")
        print(f"2. Re-run with --skip_* flags for completed steps")
        print(f"3. Run the final combination step manually if needed")
    
    else:
        print(f"\n❌ PIPELINE FAILED")
        print(f"No steps completed successfully. Check the errors above.")


if __name__ == "__main__":
    main() 