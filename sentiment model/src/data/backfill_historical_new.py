"""Updated orchestrator for the new sentiment data pipeline.

This script orchestrates data ingestion using:
- Kaggle Twitter dataset as the core (no new Twitter scraping)
- NewsAPI with FinancialBERT sentiment analysis
- Google Trends aligned with Kaggle dataset
- Data alignment and feature fusion

No longer includes Reddit or Twitter scraping.
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

WINDOW_DAYS = 7  # per-chunk window for APIs

# Updated ingestion commands for the new pipeline
INGEST_CMDS = {
    # Process Kaggle dataset (one-time or when updated)
    "kaggle": "python -m src.data.ingest_kaggle_twitter --kaggle_path {kaggle_path} --tickers {tickers} --out_dir data/processed/kaggle --start_date {start} --end_date {end}",
    
    # News ingestion with FinancialBERT
    "news": "python -m src.data.ingest_news_financialbert --tickers {tickers} --out_dir data/raw/news --start_date {start} --end_date {end} --kaggle_alignment data/processed/kaggle/kaggle_twitter_processed.csv",
    
    # Google Trends aligned with Kaggle dataset
    "trends": "python -m src.data.ingest_trends_aligned --tickers {tickers} --out_dir data/raw/trends --start_date {start} --end_date {end} --kaggle_alignment data/processed/kaggle/kaggle_twitter_processed.csv",
    
    # Market data (unchanged)
    "market": "python -m src.data.ingest_market --tickers {tickers} --out_dir data/raw/market --start_date {start} --end_date {end}",
    
    # Align all datasets
    "align": "python -m src.data.align_datasets --kaggle_data data/processed/kaggle/kaggle_twitter_processed.csv --news_data_dir data/raw/news --trends_data_dir data/raw/trends --out_dir data/processed/aligned --start_date {start} --end_date {end} --tickers {tickers}",
}


def parse_args():
    p = argparse.ArgumentParser(description="Updated pipeline orchestrator using Kaggle Twitter dataset with NewsAPI and Google Trends features.")
    p.add_argument("--since", type=str, default="2023-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--until", type=str, default=date.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "TSLA", "MP"], 
                   help="Stock tickers to process")
    p.add_argument("--kaggle_path", type=str, required=True,
                   help="Path to the downloaded Kaggle Twitter sentiment dataset CSV")
    p.add_argument("--include", nargs="+", choices=list(INGEST_CMDS.keys()), 
                   default=list(INGEST_CMDS.keys()), help="Sources to run")
    p.add_argument("--dry_run", action="store_true", help="Show commands without executing")
    p.add_argument("--continue_on_error", action="store_true", 
                   help="Continue processing other sources if one fails")
    return p.parse_args()


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
            print("[CONTINUE] Continuing with next command due to --continue_on_error flag")
            return False
        else:
            print("[ABORT] Stopping execution. Use --continue_on_error to continue on failures.")
            sys.exit(1)


def create_directories():
    """Create necessary output directories."""
    directories = [
        "data/raw/news",
        "data/raw/trends", 
        "data/raw/market",
        "data/processed/kaggle",
        "data/processed/aligned"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def validate_kaggle_dataset(kaggle_path):
    """Validate that the Kaggle dataset exists and has expected structure."""
    if not os.path.exists(kaggle_path):
        print(f"[ERROR] Kaggle dataset not found at: {kaggle_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns")
        sys.exit(1)
    
    try:
        import pandas as pd
        df = pd.read_csv(kaggle_path)
        print(f"[INFO] Kaggle dataset validation:")
        print(f"  Path: {kaggle_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['date', 'ticker']  # Adjust based on actual Kaggle dataset structure
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARN] Expected columns not found: {missing_cols}")
            print("The alignment process will attempt to auto-detect column names")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to validate Kaggle dataset: {e}")
        return False


def get_date_chunks(start_date, end_date, window_days):
    """Generate date chunks for processing."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    chunks = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=window_days), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        current = chunk_end
    
    return chunks


def main():
    args = parse_args()
    
    print("Updated Financial Sentiment Pipeline Orchestrator")
    print("=" * 60)
    print(f"Processing tickers: {args.tickers}")
    print(f"Date range: {args.since} to {args.until}")
    print(f"Kaggle dataset: {args.kaggle_path}")
    print(f"Sources to run: {args.include}")
    print(f"Dry run: {args.dry_run}")
    
    # Validate inputs
    if not validate_kaggle_dataset(args.kaggle_path):
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Format tickers for command line
    tickers_str = " ".join(args.tickers)
    
    # Process in chunks for API rate limits
    date_chunks = get_date_chunks(args.since, args.until, WINDOW_DAYS)
    print(f"\nProcessing {len(date_chunks)} date chunks:")
    for i, (start, end) in enumerate(date_chunks):
        print(f"  Chunk {i+1}: {start} to {end}")
    
    successful_runs = []
    failed_runs = []
    
    # Process each chunk
    for chunk_idx, (start, end) in enumerate(date_chunks):
        print(f"\n\n{'#'*100}")
        print(f"PROCESSING CHUNK {chunk_idx + 1}/{len(date_chunks)}: {start} to {end}")
        print(f"{'#'*100}")
        
        # Kaggle processing (only once, not per chunk)
        if chunk_idx == 0 and "kaggle" in args.include:
            cmd = INGEST_CMDS["kaggle"].format(
                kaggle_path=args.kaggle_path,
                tickers=tickers_str,
                start=args.since,  # Use full date range for Kaggle processing
                end=args.until
            )
            
            success = run_command(cmd, args.dry_run, args.continue_on_error)
            if success:
                successful_runs.append(f"kaggle_{start}_{end}")
            else:
                failed_runs.append(f"kaggle_{start}_{end}")
        
        # Process other sources for this chunk
        for source in args.include:
            if source == "kaggle":
                continue  # Already processed
            
            if source == "align":
                # Alignment runs after all data collection
                continue
            
            cmd = INGEST_CMDS[source].format(
                tickers=tickers_str,
                start=start,
                end=end
            )
            
            success = run_command(cmd, args.dry_run, args.continue_on_error)
            if success:
                successful_runs.append(f"{source}_{start}_{end}")
            else:
                failed_runs.append(f"{source}_{start}_{end}")
    
    # Run alignment after all data collection
    if "align" in args.include:
        print(f"\n\n{'#'*100}")
        print("ALIGNING ALL DATASETS")
        print(f"{'#'*100}")
        
        cmd = INGEST_CMDS["align"].format(
            start=args.since,
            end=args.until,
            tickers=tickers_str
        )
        
        success = run_command(cmd, args.dry_run, args.continue_on_error)
        if success:
            successful_runs.append("align_final")
        else:
            failed_runs.append("align_final")
    
    # Summary
    print(f"\n\n{'='*100}")
    print("EXECUTION SUMMARY")
    print(f"{'='*100}")
    print(f"Total successful runs: {len(successful_runs)}")
    print(f"Total failed runs: {len(failed_runs)}")
    
    if successful_runs:
        print("\nSuccessful runs:")
        for run in successful_runs:
            print(f"  ✓ {run}")
    
    if failed_runs:
        print("\nFailed runs:")
        for run in failed_runs:
            print(f"  ✗ {run}")
    
    if failed_runs and not args.continue_on_error:
        print("\n[RECOMMENDATION] Re-run with --continue_on_error to skip failed components")
    
    # Final dataset location
    if "align" in successful_runs or "align_final" in successful_runs:
        print(f"\n[SUCCESS] Final aligned dataset available in: data/processed/aligned/")
        print("This dataset combines:")
        print("  • Kaggle Twitter sentiment data (core)")
        print("  • NewsAPI headlines with FinancialBERT sentiment scores")
        print("  • Google Trends interest scores")
        print("  • All aligned by date and ticker")


if __name__ == "__main__":
    main() 