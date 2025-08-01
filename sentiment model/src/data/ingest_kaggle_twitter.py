"""Load and process the Kaggle Twitter sentiment dataset.

This script downloads and processes the Kaggle dataset:
https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns

The dataset serves as the core data foundation that NewsAPI and Google Trends data will be aligned with.
"""

import argparse
import os
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.date_utils import resolve_date_range, add_window_cli


def parse_args():
    parser = argparse.ArgumentParser(description='Load and process Kaggle Twitter sentiment dataset.')
    parser.add_argument('--kaggle_path', type=str, required=True, 
                       help='Path to the downloaded Kaggle dataset CSV file')
    parser.add_argument('--tickers', nargs='+', 
                       help='Filter by specific tickers (optional)')
    add_window_cli(parser)
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for processed data')
    return parser.parse_args()


def load_kaggle_dataset(kaggle_path: str) -> pd.DataFrame:
    """Load the Kaggle Twitter sentiment dataset."""
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(f"Kaggle dataset not found at {kaggle_path}")
    
    print(f"Loading Kaggle dataset from {kaggle_path}")
    df = pd.read_csv(kaggle_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def standardize_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the Kaggle dataset to match our expected format."""
    
    # Create a copy to avoid modifying original
    df_std = df.copy()
    
    # Handle the specific structure of this Kaggle dataset
    # Columns: ['Unnamed: 0', 'TWEET', 'STOCK', 'DATE', 'LAST_PRICE', '1_DAY_RETURN', ...]
    column_mapping = {
        'STOCK': 'ticker',
        'DATE': 'date', 
        'TWEET': 'text',
        'LSTM_POLARITY': 'lstm_sentiment',
        'TEXTBLOB_POLARITY': 'textblob_sentiment',
        'LAST_PRICE': 'price',
        '1_DAY_RETURN': 'return_1d',
        '2_DAY_RETURN': 'return_2d', 
        '3_DAY_RETURN': 'return_3d',
        '7_DAY_RETURN': 'return_7d'
    }
    
    # Apply the mapping for columns that exist
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df_std.columns}
    df_std = df_std.rename(columns=existing_mapping)
    
    # Clean up the data
    print("Cleaning and standardizing data...")
    
    # Handle the DATE column - filter out non-date values
    if 'date' in df_std.columns:
        print(f"Original dataset size: {len(df_std)} records")
        
        # Convert DATE column to string first to handle mixed types
        df_std['date'] = df_std['date'].astype(str)
        
        # Filter out rows where date looks like a number (like "823.48")
        # Keep only rows where date contains date-like patterns (including various formats)
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}/\d{1,2}/\d{1,2}'
        valid_date_mask = df_std['date'].str.contains(date_pattern, na=False)
        
        # Also filter out obvious non-dates (pure numbers, NaN, etc.)
        df_std['date'] = df_std['date'].astype(str)
        numeric_mask = df_std['date'].str.match(r'^\d+\.?\d*$')  # Pure numbers like "823.48"
        valid_date_mask = valid_date_mask & ~numeric_mask
        
        df_std = df_std[valid_date_mask].copy()
        print(f"After filtering invalid dates: {len(df_std)} records")
        
        # Try to parse the dates
        try:
            df_std['date'] = pd.to_datetime(df_std['date'], errors='coerce')
            # Remove rows where date parsing failed
            df_std = df_std.dropna(subset=['date'])
            df_std['date'] = df_std['date'].dt.strftime('%Y-%m-%d')
            print(f"After date parsing: {len(df_std)} records")
        except Exception as e:
            print(f"Date parsing error: {e}")
            # If parsing fails completely, try alternative approach
            df_std = df_std.dropna(subset=['date'])
    
    # Ensure ticker is string and uppercase
    if 'ticker' in df_std.columns:
        df_std['ticker'] = df_std['ticker'].astype(str).str.upper()
        # Remove any rows with invalid tickers
        df_std = df_std[df_std['ticker'].str.len() <= 10]  # Reasonable ticker length
        df_std = df_std[df_std['ticker'] != 'NAN']
    
    # Ensure text column exists and is clean
    if 'text' in df_std.columns:
        df_std['text'] = df_std['text'].astype(str)
        df_std = df_std[df_std['text'].str.len() > 0]  # Remove empty text
        df_std = df_std[df_std['text'] != 'nan']
    
    # Add a combined sentiment score (average of available sentiment scores)
    sentiment_cols = [col for col in df_std.columns if 'sentiment' in col]
    if sentiment_cols:
        # Convert sentiment columns to numeric, handling errors
        for col in sentiment_cols:
            df_std[col] = pd.to_numeric(df_std[col], errors='coerce')
        
        # Create combined sentiment (average of non-null values)
        df_std['sentiment'] = df_std[sentiment_cols].mean(axis=1, skipna=True)
    
    print(f"Final standardized dataset: {len(df_std)} records")
    print(f"Date range: {df_std['date'].min()} to {df_std['date'].max()}")
    print(f"Unique tickers: {df_std['ticker'].nunique()}")
    
    return df_std


def filter_data(df: pd.DataFrame, tickers=None, start_date=None, end_date=None) -> pd.DataFrame:
    """Filter the dataset by tickers and date range."""
    
    df_filtered = df.copy()
    
    # Filter by tickers if specified
    if tickers and 'ticker' in df_filtered.columns:
        tickers_upper = [t.upper() for t in tickers]
        df_filtered = df_filtered[df_filtered['ticker'].isin(tickers_upper)]
        print(f"Filtered to tickers: {tickers_upper}")
    
    # Filter by date range if specified
    if start_date and end_date and 'date' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['date'] >= start_date) & 
            (df_filtered['date'] <= end_date)
        ]
        print(f"Filtered to date range: {start_date} to {end_date}")
    
    return df_filtered


def save_processed_data(df: pd.DataFrame, out_dir: str):
    """Save the processed Kaggle dataset."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as CSV and JSONL
    csv_path = os.path.join(out_dir, 'kaggle_twitter_processed.csv')
    jsonl_path = os.path.join(out_dir, 'kaggle_twitter_processed.jsonl')
    
    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient='records', lines=True)
    
    print(f"Saved processed dataset to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    
    # Save summary statistics
    summary_path = os.path.join(out_dir, 'kaggle_dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Kaggle Twitter Dataset Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
        
        if 'ticker' in df.columns:
            f.write(f"Unique tickers: {df['ticker'].nunique()}\n")
            f.write(f"Tickers: {sorted(df['ticker'].unique())}\n")
        
        f.write(f"Columns: {list(df.columns)}\n")
    
    print(f"Dataset summary saved to: {summary_path}")


def main():
    args = parse_args()
    
    # Resolve date range if provided
    start_date, end_date, window_used = resolve_date_range(
        args.start_date, args.end_date, args.window
    )
    
    if start_date and end_date:
        print(f'Processing data from {start_date} to {end_date} (window={window_used})')
    
    # Load the Kaggle dataset
    df = load_kaggle_dataset(args.kaggle_path)
    
    # Standardize the dataset format
    df_std = standardize_kaggle_data(df)
    
    # Filter the data
    df_filtered = filter_data(df_std, args.tickers, start_date, end_date)
    
    print(f"Final dataset shape: {df_filtered.shape}")
    
    # Save the processed data
    save_processed_data(df_filtered, args.out_dir)


if __name__ == '__main__':
    main() 