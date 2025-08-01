"""Ingest Google Trends data aligned with Kaggle Twitter dataset.

This script fetches Google Trends interest scores for stocks/companies and aligns
the data with the Kaggle Twitter dataset by date and ticker.
"""

import argparse
import os
import json
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import pytrends
from pytrends.request import TrendReq
import pytrends.exceptions as pt_ex

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UTILS_PATH = os.path.join(SRC_PATH, 'utils')
for p in (SRC_PATH, UTILS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.date_utils import resolve_date_range, add_window_cli


def with_retries(max_attempts=5, backoff_seconds=2.0, exceptions=(Exception,)):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = backoff_seconds
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
            return None
        return wrapper
    return decorator


class GoogleTrendsDataCollector:
    """Collect Google Trends data with retry logic and batch processing."""
    
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
    
    @with_retries(max_attempts=5, backoff_seconds=2.0, exceptions=(Exception,))
    def fetch_trends_batch(self, keywords, timeframe):
        """Fetch trends for a batch of keywords (max 5)."""
        if len(keywords) > 5:
            raise ValueError("Google Trends allows maximum 5 keywords per request")
        
        attempts = 0
        delay = 1
        while True:
            try:
                self.pytrends.build_payload(keywords, timeframe=timeframe)
                data = self.pytrends.interest_over_time()
                return data
            except (pt_ex.TooManyRequestsError, pt_ex.ResponseError) as e:
                attempts += 1
                if attempts >= 5:
                    raise e
                print(f"Rate limited, waiting {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    
    def normalize_trends_data(self, data):
        """Normalize trends data to 0-1 scale."""
        if data.empty:
            return data
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'isPartial':
                max_val = data[col].max()
                if max_val > 0:
                    data[col] = data[col] / max_val
        return data


def parse_args():
    parser = argparse.ArgumentParser(description='Ingest Google Trends data aligned with Kaggle dataset.')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of stock tickers')
    parser.add_argument('--companies', nargs='+', help='List of company names (optional)')
    parser.add_argument('--custom_keywords', nargs='+', help='Custom keywords for trends (optional)')
    add_window_cli(parser)
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for trends data')
    parser.add_argument('--kaggle_alignment', type=str, 
                       help='Path to processed Kaggle dataset for date/ticker alignment')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize trends data to 0-1 scale')
    return parser.parse_args()


def get_company_names(tickers):
    """Get company names for tickers."""
    ticker_to_company = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta',
        'NVDA': 'NVIDIA',
        'JPM': 'JPMorgan',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa',
        'PG': 'Procter & Gamble',
        'UNH': 'UnitedHealth',
        'HD': 'Home Depot',
        'MA': 'Mastercard',
        'BAC': 'Bank of America',
        'MP': 'MP Materials'
    }
    
    return [ticker_to_company.get(ticker, ticker) for ticker in tickers]


def generate_search_keywords(tickers, companies, custom_keywords=None):
    """Generate search keywords for Google Trends."""
    keywords_map = {}
    
    for ticker, company in zip(tickers, companies):
        # Generate multiple keyword variations for each ticker
        ticker_keywords = [
            ticker,
            f"{ticker} stock",
            company,
            f"{company} stock",
            f"{company} share price"
        ]
        
        # Add custom keywords if provided
        if custom_keywords:
            ticker_keywords.extend([kw for kw in custom_keywords if ticker.lower() in kw.lower()])
        
        keywords_map[ticker] = ticker_keywords
    
    return keywords_map


def collect_trends_data(collector, keywords_map, start_date, end_date):
    """Collect trends data for all keywords."""
    timeframe = f'{start_date} {end_date}'
    all_trends_data = {}
    
    for ticker, keywords in keywords_map.items():
        print(f"Collecting trends data for {ticker}")
        
        # Process in batches of 5 (Google Trends limit)
        ticker_data = []
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            print(f"  Processing batch: {batch}")
            
            try:
                batch_data = collector.fetch_trends_batch(batch, timeframe)
                if batch_data is not None and not batch_data.empty:
                    # Remove 'isPartial' column if present
                    if 'isPartial' in batch_data.columns:
                        batch_data = batch_data.drop(columns=['isPartial'])
                    ticker_data.append(batch_data)
                    
                    # Add delay between requests to avoid rate limiting
                    time.sleep(1)
                    
            except Exception as e:
                print(f"  Failed to fetch batch {batch}: {e}")
                continue
        
        if ticker_data:
            # Combine all batches for this ticker
            combined_data = pd.concat(ticker_data, axis=1)
            # Take the average across all keyword variations for each date
            combined_data[f'{ticker}_trends_score'] = combined_data.mean(axis=1)
            all_trends_data[ticker] = combined_data[[f'{ticker}_trends_score']]
        else:
            print(f"  No data collected for {ticker}")
    
    return all_trends_data


def align_with_kaggle_dataset(trends_data, kaggle_path=None):
    """Align trends data with Kaggle dataset dates and tickers."""
    if not kaggle_path or not os.path.exists(kaggle_path):
        print("[INFO] No Kaggle alignment file provided")
        return trends_data
    
    try:
        # Load Kaggle dataset to get available dates and tickers
        kaggle_df = pd.read_csv(kaggle_path)
        
        # Get unique date-ticker combinations from Kaggle dataset
        kaggle_combinations = set()
        if 'date' in kaggle_df.columns and 'ticker' in kaggle_df.columns:
            kaggle_df['date'] = pd.to_datetime(kaggle_df['date']).dt.strftime('%Y-%m-%d')
            for _, row in kaggle_df.iterrows():
                kaggle_combinations.add((row['date'], row['ticker'].upper()))
        
        # Filter trends data to only include dates/tickers present in Kaggle dataset
        aligned_trends = {}
        total_original = 0
        total_aligned = 0
        
        for ticker, ticker_trends in trends_data.items():
            total_original += len(ticker_trends)
            
            # Convert index to date strings for comparison
            ticker_trends_copy = ticker_trends.copy()
            ticker_trends_copy.index = ticker_trends_copy.index.strftime('%Y-%m-%d')
            
            # Filter to only dates present in Kaggle dataset for this ticker
            aligned_dates = []
            for date_str in ticker_trends_copy.index:
                if (date_str, ticker.upper()) in kaggle_combinations:
                    aligned_dates.append(date_str)
            
            if aligned_dates:
                aligned_ticker_data = ticker_trends_copy.loc[aligned_dates]
                aligned_trends[ticker] = aligned_ticker_data
                total_aligned += len(aligned_ticker_data)
            else:
                print(f"[WARN] No aligned dates found for {ticker}")
        
        print(f"[INFO] Aligned trends data: {total_aligned} records "
              f"(from {total_original} original records)")
        return aligned_trends
        
    except Exception as e:
        print(f"[WARN] Failed to align with Kaggle dataset: {e}")
        return trends_data


def save_trends_data(trends_data, out_dir, start_date, end_date, normalize=True):
    """Save trends data in multiple formats."""
    os.makedirs(out_dir, exist_ok=True)
    
    collector = GoogleTrendsDataCollector()
    
    # Save individual ticker files
    for ticker, data in trends_data.items():
        if normalize:
            data = collector.normalize_trends_data(data)
        
        # Save as JSONL
        jsonl_path = os.path.join(out_dir, f'{ticker}_trends_{start_date}_{end_date}.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for date_idx, row in data.iterrows():
                record = {
                    'ticker': ticker,
                    'date': date_idx if isinstance(date_idx, str) else date_idx.strftime('%Y-%m-%d'),
                    'trends_score': float(row.iloc[0]) if not pd.isna(row.iloc[0]) else 0.0
                }
                json.dump(record, f)
                f.write('\n')
        
        # Save as CSV
        csv_path = os.path.join(out_dir, f'{ticker}_trends_{start_date}_{end_date}.csv')
        csv_data = data.copy()
        csv_data['ticker'] = ticker
        csv_data['date'] = csv_data.index if isinstance(csv_data.index[0], str) else csv_data.index.strftime('%Y-%m-%d')
        csv_data = csv_data.reset_index(drop=True)
        csv_data.to_csv(csv_path, index=False)
        
        print(f'Saved trends data for {ticker}:')
        print(f'  JSONL: {jsonl_path}')
        print(f'  CSV: {csv_path}')
    
    # Save combined summary
    summary_path = os.path.join(out_dir, f'trends_summary_{start_date}_{end_date}.json')
    summary = {
        'date_range': f'{start_date} to {end_date}',
        'tickers': list(trends_data.keys()),
        'records_per_ticker': {ticker: len(data) for ticker, data in trends_data.items()},
        'total_records': sum(len(data) for data in trends_data.values())
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'Summary saved to: {summary_path}')


def main():
    args = parse_args()
    
    # Resolve date range
    start_date, end_date, window_used = resolve_date_range(
        args.start_date, args.end_date, args.window
    )
    print(f'Fetching Google Trends data from {start_date} to {end_date} (window={window_used})')
    
    # Get company names
    company_names = args.companies or get_company_names(args.tickers)
    
    # Generate search keywords
    keywords_map = generate_search_keywords(args.tickers, company_names, args.custom_keywords)
    
    print("Search keywords generated:")
    for ticker, keywords in keywords_map.items():
        print(f"  {ticker}: {keywords}")
    
    # Initialize trends collector
    collector = GoogleTrendsDataCollector()
    
    # Collect trends data
    trends_data = collect_trends_data(collector, keywords_map, start_date, end_date)
    
    if not trends_data:
        print("[ERROR] No trends data collected")
        return
    
    # Align with Kaggle dataset if provided
    if args.kaggle_alignment:
        trends_data = align_with_kaggle_dataset(trends_data, args.kaggle_alignment)
    
    if not trends_data:
        print("[WARN] No trends data remaining after alignment")
        return
    
    # Save the data
    save_trends_data(trends_data, args.out_dir, start_date, end_date, args.normalize)


if __name__ == '__main__':
    main() 