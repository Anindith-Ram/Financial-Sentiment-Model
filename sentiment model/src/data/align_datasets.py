"""Align and merge Kaggle Twitter dataset with NewsAPI and Google Trends data.

This script takes the Kaggle Twitter sentiment dataset as the core and appends
supporting features from NewsAPI sentiment scores and Google Trends interest scores,
maintaining the original column structure while adding new feature columns.
"""

import argparse
import os
import pandas as pd
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.date_utils import resolve_date_range, add_window_cli


def parse_args():
    parser = argparse.ArgumentParser(description='Align Kaggle Twitter dataset with NewsAPI and Google Trends data.')
    parser.add_argument('--kaggle_data', type=str, required=True,
                       help='Path to processed Kaggle Twitter dataset (CSV)')
    parser.add_argument('--news_data_dir', type=str, 
                       help='Directory containing NewsAPI sentiment data files')
    parser.add_argument('--trends_data_dir', type=str,
                       help='Directory containing Google Trends data files')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for aligned dataset')
    parser.add_argument('--aggregation_method', choices=['mean', 'sum', 'max', 'last'], default='mean',
                       help='Method to aggregate multiple news/trends entries per day')
    parser.add_argument('--fill_missing', choices=['zero', 'forward_fill', 'backward_fill', 'interpolate'], default='zero',
                       help='Method to handle missing values in supporting data')
    parser.add_argument('--tickers', nargs='+', help='Filter to specific tickers (optional)')
    add_window_cli(parser)
    return parser.parse_args()


class DatasetAligner:
    """Class to handle alignment and merging of different datasets."""
    
    def __init__(self, aggregation_method='mean', fill_missing='zero'):
        self.aggregation_method = aggregation_method
        self.fill_missing = fill_missing
    
    def load_kaggle_dataset(self, kaggle_path: str) -> pd.DataFrame:
        """Load and prepare the Kaggle Twitter dataset."""
        print(f"Loading Kaggle dataset from {kaggle_path}")
        
        if not os.path.exists(kaggle_path):
            raise FileNotFoundError(f"Kaggle dataset not found at {kaggle_path}")
        
        df = pd.read_csv(kaggle_path)
        
        # Ensure required columns exist
        required_cols = ['date', 'ticker']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in Kaggle dataset: {missing_cols}")
        
        # Standardize date format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['ticker'] = df['ticker'].str.upper()
        
        print(f"Kaggle dataset loaded: {len(df)} records, {df['ticker'].nunique()} unique tickers")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def load_news_data(self, news_data_dir: str, tickers: List[str]) -> pd.DataFrame:
        """Load and aggregate NewsAPI sentiment data."""
        if not news_data_dir or not os.path.exists(news_data_dir):
            print("[INFO] No news data directory provided or directory not found")
            return pd.DataFrame()
        
        print(f"Loading news data from {news_data_dir}")
        
        all_news_data = []
        
        # Look for news files for each ticker
        for ticker in tickers:
            ticker_files = [f for f in os.listdir(news_data_dir) 
                          if f.startswith(f"{ticker}_news_sentiment_") and f.endswith('.csv')]
            
            for file in ticker_files:
                file_path = os.path.join(news_data_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns and 'ticker' in df.columns:
                        all_news_data.append(df)
                except Exception as e:
                    print(f"[WARN] Failed to load {file}: {e}")
        
        if not all_news_data:
            print("[INFO] No news data found")
            return pd.DataFrame()
        
        # Combine all news data
        news_df = pd.concat(all_news_data, ignore_index=True)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')
        news_df['ticker'] = news_df['ticker'].str.upper()
        
        # Aggregate by date and ticker
        news_agg = self._aggregate_news_data(news_df)
        
        print(f"News data loaded: {len(news_agg)} date-ticker combinations")
        return news_agg
    
    def load_trends_data(self, trends_data_dir: str, tickers: List[str]) -> pd.DataFrame:
        """Load and aggregate Google Trends data."""
        if not trends_data_dir or not os.path.exists(trends_data_dir):
            print("[INFO] No trends data directory provided or directory not found")
            return pd.DataFrame()
        
        print(f"Loading trends data from {trends_data_dir}")
        
        all_trends_data = []
        
        # Look for trends files for each ticker
        for ticker in tickers:
            ticker_files = [f for f in os.listdir(trends_data_dir) 
                          if f.startswith(f"{ticker}_trends_") and f.endswith('.csv')]
            
            for file in ticker_files:
                file_path = os.path.join(trends_data_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns and 'ticker' in df.columns:
                        all_trends_data.append(df)
                except Exception as e:
                    print(f"[WARN] Failed to load {file}: {e}")
        
        if not all_trends_data:
            print("[INFO] No trends data found")
            return pd.DataFrame()
        
        # Combine all trends data
        trends_df = pd.concat(all_trends_data, ignore_index=True)
        trends_df['date'] = pd.to_datetime(trends_df['date']).dt.strftime('%Y-%m-%d')
        trends_df['ticker'] = trends_df['ticker'].str.upper()
        
        # Aggregate by date and ticker  
        trends_agg = self._aggregate_trends_data(trends_df)
        
        print(f"Trends data loaded: {len(trends_agg)} date-ticker combinations")
        return trends_agg
    
    def _aggregate_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news data by date and ticker."""
        if news_df.empty:
            return news_df
        
        # Define aggregation functions for each column
        agg_functions = {}
        
        # Sentiment scores - use specified aggregation method
        sentiment_cols = [col for col in news_df.columns if 'sentiment_score' in col]
        for col in sentiment_cols:
            agg_functions[col] = self.aggregation_method
        
        # Confidence scores - use mean
        confidence_cols = [col for col in news_df.columns if 'confidence' in col]
        for col in confidence_cols:
            agg_functions[col] = 'mean'
        
        # Count number of articles per day
        agg_functions['news_article_count'] = 'count'
        
        # Group by date and ticker
        news_df['news_article_count'] = 1  # For counting
        
        if agg_functions:
            news_agg = news_df.groupby(['date', 'ticker']).agg(agg_functions).reset_index()
            
            # Flatten column names if multi-level
            if isinstance(news_agg.columns, pd.MultiIndex):
                news_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                  for col in news_agg.columns.values]
                news_agg = news_agg.rename(columns=lambda x: x.replace('_' + self.aggregation_method, '').replace('_count', '_count'))
        else:
            # If no aggregation columns found, just count articles
            news_agg = news_df.groupby(['date', 'ticker']).size().reset_index(name='news_article_count')
        
        return news_agg
    
    def _aggregate_trends_data(self, trends_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trends data by date and ticker."""
        if trends_df.empty:
            return trends_df
        
        # For trends data, usually just one value per date-ticker combination
        # But if multiple, use the specified aggregation method
        agg_functions = {}
        
        trends_cols = [col for col in trends_df.columns if 'trends_score' in col or col == 'trends_score']
        for col in trends_cols:
            agg_functions[col] = self.aggregation_method
        
        if agg_functions:
            trends_agg = trends_df.groupby(['date', 'ticker']).agg(agg_functions).reset_index()
        else:
            trends_agg = trends_df.drop_duplicates(['date', 'ticker'])
        
        return trends_agg
    
    def align_datasets(self, kaggle_df: pd.DataFrame, news_df: pd.DataFrame, 
                      trends_df: pd.DataFrame) -> pd.DataFrame:
        """Align and merge all datasets using Kaggle as the base."""
        
        print("Aligning datasets...")
        
        # Start with Kaggle dataset as the base
        aligned_df = kaggle_df.copy()
        original_count = len(aligned_df)
        
        # Merge news data
        if not news_df.empty:
            print("Merging news data...")
            aligned_df = aligned_df.merge(
                news_df, 
                on=['date', 'ticker'], 
                how='left',
                suffixes=('', '_news')
            )
            
            # Fill missing news values
            news_cols = [col for col in aligned_df.columns if col.startswith('news_')]
            aligned_df = self._fill_missing_values(aligned_df, news_cols)
            
            print(f"News merge: {len(news_df)} news records merged")
        
        # Merge trends data
        if not trends_df.empty:
            print("Merging trends data...")
            aligned_df = aligned_df.merge(
                trends_df,
                on=['date', 'ticker'],
                how='left',
                suffixes=('', '_trends')
            )
            
            # Fill missing trends values
            trends_cols = [col for col in aligned_df.columns if col.startswith('trends_') or 'trends_score' in col]
            aligned_df = self._fill_missing_values(aligned_df, trends_cols)
            
            print(f"Trends merge: {len(trends_df)} trends records merged")
        
        print(f"Final aligned dataset: {len(aligned_df)} records (started with {original_count})")
        return aligned_df
    
    def _fill_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values in specified columns."""
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if self.fill_missing == 'zero':
                df[col] = df[col].fillna(0)
            elif self.fill_missing == 'forward_fill':
                df[col] = df.groupby('ticker')[col].fillna(method='ffill')
            elif self.fill_missing == 'backward_fill':
                df[col] = df.groupby('ticker')[col].fillna(method='bfill')
            elif self.fill_missing == 'interpolate':
                df[col] = df.groupby('ticker')[col].interpolate()
                df[col] = df[col].fillna(0)  # Fill any remaining NaNs
        
        return df


def save_aligned_dataset(df: pd.DataFrame, out_dir: str, start_date: str = None, end_date: str = None):
    """Save the aligned dataset in multiple formats."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate filename suffix
    if start_date and end_date:
        suffix = f"_{start_date}_{end_date}"
    else:
        suffix = f"_{datetime.now().strftime('%Y%m%d')}"
    
    # Save as CSV
    csv_path = os.path.join(out_dir, f'aligned_sentiment_dataset{suffix}.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as Parquet for efficiency
    parquet_path = os.path.join(out_dir, f'aligned_sentiment_dataset{suffix}.parquet')
    df.to_parquet(parquet_path, index=False)
    
    # Save as JSONL for compatibility
    jsonl_path = os.path.join(out_dir, f'aligned_sentiment_dataset{suffix}.jsonl')
    df.to_json(jsonl_path, orient='records', lines=True)
    
    # Generate summary statistics
    summary = {
        'total_records': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
        'unique_tickers': df['ticker'].nunique(),
        'tickers': sorted(df['ticker'].unique().tolist()),
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'records_per_ticker': df.groupby('ticker').size().to_dict()
    }
    
    summary_path = os.path.join(out_dir, f'dataset_summary{suffix}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Aligned dataset saved:")
    print(f"  CSV: {csv_path}")
    print(f"  Parquet: {parquet_path}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Summary: {summary_path}")
    
    # Print column summary
    print(f"\nDataset Summary:")
    print(f"  Records: {len(df)}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Columns: {len(df.columns)}")
    
    # Show feature columns
    kaggle_cols = [col for col in df.columns if not (col.startswith('news_') or col.startswith('trends_'))]
    news_cols = [col for col in df.columns if col.startswith('news_')]
    trends_cols = [col for col in df.columns if col.startswith('trends_') or 'trends_score' in col]
    
    print(f"  Original columns: {len(kaggle_cols)}")
    print(f"  News features: {len(news_cols)}")
    print(f"  Trends features: {len(trends_cols)}")


def main():
    args = parse_args()
    
    # Resolve date range if provided
    start_date, end_date, window_used = resolve_date_range(
        args.start_date, args.end_date, args.window
    )
    
    if start_date and end_date:
        print(f'Processing data from {start_date} to {end_date} (window={window_used})')
    
    # Initialize aligner
    aligner = DatasetAligner(args.aggregation_method, args.fill_missing)
    
    # Load Kaggle dataset
    kaggle_df = aligner.load_kaggle_dataset(args.kaggle_data)
    
    # Filter by date range if specified
    if start_date and end_date:
        kaggle_df = kaggle_df[
            (kaggle_df['date'] >= start_date) & 
            (kaggle_df['date'] <= end_date)
        ]
        print(f"Filtered Kaggle data to date range: {len(kaggle_df)} records")
    
    # Filter by tickers if specified
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        kaggle_df = kaggle_df[kaggle_df['ticker'].isin(tickers)]
        print(f"Filtered to tickers {tickers}: {len(kaggle_df)} records")
    else:
        tickers = kaggle_df['ticker'].unique().tolist()
    
    if kaggle_df.empty:
        print("[ERROR] No Kaggle data remaining after filtering")
        return
    
    # Load supporting datasets
    news_df = aligner.load_news_data(args.news_data_dir, tickers)
    trends_df = aligner.load_trends_data(args.trends_data_dir, tickers)
    
    # Align all datasets
    aligned_df = aligner.align_datasets(kaggle_df, news_df, trends_df)
    
    # Save the result
    save_aligned_dataset(aligned_df, args.out_dir, start_date, end_date)


if __name__ == '__main__':
    main() 