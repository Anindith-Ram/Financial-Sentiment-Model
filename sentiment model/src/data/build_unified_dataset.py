"""Build a unified sentiment dataset from the aligned pipeline output.

This script takes the aligned dataset (Kaggle Twitter + NewsAPI + Google Trends)
and prepares it for machine learning training, including feature engineering,
normalization, and target variable creation.

Usage example:
python src/data/build_unified_dataset.py \
    --input_path data/processed/aligned/aligned_sentiment_dataset_2023-01-01_2023-12-31.csv \
    --out_path data/processed/ml_ready_dataset.csv \
    --target_type binary_return \
    --feature_engineering
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import sys

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.date_utils import resolve_date_range


def parse_args():
    parser = argparse.ArgumentParser(description='Build unified ML-ready dataset from aligned sentiment data.')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to aligned sentiment dataset (CSV/Parquet)')
    parser.add_argument('--out_path', type=str, required=True,
                       help='Output path for ML-ready dataset')
    parser.add_argument('--target_type', choices=['binary_return', 'return_regression', 'sentiment_classification'], 
                       default='binary_return', help='Type of target variable to create')
    parser.add_argument('--feature_engineering', action='store_true',
                       help='Apply feature engineering (lags, moving averages, etc.)')
    parser.add_argument('--normalize_features', choices=['standard', 'robust', 'minmax', 'none'], 
                       default='standard', help='Feature normalization method')
    parser.add_argument('--future_days', type=int, default=1,
                       help='Number of days ahead for target variable')
    parser.add_argument('--min_samples_per_ticker', type=int, default=30,
                       help='Minimum samples required per ticker')
    parser.add_argument('--exclude_recent_days', type=int, default=5,
                       help='Exclude recent N days due to potential data quality issues')
    return parser.parse_args()


class UnifiedDatasetBuilder:
    """Build ML-ready dataset from aligned sentiment data."""
    
    def __init__(self, target_type='binary_return', future_days=1, feature_engineering=True, 
                 normalize_features='standard'):
        self.target_type = target_type
        self.future_days = future_days
        self.feature_engineering = feature_engineering
        self.normalize_features = normalize_features
        self.scaler = None
        
    def load_data(self, input_path: str) -> pd.DataFrame:
        """Load the aligned sentiment dataset."""
        print(f"Loading aligned dataset from: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Support multiple formats
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.jsonl'):
            df = pd.read_json(input_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        
        return df
    
    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the dataset."""
        print("Cleaning and preparing dataset...")
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['ticker', 'date'])
        if len(df) < initial_size:
            print(f"Removed {initial_size - len(df)} duplicate records")
        
        # Handle missing values in key columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables based on future price movements."""
        print(f"Creating target variables (type: {self.target_type}, future_days: {self.future_days})")
        
        # Get unique tickers for price data
        tickers = df['ticker'].unique()
        
        # Download price data
        price_data = {}
        for ticker in tqdm(tickers, desc="Downloading price data"):
            try:
                # Get a wider date range to ensure we have future prices
                start_date = df['date'].min() - timedelta(days=30)
                end_date = df['date'].max() + timedelta(days=30)
                
                ticker_prices = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                if not ticker_prices.empty:
                    ticker_prices = ticker_prices.reset_index()
                    ticker_prices['date'] = ticker_prices['Date'].dt.strftime('%Y-%m-%d')
                    ticker_prices['ticker'] = ticker
                    
                    # Calculate returns
                    ticker_prices['next_day_return'] = ticker_prices['Close'].pct_change().shift(-self.future_days)
                    
                    price_data[ticker] = ticker_prices[['date', 'ticker', 'Close', 'next_day_return']]
                
            except Exception as e:
                print(f"Failed to download price data for {ticker}: {e}")
        
        # Combine all price data
        if price_data:
            all_prices = pd.concat(price_data.values(), ignore_index=True)
            
            # Merge with main dataset
            df = df.merge(all_prices, on=['date', 'ticker'], how='left')
            
            # Create target variables based on type
            if self.target_type == 'binary_return':
                df['target'] = (df['next_day_return'] > 0).astype(int)
            elif self.target_type == 'return_regression':
                df['target'] = df['next_day_return']
            elif self.target_type == 'sentiment_classification':
                # Use existing sentiment from Kaggle data if available
                sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and col != 'news_sentiment_score']
                if sentiment_cols:
                    df['target'] = df[sentiment_cols[0]]
                else:
                    print("No sentiment column found for sentiment classification target")
                    df['target'] = 0
            
            # Remove rows without target values
            initial_size = len(df)
            df = df.dropna(subset=['target'])
            print(f"Created targets: {len(df)} records with valid targets (removed {initial_size - len(df)} without targets)")
        
        else:
            print("WARNING: No price data downloaded. Cannot create return-based targets.")
            df['target'] = 0
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to create additional predictive features."""
        if not self.feature_engineering:
            return df
        
        print("Applying feature engineering...")
        
        # Sort by ticker and date for proper feature creation
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Identify feature columns
        feature_cols = []
        
        # News sentiment features
        news_cols = [col for col in df.columns if col.startswith('news_')]
        feature_cols.extend(news_cols)
        
        # Trends features
        trends_cols = [col for col in df.columns if 'trends_score' in col or col.startswith('trends_')]
        feature_cols.extend(trends_cols)
        
        # Original sentiment features
        sentiment_cols = [col for col in df.columns if 'sentiment' in col and col not in news_cols]
        feature_cols.extend(sentiment_cols)
        
        print(f"Engineering features for {len(feature_cols)} base columns")
        
        # Create engineered features for each ticker
        engineered_features = []
        
        for ticker in tqdm(df['ticker'].unique(), desc="Engineering features"):
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Skip if insufficient data
            if len(ticker_data) < 10:
                continue
            
            # Create lagged features (1, 3, 7 days)
            for col in feature_cols:
                if col in ticker_data.columns:
                    for lag in [1, 3, 7]:
                        ticker_data[f'{col}_lag_{lag}'] = ticker_data[col].shift(lag)
            
            # Create moving averages (3, 7, 14 days)
            for col in feature_cols:
                if col in ticker_data.columns:
                    for window in [3, 7, 14]:
                        ticker_data[f'{col}_ma_{window}'] = ticker_data[col].rolling(window=window, min_periods=1).mean()
            
            # Create momentum features (change over periods)
            for col in feature_cols:
                if col in ticker_data.columns:
                    for period in [3, 7]:
                        ticker_data[f'{col}_momentum_{period}'] = ticker_data[col] - ticker_data[col].shift(period)
            
            # Create volatility features (rolling std)
            for col in feature_cols:
                if col in ticker_data.columns:
                    for window in [7, 14]:
                        ticker_data[f'{col}_vol_{window}'] = ticker_data[col].rolling(window=window, min_periods=1).std()
            
            engineered_features.append(ticker_data)
        
        if engineered_features:
            df = pd.concat(engineered_features, ignore_index=True)
            print(f"Feature engineering complete: {len(df.columns)} total columns")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature columns."""
        if self.normalize_features == 'none':
            return df
        
        print(f"Normalizing features using {self.normalize_features} scaler")
        
        # Identify feature columns (exclude metadata and target)
        exclude_cols = ['date', 'ticker', 'target', 'Close', 'next_day_return']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select scaler
        if self.normalize_features == 'standard':
            self.scaler = StandardScaler()
        elif self.normalize_features == 'robust':
            self.scaler = RobustScaler()
        elif self.normalize_features == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        # Apply normalization
        if feature_cols:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            print(f"Normalized {len(feature_cols)} feature columns")
        
        return df
    
    def filter_and_validate(self, df: pd.DataFrame, min_samples_per_ticker: int, 
                           exclude_recent_days: int) -> pd.DataFrame:
        """Filter dataset and validate quality."""
        print("Filtering and validating dataset...")
        
        # Exclude recent days
        if exclude_recent_days > 0:
            cutoff_date = df['date'].max() - timedelta(days=exclude_recent_days)
            initial_size = len(df)
            df = df[df['date'] <= cutoff_date]
            print(f"Excluded recent {exclude_recent_days} days: {len(df)} records remaining (removed {initial_size - len(df)})")
        
        # Filter tickers with insufficient samples
        ticker_counts = df.groupby('ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_samples_per_ticker].index
        
        initial_tickers = df['ticker'].nunique()
        df = df[df['ticker'].isin(valid_tickers)]
        
        print(f"Filtered tickers: {len(valid_tickers)} tickers remaining "
              f"(removed {initial_tickers - len(valid_tickers)} with <{min_samples_per_ticker} samples)")
        
        # Final validation
        print(f"\nFinal dataset statistics:")
        print(f"  Records: {len(df)}")
        print(f"  Tickers: {df['ticker'].nunique()}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Features: {len([col for col in df.columns if col not in ['date', 'ticker', 'target', 'Close', 'next_day_return']])}")
        
        if self.target_type == 'binary_return':
            target_dist = df['target'].value_counts(normalize=True)
            print(f"  Target distribution: {target_dist.to_dict()}")
        
        return df


def save_dataset(df: pd.DataFrame, out_path: str, scaler=None):
    """Save the final dataset and metadata."""
    
    # Create output directory
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save main dataset
    if out_path.endswith('.parquet'):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    
    print(f"Dataset saved to: {out_path}")
    
    # Save metadata
    base_path = out_path.rsplit('.', 1)[0]
    
    # Dataset metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_records': len(df),
        'unique_tickers': df['ticker'].nunique(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
        'columns': df.columns.tolist(),
        'feature_columns': [col for col in df.columns if col not in ['date', 'ticker', 'target', 'Close', 'next_day_return']],
        'target_column': 'target',
        'missing_values': df.isnull().sum().to_dict()
    }
    
    with open(f"{base_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save scaler if available
    if scaler:
        import pickle
        with open(f"{base_path}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {base_path}_scaler.pkl")
    
    print(f"Metadata saved to: {base_path}_metadata.json")


def main():
    args = parse_args()
    
    # Initialize builder
    builder = UnifiedDatasetBuilder(
        target_type=args.target_type,
        future_days=args.future_days,
        feature_engineering=args.feature_engineering,
        normalize_features=args.normalize_features
    )
    
    # Load data
    df = builder.load_data(args.input_path)
    
    # Process data
    df = builder.clean_and_prepare(df)
    df = builder.create_target_variables(df)
    df = builder.engineer_features(df)
    df = builder.normalize_features(df)
    df = builder.filter_and_validate(df, args.min_samples_per_ticker, args.exclude_recent_days)
    
    # Save final dataset
    save_dataset(df, args.out_path, builder.scaler)
    
    print("\n" + "="*80)
    print("UNIFIED DATASET CREATION COMPLETE")
    print("="*80)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.out_path}")
    print(f"Target type: {args.target_type}")
    print(f"Feature engineering: {args.feature_engineering}")
    print(f"Normalization: {args.normalize_features}")
    print(f"Final records: {len(df)}")
    print(f"Ready for ML training!")


if __name__ == "__main__":
    main() 