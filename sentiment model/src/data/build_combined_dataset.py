"""Build combined dataset from Kaggle Twitter data + current NewsAPI + Google Trends.

This script:
1. Loads historical Kaggle Twitter sentiment data (2017-2018)
2. Loads current NewsAPI headlines 
3. Loads current Google Trends data
4. Combines them while preserving Kaggle structure
5. Applies FinancialBERT for consistent sentiment analysis across all text data

Usage:
python src/data/build_combined_dataset.py \
    --kaggle_data data/processed/kaggle/kaggle_twitter_processed.csv \
    --news_data_dir data/raw/news \
    --trends_data_dir data/raw/trends \
    --out_dir data/processed/combined \
    --tickers AAPL MSFT GOOGL TSLA
"""

import argparse
import os
import pandas as pd
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


class FinancialBERTSentimentAnalyzer:
    """Sentiment analyzer using FinancialBERT model."""
    
    def __init__(self, model_name="ahmedrachid/FinancialBERT-Sentiment-Analysis"):
        print(f"Loading FinancialBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Define label mapping for FinancialBERT
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        print(f"FinancialBERT loaded on device: {self.device}")
    
    def analyze_sentiment_batch(self, texts, batch_size=32):
        """Analyze sentiment for a batch of texts."""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="FinancialBERT analysis"):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                if not text or not str(text).strip():
                    batch_results.append({'label': 'neutral', 'score': 0.0, 'confidence': 0.0})
                    continue
                
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        str(text), 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class = torch.argmax(predictions, dim=-1).item()
                        confidence = torch.max(predictions).item()
                    
                    # Convert to sentiment score (-1 to 1)
                    if predicted_class == 0:  # negative
                        sentiment_score = -confidence
                    elif predicted_class == 1:  # neutral
                        sentiment_score = 0.0
                    else:  # positive
                        sentiment_score = confidence
                    
                    batch_results.append({
                        'label': self.label_mapping[predicted_class],
                        'score': sentiment_score,
                        'confidence': confidence
                    })
                
                except Exception as e:
                    print(f"Error analyzing text: {e}")
                    batch_results.append({'label': 'neutral', 'score': 0.0, 'confidence': 0.0})
            
            results.extend(batch_results)
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Build combined dataset with Kaggle + NewsAPI + Trends + FinancialBERT')
    parser.add_argument('--kaggle_data', type=str, required=True,
                       help='Path to processed Kaggle dataset')
    parser.add_argument('--news_data_dir', type=str,
                       help='Directory with NewsAPI data files')
    parser.add_argument('--trends_data_dir', type=str,
                       help='Directory with Google Trends data files')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--tickers', nargs='+', 
                       help='Filter to specific tickers')
    parser.add_argument('--apply_financialbert', action='store_true', default=True,
                       help='Apply FinancialBERT to all text data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for FinancialBERT processing')
    return parser.parse_args()


def load_kaggle_data(kaggle_path: str, tickers: List[str] = None) -> pd.DataFrame:
    """Load the processed Kaggle Twitter dataset."""
    print(f"Loading Kaggle dataset from: {kaggle_path}")
    
    if not os.path.exists(kaggle_path):
        print(f"[WARN] Kaggle dataset not found at {kaggle_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(kaggle_path)
    
    if tickers:
        tickers_upper = [t.upper() for t in tickers]
        df = df[df['ticker'].isin(tickers_upper)]
    
    print(f"Kaggle data: {len(df)} records, {df['ticker'].nunique()} tickers")
    if len(df) > 0:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Add data source indicator
    df['data_source'] = 'kaggle_twitter'
    
    return df


def load_news_data(news_dir: str, tickers: List[str] = None) -> pd.DataFrame:
    """Load NewsAPI data."""
    if not news_dir or not os.path.exists(news_dir):
        print("[INFO] No news data directory found")
        return pd.DataFrame()
    
    print(f"Loading news data from: {news_dir}")
    
    all_news = []
    news_files = [f for f in os.listdir(news_dir) if f.endswith('.csv')]
    
    for file in news_files:
        try:
            file_path = os.path.join(news_dir, file)
            df = pd.read_csv(file_path)
            
            if tickers:
                ticker_in_filename = None
                for ticker in tickers:
                    if ticker.upper() in file.upper():
                        ticker_in_filename = ticker.upper()
                        break
                
                if ticker_in_filename:
                    df['ticker'] = ticker_in_filename
            
            all_news.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")
    
    if not all_news:
        print("[INFO] No news data found")
        return pd.DataFrame()
    
    news_df = pd.concat(all_news, ignore_index=True)
    
    # Standardize columns for news data
    news_columns = {
        'headline': 'text',
        'title': 'text', 
        'description': 'description',
        'date': 'date',
        'ticker': 'ticker'
    }
    
    existing_mapping = {k: v for k, v in news_columns.items() if k in news_df.columns}
    news_df = news_df.rename(columns=existing_mapping)
    
    # Add data source
    news_df['data_source'] = 'newsapi'
    
    print(f"News data: {len(news_df)} records")
    return news_df


def load_trends_data(trends_dir: str, tickers: List[str] = None) -> pd.DataFrame:
    """Load Google Trends data."""
    if not trends_dir or not os.path.exists(trends_dir):
        print("[INFO] No trends data directory found") 
        return pd.DataFrame()
    
    print(f"Loading trends data from: {trends_dir}")
    
    all_trends = []
    trends_files = [f for f in os.listdir(trends_dir) if f.endswith('.csv')]
    
    for file in trends_files:
        try:
            file_path = os.path.join(trends_dir, file)
            df = pd.read_csv(file_path)
            
            if tickers:
                ticker_in_filename = None
                for ticker in tickers:
                    if ticker.upper() in file.upper():
                        ticker_in_filename = ticker.upper()
                        break
                
                if ticker_in_filename:
                    df['ticker'] = ticker_in_filename
            
            all_trends.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")
    
    if not all_trends:
        print("[INFO] No trends data found")
        return pd.DataFrame()
    
    trends_df = pd.concat(all_trends, ignore_index=True)
    trends_df['data_source'] = 'google_trends'
    
    print(f"Trends data: {len(trends_df)} records")
    return trends_df


def combine_datasets(kaggle_df: pd.DataFrame, news_df: pd.DataFrame, trends_df: pd.DataFrame) -> pd.DataFrame:
    """Combine all datasets while preserving Kaggle structure."""
    print("Combining datasets...")
    
    # Start with Kaggle as the base structure
    combined = kaggle_df.copy() if not kaggle_df.empty else pd.DataFrame()
    
    # Add news data (different time period, so just append)
    if not news_df.empty:
        # Ensure news data has required columns, fill missing ones
        for col in combined.columns:
            if col not in news_df.columns:
                news_df[col] = None
        
        # Add missing columns from news to combined
        for col in news_df.columns:
            if col not in combined.columns:
                combined[col] = None
        
        combined = pd.concat([combined, news_df[combined.columns]], ignore_index=True)
        print(f"After adding news: {len(combined)} total records")
    
    # Add trends data (different time period, append as well)
    if not trends_df.empty:
        # Handle trends data similarly
        for col in combined.columns:
            if col not in trends_df.columns:
                trends_df[col] = None
        
        for col in trends_df.columns:
            if col not in combined.columns:
                combined[col] = None
        
        combined = pd.concat([combined, trends_df[combined.columns]], ignore_index=True)
        print(f"After adding trends: {len(combined)} total records")
    
    return combined


def apply_financialbert_sentiment(df: pd.DataFrame, analyzer: FinancialBERTSentimentAnalyzer, batch_size: int = 32) -> pd.DataFrame:
    """Apply FinancialBERT sentiment analysis to all text data."""
    print("Applying FinancialBERT sentiment analysis...")
    
    if 'text' not in df.columns:
        print("[WARN] No text column found for sentiment analysis")
        return df
    
    # Get all text data
    texts = df['text'].fillna('').astype(str).tolist()
    
    # Analyze sentiment
    sentiment_results = analyzer.analyze_sentiment_batch(texts, batch_size)
    
    # Add results to dataframe
    df['financialbert_label'] = [r['label'] for r in sentiment_results]
    df['financialbert_score'] = [r['score'] for r in sentiment_results]
    df['financialbert_confidence'] = [r['confidence'] for r in sentiment_results]
    
    print(f"Applied FinancialBERT to {len(df)} records")
    
    # Show distribution
    label_dist = pd.Series([r['label'] for r in sentiment_results]).value_counts()
    print("FinancialBERT sentiment distribution:")
    for label, count in label_dist.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def save_combined_dataset(df: pd.DataFrame, out_dir: str):
    """Save the combined dataset."""
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main dataset
    csv_path = os.path.join(out_dir, f'combined_sentiment_dataset_{timestamp}.csv')
    parquet_path = os.path.join(out_dir, f'combined_sentiment_dataset_{timestamp}.parquet')
    
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    
    # Save summary
    summary = {
        'total_records': len(df),
        'data_sources': df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {},
        'tickers': df['ticker'].nunique() if 'ticker' in df.columns else 0,
        'unique_tickers': sorted(df['ticker'].unique().tolist()) if 'ticker' in df.columns else [],
        'columns': df.columns.tolist(),
        'date_range': {
            'min': df['date'].min() if 'date' in df.columns else None,
            'max': df['date'].max() if 'date' in df.columns else None
        }
    }
    
    summary_path = os.path.join(out_dir, f'dataset_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCombined dataset saved:")
    print(f"  CSV: {csv_path}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Summary: {summary_path}")
    
    return csv_path


def main():
    args = parse_args()
    
    print("Building Combined Sentiment Dataset")
    print("=" * 50)
    print(f"Kaggle data: {args.kaggle_data}")
    print(f"News data dir: {args.news_data_dir}")
    print(f"Trends data dir: {args.trends_data_dir}")
    print(f"Output dir: {args.out_dir}")
    print(f"Tickers: {args.tickers}")
    
    # Load all datasets
    kaggle_df = load_kaggle_data(args.kaggle_data, args.tickers)
    news_df = load_news_data(args.news_data_dir, args.tickers)
    trends_df = load_trends_data(args.trends_data_dir, args.tickers)
    
    # Combine datasets
    combined_df = combine_datasets(kaggle_df, news_df, trends_df)
    
    if combined_df.empty:
        print("[ERROR] No data to process")
        return
    
    # Apply FinancialBERT if requested
    if args.apply_financialbert:
        analyzer = FinancialBERTSentimentAnalyzer()
        combined_df = apply_financialbert_sentiment(combined_df, analyzer, args.batch_size)
    
    # Save result
    output_path = save_combined_dataset(combined_df, args.out_dir)
    
            print(f"\n[SUCCESS] Combined dataset creation complete!")
        print(f"[CHART] Total records: {len(combined_df)}")
    print(f"📁 Output: {output_path}")


if __name__ == '__main__':
    main() 