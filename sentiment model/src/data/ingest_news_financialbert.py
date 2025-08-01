"""Ingest news from NewsAPI and compute sentiment using FinancialBERT.

This script fetches news headlines for specific stocks/companies and computes
sentiment scores using AhmedRachid's FinancialBERT model:
https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis

The output is designed to be aligned with the Kaggle Twitter dataset by date and ticker.
"""

import argparse
import os
import json
import requests
import sys
from datetime import datetime, date, timedelta
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Ensure utils path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UTILS_PATH = os.path.join(SRC_PATH, 'utils')
for p in (SRC_PATH, UTILS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.api_utils import get_key
from utils.date_utils import resolve_date_range, add_window_cli


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
        # Model typically outputs: negative, neutral, positive
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        print(f"FinancialBERT loaded on device: {self.device}")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text."""
        if not text or not text.strip():
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
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
            
            return {
                'label': self.label_mapping[predicted_class],
                'score': sentiment_score,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def analyze_batch(self, texts, batch_size=32):
        """Analyze sentiment for a batch of texts."""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i+batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch]
            results.extend(batch_results)
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Ingest news with FinancialBERT sentiment analysis.')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of stock tickers')
    parser.add_argument('--companies', nargs='+', help='List of company names (optional, derived from tickers if not provided)')
    add_window_cli(parser)
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for processed news data')
    parser.add_argument('--kaggle_alignment', type=str, 
                       help='Path to processed Kaggle dataset for date/ticker alignment')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for sentiment analysis')
    return parser.parse_args()


def get_company_names(tickers):
    """Get company names for tickers. This is a simple mapping - can be enhanced."""
    # Basic ticker to company name mapping - can be extended
    ticker_to_company = {
        'AAPL': 'Apple Inc',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta Platforms',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa Inc',
        'PG': 'Procter & Gamble',
        'UNH': 'UnitedHealth Group',
        'HD': 'Home Depot',
        'MA': 'Mastercard',
        'BAC': 'Bank of America',
        'MP': 'MP Materials'
    }
    
    return [ticker_to_company.get(ticker, ticker) for ticker in tickers]


def fetch_newsapi_headlines(ticker, company_name, since, until, api_key):
    """Fetch news headlines for a ticker/company from NewsAPI."""
    # Create search queries for both ticker and company name
    queries = [f'"{ticker}"', f'"{company_name}"', f'{company_name} stock']
    
    all_articles = []
    seen_titles = set()
    
    for query in queries:
        params = {
            'q': query,
            'from': since,
            'to': until,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,
            'apiKey': api_key
        }
        
        try:
            resp = requests.get('https://newsapi.org/v2/everything', params=params, timeout=30)
            if resp.status_code != 200:
                print(f"[WARN] NewsAPI request failed for {query}: {resp.text[:100]}")
                continue
                
            data = resp.json()
            for art in data.get('articles', []):
                title = art.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_articles.append({
                        'ticker': ticker,
                        'company': company_name,
                        'headline': title,
                        'description': art.get('description', ''),
                        'date': art.get('publishedAt', '')[:10],  # Extract date part
                        'publisher': art.get('source', {}).get('name', ''),
                        'url': art.get('url', '')
                    })
                    
        except Exception as e:
            print(f"[ERROR] Failed to fetch news for {query}: {e}")
    
    return all_articles


def align_with_kaggle_dataset(news_data, kaggle_path=None):
    """Align news data with Kaggle dataset dates and tickers."""
    if not kaggle_path or not os.path.exists(kaggle_path):
        print("[INFO] No Kaggle alignment file provided or file not found")
        return news_data
    
    try:
        # Load Kaggle dataset to get available dates and tickers
        kaggle_df = pd.read_csv(kaggle_path)
        
        # Get unique date-ticker combinations from Kaggle dataset
        kaggle_combinations = set()
        if 'date' in kaggle_df.columns and 'ticker' in kaggle_df.columns:
            for _, row in kaggle_df.iterrows():
                kaggle_combinations.add((row['date'], row['ticker'].upper()))
        
        # Filter news data to only include dates/tickers present in Kaggle dataset
        aligned_news = []
        for article in news_data:
            key = (article['date'], article['ticker'].upper())
            if key in kaggle_combinations:
                aligned_news.append(article)
        
        print(f"[INFO] Aligned {len(aligned_news)} news articles with Kaggle dataset "
              f"(from {len(news_data)} total articles)")
        return aligned_news
        
    except Exception as e:
        print(f"[WARN] Failed to align with Kaggle dataset: {e}")
        return news_data


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Resolve date range
    start_date, end_date, window_used = resolve_date_range(
        args.start_date, args.end_date, args.window
    )
    print(f'Fetching news from {start_date} to {end_date} (window={window_used})')
    
    # Get API key
    try:
        newsapi_key = os.getenv('NEWSAPI_KEY') or get_key('newsapi_key')
    except KeyError:
        print("[ERROR] NewsAPI key not found. Set NEWSAPI_KEY environment variable.")
        return
    
    # Get company names
    company_names = args.companies or get_company_names(args.tickers)
    
    # Initialize FinancialBERT
    sentiment_analyzer = FinancialBERTSentimentAnalyzer()
    
    # Collect all news data
    all_news_data = []
    
    for ticker, company in zip(args.tickers, company_names):
        print(f"Fetching news for {ticker} ({company})")
        
        # Fetch news articles
        articles = fetch_newsapi_headlines(ticker, company, start_date, end_date, newsapi_key)
        all_news_data.extend(articles)
    
    print(f"Total articles fetched: {len(all_news_data)}")
    
    # Align with Kaggle dataset if provided
    if args.kaggle_alignment:
        all_news_data = align_with_kaggle_dataset(all_news_data, args.kaggle_alignment)
    
    if not all_news_data:
        print("[WARN] No news articles found or all filtered out during alignment")
        return
    
    # Analyze sentiment for all headlines
    headlines = [article['headline'] for article in all_news_data]
    sentiment_results = sentiment_analyzer.analyze_batch(headlines, args.batch_size)
    
    # Add sentiment scores to articles
    for article, sentiment in zip(all_news_data, sentiment_results):
        article.update({
            'news_sentiment_label': sentiment['label'],
            'news_sentiment_score': sentiment['score'],
            'news_confidence': sentiment['confidence']
        })
    
    # Save results
    # Group by ticker and save separate files
    ticker_groups = {}
    for article in all_news_data:
        ticker = article['ticker']
        if ticker not in ticker_groups:
            ticker_groups[ticker] = []
        ticker_groups[ticker].append(article)
    
    for ticker, articles in ticker_groups.items():
        # Save as JSONL
        jsonl_path = os.path.join(args.out_dir, f'{ticker}_news_sentiment_{start_date}_{end_date}.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for article in articles:
                json.dump(article, f)
                f.write('\n')
        
        # Save as CSV for easier analysis
        csv_path = os.path.join(args.out_dir, f'{ticker}_news_sentiment_{start_date}_{end_date}.csv')
        df = pd.DataFrame(articles)
        df.to_csv(csv_path, index=False)
        
        print(f'Saved {len(articles)} articles for {ticker}:')
        print(f'  JSONL: {jsonl_path}')
        print(f'  CSV: {csv_path}')
    
    # Save aggregated summary
    summary_path = os.path.join(args.out_dir, f'news_sentiment_summary_{start_date}_{end_date}.json')
    summary = {
        'total_articles': len(all_news_data),
        'date_range': f'{start_date} to {end_date}',
        'tickers': list(ticker_groups.keys()),
        'articles_per_ticker': {k: len(v) for k, v in ticker_groups.items()},
        'sentiment_distribution': {
            'positive': sum(1 for a in all_news_data if a['news_sentiment_label'] == 'positive'),
            'neutral': sum(1 for a in all_news_data if a['news_sentiment_label'] == 'neutral'),
            'negative': sum(1 for a in all_news_data if a['news_sentiment_label'] == 'negative'),
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main() 