import argparse
import os
import json
import requests
from datetime import datetime, date, timedelta
from tqdm import tqdm
import sys
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UTILS_PATH = os.path.join(SRC_PATH, 'utils')
for p in (SRC_PATH, UTILS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)
from tickers import get_sp500_tickers
from utils.api_utils import get_key
from utils.date_utils import resolve_date_range, add_window_cli

NEWS_SOURCES = {
    'newsapi': 'https://newsapi.org/v2/everything',
    'marketaux': 'https://api.marketaux.com/v1/news/all'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Ingest news from multiple sources.')
    parser.add_argument('--sources', nargs='+', choices=NEWS_SOURCES.keys(), default=['newsapi'],
                        help='List of news sources (default: newsapi)')
    parser.add_argument('--tickers', nargs='+', default=get_sp500_tickers(), help='List of stock tickers')
    add_window_cli(parser)
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for JSONL files')
    return parser.parse_args()

def fetch_newsapi_news(ticker, since, until, api_key):
    params = {
        'q': ticker,
        'from': since,
        'to': until,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,
        'apiKey': api_key
    }
    resp = requests.get(NEWS_SOURCES['newsapi'], params=params, timeout=30)
    if resp.status_code != 200:
        print(f"[WARN] NewsAPI request failed for {ticker}: {resp.text[:100]}")
        return []
    data = resp.json()
    articles = []
    for art in data.get('articles', []):
        articles.append({
            'headline': art['title'],
            'summary': art.get('description'),
            'date': art['publishedAt'],
            'publisher': art.get('source', {}).get('name')
        })
    return articles


def fetch_marketaux_news(ticker, since, until, api_key):
    params = {
        'symbols': ticker,
        'filter_entities': 'true',
        'language': 'en',
        'from': since,
        'to': until,
        'api_token': api_key
    }
    resp = requests.get(NEWS_SOURCES['marketaux'], params=params, timeout=30)
    if resp.status_code != 200:
        print(f"[WARN] Marketaux request failed for {ticker}: {resp.text[:100]}")
        return []
    data = resp.json()
    return [{
        'headline': a['title'],
        'summary': a.get('description'),
        'date': a['published_at'],
        'publisher': a.get('source')
    } for a in data.get('data', [])]

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    start_date, end_date, window_used = resolve_date_range(args.start_date, args.end_date, args.window)
    print(f'Fetching news from {start_date} to {end_date} (window={window_used})')
    newsapi_key = os.getenv('NEWSAPI_KEY') or None
    marketaux_key = os.getenv('MARKETAUX_KEY') or None
    try:
        if not newsapi_key and 'newsapi' in args.sources:
            newsapi_key = get_key('newsapi_key')
    except KeyError:
        pass

    try:
        if not marketaux_key and 'marketaux' in args.sources:
            marketaux_key = get_key('marketaux_key')
    except KeyError:
        pass

    for ticker in args.tickers:
        all_articles = []
        seen_titles = set()
        for source in args.sources:
            if source == 'newsapi' and newsapi_key:
                fetched = fetch_newsapi_news(ticker, start_date, end_date, newsapi_key)
            elif source == 'marketaux' and marketaux_key:
                fetched = fetch_marketaux_news(ticker, start_date, end_date, marketaux_key)
            else:
                print(f"[WARN] API key missing for {source}. Skipping.")
                fetched = []
            for art in fetched:
                if art['headline'] not in seen_titles:
                    seen_titles.add(art['headline'])
                    all_articles.append(art)
        out_path = os.path.join(args.out_dir, f'{ticker}_news_{start_date}_{end_date}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for article in all_articles:
                json.dump(article, f)
                f.write('\n')
        print(f'Saved news for {ticker} to {out_path}')

if __name__ == '__main__':
    main() 