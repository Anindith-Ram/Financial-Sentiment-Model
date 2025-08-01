import argparse
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import pytrends
from pytrends.request import TrendReq
import pytrends.exceptions as pt_ex
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.date_utils import resolve_date_range, add_window_cli

pytrends = TrendReq(hl='en-US', tz=360)

def with_retries(max_attempts=5, backoff_seconds=2.0, exceptions=(Exception,)):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(backoff_seconds * (2 ** attempt))
        return wrapper
    return decorator

def parse_args():
    parser = argparse.ArgumentParser(description='Ingest Google Trends data for keywords.')
    parser.add_argument('--keywords', nargs='+', required=True, help='Keywords to search for trends')
    add_window_cli(parser)
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for trends data')
    return parser.parse_args()

def normalize_trends(series):
    """Normalize trends data to 0-1 scale."""
    if series.max() > 0:
        return series / series.max()
    return series

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    start_date, end_date, window_used = resolve_date_range(args.start_date, args.end_date, args.window)
    print(f'Fetching Google Trends data from {start_date} to {end_date} (window={window_used})')

    def fetch_batch(batch):
        """Fetch trends for a <=5 keyword batch with retry/backoff."""
        @with_retries(max_attempts=5, backoff_seconds=2.0, exceptions=(Exception,))
        def _inner():
            attempts = 0
            delay = 1
            while True:
                try:
                    pytrends.build_payload(batch, timeframe=f'{start_date} {end_date}')
                    return pytrends.interest_over_time()
                except (pt_ex.TooManyRequestsError, pt_ex.ResponseError):
                    attempts += 1
                    if attempts >= 5:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return _inner()

    # Google allows max 5 terms; batch if more
    batches = [args.keywords[i:i+5] for i in range(0, len(args.keywords), 5)]
    frames = []
    for b in batches:
        frames.append(fetch_batch(b))
    trends_data = pd.concat(frames, axis=1)

    if 'isPartial' in trends_data.columns:
        trends_data = trends_data.drop(columns=['isPartial'])
    trends_data = trends_data.apply(normalize_trends)
    out_path = os.path.join(args.out_dir, f'trends_{start_date}_{end_date}.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for date, row in tqdm(trends_data.iterrows(), desc='Saving trends data'):
            data = {
                'date': date.strftime('%Y-%m-%d'),
                'trends': row.to_dict()
            }
            json.dump(data, f)
            f.write('\n')
    print(f'Saved Google Trends data to {out_path}')

if __name__ == '__main__':
    main() 