import argparse
import os
import json
import sys
from datetime import datetime, date
from tqdm import tqdm
import yfinance as yf
import pandas as pd

# Ensure src and src/utils are in path before imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UTILS_PATH = os.path.join(SRC_PATH, 'utils')
for p in (SRC_PATH, UTILS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.api_utils import with_retries
from utils.tickers import get_sp500_tickers
from utils.date_utils import resolve_date_range, add_window_cli

def parse_args():
    parser = argparse.ArgumentParser(description='Ingest market proxies for given tickers.')
    parser.add_argument('--tickers', nargs='+', default=get_sp500_tickers(), help='List of stock tickers')
    add_window_cli(parser)
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for JSONL files')
    return parser.parse_args()

@with_retries(max_attempts=3, backoff_seconds=1.0)
def _fetch_yfinance_metrics(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    short_interest = None
    if info.get('sharesShort') and info.get('sharesOutstanding'):
        short_interest = info['sharesShort'] / max(info['sharesOutstanding'], 1)

    put_call_ratio = None
    options_volume = None
    try:
        expirations = tk.options
        if expirations:
            chain = tk.option_chain(expirations[0])
            puts_vol = chain.puts['volume'].sum()
            calls_vol = chain.calls['volume'].sum()
            options_volume = float(puts_vol + calls_vol)
            if calls_vol > 0:
                put_call_ratio = float(puts_vol / calls_vol)
    except Exception:
        pass

    return {
        'short_interest': short_interest,
        'put_call_ratio': put_call_ratio,
        'options_volume': options_volume
    }


def fetch_market_data(ticker: str, start: str, end: str) -> dict:
    """Fetch market proxy data using yfinance only."""
    data = _fetch_yfinance_metrics(ticker)
    missing = len([v for v in data.values() if v is None])
    if missing:
        print(f'⚠️  {ticker}: {missing}/{len(data)} metrics missing for {start}–{end}')
    return data

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    start_date, end_date, window_used = resolve_date_range(args.start_date, args.end_date, args.window)
    print(f'Fetching market proxy data from {start_date} to {end_date} (window={window_used})')
    for ticker in args.tickers:
        out_path = os.path.join(args.out_dir, f'{ticker}_market_{start_date}_{end_date}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            market_data = fetch_market_data(ticker, start_date, end_date)
            data = {
                'ticker': ticker,
                'date_range': f'{start_date} to {end_date}',
                'market_data': market_data
            }
            json.dump(data, f)
            f.write('\n')
        print(f'Saved market proxies for {ticker} to {out_path}')

if __name__ == '__main__':
    main() 