"""Create (X, y) dataset by merging lagged sentiment features with next-day stock returns.

Usage
-----
python src/features/create_xy_dataset.py \
    --input_path data/processed/lagged_sentiment_features.parquet \
    --output_path data/processed/final_xy.parquet \
    --horizon 1

Assumptions
-----------
* Input parquet contains columns: `date` (YYYY-MM-DD), `ticker`, sentiment features.
* Returns are computed from adjusted close prices via yfinance.
* The label y is `return_t+horizon` where horizon>=1 trading days.
"""

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Merge sentiment features with future price movement and add sample weights")
    p.add_argument("--input_path", required=True, help="Path to lagged_sentiment_features.parquet")
    p.add_argument("--output_path", required=True, help="Destination parquet file")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days (default=1, next-day)")
    p.add_argument("--tau", type=int, default=180, help="Exponential decay half-life in days for sample weights")
    return p.parse_args()


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted close prices using yfinance."""
    data = yf.download(tickers, start=start, end=end, progress=False, interval="1d", group_by="ticker", auto_adjust=True)
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([tickers, data.columns])
    prices = (
        data["Adj Close"]
        .stack()
        .rename("price")
        .reset_index()
        .rename(columns={"level_1": "ticker"})
    )
    prices["date"] = prices["Date"].dt.strftime("%Y-%m-%d")
    prices = prices.drop(columns=["Date"])
    return prices


def main():
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("Input dataset must contain 'ticker' and 'date' columns")

    tickers = df["ticker"].unique().tolist()
    min_date = df["date"].min()
    max_date = pd.to_datetime(df["date"].max()) + timedelta(days=args.horizon + 5)

    price_df = fetch_prices(tickers, start=min_date, end=max_date.strftime("%Y-%m-%d"))
    price_df = price_df.sort_values(["ticker", "date"])

    # compute next-day open gap: (next_open - close) / close
    price_df['close'] = price_df['price']
    price_df['open_next'] = price_df.groupby('ticker')['price'].shift(-1)
    price_df['gap'] = (price_df['open_next'] - price_df['close']) / price_df['close']
    price_df[f'return_t+{args.horizon}'] = price_df['gap']

    merged = df.merge(price_df[["date", "ticker", f"return_t+{args.horizon}"]], on=["date", "ticker"], how="left")
    merged = merged.dropna(subset=[f"return_t+{args.horizon}"]).reset_index(drop=True)

    # Add age (days since) and exponential-decay sample weight
    today_str = pd.to_datetime("today").strftime("%Y-%m-%d")
    age_days = (pd.to_datetime(today_str) - pd.to_datetime(merged["date"])).dt.days
    merged["age_days"] = age_days
    merged["sample_weight"] = np.exp(-age_days / args.tau)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_path, index=False)
    print(f"Saved X,y dataset to {args.output_path} (rows={len(merged)})")


if __name__ == "__main__":
    main() 