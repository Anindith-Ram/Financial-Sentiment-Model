"""Add lagged and rolling-window features to the fused sentiment dataset.

Usage
-----
python src/features/add_lagged_features.py \
    --input_path data/processed/fused_sentiment_scores.parquet \
    --output_path data/processed/lagged_sentiment_features.parquet \
    --lags 1 3 7 \
    --windows 3 7

The script assumes the dataset contains a `date` column in YYYY-MM-DD ISO
format (string) and optionally a `ticker` column.  If no ticker column is
present, computations are performed globally.
"""

import argparse
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Create lagged & rolling features from composite scores")
    p.add_argument("--input_path", required=True, help="Path to fused_sentiment_scores.parquet")
    p.add_argument("--output_path", required=True, help="Destination parquet path for lagged features")
    p.add_argument("--lags", nargs="+", type=int, default=[1], help="Lag periods (in days) to add")
    p.add_argument("--windows", nargs="+", type=int, default=[3, 7], help="Rolling window sizes (in days)")
    return p.parse_args()


def _detect_date_column(df: pd.DataFrame):
    for col in ["date", "timestamp", "created_at"]:
        if col in df.columns:
            return col
    raise ValueError("No date-like column (date/timestamp/created_at) found in dataset")


def add_features(df: pd.DataFrame, lags: list[int], windows: list[int]):
    # Identify composite score column
    if "composite_score" not in df.columns:
        raise ValueError("Dataset must contain 'composite_score' column")

    has_ticker = "ticker" in df.columns
    date_col = _detect_date_column(df)

    df[date_col] = pd.to_datetime(df[date_col]).dt.floor("D")
    sort_cols = ["ticker", date_col] if has_ticker else [date_col]
    df = df.sort_values(sort_cols)

    group_obj = df.groupby("ticker") if has_ticker else [(None, df)]

    frames = []
    for key, grp in group_obj:
        g = grp.copy()
        for lag in lags:
            g[f"composite_lag_{lag}"] = g["composite_score"].shift(lag)
        for win in windows:
            g[f"composite_roll_mean_{win}"] = (
                g["composite_score"].rolling(win).mean()
            )
            g[f"composite_roll_std_{win}"] = (
                g["composite_score"].rolling(win).std()
            )
        frames.append(g)

    out = pd.concat(frames).reset_index(drop=True)
    # Drop rows where lagged features are NaN due to insufficient history
    min_lag = max(max(lags, default=0), max(windows, default=0))
    if min_lag:
        out = out.groupby("ticker").apply(lambda x: x.iloc[min_lag:]).reset_index(drop=True) if has_ticker else out.iloc[min_lag:]
    return out


def main():
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    df_feat = add_features(df, args.lags, args.windows)
    df_feat.to_parquet(args.output_path, index=False)
    print(f"Saved lagged features to {args.output_path}")


if __name__ == "__main__":
    main() 