import argparse
import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

NORMALIZATION_METHODS = ['z-score', 'min-max']
FUSION_METHODS = ['weighted_average', 'pca']

WEIGHTS = {
    'twitter_sentiment': 0.2,
    'reddit_sentiment': 0.1,
    'news_sentiment': 0.2,
    'sec_sentiment': 0.1,
    'google_trend': 0.1,
    'wiki_trend': 0.1,
    'put_call_ratio': 0.1,
    'short_interest': 0.1
}

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize and fuse sentiment scores into a composite score.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with raw sentiment JSONL files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed sentiment scores')
    parser.add_argument('--method', type=str, required=True, choices=NORMALIZATION_METHODS + FUSION_METHODS, help='Normalization and fusion method')
    return parser.parse_args()

def normalize_scores(df, method):
    if method == 'z-score':
        scaler = StandardScaler()
    elif method == 'min-max':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f'Unknown normalization method: {method}')
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def fuse_scores(df, method):
    if method == 'weighted_average':
        return df.dot(pd.Series(WEIGHTS))
    elif method == 'pca':
        pca = PCA(n_components=1)
        return pca.fit_transform(df).flatten()
    else:
        raise ValueError(f'Unknown fusion method: {method}')

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    all_data = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.jsonl'):
            with open(os.path.join(args.input_dir, file), 'r', encoding='utf-8') as f:
                for line in f:
                    all_data.append(json.loads(line))
    df = pd.DataFrame(all_data)
    # identify numeric feature cols (excluding date/ticker)
    ignore = {'date', 'ticker'}
    numeric_cols = [c for c in df.columns if c not in ignore and df[c].dtype != object]

    # per-ticker z-score normalization
    for col in numeric_cols:
        df[col + '_z'] = df.groupby('ticker')[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    z_cols = [c for c in df.columns if c.endswith('_z')]
    df['composite_score'] = fuse_scores(df[z_cols], 'weighted_average')
    out_path = os.path.join(args.output_dir, 'fused_sentiment_scores.parquet')
    df.to_parquet(out_path, index=False)
    print(f'Saved fused sentiment scores to {out_path}')

if __name__ == '__main__':
    main() 