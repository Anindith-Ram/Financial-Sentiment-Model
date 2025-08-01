# Unified Investor Sentiment Signal

This project generates a unified investor sentiment signal from multiple data sources for machine learning training and backtesting trading strategies.

**🔄 PIPELINE UPDATED**: This pipeline has been modernized to use a Kaggle Twitter dataset as the core foundation, with NewsAPI sentiment analysis via FinancialBERT and aligned Google Trends data.

## Project Structure

- `data/raw/`: Raw data (news, trends, market data)
- `data/processed/`: Structured sentiment scores and aligned datasets
- `src/data/`: Data ingestion and processing scripts
- `src/features/`: Normalization + signal fusion
- `src/models/`: ML training + evaluation
- `src/backtest/`: Backtesting strategies
- `utils/`: Shared helpers (logging, cleaning, config)
- `notebooks/`: EDA and visualizations

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Kaggle Dataset:**
   Download the Twitter sentiment dataset from:
   https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns

## Environment variables & API keys

The ingestion scripts rely on external data providers. Keys can be supplied either as classic environment variables **or** via a YAML / JSON file referenced by `APP_CONFIG`.

Required keys (depending on which scripts you run):

* `NEWSAPI_KEY` *(for news headlines)*
* `FINNHUB_API_KEY` *(optional - for additional market data)*
* `POLYGON_API_KEY` *(optional - for market data)*
* `OPENAI_API_KEY` *(optional – used for LLM-based labelling)*

See `configs/my_keys.yaml` for a template:

```yaml
newsapi_key: "YOUR_NEWSAPI_KEY"
finnhub_api_key: "YOUR_FINNHUB_API_KEY"
polygon_api_key: "YOUR_POLYGON_API_KEY"
openai_api_key: "YOUR_OPENAI_API_KEY"
```

Export a config path instead of cluttering your shell:

```bash
export APP_CONFIG=$(pwd)/configs/my_keys.yaml
```

### Rolling windows (`--window`)
All ingesters accept a `--window` flag (e.g. `90d`, `18m`, `2y`) alongside the explicit `--start_date` / `--end_date`.  
When the provider returns fewer data points than expected for the selected window, the script prints a warning so you can adjust the horizon or check the upstream limits.

## Updated Data Pipeline

### Complete Pipeline (Recommended)

Run the entire updated pipeline with the new orchestrator:

```bash
python src/data/backfill_historical_new.py \
    --kaggle_path path/to/kaggle_twitter_sentiment.csv \
    --tickers AAPL MSFT GOOGL TSLA \
    --since 2023-01-01 \
    --until 2023-12-31
```

This will:
1. Process the Kaggle Twitter dataset
2. Fetch news headlines with FinancialBERT sentiment analysis
3. Collect Google Trends data aligned with the Twitter dataset
4. Align all datasets by date and ticker
5. Create a unified sentiment dataset

### Individual Components

You can also run components separately:

#### Kaggle Twitter Dataset Processing
```bash
python src/data/ingest_kaggle_twitter.py \
    --kaggle_path kaggle_dataset.csv \
    --tickers AAPL TSLA \
    --out_dir data/processed/kaggle
```

#### News Headlines with FinancialBERT Sentiment
```bash
python src/data/ingest_news_financialbert.py \
    --tickers AAPL TSLA \
    --since 2023-01-01 \
    --until 2023-01-31 \
    --out_dir data/raw/news
```

#### Google Trends (Aligned)
```bash
python src/data/ingest_trends_aligned.py \
    --tickers AAPL TSLA \
    --since 2023-01-01 \
    --until 2023-01-31 \
    --out_dir data/raw/trends
```

#### Market Data (Unchanged)
```bash
python src/data/ingest_market.py \
    --tickers AAPL TSLA \
    --since 2023-01-01 \
    --until 2023-01-31 \
    --out_dir data/raw/market
```

#### Dataset Alignment
```bash
python src/data/align_datasets.py \
    --kaggle_data data/processed/kaggle/kaggle_twitter_processed.csv \
    --news_data_dir data/raw/news \
    --trends_data_dir data/raw/trends \
    --out_dir data/processed/aligned
```

#### ML-Ready Dataset Creation
```bash
python src/data/build_unified_dataset.py \
    --input_path data/processed/aligned/aligned_sentiment_dataset.csv \
    --out_path data/processed/ml_ready_dataset.csv \
    --target_type binary_return \
    --feature_engineering
```

## Key Features

### 🔗 **Data Sources**
- **Kaggle Twitter Dataset**: Pre-existing high-quality Twitter sentiment data (core)
- **NewsAPI**: Real-time news headlines with FinancialBERT sentiment analysis
- **Google Trends**: Interest scores aligned with Twitter dataset dates/tickers
- **Market Data**: Stock prices for target variable creation

### 🧠 **FinancialBERT Integration**
- Domain-specific sentiment analysis for financial text
- Model: `ahmedrachid/FinancialBERT-Sentiment-Analysis`
- Outputs: sentiment scores (-1 to 1), labels, and confidence scores

### 🎯 **Dataset Alignment**
- All data sources aligned by date and ticker
- Kaggle dataset serves as the foundation
- Supporting features appended without disrupting original schema

### 🛠 **Feature Engineering**
- Lagged features (1, 3, 7 days)
- Moving averages (3, 7, 14 days) 
- Momentum indicators
- Volatility measures

## Data Processing

### Feature Fusion

The pipeline creates unified datasets with these feature categories:

- **Original Features**: From Kaggle Twitter dataset (preserved)
- **News Features**: `news_sentiment_score`, `news_sentiment_label`, `news_confidence`, `news_article_count`
- **Trends Features**: `trends_score` (normalized 0-1)
- **Engineered Features**: Lags, moving averages, momentum, volatility (optional)

### Model Training

Train models using the unified features:

```bash
python src/models/train.py \
    --data_path data/processed/ml_ready_dataset.csv \
    --config configs/train.yaml
```

### Evaluation

Evaluate model performance:

```bash
python src/models/evaluate.py \
    --model_path models/trained_model.pkl \
    --test_data data/processed/test_dataset.csv
```

## Migration from Old Pipeline

If you were using the previous version:

1. **Download the Kaggle dataset** (replaces Twitter scraping)
2. **Update dependencies**: `pip install -r requirements.txt`
3. **Use new orchestrator**: `backfill_historical_new.py`
4. **Update any custom scripts** to use the new dataset structure

## Benefits of Updated Pipeline

- ✅ **No API Rate Limits**: Uses pre-existing Twitter data
- ✅ **Domain-Specific Sentiment**: FinancialBERT for financial text
- ✅ **Reliable Data Sources**: Focused on News + Trends
- ✅ **Modular Design**: Run components independently
- ✅ **Feature Rich**: Comprehensive sentiment signals
- ✅ **Maintainable**: Clean, documented codebase

For detailed information about the pipeline update, see `PIPELINE_UPDATE_SUMMARY.md`.