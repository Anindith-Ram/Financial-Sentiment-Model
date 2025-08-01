# Getting Started with the Financial Sentiment Pipeline

## Quick Start Guide

### 1. 📥 Download Kaggle Dataset
1. Go to: https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns
2. Download the CSV file
3. Place it as: `data/external/kaggle_twitter_sentiment.csv`

### 2. 🔑 Set Up API Key
Get a free NewsAPI key from https://newsapi.org/ and set it:

**Windows:**
```cmd
set NEWSAPI_KEY=your_actual_key_here
```

**Linux/Mac:**
```bash
export NEWSAPI_KEY=your_actual_key_here
```

### 3. 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. 🚀 Run the Pipeline

**Easy Way (Windows):**
```cmd
run_new_pipeline.bat
```

**Manual Command:**
```bash
python src/data/orchestrate_combined_pipeline.py \
    --kaggle_path data/external/kaggle_twitter_sentiment.csv \
    --tickers AAPL MSFT GOOGL TSLA \
    --news_since 2024-01-01 \
    --news_until 2024-12-31
```

**If you already processed Kaggle data:**
```bash
python src/data/orchestrate_combined_pipeline.py \
    --kaggle_path data/external/kaggle_twitter_sentiment.csv \
    --tickers AAPL MSFT GOOGL TSLA \
    --news_since 2024-01-01 \
    --news_until 2024-12-31 \
    --skip_kaggle
```

### 5. 🎯 Create ML-Ready Dataset
After the pipeline completes:
```bash
python src/data/build_unified_dataset.py \
    --input_path data/processed/aligned/aligned_sentiment_dataset_2023-01-01_2023-12-31.csv \
    --out_path data/processed/ml_ready_dataset.csv \
    --target_type binary_return \
    --feature_engineering
```

## 📊 Expected Output Structure

After running, you'll have:
```
data/
├── external/
│   └── kaggle_twitter_sentiment.csv      # Your downloaded dataset
├── processed/
│   ├── kaggle/
│   │   └── kaggle_twitter_processed.csv  # Processed Kaggle data
│   └── aligned/
│       └── aligned_sentiment_dataset_*.csv # Final aligned dataset
├── raw/
│   ├── news/                             # NewsAPI + FinancialBERT results
│   └── trends/                           # Google Trends data
└── processed/
    └── ml_ready_dataset.csv              # Ready for ML training
```

## 🎯 Key Features You'll Get

- **Twitter Sentiment**: From Kaggle dataset (core foundation)
- **News Sentiment**: FinancialBERT analysis of news headlines
- **Trends Data**: Google search interest aligned by date/ticker
- **All Aligned**: Everything matched by date and ticker

## 🔧 Troubleshooting

**No Kaggle data found:**
- Make sure the CSV is at `data/external/kaggle_twitter_sentiment.csv`
- Check the file isn't corrupted

**NewsAPI errors:**
- Verify your API key is set correctly
- Check you haven't exceeded the free tier limit (1000 requests/month)

**Google Trends rate limits:**
- The script includes retry logic and delays
- For large date ranges, consider smaller chunks

**FinancialBERT download:**
- First run will download the model (~400MB)
- Ensure you have internet connection and disk space

## 🎉 Success Indicators

Pipeline completed successfully when you see:
```
[SUCCESS] Final aligned dataset available in: data/processed/aligned/
This dataset combines:
  • Kaggle Twitter sentiment data (core)
  • NewsAPI headlines with FinancialBERT sentiment scores
  • Google Trends interest scores
  • All aligned by date and ticker
```

## 📝 Next Steps

1. Explore the aligned dataset in `data/processed/aligned/`
2. Train ML models using `src/models/train.py`
3. Evaluate performance with `src/models/evaluate.py`
4. Run backtests using `src/backtest/run.py`

For detailed documentation, see `PIPELINE_UPDATE_SUMMARY.md` 