@echo off
echo Starting Combined Sentiment Pipeline
echo.

REM Set your NewsAPI key
set NEWSAPI_KEY=438b09a89fd141718796f1a9e79ad88d

echo Running combined sentiment pipeline...
echo.

REM Run the orchestrator (skipping Kaggle since we already processed it)
python src/data/orchestrate_combined_pipeline.py ^
    --kaggle_path data/external/kaggle_twitter_sentiment.csv ^
    --tickers AAPL MSFT GOOGL TSLA ^
    --news_since 2024-01-01 ^
    --news_until 2024-12-31 ^
    --skip_kaggle ^
    --continue_on_error

echo.
echo Pipeline complete! Check data/processed/combined/ for results.
pause 