@echo off
echo Setting up Financial Sentiment Pipeline Environment
echo.

REM Set your NewsAPI key here (get free key from https://newsapi.org/)
set NEWSAPI_KEY="______"

REM Optional: Set other API keys if you have them
REM set FINNHUB_API_KEY=your_finnhub_key_here
REM set POLYGON_API_KEY=your_polygon_key_here

echo Environment variables set!
echo.
echo IMPORTANT: Replace 'your_newsapi_key_here' with your actual NewsAPI key
echo.
echo To run the pipeline:
echo python src/data/backfill_historical_new.py --kaggle_path data/external/kaggle_twitter_sentiment.csv --tickers AAPL MSFT GOOGL TSLA --since 2023-01-01 --until 2023-12-31
echo. 
