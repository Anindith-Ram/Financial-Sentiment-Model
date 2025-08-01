"""
Unified Data Collection with Adjusted Close Integration
Combines enhanced features with streamlined functionality
"""
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from config.config import (
    TICKERS_CSV, N_TICKERS, START, END, SEQ_LEN, HORIZON, 
    PATTERNS, DATA_OUTPUT_PATH, USE_RAW_COLS, USE_ADJ_COLS
)
from src.utils.helpers import label_class


def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data with explicit raw and adjusted columns (Professional Pipeline)
    
    Args:
        ticker (str): Stock ticker
        start_date (str): Start date
        end_date (str): End date
        
    Returns:
        pd.DataFrame: DataFrame with explicit raw and adjusted columns
    """
    try:
        # Call 1: Raw data (auto_adjust=False) - Traditional candlestick analysis
        print(f"Downloading raw data for {ticker}...")
        raw_df = yf.download(ticker, start_date, end_date, interval='1d', 
                           auto_adjust=False, progress=False)
        
        # Call 2: Adjusted data (auto_adjust=True) - Economic accuracy
        print(f"Downloading adjusted data for {ticker}...")
        adj_df = yf.download(ticker, start_date, end_date, interval='1d', 
                           auto_adjust=True, progress=False)
        
        if raw_df.empty or adj_df.empty:
            print(f"No data available for {ticker}")
            return None
        
        # Ensure same date range
        common_dates = raw_df.index.intersection(adj_df.index)
        if len(common_dates) == 0:
            print(f"No overlapping dates for {ticker}")
            return None
        
        raw_df = raw_df.loc[common_dates]
        adj_df = adj_df.loc[common_dates]
        
        # Handle MultiIndex columns from yfinance
        # If downloading single ticker, yfinance returns MultiIndex columns like ('Close', 'AAPL')
        if raw_df.columns.nlevels > 1:
            # Extract ticker name and flatten columns
            raw_df.columns = raw_df.columns.get_level_values(0)
        if adj_df.columns.nlevels > 1:
            adj_df.columns = adj_df.columns.get_level_values(0)
        
        # Create explicit column structure
        final_df = pd.DataFrame({
            'Date': common_dates,
            # Raw OHLCV (for candlestick patterns)
            'Open_raw': raw_df['Open'].values,
            'High_raw': raw_df['High'].values,
            'Low_raw': raw_df['Low'].values,
            'Close_raw': raw_df['Close'].values,
            'Volume_raw': raw_df['Volume'].values,
            # Adjusted OHLCV (for technical indicators)
            'Open_adj': adj_df['Open'].values,
            'High_adj': adj_df['High'].values,
            'Low_adj': adj_df['Low'].values,
            'Close_adj': adj_df['Close'].values,
            'Volume_adj': adj_df['Volume'].values
        })
        
        # Reset index to make Date a regular column
        final_df.reset_index(drop=True, inplace=True)
        
        print(f"Successfully downloaded {len(final_df)} days of data for {ticker}")
        return final_df
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None


def calculate_all_features(df):
    """
    Calculate all features using explicit raw/adjusted column families
    
    Args:
        df (pd.DataFrame): DataFrame with explicit raw and adjusted columns
        
    Returns:
        pd.DataFrame: DataFrame with all features calculated from appropriate sources
    """
    df = df.copy()
    
    # Verify required columns exist
    required_raw = ['Open_raw', 'High_raw', 'Low_raw', 'Close_raw', 'Volume_raw']
    required_adj = ['Open_adj', 'High_adj', 'Low_adj', 'Close_adj', 'Volume_adj']
    
    missing_raw = [col for col in required_raw if col not in df.columns]
    missing_adj = [col for col in required_adj if col not in df.columns]
    
    if missing_raw or missing_adj:
        raise ValueError(f"Missing required columns. Raw: {missing_raw}, Adj: {missing_adj}")
    
    print("Calculating features with explicit data sources...")
    
    # 1. Candlestick patterns using RAW data (traditional approach)
    print("Computing candlestick patterns on raw OHLC...")
    try:
        from src.utils.pattern_validator import calculate_patterns_with_validation
        
        # Create temporary dataframe for pattern calculation
        pattern_df = pd.DataFrame({
            'Open': df['Open_raw'],
            'High': df['High_raw'], 
            'Low': df['Low_raw'],
            'Close': df['Close_raw']
        })
        
        # Calculate patterns with validation
        pattern_df_with_flags, pattern_info = calculate_patterns_with_validation(pattern_df, PATTERNS)
        
        # Add pattern flags to main dataframe
        for pattern in PATTERNS:
            if pattern in pattern_df_with_flags.columns:
                df[pattern] = pattern_df_with_flags[pattern]
                
    except ImportError:
        print("Pattern validator not available, using basic TA-Lib...")
        for pattern in PATTERNS:
            try:
                pattern_result = getattr(talib, pattern)(
                    df['Open_raw'], df['High_raw'], df['Low_raw'], df['Close_raw']
                )
                df[pattern] = pattern_result.ne(0).astype(int)
            except Exception as e:
                print(f"Warning: Error calculating {pattern}: {e}")
                df[pattern] = 0
    
    # 2. Technical indicators using ADJUSTED data (economic accuracy)
    print("Computing technical indicators on adjusted OHLC...")
    
    # Basic indicators
    close_adj = df['Close_adj'].values
    high_adj = df['High_adj'].values
    low_adj = df['Low_adj'].values
    volume_adj = df['Volume_adj'].values
    
    try:
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = talib.SMA(close_adj, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close_adj, timeperiod=period)
        
        # Momentum indicators
        df['rsi_14'] = talib.RSI(close_adj, timeperiod=14)
        df['rsi_21'] = talib.RSI(close_adj, timeperiod=21)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_adj)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Volatility indicators
        df['atr_14'] = talib.ATR(high_adj, low_adj, close_adj, timeperiod=14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_adj, timeperiod=20)
        
        # Volume indicators
        df['ad_line'] = talib.AD(high_adj, low_adj, close_adj, volume_adj)
        df['obv'] = talib.OBV(close_adj, volume_adj)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high_adj, low_adj, close_adj)
        
    except Exception as e:
        print(f"Warning: Error calculating some technical indicators: {e}")
    
    # 3. Enhanced features using ADJUSTED data (essential for accuracy)
    print("Computing enhanced features on adjusted data...")
    
    # Returns (critical to use adjusted data)
    df['daily_return'] = df['Close_adj'].pct_change()
    df['log_return'] = np.log(df['Close_adj'] / df['Close_adj'].shift(1))
    
    # Volatility measures
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['daily_return'].rolling(window).std()
        df[f'avg_return_{window}'] = df['daily_return'].rolling(window).mean()
    
    # Price relationships (using adjusted data)
    df['hl_ratio'] = (df['High_adj'] - df['Low_adj']) / df['Close_adj']
    df['oc_ratio'] = (df['Open_adj'] - df['Close_adj']) / df['Close_adj']
    df['price_change'] = (df['Close_adj'] - df['Open_adj']) / df['Open_adj']
    
    # Support/Resistance levels
    for window in [10, 20]:
        df[f'resistance_{window}'] = df['High_adj'].rolling(window).max()
        df[f'support_{window}'] = df['Low_adj'].rolling(window).min()
        df[f'price_position_{window}'] = (df['Close_adj'] - df[f'support_{window}']) / (df[f'resistance_{window}'] - df[f'support_{window}'])
    
    # Gap analysis (using adjusted data for accuracy)
    df['gap'] = (df['Open_adj'] - df['Close_adj'].shift(1)) / df['Close_adj'].shift(1)
    df['gap_up'] = (df['gap'] > 0.01).astype(int)  # 1% gap up
    df['gap_down'] = (df['gap'] < -0.01).astype(int)  # 1% gap down
    
    # Volume analysis
    df['volume_sma_20'] = df['Volume_adj'].rolling(20).mean()
    df['volume_ratio'] = df['Volume_adj'] / df['volume_sma_20']
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    
    print(f"Feature calculation complete. Total columns: {len(df.columns)}")
    return df


def build_dataset(csv_output=None):
    """
    Build the dataset using explicit raw/adjusted column families (Professional Pipeline)
    
    Args:
        csv_output (str, optional): Output CSV file path
    
    Returns:
        str: Path to the created CSV file
    """
    if csv_output is None:
        csv_output = DATA_OUTPUT_PATH
    
    print(f"Building professional dataset from {N_TICKERS} tickers...")
    print(f"Date range: {START} to {END}")
    print("Using explicit raw/adjusted column families")
    
    # Get S&P 500 tickers
    tickers = pd.read_csv(TICKERS_CSV)["Symbol"].tolist()[:N_TICKERS]
    
    rows = []
    needed_length = SEQ_LEN + HORIZON + 1
    feature_columns = None  # Will be set from first successful ticker
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
        
        try:
            # Download stock data with explicit raw/adjusted columns
            df = download_stock_data(ticker, START, END)
            
            if df is None or len(df) < needed_length:
                print(f"Skipping {ticker}: insufficient data")
                continue
            
            # Calculate all features using appropriate data sources
            df = calculate_all_features(df)
            df.dropna(inplace=True)
            
            if len(df) < needed_length:
                print(f"Skipping {ticker}: insufficient data after feature calculation")
                continue
            
            # Get feature columns from first successful ticker
            if feature_columns is None:
                # Exclude metadata columns, keep all feature columns
                exclude_cols = ['Date']
                feature_columns = [col for col in df.columns if col not in exclude_cols]
                print(f"Using {len(feature_columns)} features per timestep:")
                
                # Show data lineage
                raw_features = [col for col in feature_columns if col.endswith('_raw')]
                adj_features = [col for col in feature_columns if col.endswith('_adj')]
                pattern_features = [col for col in feature_columns if col in PATTERNS]
                indicator_features = [col for col in feature_columns if col not in raw_features + adj_features + pattern_features]
                
                print(f"  - Raw OHLCV features: {len(raw_features)}")
                print(f"  - Adjusted OHLCV features: {len(adj_features)}")
                print(f"  - Candlestick patterns: {len(pattern_features)} (from raw data)")
                print(f"  - Technical indicators: {len(indicator_features)} (from adjusted data)")
            
            # Create sliding windows
            for j in range(len(df) - needed_length):
                window = df.iloc[j:j+SEQ_LEN]
                # Use adjusted close for label calculation (economic accuracy)
                current_close = df['Close_adj'].iloc[j+SEQ_LEN-1]
                future_close = df['Close_adj'].iloc[j+SEQ_LEN+HORIZON-1]
                
                # Calculate percentage change using adjusted close
                pct_change = (future_close - current_close) / current_close
                label = label_class(pct_change)
                
                # Extract all features
                features = window[feature_columns].values.flatten()
                
                # Create row: [ticker, features..., label]
                row = [ticker] + features.tolist() + [label]
                rows.append(row)
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if not rows:
        raise ValueError("No data was successfully processed")
    
    # Create column names with timestep suffixes
    columns = []
    for t in range(SEQ_LEN):
        for feature in feature_columns:
            columns.append(f"{feature}_t{t}")
    
    # Create DataFrame and save
    final_columns = ['Ticker'] + columns + ['Label']
    df_final = pd.DataFrame(rows, columns=final_columns)
    
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Label distribution:")
    print(df_final['Label'].value_counts().sort_index())
    
    # Save dataset
    df_final.to_csv(csv_output, index=False)
    print(f"Dataset saved to: {csv_output}")
    
    # Create detailed feature summary
    summary_file = csv_output.replace('.csv', '_feature_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Professional Dataset Feature Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features per timestep: {len(feature_columns)}\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"Total feature dimensions: {len(feature_columns) * SEQ_LEN}\n\n")
        
        f.write("Data Source Strategy:\n")
        f.write("- Candlestick patterns: Calculated on raw OHLC (traditional visual analysis)\n")
        f.write("- Technical indicators: Calculated on adjusted OHLC (economic accuracy)\n")
        f.write("- Returns & volatility: Calculated on adjusted data (essential for modeling)\n")
        f.write("- Price labels: Based on adjusted close (real economic returns)\n\n")
        
        # Categorize features by source
        raw_features = [f for f in feature_columns if f.endswith('_raw')]
        adj_features = [f for f in feature_columns if f.endswith('_adj')]
        pattern_features = [f for f in feature_columns if f in PATTERNS]
        other_features = [f for f in feature_columns if f not in raw_features + adj_features + pattern_features]
        
        f.write("Feature Categories:\n\n")
        
        f.write(f"Raw OHLCV Features ({len(raw_features)}):\n")
        for i, feature in enumerate(raw_features, 1):
            f.write(f"  {i:2d}. {feature}\n")
        
        f.write(f"\nAdjusted OHLCV Features ({len(adj_features)}):\n")
        for i, feature in enumerate(adj_features, 1):
            f.write(f"  {i:2d}. {feature}\n")
        
        f.write(f"\nCandlestick Patterns ({len(pattern_features)}) - from raw data:\n")
        for i, feature in enumerate(pattern_features, 1):
            f.write(f"  {i:2d}. {feature}\n")
        
        f.write(f"\nTechnical Indicators ({len(other_features)}) - from adjusted data:\n")
        for i, feature in enumerate(other_features, 1):
            f.write(f"  {i:2d}. {feature}\n")
    
    print(f"Feature summary saved to: {summary_file}")
    
    return csv_output


def load_recent_data(ticker, days=10):
    """
    Load recent data for a specific ticker for prediction (Professional Pipeline)
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days of historical data to load
        
    Returns:
        pd.DataFrame: DataFrame with recent data and all features
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days+20)).strftime('%Y-%m-%d')  # Extra days for indicators
    
    # Download data with explicit raw/adjusted columns
    df = download_stock_data(ticker, start_date, end_date)
    
    if df is None:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Calculate all features using professional approach
    df = calculate_all_features(df)
    
    # Return the most recent 'days' worth of data
    return df.tail(days)


def compare_adjusted_vs_raw_analysis(ticker="AAPL", days=252):
    """
    Compare raw vs adjusted data impact using professional pipeline
    
    Args:
        ticker (str): Ticker to analyze
        days (int): Number of days to analyze
        
    Returns:
        dict: Comparison analysis
    """
    print(f"Analyzing {ticker}: Professional Raw vs Adjusted Comparison")
    
    # Download data with explicit columns
    df = download_stock_data(ticker, 
                           (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                           datetime.now().strftime('%Y-%m-%d'))
    
    if df is None:
        return None
    
    # Calculate returns using both raw and adjusted data
    df['raw_return'] = df['Close_raw'].pct_change()
    df['adj_return'] = df['Close_adj'].pct_change()
    
    # Calculate cumulative returns
    df['raw_cumulative'] = (1 + df['raw_return']).cumprod()
    df['adj_cumulative'] = (1 + df['adj_return']).cumprod()
    
    # Calculate technical indicators on both
    df['raw_rsi'] = talib.RSI(df['Close_raw'], timeperiod=14)
    df['adj_rsi'] = talib.RSI(df['Close_adj'], timeperiod=14)
    
    # Calculate adjustment factor for analysis
    df['adjustment_factor'] = df['Close_adj'] / df['Close_raw']
    
    # Analysis
    analysis = {
        'ticker': ticker,
        'days_analyzed': len(df),
        'raw_total_return': df['raw_cumulative'].iloc[-1] - 1,
        'adj_total_return': df['adj_cumulative'].iloc[-1] - 1,
        'return_difference': abs(df['adj_cumulative'].iloc[-1] - df['raw_cumulative'].iloc[-1]),
        'avg_adjustment_factor': df['adjustment_factor'].mean(),
        'max_adjustment_factor': df['adjustment_factor'].max(),
        'min_adjustment_factor': df['adjustment_factor'].min(),
        'adjustment_volatility': df['adjustment_factor'].std(),
        'rsi_correlation': df['raw_rsi'].corr(df['adj_rsi']),
        'price_correlation': df['Close_raw'].corr(df['Close_adj']),
        'return_correlation': df['raw_return'].corr(df['adj_return'])
    }
    
    print(f"Analysis Results for {ticker}:")
    print(f"  Raw total return: {analysis['raw_total_return']:.4f}")
    print(f"  Adjusted total return: {analysis['adj_total_return']:.4f}")
    print(f"  Return difference: {analysis['return_difference']:.4f}")
    print(f"  Average adjustment factor: {analysis['avg_adjustment_factor']:.4f}")
    print(f"  Adjustment volatility: {analysis['adjustment_volatility']:.4f}")
    print(f"  RSI correlation: {analysis['rsi_correlation']:.4f}")
    print(f"  Price correlation: {analysis['price_correlation']:.4f}")
    print(f"  Return correlation: {analysis['return_correlation']:.4f}")
    
    return analysis 