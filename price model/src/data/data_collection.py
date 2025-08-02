"""
Unified Data Collection with Adjusted Close Integration
Combines enhanced features with streamlined functionality
"""
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import signal
import sys
from datetime import datetime, timedelta
from config.config import (
    TICKERS_CSV, N_TICKERS, START, END, SEQ_LEN, HORIZON, 
    PATTERNS, DATA_OUTPUT_PATH, USE_RAW_COLS, USE_ADJ_COLS
)
from src.utils.helpers import label_class
import os

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    print(f"\n⚠️  Interrupt signal received. Saving collected data and shutting down gracefully...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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


def validate_and_clean_data(df):
    """
    Validate and clean data before feature calculation
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()
    
    # Remove any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove any negative prices (invalid data)
    price_columns = ['Open_raw', 'High_raw', 'Low_raw', 'Close_raw', 
                    'Open_adj', 'High_adj', 'Low_adj', 'Close_adj']
    for col in price_columns:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    
    # Remove any negative volumes
    volume_columns = ['Volume_raw', 'Volume_adj']
    for col in volume_columns:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    
    # Forward fill any missing values (but not too many consecutive)
    df = df.fillna(method='ffill', limit=5)
    
    # Remove rows with too many missing values
    threshold = len(df.columns) * 0.1  # 10% threshold
    df = df.dropna(thresh=threshold)
    
    # Ensure data types are correct
    for col in price_columns + volume_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def verify_data_quality(df, ticker):
    """
    Verify data quality and identify potential issues
    
    Args:
        df (pd.DataFrame): DataFrame to verify
        ticker (str): Ticker symbol for logging
    """
    print(f"\n🔍 VERIFYING DATA QUALITY FOR {ticker}")
    print("=" * 50)
    
    # Check basic data
    print(f"📊 Data shape: {df.shape}")
    print(f"📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"📊 Total days: {len(df)}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    print(f"📊 Missing values per column:")
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  ⚠️  {col}: {missing} missing values")
    
    # Check price data quality
    price_issues = []
    for col in ['Open_raw', 'High_raw', 'Low_raw', 'Close_raw', 'Open_adj', 'High_adj', 'Low_adj', 'Close_adj']:
        if col in df.columns:
            # Check for negative prices
            negative_prices = (df[col] <= 0).sum()
            if negative_prices > 0:
                price_issues.append(f"{col}: {negative_prices} negative/zero prices")
            
            # Check for extreme values
            mean_price = df[col].mean()
            extreme_values = ((df[col] > mean_price * 10) | (df[col] < mean_price * 0.1)).sum()
            if extreme_values > 0:
                price_issues.append(f"{col}: {extreme_values} extreme values")
    
    if price_issues:
        print(f"⚠️  Price data issues:")
        for issue in price_issues:
            print(f"  {issue}")
    else:
        print("✅ Price data looks good")
    
    # Check OHLC relationships
    if all(col in df.columns for col in ['High_raw', 'Low_raw', 'Open_raw', 'Close_raw']):
        invalid_ohlc = 0
        for _, row in df.iterrows():
            if (row['High_raw'] < row['Low_raw'] or 
                row['High_raw'] < row['Open_raw'] or 
                row['High_raw'] < row['Close_raw'] or
                row['Low_raw'] > row['Open_raw'] or 
                row['Low_raw'] > row['Close_raw']):
                invalid_ohlc += 1
        
        if invalid_ohlc > 0:
            print(f"⚠️  OHLC relationship issues: {invalid_ohlc} invalid rows")
        else:
            print("✅ OHLC relationships are valid")
    
    # Check for sufficient data for patterns
    if len(df) < 20:
        print(f"⚠️  Insufficient data for pattern detection: {len(df)} days (need at least 20)")
    else:
        print(f"✅ Sufficient data for pattern detection: {len(df)} days")
    
    # Sample data values
    print(f"\n📊 Sample data values:")
    for col in ['Open_raw', 'High_raw', 'Low_raw', 'Close_raw']:
        if col in df.columns:
            sample_values = df[col].head(3).values
            print(f"  {col}: {sample_values}")
    
    print("=" * 50)


def calculate_swing_trading_features(df):
    """
    Calculate features optimized for swing trading (1-3 days)
    Focuses on 1-day return prediction with swing trading indicators
    
    Args:
        df (pd.DataFrame): DataFrame with explicit raw and adjusted columns
        
    Returns:
        pd.DataFrame: DataFrame with swing trading optimized features
    """
    df = df.copy()
    
    # Validate and clean data first
    df = validate_and_clean_data(df)
    
    print("📈 Calculating swing trading features (1-3 day focus)...")
    print(f"📊 Input data shape: {df.shape}")
    print(f"📊 Data columns: {list(df.columns)}")
    
    # Convert to float64 for TA-Lib compatibility
    close_adj = df['Close_adj'].values.astype(np.float64)
    high_adj = df['High_adj'].values.astype(np.float64)
    low_adj = df['Low_adj'].values.astype(np.float64)
    volume_adj = df['Volume_adj'].values.astype(np.float64)
    
    # 1. TREND ANALYSIS (Swing traders focus on medium-term trends)
    print("📊 Computing trend analysis...")
    trend_features = []
    
    try:
        # Medium-term moving averages (swing traders use 10, 20, 50)
        df['sma_10'] = talib.SMA(close_adj, timeperiod=10)
        df['sma_20'] = talib.SMA(close_adj, timeperiod=20)
        df['sma_50'] = talib.SMA(close_adj, timeperiod=50)
        df['ema_10'] = talib.EMA(close_adj, timeperiod=10)
        df['ema_20'] = talib.EMA(close_adj, timeperiod=20)
        
        # Trend strength and position
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['above_sma_20'] = (df['Close_adj'] > df['sma_20']).astype(int)
        df['above_sma_50'] = (df['Close_adj'] > df['sma_50']).astype(int)
        df['price_vs_ema_10'] = (df['Close_adj'] - df['ema_10']) / df['ema_10']
        
        # Golden/Death Cross signals (swing traders love these)
        df['golden_cross'] = (df['sma_10'] > df['sma_50']) & (df['sma_10'].shift(1) <= df['sma_50'].shift(1))
        df['death_cross'] = (df['sma_10'] < df['sma_50']) & (df['sma_10'].shift(1) >= df['sma_50'].shift(1))
        
        trend_features = ['sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'trend_strength', 
                        'above_sma_20', 'above_sma_50', 'price_vs_ema_10', 'golden_cross', 'death_cross']
        print(f"✅ Trend features added: {len(trend_features)} features")
        print(f"   📈 SMA periods: 10, 20, 50")
        print(f"   📈 EMA periods: 10, 20")
        print(f"   📈 Trend strength calculation: (SMA20 - SMA50) / SMA50")
        print(f"   📈 Golden/Death cross detection")
        
    except Exception as e:
        print(f"⚠️  Error calculating trend indicators: {e}")
    
    # 2. MOMENTUM (Swing traders use longer periods)
    print("⚡ Computing momentum indicators...")
    momentum_features = []
    
    try:
        # RSI with swing trading focus
        df['rsi_14'] = talib.RSI(close_adj, timeperiod=14)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_neutral'] = ((df['rsi_14'] >= 30) & (df['rsi_14'] <= 70)).astype(int)
        
        # MACD (swing traders use for trend confirmation)
        macd, macd_signal, macd_hist = talib.MACD(close_adj)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_bullish'] = (macd > macd_signal).astype(int)
        df['macd_bearish'] = (macd < macd_signal).astype(int)
        
        # Stochastic (swing traders use for entry/exit)
        stoch_k, stoch_d = talib.STOCH(high_adj, low_adj, close_adj)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_oversold'] = (stoch_k < 20).astype(int)
        df['stoch_overbought'] = (stoch_k > 80).astype(int)
        
        momentum_features = ['rsi_14', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
                           'macd', 'macd_signal', 'macd_histogram', 'macd_bullish', 'macd_bearish',
                           'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought']
        print(f"✅ Momentum features added: {len(momentum_features)} features")
        print(f"   ⚡ RSI: 14-period with oversold/overbought zones")
        print(f"   ⚡ MACD: 12,26,9 with bullish/bearish signals")
        print(f"   ⚡ Stochastic: 14,3 with oversold/overbought zones")
        
    except Exception as e:
        print(f"⚠️  Error calculating momentum indicators: {e}")
    
    # 3. VOLATILITY (Swing traders focus on volatility for position sizing)
    print("📊 Computing volatility analysis...")
    volatility_features = []
    
    try:
        # ATR for volatility measurement
        df['atr_14'] = talib.ATR(high_adj, low_adj, close_adj, timeperiod=14)
        df['atr_high'] = (df['atr_14'] > df['atr_14'].rolling(20).mean() * 1.5).astype(int)
        df['atr_low'] = (df['atr_14'] < df['atr_14'].rolling(20).mean() * 0.5).astype(int)
        
        # Bollinger Bands for volatility and mean reversion
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_adj, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['Close_adj'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < 0.1).astype(int)
        df['bb_expansion'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] > 0.2).astype(int)
        
        # Price near Bollinger Bands (swing trading signals)
        df['near_bb_upper'] = (df['Close_adj'] > df['bb_upper'] * 0.98).astype(int)
        df['near_bb_lower'] = (df['Close_adj'] < df['bb_lower'] * 1.02).astype(int)
        
        volatility_features = ['atr_14', 'atr_high', 'atr_low', 'bb_upper', 'bb_middle', 'bb_lower',
                             'bb_position', 'bb_squeeze', 'bb_expansion', 'near_bb_upper', 'near_bb_lower']
        print(f"✅ Volatility features added: {len(volatility_features)} features")
        print(f"   📊 ATR: 14-period with high/low volatility flags")
        print(f"   📊 Bollinger Bands: 20-period with squeeze/expansion detection")
        print(f"   📊 BB Position: Price position within bands")
        
    except Exception as e:
        print(f"⚠️  Error calculating volatility indicators: {e}")
    
    # 4. SUPPORT/RESISTANCE (Swing traders focus on key levels)
    print("🎯 Computing support/resistance levels...")
    support_resistance_features = []
    
    try:
        # Key levels (swing traders use 20-day lookback)
        df['resistance_20'] = df['High_adj'].rolling(20).max()
        df['support_20'] = df['Low_adj'].rolling(20).min()
        df['price_position'] = (df['Close_adj'] - df['support_20']) / (df['resistance_20'] - df['support_20'])
        
        # Near support/resistance (swing trading entry/exit points)
        df['near_resistance'] = (df['Close_adj'] > df['resistance_20'] * 0.98).astype(int)
        df['near_support'] = (df['Close_adj'] < df['support_20'] * 1.02).astype(int)
        
        # Breakout signals (swing traders love breakouts)
        df['breakout_up'] = (df['Close_adj'] > df['resistance_20'].shift(1)).astype(int)
        df['breakout_down'] = (df['Close_adj'] < df['support_20'].shift(1)).astype(int)
        
        support_resistance_features = ['resistance_20', 'support_20', 'price_position', 'near_resistance', 
                                    'near_support', 'breakout_up', 'breakout_down']
        print(f"✅ Support/Resistance features added: {len(support_resistance_features)} features")
        print(f"   🎯 20-day resistance/support levels")
        print(f"   🎯 Price position within range")
        print(f"   🎯 Breakout detection (up/down)")
        
    except Exception as e:
        print(f"⚠️  Error calculating support/resistance: {e}")
    
    # 5. VOLUME (Swing traders use volume for confirmation)
    print("📊 Computing volume analysis...")
    volume_features = []
    
    try:
        # Volume confirmation
        df['volume_sma_20'] = talib.SMA(volume_adj, timeperiod=20)
        df['volume_ratio'] = volume_adj / df['volume_sma_20']
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        df['low_volume'] = (df['volume_ratio'] < 0.5).astype(int)
        
        # OBV for trend confirmation
        df['obv'] = talib.OBV(close_adj, volume_adj)
        
        # Volume-price relationship
        df['mfi'] = talib.MFI(high_adj, low_adj, close_adj, volume_adj, timeperiod=14)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        
        volume_features = ['volume_sma_20', 'volume_ratio', 'high_volume', 'low_volume', 'obv', 'mfi', 
                         'mfi_oversold', 'mfi_overbought']
        print(f"✅ Volume features added: {len(volume_features)} features")
        print(f"   📊 Volume SMA: 20-period average")
        print(f"   📊 Volume ratio: Current vs 20-day average")
        print(f"   📊 OBV: On Balance Volume for trend confirmation")
        print(f"   📊 MFI: Money Flow Index (14-period)")
        
    except Exception as e:
        print(f"⚠️  Error calculating volume indicators: {e}")
    
    # 6. PRICE ACTION (Swing traders focus on multi-day patterns)
    print("💹 Computing price action features...")
    price_action_features = []
    
    # Returns (swing traders focus on 1-3 day moves)
    df['daily_return'] = df['Close_adj'].pct_change()
    df['return_3d'] = df['Close_adj'].pct_change(3)
    df['return_5d'] = df['Close_adj'].pct_change(5)
    
    # Gap analysis (swing traders use gaps for entry/exit)
    df['gap'] = (df['Open_adj'] - df['Close_adj'].shift(1)) / df['Close_adj'].shift(1)
    df['gap_up'] = (df['gap'] > 0.01).astype(int)
    df['gap_down'] = (df['gap'] < -0.01).astype(int)
    
    # Price relationships
    df['hl_ratio'] = (df['High_adj'] - df['Low_adj']) / df['Close_adj']
    df['oc_ratio'] = (df['Open_adj'] - df['Close_adj']) / df['Close_adj']
    
    # Swing trading specific patterns
    df['higher_high'] = (df['High_adj'] > df['High_adj'].shift(1)).astype(int)
    df['lower_low'] = (df['Low_adj'] < df['Low_adj'].shift(1)).astype(int)
    df['higher_low'] = (df['Low_adj'] > df['Low_adj'].shift(1)).astype(int)
    df['lower_high'] = (df['High_adj'] < df['High_adj'].shift(1)).astype(int)
    
    price_action_features = ['daily_return', 'return_3d', 'return_5d', 'gap', 'gap_up', 'gap_down',
                           'hl_ratio', 'oc_ratio', 'higher_high', 'lower_low', 'higher_low', 'lower_high']
    print(f"✅ Price action features added: {len(price_action_features)} features")
    print(f"   💹 Returns: 1-day, 3-day, 5-day")
    print(f"   💹 Gap analysis: Up/down gap detection")
    print(f"   💹 Price relationships: High-low, Open-close ratios")
    print(f"   💹 Swing patterns: Higher highs/lows, Lower highs/lows")
    
    # 7. CANDLESTICK PATTERNS (Debug this section)
    print("🕯️ Computing candlestick patterns...")
    pattern_features = []
    
    try:
        # Key candlestick patterns for swing trading
        key_patterns = [
            'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLHANGINGMAN', 'CDLSHOOTINGSTAR',
            'CDLDOJI', 'CDLENGULFING', 'CDLPIERCING', 'CDLDARKCLOUDCOVER',
            'CDLMORNINGSTAR', 'CDLEVENINGSTAR'
        ]
        
        open_raw = df['Open_raw'].values.astype(np.float64)
        high_raw = df['High_raw'].values.astype(np.float64)
        low_raw = df['Low_raw'].values.astype(np.float64)
        close_raw = df['Close_raw'].values.astype(np.float64)
        
        print(f"🕯️ Raw data sample - Open: {open_raw[:5]}, High: {high_raw[:5]}, Low: {low_raw[:5]}, Close: {close_raw[:5]}")
        
        for pattern in key_patterns:
            try:
                pattern_result = getattr(talib, pattern)(
                    open_raw, high_raw, low_raw, close_raw
                )
                # Convert numpy array to pandas series and then to boolean
                df[f'pattern_{pattern}'] = (pattern_result != 0).astype(int)
                pattern_count = df[f'pattern_{pattern}'].sum()
                print(f"🕯️ {pattern}: {pattern_count} patterns found")
                pattern_features.append(f'pattern_{pattern}')
            except Exception as e:
                print(f"⚠️  Error calculating {pattern}: {e}")
                df[f'pattern_{pattern}'] = 0
                pattern_features.append(f'pattern_{pattern}')
        
        print(f"✅ Candlestick patterns added: {len(pattern_features)} features")
        print(f"   🕯️ 10 key patterns for swing trading")
        print(f"   🕯️ Calculated from raw OHLC data")
        
    except Exception as e:
        print(f"⚠️  Error calculating candlestick patterns: {e}")
    
    # Summary
    total_features = len(trend_features) + len(momentum_features) + len(volatility_features) + \
                   len(support_resistance_features) + len(volume_features) + len(price_action_features) + \
                   len(pattern_features)
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"📈 Trend features: {len(trend_features)}")
    print(f"⚡ Momentum features: {len(momentum_features)}")
    print(f"📊 Volatility features: {len(volatility_features)}")
    print(f"🎯 Support/Resistance features: {len(support_resistance_features)}")
    print(f"📊 Volume features: {len(volume_features)}")
    print(f"💹 Price action features: {len(price_action_features)}")
    print(f"🕯️ Candlestick patterns: {len(pattern_features)}")
    print(f"📊 TOTAL FEATURES: {total_features}")
    print(f"📊 Final data shape: {df.shape}")
    
    print(f"✅ Swing trading features complete. Total features: {len(df.columns)}")
    return df


def calculate_focused_features(df):
    """
    DEPRECATED: Use calculate_swing_trading_features instead
    This function is kept for backward compatibility but should not be used
    """
    print("⚠️  DEPRECATED: Use calculate_swing_trading_features instead")
    return calculate_swing_trading_features(df)


def calculate_all_features(df):
    """
    DEPRECATED: Use calculate_swing_trading_features instead
    This function is kept for backward compatibility but should not be used
    """
    print("⚠️  DEPRECATED: Use calculate_swing_trading_features instead")
    return calculate_swing_trading_features(df)


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
            
            # Validate and clean data
            df = validate_and_clean_data(df)
            
            # Verify data quality (detailed debugging)
            verify_data_quality(df, ticker)

            # Calculate all features using appropriate data sources
            df = calculate_swing_trading_features(df)  # Use swing trading features
            
            if df is None or len(df) < needed_length:
                print(f"Skipping {ticker}: insufficient data")
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


def build_dataset_incremental(csv_output=None, save_interval=100):
    """
    Build the dataset incrementally, saving every N tickers and allowing graceful interruption
    
    Args:
        csv_output (str, optional): Output CSV file path
        save_interval (int): Save data every N tickers (default: 100)
    
    Returns:
        str: Path to the created CSV file
    """
    global shutdown_requested
    
    if csv_output is None:
        csv_output = DATA_OUTPUT_PATH
    
    print(f"Building professional dataset incrementally from {N_TICKERS} tickers...")
    print(f"Date range: {START} to {END}")
    print(f"Saving data every {save_interval} tickers")
    print("Press Ctrl+C to stop gracefully and save collected data")
    print("Using explicit raw/adjusted column families")
    
    # Get S&P 500 tickers
    tickers = pd.read_csv(TICKERS_CSV)["Symbol"].tolist()[:N_TICKERS]
    
    all_rows = []
    needed_length = SEQ_LEN + HORIZON + 1
    feature_columns = None  # Will be set from first successful ticker
    
    # Create backup file path
    backup_file = csv_output.replace('.csv', '_backup.csv')
    
    for i, ticker in enumerate(tickers):
        # Check for graceful shutdown
        if shutdown_requested:
            print(f"\n🛑 Graceful shutdown requested. Saving {len(all_rows)} collected samples...")
            break
            
        if i % 50 == 0:
            print(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
        
        try:
            # Download stock data with explicit raw/adjusted columns
            df = download_stock_data(ticker, START, END)
            
            if df is None or len(df) < needed_length:
                print(f"Skipping {ticker}: insufficient data")
                continue
            
            # Validate and clean data
            df = validate_and_clean_data(df)
            
            # Verify data quality (detailed debugging)
            verify_data_quality(df, ticker)

            # Calculate all features using appropriate data sources
            df = calculate_swing_trading_features(df)  # Use swing trading features
            
            if df is None or len(df) < needed_length:
                print(f"Skipping {ticker}: insufficient data")
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
            ticker_rows = []
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
                ticker_rows.append(row)
            
            # Add ticker rows to all rows
            all_rows.extend(ticker_rows)
            
            # Save incrementally every save_interval tickers
            if (i + 1) % save_interval == 0:
                print(f"\n💾 Saving intermediate data after {i+1} tickers ({len(all_rows)} samples)...")
                _save_incremental_data(all_rows, feature_columns, backup_file, i+1)
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if not all_rows:
        raise ValueError("No data was successfully processed")
    
    # Final save
    print(f"\n💾 Saving final dataset with {len(all_rows)} samples...")
    final_path = _save_incremental_data(all_rows, feature_columns, csv_output, len(tickers))
    
    # Create detailed feature summary
    summary_file = csv_output.replace('.csv', '_feature_summary.txt')
    _create_feature_summary(summary_file, feature_columns, len(all_rows))
    
    print(f"✅ Dataset saved to: {final_path}")
    print(f"📊 Total samples collected: {len(all_rows)}")
    
    return final_path


def _save_incremental_data(rows, feature_columns, output_path, tickers_processed):
    """
    Save incremental data to CSV
    
    Args:
        rows (list): List of data rows
        feature_columns (list): Feature column names
        output_path (str): Output file path
        tickers_processed (int): Number of tickers processed
    
    Returns:
        str: Path to saved file
    """
    # Create column names with timestep suffixes
    columns = []
    for t in range(SEQ_LEN):
        for feature in feature_columns:
            columns.append(f"{feature}_t{t}")
    
    # Create DataFrame and save
    final_columns = ['Ticker'] + columns + ['Label']
    df_final = pd.DataFrame(rows, columns=final_columns)
    
    print(f"  Dataset shape: {df_final.shape}")
    print(f"  Label distribution:")
    print(df_final['Label'].value_counts().sort_index())
    
    # Save dataset
    df_final.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return output_path


def _create_feature_summary(summary_file, feature_columns, total_samples):
    """
    Create detailed feature summary
    
    Args:
        summary_file (str): Path to summary file
        feature_columns (list): Feature column names
        total_samples (int): Total number of samples
    """
    with open(summary_file, 'w') as f:
        f.write("Professional Dataset Feature Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features per timestep: {len(feature_columns)}\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"Total feature dimensions: {len(feature_columns) * SEQ_LEN}\n")
        f.write(f"Total samples: {total_samples}\n\n")
        
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


def build_dataset_smart_update(csv_output=None, days_back=30):
    """
    Smart dataset update that only collects recent data for new tickers and updates existing ones
    
    Args:
        csv_output (str, optional): Output CSV file path
        days_back (int): Number of days of recent data to collect for updates
    
    Returns:
        str: Path to the created CSV file
    """
    if csv_output is None:
        csv_output = DATA_OUTPUT_PATH
    
    print(f"🔄 Smart dataset update - collecting recent data only...")
    print(f"📅 Collecting {days_back} days of recent data for updates")
    
    # Check if existing dataset exists
    existing_data = None
    existing_tickers = set()
    
    if os.path.exists(csv_output):
        print(f"📁 Found existing dataset: {csv_output}")
        existing_data = pd.read_csv(csv_output)
        existing_tickers = set(existing_data['Ticker'].unique())
        print(f"📊 Existing dataset has {len(existing_tickers)} tickers and {len(existing_data)} samples")
    elif os.path.exists(csv_output.replace('.csv', '_backup.csv')):
        backup_file = csv_output.replace('.csv', '_backup.csv')
        print(f"📁 Found backup dataset: {backup_file}")
        existing_data = pd.read_csv(backup_file)
        existing_tickers = set(existing_data['Ticker'].unique())
        print(f"📊 Backup dataset has {len(existing_tickers)} tickers and {len(existing_data)} samples")
        # Move backup to main file
        import shutil
        shutil.move(backup_file, csv_output)
        print(f"✅ Moved backup to main dataset: {csv_output}")
    else:
        print("📁 No existing dataset found - will create new one")
    
    # Get current S&P 500 tickers
    tickers = pd.read_csv(TICKERS_CSV)["Symbol"].tolist()[:N_TICKERS]
    current_tickers = set(tickers)
    
    # Identify new tickers
    new_tickers = current_tickers - existing_tickers
    removed_tickers = existing_tickers - current_tickers
    
    print(f"\n📈 Ticker Analysis:")
    print(f"  - Current S&P 500 tickers: {len(current_tickers)}")
    print(f"  - Existing dataset tickers: {len(existing_tickers)}")
    print(f"  - New tickers to add: {len(new_tickers)}")
    print(f"  - Removed tickers: {len(removed_tickers)}")
    
    if new_tickers:
        print(f"  - New tickers: {', '.join(sorted(new_tickers))}")
    if removed_tickers:
        print(f"  - Removed tickers: {', '.join(sorted(removed_tickers))}")
    
    # Calculate date range for recent data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back + 20)).strftime('%Y-%m-%d')  # Extra days for indicators
    
    all_rows = []
    feature_columns = None
    
    # Process new tickers (full historical data)
    if new_tickers:
        print(f"\n🆕 Collecting full historical data for {len(new_tickers)} new tickers...")
        for i, ticker in enumerate(new_tickers):
            print(f"Processing new ticker {i+1}/{len(new_tickers)}: {ticker}")
            
            try:
                # Download full historical data for new tickers
                df = download_stock_data(ticker, START, END)
                
                if df is None or len(df) < SEQ_LEN + HORIZON + 1:
                    print(f"Skipping {ticker}: insufficient data")
                    continue
                
                df = validate_and_clean_data(df)
                verify_data_quality(df, ticker)
                df = calculate_swing_trading_features(df)
                
                if df is None or len(df) < SEQ_LEN + HORIZON + 1:
                    print(f"Skipping {ticker}: insufficient data after processing")
                    continue
                
                # Set feature columns from first successful ticker
                if feature_columns is None:
                    exclude_cols = ['Date']
                    feature_columns = [col for col in df.columns if col not in exclude_cols]
                
                # Create sliding windows for new ticker
                ticker_rows = _create_sliding_windows(df, ticker, feature_columns)
                all_rows.extend(ticker_rows)
                
            except Exception as e:
                print(f"Error processing new ticker {ticker}: {e}")
                continue
    
    # Process existing tickers (recent data only)
    if existing_tickers:
        print(f"\n🔄 Updating {len(existing_tickers)} existing tickers with recent data...")
        
        # Remove data for removed tickers
        if removed_tickers:
            print(f"🗑️ Removing data for {len(removed_tickers)} removed tickers...")
            existing_data = existing_data[~existing_data['Ticker'].isin(removed_tickers)]
        
        # Get feature columns from existing data if not set
        if feature_columns is None:
            # Extract feature columns from existing data
            exclude_cols = ['Ticker', 'Label']
            all_cols = existing_data.columns.tolist()
            feature_columns = [col for col in all_cols if col not in exclude_cols]
            # Convert back to original feature names (remove timestep suffixes)
            base_features = set()
            for col in feature_columns:
                if '_t' in col:
                    base_feature = col.rsplit('_t', 1)[0]
                    base_features.add(base_feature)
                else:
                    base_features.add(col)
            feature_columns = list(base_features)
        
        # Update existing tickers with recent data
        for i, ticker in enumerate(existing_tickers):
            if ticker in removed_tickers:
                continue
                
            print(f"Updating ticker {i+1}/{len(existing_tickers)}: {ticker}")
            
            try:
                # Download recent data only
                df = download_stock_data(ticker, start_date, end_date)
                
                if df is None or len(df) < SEQ_LEN + HORIZON + 1:
                    print(f"Skipping {ticker}: insufficient recent data")
                    continue
                
                df = validate_and_clean_data(df)
                verify_data_quality(df, ticker)
                df = calculate_swing_trading_features(df)
                
                if df is None or len(df) < SEQ_LEN + HORIZON + 1:
                    print(f"Skipping {ticker}: insufficient data after processing")
                    continue
                
                # Create sliding windows for recent data
                ticker_rows = _create_sliding_windows(df, ticker, feature_columns)
                all_rows.extend(ticker_rows)
                
            except Exception as e:
                print(f"Error updating ticker {ticker}: {e}")
                continue
    
    if not all_rows:
        raise ValueError("No data was successfully processed")
    
    # Combine with existing data
    if existing_data is not None and len(existing_data) > 0:
        print(f"\n🔗 Combining new data with existing dataset...")
        print(f"  - New samples: {len(all_rows)}")
        print(f"  - Existing samples: {len(existing_data)}")
        
        # Create new data DataFrame
        columns = []
        for t in range(SEQ_LEN):
            for feature in feature_columns:
                columns.append(f"{feature}_t{t}")
        final_columns = ['Ticker'] + columns + ['Label']
        new_df = pd.DataFrame(all_rows, columns=final_columns)
        
        # Combine datasets
        combined_df = pd.concat([existing_data, new_df], ignore_index=True)
        print(f"  - Combined samples: {len(combined_df)}")
        
        # Remove duplicates (keep newer data)
        combined_df = combined_df.drop_duplicates(subset=['Ticker'], keep='last')
        print(f"  - After deduplication: {len(combined_df)}")
        
        final_df = combined_df
    else:
        # Create new dataset
        columns = []
        for t in range(SEQ_LEN):
            for feature in feature_columns:
                columns.append(f"{feature}_t{t}")
        final_columns = ['Ticker'] + columns + ['Label']
        final_df = pd.DataFrame(all_rows, columns=final_columns)
    
    # Save final dataset
    print(f"\n💾 Saving updated dataset...")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Label distribution:")
    print(final_df['Label'].value_counts().sort_index())
    
    final_df.to_csv(csv_output, index=False)
    print(f"✅ Updated dataset saved to: {csv_output}")
    
    # Create feature summary
    summary_file = csv_output.replace('.csv', '_feature_summary.txt')
    _create_feature_summary(summary_file, feature_columns, len(final_df))
    
    return csv_output


def _create_sliding_windows(df, ticker, feature_columns):
    """
    Create sliding windows for a ticker
    
    Args:
        df (pd.DataFrame): Processed dataframe with features
        ticker (str): Ticker symbol
        feature_columns (list): Feature column names
    
    Returns:
        list: List of data rows
    """
    rows = []
    needed_length = SEQ_LEN + HORIZON + 1
    
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
    
    return rows


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
    
    # Validate and clean data
    df = validate_and_clean_data(df)

    # Calculate all features using professional approach
    df = calculate_swing_trading_features(df)
    
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
    
    # Validate and clean data
    df = validate_and_clean_data(df)

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