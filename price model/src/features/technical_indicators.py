"""
Technical Indicators and Enhanced Feature Engineering
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional


class TechnicalIndicators:
    """
    Calculate various technical indicators for enhanced feature engineering
    """
    
    def __init__(self):
        self.indicators = {}
    
    def add_price_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based technical indicators
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with price indicators added
        """
        df = df.copy()
        
        # Price ratios and relationships
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['oc_ratio'] = (df['Open'] - df['Close']) / df['Close']
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # Price position within the day's range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['price_position'] = df['price_position'].fillna(0.5)  # Handle when High == Low
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame, periods: List[int] = [5, 10, 14, 20]) -> pd.DataFrame:
        """
        Add momentum-based indicators
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            periods (List[int]): Periods for calculations
            
        Returns:
            pd.DataFrame: DataFrame with momentum indicators
        """
        df = df.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        for period in periods:
            # RSI (Relative Strength Index)
            df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            
            # Rate of Change
            df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
            
            # Williams %R
            df[f'willr_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
            
            # Commodity Channel Index
            df[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based indicators
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with volatility indicators
        """
        df = df.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range
        df['atr'] = talib.ATR(high, low, close)
        df['atr_ratio'] = df['atr'] / close
        
        # Historical volatility (rolling standard deviation)
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['Close'].rolling(period).std() / df['Close']
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with volume indicators
        """
        df = df.copy()
        close = df['Close'].values
        volume = df['Volume'].values
        high = df['High'].values
        low = df['Low'].values
        
        # Volume indicators
        df['obv'] = talib.OBV(close, volume)  # On-Balance Volume
        df['ad'] = talib.AD(high, low, close, volume)  # Accumulation/Distribution
        
        # Volume ratios
        df['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
        df['volume_ratio'] = volume / df['volume_sma_10']
        
        # Price-Volume relationship
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap'] = df['price_volume'].rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based indicators
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with trend indicators
        """
        df = df.copy()
        close = df['Close'].values
        
        # Moving averages
        periods = [5, 10, 20, 50]
        for period in periods:
            df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # Price relative to moving averages
            df[f'price_sma_{period}_ratio'] = close / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = close / df[f'ema_{period}']
        
        # Moving average crossovers
        df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['High'], df['Low'])
        df['sar_trend'] = (close > df['sar']).astype(int)
        
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with all indicators
        """
        print("Calculating price-based features...")
        df = self.add_price_based_features(df)
        
        print("Calculating momentum indicators...")
        df = self.add_momentum_indicators(df)
        
        print("Calculating volatility indicators...")
        df = self.add_volatility_indicators(df)
        
        print("Calculating volume indicators...")
        df = self.add_volume_indicators(df)
        
        print("Calculating trend indicators...")
        df = self.add_trend_indicators(df)
        
        # Fill NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get organized feature groups for analysis
        
        Returns:
            dict: Feature groups
        """
        return {
            'price_features': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'price_based': ['hl_ratio', 'oc_ratio', 'body_size', 'upper_shadow', 
                           'lower_shadow', 'price_position'],
            'momentum': [f'rsi_{p}' for p in [5, 10, 14, 20]] + 
                       [f'roc_{p}' for p in [5, 10, 14, 20]] + 
                       [f'willr_{p}' for p in [5, 10, 14, 20]] +
                       [f'cci_{p}' for p in [5, 10, 14, 20]] +
                       ['macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d'],
            'volatility': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                          'atr', 'atr_ratio'] + [f'volatility_{p}' for p in [5, 10, 20]],
            'volume': ['obv', 'ad', 'volume_sma_10', 'volume_ratio', 'price_volume', 'vwap'],
            'trend': [f'sma_{p}' for p in [5, 10, 20, 50]] + 
                     [f'ema_{p}' for p in [5, 10, 20, 50]] +
                     [f'price_sma_{p}_ratio' for p in [5, 10, 20, 50]] +
                     [f'price_ema_{p}_ratio' for p in [5, 10, 20, 50]] +
                     ['sma_5_20_cross', 'ema_5_20_cross', 'sar', 'sar_trend']
        }


def create_enhanced_features(df: pd.DataFrame, include_patterns: bool = True) -> pd.DataFrame:
    """
    Create enhanced feature set with technical indicators and candlestick patterns
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        include_patterns (bool): Whether to include candlestick patterns
        
    Returns:
        pd.DataFrame: Enhanced feature dataframe
    """
    # Calculate technical indicators
    indicator_calc = TechnicalIndicators()
    df_enhanced = indicator_calc.calculate_all_indicators(df)
    
    # Add candlestick patterns if requested
    if include_patterns:
        try:
            from src.utils.pattern_validator import calculate_patterns_with_validation
            from config.config import PATTERNS
            df_enhanced, pattern_info = calculate_patterns_with_validation(df_enhanced, PATTERNS)
        except ImportError:
            print("Warning: Pattern validator not available, skipping patterns")
    
    return df_enhanced


def get_feature_importance_analysis(df: pd.DataFrame, target_col: str = 'Label') -> pd.DataFrame:
    """
    Analyze feature importance using correlation and mutual information
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Feature importance analysis
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [target_col, 'Ticker']]
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Calculate correlations
        correlations = X.corrwith(y).abs()
        
        # Calculate mutual information
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        
        # Create analysis dataframe
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'correlation': correlations.values,
            'mutual_info': mi_scores
        })
        
        importance_df['combined_score'] = (
            importance_df['correlation'] * 0.5 + 
            importance_df['mutual_info'] * 0.5
        )
        
        return importance_df.sort_values('combined_score', ascending=False)
        
    except ImportError:
        print("Warning: sklearn not available for feature importance analysis")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the technical indicators
    try:
        from src.data.data_collection import load_recent_data
        
        df = load_recent_data("AAPL", days=100)
        df_enhanced = create_enhanced_features(df)
        
        print(f"Original features: {len(df.columns)}")
        print(f"Enhanced features: {len(df_enhanced.columns)}")
        print(f"Added {len(df_enhanced.columns) - len(df.columns)} new features")
        
        # Show feature groups
        indicator_calc = TechnicalIndicators()
        feature_groups = indicator_calc.get_feature_groups()
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df_enhanced.columns]
            print(f"{group_name}: {len(available_features)} features")
        
    except Exception as e:
        print(f"Error testing technical indicators: {e}") 