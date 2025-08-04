"""
Unified Data Collection with Adjusted Close Integration
Combines enhanced features with streamlined functionality

🔧 ENHANCED WITH SAFE INDICATOR FUNCTIONS
- Prevents NaN propagation from insufficient data
- Adaptive indicator periods based on data availability
- Comprehensive data quality validation
- Graceful handling of technical indicator failures
- Automatic data quality improvements (optional)
"""
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import signal
import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
from config.config import (
    TICKERS_CSV, N_TICKERS, START, END, SEQ_LEN, HORIZON, 
    PATTERNS, DATA_OUTPUT_PATH, USE_RAW_COLS, USE_ADJ_COLS
)
from src.utils.helpers import label_class
import warnings
warnings.filterwarnings('ignore')

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    print(f"\n[WARNING] Interrupt signal received. Saving collected data and shutting down gracefully...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ===== DATA QUALITY IMPROVEMENT FUNCTIONS =====

def analyze_data_quality(df):
    """
    Comprehensive data quality analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Quality analysis results
    """
    print("[SEARCH] Analyzing data quality...")
    
    analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'nan_analysis': {},
        'extreme_values_analysis': {},
        'class_imbalance_analysis': {},
        'recommendations': []
    }
    
    # 1. NaN Analysis
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()
    nan_ratio = total_nan / (len(df) * len(df.columns))
    
    analysis['nan_analysis'] = {
        'total_nan': total_nan,
        'nan_ratio': nan_ratio,
        'columns_with_nan': nan_counts[nan_counts > 0].to_dict(),
        'severity': 'high' if nan_ratio > 0.1 else 'medium' if nan_ratio > 0.01 else 'low'
    }
    
    print(f"[CHART] NaN Analysis:")
    print(f"  Total NaN values: {total_nan:,}")
    print(f"  NaN ratio: {nan_ratio:.2%}")
    print(f"  Columns with NaN: {len(nan_counts[nan_counts > 0])}")
    
    # 2. Extreme Values Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    extreme_threshold = 1e10
    extreme_counts = {}
    total_extreme = 0
    
    for col in numeric_cols:
        if col in df.columns:
            extreme_mask = df[col].abs() > extreme_threshold
            extreme_count = extreme_mask.sum()
            extreme_counts[col] = int(extreme_count)
            total_extreme += extreme_count
    
    analysis['extreme_values_analysis'] = {
        'total_extreme': total_extreme,
        'extreme_by_column': extreme_counts,
        'columns_with_extreme': len([c for c, count in extreme_counts.items() if count > 0]),
        'severity': 'high' if total_extreme > 10000 else 'medium' if total_extreme > 1000 else 'low'
    }
    
    print(f"[CHART] Extreme Values Analysis:")
    print(f"  Total extreme values: {total_extreme:,}")
    print(f"  Columns with extreme values: {analysis['extreme_values_analysis']['columns_with_extreme']}")
    
    # 3. Class Imbalance Analysis
    if 'Label' in df.columns:
        class_counts = df['Label'].value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        analysis['class_imbalance_analysis'] = {
            'class_distribution': class_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax(),
            'severity': 'high' if imbalance_ratio > 5 else 'medium' if imbalance_ratio > 2 else 'low'
        }
        
        print(f"[CHART] Class Imbalance Analysis:")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"  Minority class: {analysis['class_imbalance_analysis']['minority_class']} ({min_class_count:,} samples)")
        print(f"  Majority class: {analysis['class_imbalance_analysis']['majority_class']} ({max_class_count:,} samples)")
    
    return analysis


def handle_nan_values(df, strategy='auto'):
    """
    Handle NaN values using various strategies
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): 'auto', 'drop', 'fill_mean', 'fill_median', 'interpolate'
        
    Returns:
        pd.DataFrame: DataFrame with NaN values handled
    """
    print(f"[TOOLS] Handling NaN values with strategy: {strategy}")
    
    df_clean = df.copy()
    original_nan = df.isnull().sum().sum()
    
    if strategy == 'auto':
        # Automatic strategy based on data characteristics
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                col_type = df_clean[col].dtype
                
                if col_type in ['object', 'string']:
                    # For categorical/text data, fill with mode or 'Unknown'
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_val)
                else:
                    # For numeric data, use interpolation for time series, median for others
                    if col in ['Open', 'High', 'Low', 'Close', 'Volume'] or 'Close' in col:
                        df_clean[col] = df_clean[col].interpolate(method='linear')
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    elif strategy == 'drop':
        # Drop rows with any NaN values
        df_clean = df_clean.dropna()
    
    elif strategy == 'fill_mean':
        # Fill numeric columns with mean
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    elif strategy == 'fill_median':
        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    elif strategy == 'interpolate':
        # Use interpolation for all numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].interpolate(method='linear')
    
    remaining_nan = df_clean.isnull().sum().sum()
    print(f"[SUCCESS] NaN values reduced from {original_nan:,} to {remaining_nan:,}")
    
    return df_clean


def handle_extreme_values(df, method='isolation_forest', threshold=1e10):
    """
    Handle extreme values using various methods
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): 'isolation_forest', 'iqr', 'zscore', 'winsorize'
        threshold (float): Threshold for extreme values
        
    Returns:
        pd.DataFrame: DataFrame with extreme values handled
    """
    print(f"[TOOLS] Handling extreme values with method: {method}")
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
            # Use Isolation Forest to detect outliers
            for col in numeric_cols:
                if col in df_clean.columns:
                    # Remove NaN values for outlier detection
                    col_data = df_clean[col].dropna()
                    if len(col_data) > 0:
                        # Reshape for sklearn
                        X = col_data.values.reshape(-1, 1)
                        
                        # Fit isolation forest
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers = iso_forest.fit_predict(X)
                        
                        # Replace outliers with median
                        outlier_indices = col_data.index[outliers == -1]
                        if len(outlier_indices) > 0:
                            median_val = col_data.median()
                            df_clean.loc[outlier_indices, col] = median_val
                            print(f"  [CHART] {col}: Replaced {len(outlier_indices)} outliers with median")
        except ImportError:
            print("[WARNING] sklearn not available, skipping extreme value handling")
    
    elif method == 'iqr':
        # Use IQR method to detect and handle outliers
        for col in numeric_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif method == 'zscore':
        # Use Z-score method
        for col in numeric_cols:
            if col in df_clean.columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outlier_mask = z_scores > 3
                
                if outlier_mask.sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean.loc[outlier_mask, col] = median_val
    
    elif method == 'winsorize':
        # Winsorize extreme values
        for col in numeric_cols:
            if col in df_clean.columns:
                q_low = df_clean[col].quantile(0.01)
                q_high = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower=q_low, upper=q_high)
    
    return df_clean


def handle_class_imbalance(df, method='smote', target_col='Label'):
    """
    Handle class imbalance using various techniques
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): 'smote', 'undersample', 'oversample', 'class_weights'
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Balanced dataframe
    """
    if target_col not in df.columns:
        print(f"[WARNING] Target column '{target_col}' not found. Skipping class balancing.")
        return df
    
    print(f"[TOOLS] Handling class imbalance with method: {method}")
    
    # Analyze current class distribution
    class_counts = df[target_col].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    print(f"[CHART] Original class distribution:")
    for class_label, count in class_counts.items():
        print(f"  Class {class_label}: {count:,} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if method == 'undersample':
        # Undersample majority class
        minority_class = class_counts.idxmin()
        minority_count = class_counts.min()
        
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_col] == class_label]
            if len(class_df) > minority_count:
                # Undersample to match minority class
                class_df = class_df.sample(n=minority_count, random_state=42)
            balanced_dfs.append(class_df)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        print(f"[SUCCESS] Undersampled to {len(df_balanced):,} samples")
    
    elif method == 'oversample':
        # Oversample minority class
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_col] == class_label]
            if len(class_df) < majority_count:
                # Oversample to match majority class
                from sklearn.utils import resample
                class_df = resample(class_df, 
                                  n_samples=majority_count, 
                                  random_state=42, 
                                  replace=True)
            balanced_dfs.append(class_df)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        print(f"[SUCCESS] Oversampled to {len(df_balanced):,} samples")
    
    elif method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols]
            y = df[target_col]
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Reconstruct dataframe
            df_balanced = pd.DataFrame(X_balanced, columns=feature_cols)
            df_balanced[target_col] = y_balanced
            
            print(f"[SUCCESS] SMOTE applied: {len(df_balanced):,} samples")
            
        except ImportError:
            print("[WARNING] SMOTE not available. Falling back to oversampling.")
            return handle_class_imbalance(df, method='oversample', target_col=target_col)
    
    else:  # class_weights
        # Return original data with class weight information
        df_balanced = df.copy()
        print(f"[INFO] Using class weights approach (no data modification)")
    
    # Show new distribution
    new_class_counts = df_balanced[target_col].value_counts()
    new_imbalance_ratio = new_class_counts.max() / new_class_counts.min()
    
    print(f"[CHART] New class distribution:")
    for class_label, count in new_class_counts.items():
        print(f"  Class {class_label}: {count:,} samples")
    print(f"  New imbalance ratio: {new_imbalance_ratio:.1f}:1")
    
    return df_balanced


def apply_data_quality_pipeline(df, nan_strategy='auto', extreme_method='isolation_forest', 
                              balance_method='smote', target_col='Label'):
    """
    Apply complete data quality improvement pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        nan_strategy (str): NaN handling strategy
        extreme_method (str): Extreme value handling method
        balance_method (str): Class balancing method
        target_col (str): Target column for balancing
        
    Returns:
        pd.DataFrame: Cleaned and balanced dataframe
    """
    print("[ROCKET] Starting Data Quality Improvement Pipeline")
    print("=" * 60)
    
    # Step 1: Analyze current quality
    analysis = analyze_data_quality(df)
    
    # Step 2: Handle NaN values
    print(f"\n[STEP 1] Handling NaN values...")
    df_clean = handle_nan_values(df, strategy=nan_strategy)
    
    # Step 3: Handle extreme values
    print(f"\n[STEP 2] Handling extreme values...")
    df_clean = handle_extreme_values(df_clean, method=extreme_method)
    
    # Step 4: Handle class imbalance
    print(f"\n[STEP 3] Handling class imbalance...")
    df_balanced = handle_class_imbalance(df_clean, method=balance_method, target_col=target_col)
    
    # Step 5: Final analysis
    print(f"\n[STEP 4] Final quality analysis...")
    final_analysis = analyze_data_quality(df_balanced)
    
    # Summary
    print(f"\n[FINISH] Data Quality Improvement Complete!")
    print(f"[CHART] Summary:")
    print(f"  Original rows: {analysis['total_rows']:,}")
    print(f"  Final rows: {final_analysis['total_rows']:,}")
    print(f"  NaN reduction: {analysis['nan_analysis']['total_nan']:,} → {final_analysis['nan_analysis']['total_nan']:,}")
    print(f"  Extreme values: {analysis['extreme_values_analysis']['total_extreme']:,} → {final_analysis['extreme_values_analysis']['total_extreme']:,}")
    
    if 'class_imbalance_analysis' in analysis and 'class_imbalance_analysis' in final_analysis:
        original_ratio = analysis['class_imbalance_analysis']['imbalance_ratio']
        final_ratio = final_analysis['class_imbalance_analysis']['imbalance_ratio']
        print(f"  Class imbalance: {original_ratio:.1f}:1 → {final_ratio:.1f}:1")
    
    return df_balanced


def save_quality_report(df_original, df_cleaned, output_path):
    """
    Save a comprehensive quality improvement report
    
    Args:
        df_original (pd.DataFrame): Original dataframe
        df_cleaned (pd.DataFrame): Cleaned dataframe
        output_path (str): Path to save report
    """
    report = []
    report.append("DATA QUALITY IMPROVEMENT REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Original vs Cleaned comparison
    report.append("BEFORE vs AFTER COMPARISON:")
    report.append(f"  Original rows: {len(df_original):,}")
    report.append(f"  Cleaned rows: {len(df_cleaned):,}")
    report.append(f"  Original columns: {len(df_original.columns)}")
    report.append(f"  Cleaned columns: {len(df_cleaned.columns)}")
    report.append("")
    
    # NaN comparison
    original_nan = df_original.isnull().sum().sum()
    cleaned_nan = df_cleaned.isnull().sum().sum()
    report.append("NaN VALUES:")
    report.append(f"  Original: {original_nan:,}")
    report.append(f"  Cleaned: {cleaned_nan:,}")
    report.append(f"  Reduction: {original_nan - cleaned_nan:,} ({((original_nan - cleaned_nan) / original_nan * 100):.1f}%)")
    report.append("")
    
    # Class distribution comparison
    if 'Label' in df_original.columns and 'Label' in df_cleaned.columns:
        report.append("CLASS DISTRIBUTION:")
        original_dist = df_original['Label'].value_counts()
        cleaned_dist = df_cleaned['Label'].value_counts()
        
        for class_label in sorted(original_dist.index):
            original_count = original_dist.get(class_label, 0)
            cleaned_count = cleaned_dist.get(class_label, 0)
            report.append(f"  Class {class_label}: {original_count:,} → {cleaned_count:,}")
        
        original_ratio = original_dist.max() / original_dist.min()
        cleaned_ratio = cleaned_dist.max() / cleaned_dist.min()
        report.append(f"  Imbalance ratio: {original_ratio:.1f}:1 → {cleaned_ratio:.1f}:1")
        report.append("")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"[SAVE] Quality report saved to: {output_path}")


# ===== SAFE INDICATOR FUNCTIONS =====

def safe_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate SMA with safe handling of insufficient data
    
    Args:
        data: Price data array
        period: SMA period
        
    Returns:
        SMA array with NaN for insufficient data
    """
    if len(data) < period:
        return np.full(len(data), np.nan)
    return talib.SMA(data, timeperiod=period)


def safe_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA with safe handling of insufficient data"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    return talib.EMA(data, timeperiod=period)


def safe_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with safe handling of insufficient data"""
    if len(data) < period + 1:  # RSI needs period + 1
        return np.full(len(data), np.nan)
    return talib.RSI(data, timeperiod=period)


def safe_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD with safe handling of insufficient data"""
    if len(data) < slow:  # Need at least slow period
        nan_array = np.full(len(data), np.nan)
        return nan_array, nan_array, nan_array
    return talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)


def safe_bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands with safe handling of insufficient data"""
    if len(data) < period:
        nan_array = np.full(len(data), np.nan)
        return nan_array, nan_array, nan_array
    return talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)


def safe_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Stochastic with safe handling of insufficient data"""
    if len(high) < k_period:
        nan_array = np.full(len(high), np.nan)
        return nan_array, nan_array
    return talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)


def safe_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Williams %R with safe handling of insufficient data"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    return talib.WILLR(high, low, close, timeperiod=period)


def safe_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate CCI with safe handling of insufficient data"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    return talib.CCI(high, low, close, timeperiod=period)


def safe_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ADX with safe handling of insufficient data"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    return talib.ADX(high, low, close, timeperiod=period)


def safe_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with safe handling of insufficient data"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    return talib.ATR(high, low, close, timeperiod=period)


def get_adaptive_periods(data_length: int) -> Dict[str, int]:
    """
    Get adaptive indicator periods based on available data
    
    Args:
        data_length: Number of available data points
        
    Returns:
        Dictionary of indicator periods
    """
    if data_length < 20:
        return {
            'sma': 5,
            'ema': 5,
            'rsi': 7,
            'macd_fast': 6,
            'macd_slow': 12,
            'bb': 10,
            'stoch': 7,
            'williams_r': 7,
            'cci': 10,
            'adx': 7,
            'atr': 7
        }
    elif data_length < 50:
        return {
            'sma': 10,
            'ema': 10,
            'rsi': 10,
            'macd_fast': 8,
            'macd_slow': 16,
            'bb': 15,
            'stoch': 10,
            'williams_r': 10,
            'cci': 15,
            'adx': 10,
            'atr': 10
        }
    else:
        return {
            'sma': 20,
            'ema': 20,
            'rsi': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'bb': 20,
            'stoch': 14,
            'williams_r': 14,
            'cci': 20,
            'adx': 14,
            'atr': 14
        }


def validate_stock_data(df: pd.DataFrame, ticker: str, min_length: int = 100) -> Dict[str, any]:
    """
    Comprehensive validation of stock data quality
    
    Args:
        df: Stock data DataFrame
        ticker: Stock ticker for logging
        min_length: Minimum required data length
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'ticker': ticker,
        'passed': True,
        'issues': [],
        'metrics': {}
    }
    
    # Basic data checks
    if df.empty:
        validation_results['passed'] = False
        validation_results['issues'].append('Empty DataFrame')
        return validation_results
    
    # Length check
    data_length = len(df)
    validation_results['metrics']['data_length'] = data_length
    
    if data_length < min_length:
        validation_results['passed'] = False
        validation_results['issues'].append(f'Insufficient data: {data_length} < {min_length}')
    
    # Required columns check
    required_cols = ['Open_adj', 'High_adj', 'Low_adj', 'Close_adj', 'Volume_adj']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation_results['passed'] = False
        validation_results['issues'].append(f'Missing columns: {missing_cols}')
    
    # Missing values check
    missing_count = df[required_cols].isnull().sum().sum()
    missing_ratio = missing_count / (len(df) * len(required_cols))
    validation_results['metrics']['missing_ratio'] = missing_ratio
    
    if missing_ratio > 0.05:  # More than 5% missing
        validation_results['passed'] = False
        validation_results['issues'].append(f'Too many missing values: {missing_ratio:.2%}')
    
    # Extreme values check
    extreme_threshold = 1e10
    extreme_count = 0
    
    for col in required_cols:
        if col in df.columns:
            extreme_mask = df[col].abs() > extreme_threshold
            extreme_count += extreme_mask.sum()
    
    validation_results['metrics']['extreme_values'] = extreme_count
    
    if extreme_count > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f'Extreme values detected: {extreme_count}')
    
    return validation_results


def validate_dataset_quality(csv_file: str) -> Dict[str, any]:
    """
    Validate entire dataset quality
    
    Args:
        csv_file: Path to dataset CSV
        
    Returns:
        Dictionary with validation results
    """
    print(f"[SEARCH] Validating dataset: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return {
            'passed': False,
            'error': f'Failed to read CSV: {e}',
            'issues': ['CSV read error']
        }
    
    validation_results = {
        'passed': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    nan_ratio = nan_count / (len(df) * len(df.columns))
    validation_results['metrics']['nan_count'] = nan_count
    validation_results['metrics']['nan_ratio'] = nan_ratio
    
    if nan_count > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f'Found {nan_count} NaN values ({nan_ratio:.2%})')
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    validation_results['metrics']['inf_count'] = inf_count
    
    if inf_count > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f'Found {inf_count} infinite values')
    
    # Check for extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    extreme_threshold = 1e10
    extreme_count = 0
    
    for col in numeric_cols:
        extreme_mask = df[col].abs() > extreme_threshold
        extreme_count += extreme_mask.sum()
    
    validation_results['metrics']['extreme_count'] = extreme_count
    
    if extreme_count > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f'Found {extreme_count} extreme values')
    
    # Check data ranges
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        validation_results['metrics'][f'{col}_range'] = (col_min, col_max)
    
    # Check class distribution (if Label column exists)
    if 'Label' in df.columns:
        class_counts = df['Label'].value_counts()
        validation_results['metrics']['class_distribution'] = class_counts.to_dict()
        
        # Check for class imbalance
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        validation_results['metrics']['imbalance_ratio'] = imbalance_ratio
        
        if imbalance_ratio > 5:  # More than 5:1 imbalance
            validation_results['warnings'].append(f'Class imbalance detected: {imbalance_ratio:.1f}:1')
    
    # Check feature statistics
    feature_cols = [col for col in df.columns if col not in ['Ticker', 'Label']]
    if feature_cols:
        feature_stats = df[feature_cols].describe()
        validation_results['metrics']['feature_statistics'] = feature_stats.to_dict()
    
    print(f"[SUCCESS] Dataset validation complete")
    print(f"[CHART] Total rows: {validation_results['total_rows']}")
    print(f"[CHART] Total columns: {validation_results['total_columns']}")
    print(f"[CHART] NaN count: {validation_results['metrics']['nan_count']}")
    print(f"[CHART] Passed: {validation_results['passed']}")
    
    if validation_results['issues']:
        print(f"[ERROR] Issues found:")
        for issue in validation_results['issues']:
            print(f"   - {issue}")
    
    if validation_results['warnings']:
        print(f"[WARNING] Warnings:")
        for warning in validation_results['warnings']:
            print(f"   - {warning}")
    
    return validation_results


def get_data_quality_report(csv_file: str) -> str:
    """
    Generate comprehensive data quality report
    
    Args:
        csv_file: Path to dataset CSV
        
    Returns:
        Formatted quality report
    """
    validation_results = validate_dataset_quality(csv_file)
    
    report = f"""
[CHART] DATA QUALITY REPORT
{'='*50}

[CHART] Dataset Overview:
   File: {csv_file}
   Total Rows: {validation_results['total_rows']:,}
   Total Columns: {validation_results['total_columns']}
   Status: {'[SUCCESS] PASSED' if validation_results['passed'] else '[ERROR] FAILED'}

[SEARCH] Quality Metrics:
   NaN Values: {validation_results['metrics']['nan_count']:,}
   NaN Ratio: {validation_results['metrics']['nan_ratio']:.2%}
   Infinite Values: {validation_results['metrics']['inf_count']:,}
   Extreme Values: {validation_results['metrics']['extreme_count']:,}

"""
    
    if validation_results['issues']:
        report += f"[ERROR] Issues Found:\n"
        for issue in validation_results['issues']:
            report += f"   • {issue}\n"
    
    if validation_results['warnings']:
        report += f"[WARNING] Warnings:\n"
        for warning in validation_results['warnings']:
            report += f"   • {warning}\n"
    
    if 'class_distribution' in validation_results['metrics']:
        report += f"\n[CHART] Class Distribution:\n"
        for class_label, count in validation_results['metrics']['class_distribution'].items():
            report += f"   Class {class_label}: {count:,} samples\n"
    
    report += f"\n{'='*50}\n"
    
    return report


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
    
    🔧 ENHANCED WITH COMPREHENSIVE VALIDATION
    - Uses validate_stock_data for systematic checks
    - Adaptive period recommendations
    - Detailed quality metrics and reporting
    
    Args:
        df (pd.DataFrame): DataFrame to verify
        ticker (str): Ticker symbol for logging
    """
    print(f"\n[SEARCH] VERIFYING DATA QUALITY FOR {ticker}")
    print("=" * 50)
    
    # Use enhanced validation
    validation_results = validate_stock_data(df, ticker, min_length=100)
    
    # Check basic data
    print(f"[CHART] Data shape: {df.shape}")
    print(f"[CHART] Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"[CHART] Total days: {len(df)}")
    
    # Display validation results
    print(f"[CHART] Validation Status: {'[SUCCESS] PASSED' if validation_results['passed'] else '[ERROR] FAILED'}")
    print(f"[CHART] Data Length: {validation_results['metrics']['data_length']}")
    print(f"[CHART] Missing Ratio: {validation_results['metrics']['missing_ratio']:.2%}")
    print(f"[CHART] Extreme Values: {validation_results['metrics']['extreme_values']}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    print(f"[CHART] Missing values per column:")
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  [WARNING] {col}: {missing} missing values")
    
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
        print(f"[WARNING] Price data issues:")
        for issue in price_issues:
            print(f"  {issue}")
    else:
        print("[SUCCESS] Price data looks good")
    
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
            print(f"[WARNING] OHLC relationship issues: {invalid_ohlc} invalid rows")
        else:
            print("[SUCCESS] OHLC relationships are valid")
    
    # Check for sufficient data for patterns
    data_length = len(df)
    if data_length < 20:
        print(f"[WARNING] Insufficient data for pattern detection: {data_length} days (need at least 20)")
    else:
        print(f"[SUCCESS] Sufficient data for pattern detection: {data_length} days")
    
    # Get adaptive periods recommendation
    periods = get_adaptive_periods(data_length)
    print(f"[TOOLS] Recommended adaptive periods for {data_length} data points:")
    print(f"   [CHART] SMA/EMA: {periods['sma']}, {periods['ema']}")
    print(f"   [CHART] RSI: {periods['rsi']}")
    print(f"   [CHART] MACD: {periods['macd_fast']}, {periods['macd_slow']}")
    print(f"   [CHART] Bollinger Bands: {periods['bb']}")
    print(f"   [CHART] Stochastic: {periods['stoch']}")
    
    # Display validation issues if any
    if validation_results['issues']:
        print(f"\n[ERROR] Validation Issues:")
        for issue in validation_results['issues']:
            print(f"   • {issue}")
    
    # Sample data values
    print(f"\n[CHART] Sample data values:")
    for col in ['Open_raw', 'High_raw', 'Low_raw', 'Close_raw']:
        if col in df.columns:
            sample_values = df[col].head(3).values
            print(f"  {col}: {sample_values}")
    
    print("=" * 50)


def calculate_swing_trading_features(df):
    """
    Calculate features optimized for swing trading (1-3 days)
    Focuses on 1-day return prediction with swing trading indicators
    
    🔧 ENHANCED WITH SAFE INDICATOR FUNCTIONS
    - Prevents NaN propagation from insufficient data
    - Uses adaptive periods based on data availability
    - Comprehensive error handling and validation
    
    Args:
        df (pd.DataFrame): DataFrame with explicit raw and adjusted columns
        
    Returns:
        pd.DataFrame: DataFrame with swing trading optimized features
    """
    df = df.copy()
    
    # Validate and clean data first
    df = validate_and_clean_data(df)
    
    print("[CHART] Calculating swing trading features (1-3 day focus)...")
    print(f"[CHART] Input data shape: {df.shape}")
    print(f"[CHART] Data columns: {list(df.columns)}")
    
    # Convert to float64 for TA-Lib compatibility
    close_adj = df['Close_adj'].values.astype(np.float64)
    high_adj = df['High_adj'].values.astype(np.float64)
    low_adj = df['Low_adj'].values.astype(np.float64)
    volume_adj = df['Volume_adj'].values.astype(np.float64)
    
    # Get adaptive periods based on data length
    data_length = len(close_adj)
    periods = get_adaptive_periods(data_length)
    
    print(f"[TOOLS] Using adaptive periods for {data_length} data points:")
    print(f"   [CHART] Periods: {periods}")
    
    # 1. TREND ANALYSIS (Swing traders focus on medium-term trends)
    print("[CHART] Computing trend analysis with safe indicators...")
    trend_features = []
    
    try:
        # Safe moving averages with adaptive periods
        df['sma_10'] = safe_sma(close_adj, periods['sma'])
        df['sma_20'] = safe_sma(close_adj, periods['sma'] * 2)
        df['sma_50'] = safe_sma(close_adj, periods['sma'] * 5)
        df['ema_10'] = safe_ema(close_adj, periods['ema'])
        df['ema_20'] = safe_ema(close_adj, periods['ema'] * 2)
        
        # Trend strength and position (with safe division)
        df['trend_strength'] = np.where(
            df['sma_50'] != 0,
            (df['sma_20'] - df['sma_50']) / df['sma_50'],
            np.nan
        )
        df['above_sma_20'] = (df['Close_adj'] > df['sma_20']).astype(int)
        df['above_sma_50'] = (df['Close_adj'] > df['sma_50']).astype(int)
        df['price_vs_ema_10'] = np.where(
            df['ema_10'] != 0,
            (df['Close_adj'] - df['ema_10']) / df['ema_10'],
            np.nan
        )
        
        # Golden/Death Cross signals (swing traders love these)
        df['golden_cross'] = (df['sma_10'] > df['sma_50']) & (df['sma_10'].shift(1) <= df['sma_50'].shift(1))
        df['death_cross'] = (df['sma_10'] < df['sma_50']) & (df['sma_10'].shift(1) >= df['sma_50'].shift(1))
        
        trend_features = ['sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'trend_strength', 
                        'above_sma_20', 'above_sma_50', 'price_vs_ema_10', 'golden_cross', 'death_cross']
        print(f"[SUCCESS] Safe trend features added: {len(trend_features)} features")
        print(f"   [CHART] Adaptive SMA periods: {periods['sma']}, {periods['sma']*2}, {periods['sma']*5}")
        print(f"   [CHART] Adaptive EMA periods: {periods['ema']}, {periods['ema']*2}")
        print(f"   [CHART] Safe trend strength calculation")
        print(f"   [CHART] Golden/Death cross detection")
        
    except Exception as e:
        print(f"[WARNING] Error calculating trend indicators: {e}")
    
    # 2. MOMENTUM (Swing traders use longer periods)
    print("[LIGHTNING] Computing momentum indicators with safe functions...")
    momentum_features = []
    
    try:
        # Safe RSI with adaptive period
        df['rsi'] = safe_rsi(close_adj, periods['rsi'])
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
        
        # Safe MACD with adaptive periods
        macd, macd_signal, macd_hist = safe_macd(close_adj, periods['macd_fast'], periods['macd_slow'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_bullish'] = (macd > macd_signal).astype(int)
        df['macd_bearish'] = (macd < macd_signal).astype(int)
        
        # Safe Stochastic with adaptive periods
        stoch_k, stoch_d = safe_stochastic(high_adj, low_adj, close_adj, periods['stoch'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_oversold'] = (stoch_k < 20).astype(int)
        df['stoch_overbought'] = (stoch_k > 80).astype(int)
        
        # Additional momentum indicators
        df['williams_r'] = safe_williams_r(high_adj, low_adj, close_adj, periods['williams_r'])
        df['cci'] = safe_cci(high_adj, low_adj, close_adj, periods['cci'])
        df['adx'] = safe_adx(high_adj, low_adj, close_adj, periods['adx'])
        
        momentum_features = ['rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
                           'macd', 'macd_signal', 'macd_histogram', 'macd_bullish', 'macd_bearish',
                           'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
                           'williams_r', 'cci', 'adx']
        print(f"[SUCCESS] Safe momentum features added: {len(momentum_features)} features")
        print(f"   [LIGHTNING] Adaptive RSI: {periods['rsi']}-period with oversold/overbought zones")
        print(f"   [LIGHTNING] Adaptive MACD: {periods['macd_fast']},{periods['macd_slow']},9 with bullish/bearish signals")
        print(f"   [LIGHTNING] Adaptive Stochastic: {periods['stoch']},3 with oversold/overbought zones")
        print(f"   [LIGHTNING] Additional: Williams %R, CCI, ADX")
        
    except Exception as e:
        print(f"[WARNING] Error calculating momentum indicators: {e}")
    
    # 3. VOLATILITY (Swing traders focus on volatility for position sizing)
    print("[CHART] Computing volatility analysis with safe functions...")
    volatility_features = []
    
    try:
        # Safe ATR for volatility measurement
        df['atr'] = safe_atr(high_adj, low_adj, close_adj, periods['atr'])
        
        # Safe rolling mean for ATR comparison (with minimum periods)
        atr_rolling_mean = df['atr'].rolling(min(20, len(df)//2), min_periods=1).mean()
        df['atr_high'] = (df['atr'] > atr_rolling_mean * 1.5).astype(int)
        df['atr_low'] = (df['atr'] < atr_rolling_mean * 0.5).astype(int)
        
        # Safe Bollinger Bands for volatility and mean reversion
        bb_upper, bb_middle, bb_lower = safe_bollinger_bands(close_adj, periods['bb'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Safe BB position calculation
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = np.where(
            bb_range != 0,
            (df['Close_adj'] - df['bb_lower']) / bb_range,
            np.nan
        )
        
        # Safe BB squeeze/expansion detection
        bb_width = np.where(
            df['bb_middle'] != 0,
            (df['bb_upper'] - df['bb_lower']) / df['bb_middle'],
            np.nan
        )
        df['bb_squeeze'] = (bb_width < 0.1).astype(int)
        df['bb_expansion'] = (bb_width > 0.2).astype(int)
        
        # Price near Bollinger Bands (swing trading signals)
        df['near_bb_upper'] = (df['Close_adj'] > df['bb_upper'] * 0.98).astype(int)
        df['near_bb_lower'] = (df['Close_adj'] < df['bb_lower'] * 1.02).astype(int)
        
        volatility_features = ['atr', 'atr_high', 'atr_low', 'bb_upper', 'bb_middle', 'bb_lower',
                             'bb_position', 'bb_squeeze', 'bb_expansion', 'near_bb_upper', 'near_bb_lower']
        print(f"[SUCCESS] Safe volatility features added: {len(volatility_features)} features")
        print(f"   [CHART] Adaptive ATR: {periods['atr']}-period with high/low volatility flags")
        print(f"   [CHART] Adaptive Bollinger Bands: {periods['bb']}-period with squeeze/expansion detection")
        print(f"   [CHART] Safe BB Position: Price position within bands")
        
    except Exception as e:
        print(f"[WARNING] Error calculating volatility indicators: {e}")
    
    # 4. SUPPORT/RESISTANCE (Swing traders focus on key levels)
    print("[TARGET] Computing support/resistance levels...")
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
        print(f"[SUCCESS] Support/Resistance features added: {len(support_resistance_features)} features")
        print(f"   [TARGET] 20-day resistance/support levels")
        print(f"   [TARGET] Price position within range")
        print(f"   [TARGET] Breakout detection (up/down)")
        
    except Exception as e:
        print(f"[WARNING] Error calculating support/resistance: {e}")
    
    # 5. VOLUME (Swing traders use volume for confirmation)
    print("[CHART] Computing volume analysis...")
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
        print(f"[SUCCESS] Volume features added: {len(volume_features)} features")
        print(f"   [CHART] Volume SMA: 20-period average")
        print(f"   [CHART] Volume ratio: Current vs 20-day average")
        print(f"   [CHART] OBV: On Balance Volume for trend confirmation")
        print(f"   [CHART] MFI: Money Flow Index (14-period)")
        
    except Exception as e:
        print(f"⚠️  Error calculating volume indicators: {e}")
    
    # 6. PRICE ACTION (Swing traders focus on multi-day patterns)
    print("[CHART] Computing price action features...")
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
    
    print(f"[SUCCESS] Price action features added: {len(price_action_features)} features")
    print(f"   [CHART] Returns: 1-day, 3-day, 5-day")
    print(f"   [CHART] Gap analysis: Up/down gap detection")
    print(f"   [CHART] Price relationships: High-low, Open-close ratios")
    print(f"   [CHART] Swing patterns: Higher highs/lows, Lower highs/lows")
    
    # 7. CANDLESTICK PATTERNS (Debug this section)
    print("[PATTERN] Computing candlestick patterns...")
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
        
        print(f"[PATTERN] Raw data sample - Open: {open_raw[:5]}, High: {high_raw[:5]}, Low: {low_raw[:5]}, Close: {close_raw[:5]}")
        
        for pattern in key_patterns:
            try:
                pattern_result = getattr(talib, pattern)(
                    open_raw, high_raw, low_raw, close_raw
                )
                # Convert numpy array to pandas series and then to boolean
                df[f'pattern_{pattern}'] = (pattern_result != 0).astype(int)
                pattern_count = df[f'pattern_{pattern}'].sum()
                print(f"[PATTERN] {pattern}: {pattern_count} patterns found")
                pattern_features.append(f'pattern_{pattern}')
            except Exception as e:
                print(f"⚠️  Error calculating {pattern}: {e}")
                df[f'pattern_{pattern}'] = 0
                pattern_features.append(f'pattern_{pattern}')
        
        print(f"[SUCCESS] Candlestick patterns added: {len(pattern_features)} features")
        print(f"   [PATTERN] 10 key patterns for swing trading")
        print(f"   [PATTERN] Calculated from raw OHLC data")
        
    except Exception as e:
        print(f"⚠️  Error calculating candlestick patterns: {e}")
    
    # Summary
    total_features = len(trend_features) + len(momentum_features) + len(volatility_features) + \
                   len(support_resistance_features) + len(volume_features) + len(price_action_features) + \
                   len(pattern_features)
    
    print(f"\n[CHART] FEATURE SUMMARY:")
    print(f"[TREND] Trend features: {len(trend_features)}")
    print(f"[MOMENTUM] Momentum features: {len(momentum_features)}")
    print(f"[VOLATILITY] Volatility features: {len(volatility_features)}")
    print(f"[SUPPORT] Support/Resistance features: {len(support_resistance_features)}")
    print(f"[VOLUME] Volume features: {len(volume_features)}")
    print(f"[PRICE] Price action features: {len(price_action_features)}")
    print(f"[PATTERN] Candlestick patterns: {len(pattern_features)}")
    print(f"[TOTAL] TOTAL FEATURES: {total_features}")
    print(f"[SHAPE] Final data shape: {df.shape}")
    
    print(f"[SUCCESS] Swing trading features complete. Total features: {len(df.columns)}")
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


def build_dataset(csv_output=None, apply_quality_fixes=True, nan_strategy='auto', 
                 extreme_method='isolation_forest', balance_method='smote'):
    """
    Build the dataset using explicit raw/adjusted column families (Professional Pipeline)
    
    🔧 ENHANCED WITH COMPREHENSIVE VALIDATION AND QUALITY IMPROVEMENTS
    - Safe indicator functions prevent NaN propagation
    - Adaptive periods based on data availability
    - Dataset quality validation and reporting
    - Graceful handling of insufficient data
    - Automatic data quality improvements (optional)
    
    Args:
        csv_output (str, optional): Output CSV file path
        apply_quality_fixes (bool): Whether to apply quality improvements (default: True)
        nan_strategy (str): NaN handling strategy ('auto', 'drop', 'fill_mean', 'fill_median', 'interpolate')
        extreme_method (str): Extreme value handling method ('isolation_forest', 'iqr', 'zscore', 'winsorize')
        balance_method (str): Class balancing method ('smote', 'undersample', 'oversample', 'class_weights')
    
    Returns:
        str: Path to the created CSV file
    """
    if csv_output is None:
        csv_output = DATA_OUTPUT_PATH
    
    print(f"Building professional dataset from {N_TICKERS} tickers...")
    print(f"Date range: {START} to {END}")
    print("Using explicit raw/adjusted column families")
    print("🔧 Enhanced with safe indicator functions and validation")
    if apply_quality_fixes:
        print("🔧 Quality improvements: ENABLED")
        print(f"   - NaN strategy: {nan_strategy}")
        print(f"   - Extreme value method: {extreme_method}")
        print(f"   - Class balancing: {balance_method}")
    else:
        print("🔧 Quality improvements: DISABLED")
    
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
    
    # Apply quality improvements if enabled
    if apply_quality_fixes:
        print(f"\n[ROCKET] Applying data quality improvements...")
        print(f"[CHART] Original dataset shape: {df_final.shape}")
        
        # Store original for comparison
        df_original = df_final.copy()
        
        # Apply quality improvement pipeline
        df_final = apply_data_quality_pipeline(
            df_final,
            nan_strategy=nan_strategy,
            extreme_method=extreme_method,
            balance_method=balance_method,
            target_col='Label'
        )
        
        print(f"[CHART] Cleaned dataset shape: {df_final.shape}")
        
        # Save quality improvement report
        quality_report_path = csv_output.replace('.csv', '_quality_improvement_report.txt')
        save_quality_report(df_original, df_final, quality_report_path)
        
        # Create cleaned dataset filename
        cleaned_csv_output = csv_output.replace('.csv', '_cleaned.csv')
        print(f"[SAVE] Saving cleaned dataset to: {cleaned_csv_output}")
        df_final.to_csv(cleaned_csv_output, index=False)
        
        # Use cleaned dataset for further processing
        csv_output = cleaned_csv_output
    else:
        # Save original dataset
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
    
    # 🔧 ENHANCED: Validate final dataset quality
    print(f"\n🔍 Validating final dataset quality...")
    validation_results = validate_dataset_quality(csv_output)
    
    if validation_results['passed']:
        print(f"[SUCCESS] Dataset validation PASSED")
        print(f"[CHART] Final dataset: {validation_results['total_rows']:,} samples")
        print(f"[CHART] Features per sample: {validation_results['total_columns'] - 2}")  # Exclude Ticker and Label
    else:
        print(f"❌ Dataset validation FAILED")
        print(f"⚠️  Issues found:")
        for issue in validation_results['issues']:
            print(f"   - {issue}")
    
    # Generate quality report
    quality_report = get_data_quality_report(csv_output)
    report_file = csv_output.replace('.csv', '_quality_report.txt')
    with open(report_file, 'w') as f:
        f.write(quality_report)
    print(f"[CHART] Quality report saved to: {report_file}")
    
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
    print(f"Sequence length: {SEQ_LEN} days (1-day swing trading optimized)")
    print(f"Saving data every {save_interval} tickers (backup only)")
    print("Press Ctrl+C to stop gracefully and save collected data")
    print("Using explicit raw/adjusted column families")
    
    # Get S&P 500 tickers
    tickers = pd.read_csv(TICKERS_CSV)["Symbol"].tolist()[:N_TICKERS]
    
    all_rows = []
    needed_length = SEQ_LEN + HORIZON + 1
    feature_columns = None  # Will be set from first successful ticker
    
    # Create backup file path with date
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = csv_output.replace('.csv', f'_backup_{timestamp}.csv')
    
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
            
            # Save incrementally every save_interval tickers (but not the last batch)
            if (i + 1) % save_interval == 0 and (i + 1) < len(tickers):
                print(f"\n[BACKUP] Saving intermediate backup after {i+1} tickers ({len(all_rows)} samples)...")
                backup_path = csv_output.replace('.csv', f'_backup_{i+1}.csv')
                _save_incremental_data(all_rows, feature_columns, backup_path, i+1)
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if not all_rows:
        raise ValueError("No data was successfully processed")
    
    # Final save
    print(f"\n💾 Saving final dataset with {len(all_rows)} samples...")
    final_path = _save_incremental_data(all_rows, feature_columns, csv_output, len(tickers))
    
    # Clean up old backups
    _cleanup_old_backups(csv_output, keep_recent=3)
    
    # Create feature summary
    summary_file = csv_output.replace('.csv', '_feature_summary.txt')
    _create_feature_summary(summary_file, feature_columns, len(all_rows))
    
    print(f"[SUCCESS] Dataset saved to: {final_path}")
    print(f"[CHART] Total samples collected: {len(all_rows)}")
    
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
        f.write("- Price labels: Based on adjusted close (real economic returns)\n")
        f.write("- Sequence length: 5 days (optimized for 1-day swing trading)\n\n")
        
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


def _cleanup_old_backups(csv_output, keep_recent=3):
    """
    Clean up old backup files, keeping only the most recent ones
    
    Args:
        csv_output (str): Main CSV file path
        keep_recent (int): Number of recent backups to keep
    """
    import glob
    import os
    
    # Find all backup files
    backup_pattern = csv_output.replace('.csv', '_backup_*.csv')
    backup_files = glob.glob(backup_pattern)
    
    if len(backup_files) > keep_recent:
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Remove old backups
        for old_backup in backup_files[keep_recent:]:
            try:
                os.remove(old_backup)
                print(f"🗑️ Removed old backup: {os.path.basename(old_backup)}")
            except Exception as e:
                print(f"⚠️ Could not remove old backup {old_backup}: {e}")


def _find_latest_backup(csv_output):
    """
    Find the most recent backup file
    
    Args:
        csv_output (str): Main CSV file path
    
    Returns:
        str: Path to latest backup file, or None if not found
    """
    import glob
    import os
    
    # Check for dated backups first
    backup_pattern = csv_output.replace('.csv', '_backup_*.csv')
    backup_files = glob.glob(backup_pattern)
    
    if backup_files:
        # Return the most recent backup
        latest_backup = max(backup_files, key=lambda x: os.path.getmtime(x))
        return latest_backup
    
    # Check for old-style backup
    old_backup = csv_output.replace('.csv', '_backup.csv')
    if os.path.exists(old_backup):
        return old_backup
    
    return None


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
        print(f"[CHART] Existing dataset has {len(existing_tickers)} tickers and {len(existing_data)} samples")
    elif _find_latest_backup(csv_output):
        backup_file = _find_latest_backup(csv_output)
        print(f"📁 Found backup dataset: {backup_file}")
        existing_data = pd.read_csv(backup_file)
        existing_tickers = set(existing_data['Ticker'].unique())
        print(f"[CHART] Backup dataset has {len(existing_tickers)} tickers and {len(existing_data)} samples")
        # Move backup to main file
        import shutil
        shutil.move(backup_file, csv_output)
        print(f"[SUCCESS] Moved backup to main dataset: {csv_output}")
    else:
        print("📁 No existing dataset found - will create new one")
    
    # Get current S&P 500 tickers
    tickers = pd.read_csv(TICKERS_CSV)["Symbol"].tolist()[:N_TICKERS]
    current_tickers = set(tickers)
    
    # Identify new tickers
    new_tickers = current_tickers - existing_tickers
    removed_tickers = existing_tickers - current_tickers
    
    print(f"\n[ANALYSIS] Ticker Analysis:")
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
    print(f"[SUCCESS] Updated dataset saved to: {csv_output}")
    
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


if __name__ == "__main__":
    """
    Main execution with command-line argument support for quality improvements
    """
    parser = argparse.ArgumentParser(description='Build financial dataset with optional quality improvements')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--skip-quality-fixes', action='store_true', 
                       help='Skip data quality improvements (default: enabled)')
    parser.add_argument('--nan-strategy', type=str, default='auto',
                       choices=['auto', 'drop', 'fill_mean', 'fill_median', 'interpolate'],
                       help='NaN handling strategy (default: auto)')
    parser.add_argument('--extreme-method', type=str, default='isolation_forest',
                       choices=['isolation_forest', 'iqr', 'zscore', 'winsorize'],
                       help='Extreme value handling method (default: isolation_forest)')
    parser.add_argument('--balance-method', type=str, default='smote',
                       choices=['smote', 'undersample', 'oversample', 'class_weights'],
                       help='Class balancing method (default: smote)')
    parser.add_argument('--incremental', action='store_true',
                       help='Use incremental dataset building')
    parser.add_argument('--smart-update', action='store_true',
                       help='Use smart update dataset building')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days back for smart update (default: 30)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save interval for incremental building (default: 100)')
    
    args = parser.parse_args()
    
    # Determine quality fixes setting
    apply_quality_fixes = not args.skip_quality_fixes
    
    print("=" * 60)
    print("FINANCIAL DATASET BUILDER WITH QUALITY IMPROVEMENTS")
    print("=" * 60)
    
    try:
        if args.incremental:
            print("[INFO] Using incremental dataset building...")
            result = build_dataset_incremental(
                csv_output=args.output,
                save_interval=args.save_interval
            )
        elif args.smart_update:
            print("[INFO] Using smart update dataset building...")
            result = build_dataset_smart_update(
                csv_output=args.output,
                days_back=args.days_back
            )
        else:
            print("[INFO] Using standard dataset building...")
            result = build_dataset(
                csv_output=args.output,
                apply_quality_fixes=apply_quality_fixes,
                nan_strategy=args.nan_strategy,
                extreme_method=args.extreme_method,
                balance_method=args.balance_method
            )
        
        print(f"\n[SUCCESS] Dataset building completed!")
        print(f"[CHART] Output file: {result}")
        
        if apply_quality_fixes:
            print(f"[INFO] Quality improvements were applied")
            print(f"[INFO] Check the quality improvement report for details")
        else:
            print(f"[INFO] Quality improvements were skipped")
        
    except KeyboardInterrupt:
        print(f"\n[WARNING] Process interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Dataset building failed: {e}")
        sys.exit(1)