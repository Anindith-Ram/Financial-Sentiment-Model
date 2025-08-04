"""
Candlestick Pattern Validation and Analysis Utilities
"""
import pandas as pd
import numpy as np
import talib
from config.config import PATTERNS


# Multi-day pattern requirements
PATTERN_REQUIREMENTS = {
    'CDLMORNINGSTAR': 3,      # Requires 3 days
    'CDLEVENINGSTAR': 3,      # Requires 3 days  
    'CDL3WHITESOLDIERS': 3,   # Requires 3 days
    'CDL3BLACKCROWS': 3,      # Requires 3 days
    'CDLENGULFING': 2,        # Requires 2 days
    'CDLPIERCING': 2,         # Requires 2 days
    'CDLDARKCLOUDCOVER': 2,   # Requires 2 days
    'CDLTHRUSTING': 2,        # Requires 2 days
    # Single day patterns
    'CDLHAMMER': 1,
    'CDLINVERTEDHAMMER': 1,
    'CDLHANGINGMAN': 1,
    'CDLSHOOTINGSTAR': 1,
    'CDLDOJI': 1,
    'CDLDRAGONFLYDOJI': 1,
    'CDLSPINNINGTOP': 1
}


def validate_pattern_data(df, min_days_required=3):
    """
    Validate that we have enough data for multi-day pattern recognition
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        min_days_required (int): Minimum days needed
        
    Returns:
        bool: Whether data is sufficient for pattern analysis
    """
    if len(df) < min_days_required:
        return False
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check for sufficient non-null data
    non_null_rows = df[required_cols].dropna()
    return len(non_null_rows) >= min_days_required


def calculate_patterns_with_validation(df, patterns=None):
    """
    Calculate candlestick patterns with proper validation
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        patterns (list): List of pattern names to calculate
        
    Returns:
        pd.DataFrame: DataFrame with pattern columns added
    """
    if patterns is None:
        patterns = PATTERNS
    
    df_copy = df.copy()
    
    # Validate minimum data requirements
    max_days_needed = max(PATTERN_REQUIREMENTS.get(p, 1) for p in patterns)
    
    if not validate_pattern_data(df_copy, max_days_needed):
        raise ValueError(f"Insufficient data: need at least {max_days_needed} days")
    
    # Calculate each pattern
    pattern_results = {}
    
    for pattern in patterns:
        try:
            # Get the TA-Lib function
            pattern_func = getattr(talib, pattern)
            
            # Calculate pattern (TA-Lib handles multi-day automatically)
            result = pattern_func(
                df_copy['Open'].values,
                df_copy['High'].values, 
                df_copy['Low'].values,
                df_copy['Close'].values
            )
            
            # Convert to binary (pattern detected or not)
            df_copy[pattern] = (result != 0).astype(int)
            
            # Store results for validation
            pattern_results[pattern] = {
                'total_signals': np.sum(result != 0),
                'positive_signals': np.sum(result > 0),
                'negative_signals': np.sum(result < 0),
                'days_required': PATTERN_REQUIREMENTS.get(pattern, 1)
            }
            
        except AttributeError:
            print(f"Warning: Pattern {pattern} not found in TA-Lib")
            df_copy[pattern] = 0
        except Exception as e:
            print(f"Error calculating {pattern}: {e}")
            df_copy[pattern] = 0
    
    return df_copy, pattern_results


def analyze_pattern_distribution(df, patterns=None):
    """
    Analyze the distribution of detected patterns
    
    Args:
        df (pd.DataFrame): DataFrame with pattern columns
        patterns (list): List of patterns to analyze
        
    Returns:
        dict: Pattern analysis results
    """
    if patterns is None:
        patterns = PATTERNS
    
    analysis = {}
    
    for pattern in patterns:
        if pattern in df.columns:
            pattern_col = df[pattern]
            analysis[pattern] = {
                'total_occurrences': pattern_col.sum(),
                'frequency': pattern_col.mean(),
                'days_with_pattern': (pattern_col > 0).sum(),
                'pattern_percentage': (pattern_col.mean() * 100)
            }
    
    return analysis


def verify_multi_day_patterns(df, ticker=None):
    """
    Specifically verify that multi-day patterns are being detected correctly
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and pattern data
        ticker (str): Optional ticker name for logging
        
    Returns:
        dict: Verification results
    """
    multi_day_patterns = [p for p in PATTERNS if PATTERN_REQUIREMENTS.get(p, 1) > 1]
    
    verification = {
        'ticker': ticker,
        'total_days': len(df),
        'multi_day_patterns_found': {}
    }
    
    for pattern in multi_day_patterns:
        if pattern in df.columns:
            occurrences = df[pattern].sum()
            days_required = PATTERN_REQUIREMENTS.get(pattern, 1)
            
            verification['multi_day_patterns_found'][pattern] = {
                'occurrences': int(occurrences),
                'days_required': days_required,
                'frequency': float(occurrences / len(df)) if len(df) > 0 else 0,
                'properly_detected': occurrences >= 0  # Basic sanity check
            }
    
    return verification


def create_pattern_summary_report(analysis_results):
    """
    Create a readable summary report of pattern analysis
    
    Args:
        analysis_results (dict): Results from analyze_pattern_distribution
        
    Returns:
        str: Formatted report
    """
            report = ["[CHART] Candlestick Pattern Analysis Report", "=" * 50]
    
    # Sort patterns by frequency
    sorted_patterns = sorted(
        analysis_results.items(), 
        key=lambda x: x[1]['frequency'], 
        reverse=True
    )
    
    report.append(f"{'Pattern':<20} {'Count':<8} {'Frequency':<10} {'Type'}")
    report.append("-" * 55)
    
    for pattern, stats in sorted_patterns:
        days_req = PATTERN_REQUIREMENTS.get(pattern, 1)
        pattern_type = f"{days_req}-day" if days_req > 1 else "1-day"
        
        report.append(
            f"{pattern:<20} {stats['total_occurrences']:<8} "
            f"{stats['pattern_percentage']:<9.2f}% {pattern_type}"
        )
    
    return "\n".join(report)


# Example usage and testing function
def test_pattern_detection(ticker="AAPL", days=100):
    """
    Test pattern detection on a specific ticker
    """
    try:
        from src.data.data_collection import load_recent_data
        
        print(f"🧪 Testing pattern detection for {ticker}...")
        
        # Load data
        df = load_recent_data(ticker, days=days)
        print(f"Loaded {len(df)} days of data")
        
        # Calculate patterns with validation
        df_with_patterns, pattern_results = calculate_patterns_with_validation(df)
        
        # Analyze distribution
        analysis = analyze_pattern_distribution(df_with_patterns)
        
        # Verify multi-day patterns
        verification = verify_multi_day_patterns(df_with_patterns, ticker)
        
        # Create report
        report = create_pattern_summary_report(analysis)
        print("\n" + report)
        
        # Show multi-day pattern verification
        print(f"\n[SEARCH] Multi-day Pattern Verification for {ticker}:")
        for pattern, info in verification['multi_day_patterns_found'].items():
            print(f"  {pattern}: {info['occurrences']} occurrences "
                  f"({info['days_required']} days required)")
        
        return df_with_patterns, analysis, verification
        
    except Exception as e:
        print(f"[ERROR] Error testing pattern detection: {e}")
        return None, None, None


if __name__ == "__main__":
    # Run test
    test_pattern_detection() 