#!/usr/bin/env python3
"""
📊 PROFESSIONAL DATA COLLECTION
===============================

Professional data collection system that stores data in the correct locations:
- Data files → price model/data/
- Configuration → Uses project-level settings
- Quality reporting → Structured output

Features:
- Smart data collection with NaN optimization
- Technical indicators with adaptive periods
- Quality scoring and validation
- Professional error handling
- Progress tracking with tqdm
"""

import pandas as pd
import numpy as np
import yfinance as yf
import talib
from pathlib import Path
import sys
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Professional imports
try:
    from src.utils.logger import setup_logger
    from src.utils.errors import DataError, ErrorHandler
    logger = setup_logger(__name__)
    error_handler = ErrorHandler()
    PROFESSIONAL_FEATURES = True
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    PROFESSIONAL_FEATURES = False

warnings.filterwarnings('ignore')


class ProfessionalDataCollector:
    """
    Professional data collection system with proper file organization
    """
    
    def __init__(self):
        """Initialize the data collector"""
        
        # Paths - Store data in project-level directories
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            "start_date": "2020-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "sequence_length": 5,
            "tickers": self._get_default_tickers()
        }
        
        logger.info(f"🏗️ Data Collector initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"🤖 Models directory: {self.models_dir}")
    
    def _get_default_tickers(self) -> List[str]:
        """Get default list of tickers"""
        
        # Try to load from CSV if it exists
        tickers_file = self.project_root / "data" / "tickers.csv"
        if tickers_file.exists():
            try:
                df = pd.read_csv(tickers_file)
                tickers = df['Ticker'].tolist()
                logger.info(f"📋 Loaded {len(tickers)} tickers from {tickers_file.name}")
                return tickers
            except Exception as e:
                logger.warning(f"Could not load tickers from CSV: {e}")
        
        # Default tickers if file doesn't exist
        default_tickers = [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            # Finance 
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP',
            # Industrial
            'BA', 'CAT', 'GE', 'MMM',
            # Energy
            'XOM', 'CVX', 'COP'
        ]
        
        logger.info(f"📋 Using {len(default_tickers)} default tickers")
        return default_tickers
    
    def collect_ticker_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect data for a single ticker"""
        
        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"⚠️ No data found for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            
            # Ensure proper column names
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
            
            # Remove unnecessary columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
            
            logger.debug(f"✅ Collected {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            if PROFESSIONAL_FEATURES:
                error_handler.handle_error(e, {"ticker": ticker, "operation": "data_collection"})
            logger.error(f"❌ Failed to collect data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with NaN optimization"""
        
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Convert to numpy arrays for talib
        open_price = data['Open'].values
        high_price = data['High'].values
        low_price = data['Low'].values
        close_price = data['Close'].values
        volume = data['Volume'].values
        
        try:
            # Trend indicators with adaptive periods
            min_period_short = min(20, len(data) // 4)
            min_period_long = min(50, len(data) // 2)
            
            if len(data) >= min_period_short:
                data['sma_20'] = talib.SMA(close_price, timeperiod=min_period_short)
                data['ema_20'] = talib.EMA(close_price, timeperiod=min_period_short)
            
            if len(data) >= min_period_long:
                data['sma_50'] = talib.SMA(close_price, timeperiod=min_period_long)
                data['ema_50'] = talib.EMA(close_price, timeperiod=min_period_long)
            
            # Momentum indicators
            if len(data) >= 14:
                data['rsi'] = talib.RSI(close_price, timeperiod=14)
                data['stoch_k'], data['stoch_d'] = talib.STOCH(high_price, low_price, close_price)
                data['williams_r'] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
            
            # MACD
            if len(data) >= 26:
                data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(close_price)
            
            # Volatility indicators
            if len(data) >= 20:
                data['atr'] = talib.ATR(high_price, low_price, close_price, timeperiod=20)
                data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(close_price, timeperiod=20)
                data['bb_width'] = data['bb_upper'] - data['bb_lower']
                data['bb_position'] = (close_price - data['bb_lower']) / data['bb_width']
            
            # Volume indicators
            if len(data) >= 20:
                data['obv'] = talib.OBV(close_price, volume)
                data['ad_line'] = talib.AD(high_price, low_price, close_price, volume)
            
            # Candlestick patterns
            data['pattern_hammer'] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
            data['pattern_engulfing'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
            data['pattern_morning_star'] = talib.CDLMORNINGSTAR(open_price, high_price, low_price, close_price)
            data['pattern_evening_star'] = talib.CDLEVENINGSTAR(open_price, high_price, low_price, close_price)
            
            # Price action features
            data['daily_return'] = data['Close'].pct_change()
            data['high_low_ratio'] = data['High'] / data['Low']
            data['close_open_ratio'] = data['Close'] / data['Open']
            
            logger.debug(f"✅ Calculated technical indicators")
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
        
        return data
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 5) -> pd.DataFrame:
        """Create sequences and labels for training"""
        
        if len(df) < sequence_length + 1:
            logger.warning(f"⚠️ Insufficient data for sequences: {len(df)} < {sequence_length + 1}")
            return pd.DataFrame()
        
        sequences = []
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker']]
        
        for i in range(len(df) - sequence_length):
            # Get sequence data
            sequence_data = df.iloc[i:i+sequence_length][feature_cols].values.flatten()
            
            # Calculate label (price direction for next day)
            current_close = df.iloc[i+sequence_length-1]['Close']
            next_close = df.iloc[i+sequence_length]['Close']
            
            # 5-class labeling
            pct_change = (next_close - current_close) / current_close
            
            if pct_change <= -0.02:
                label = 0  # Strong Down
            elif pct_change <= -0.005:
                label = 1  # Down
            elif pct_change <= 0.005:
                label = 2  # Neutral
            elif pct_change <= 0.02:
                label = 3  # Up
            else:
                label = 4  # Strong Up
            
            # Create sequence record
            sequence_record = {f'feature_{j}': val for j, val in enumerate(sequence_data)}
            sequence_record['Label'] = label
            sequence_record['Ticker'] = df.iloc[i+sequence_length]['Ticker']
            
            sequences.append(sequence_record)
        
        if sequences:
            sequences_df = pd.DataFrame(sequences)
            logger.debug(f"✅ Created {len(sequences_df)} sequences")
            return sequences_df
        else:
            return pd.DataFrame()
    
    def collect_dataset(self, mode: str = "smart_update", **kwargs) -> bool:
        """Main data collection function"""
        
        logger.info(f"🚀 Starting data collection (mode: {mode})")
        
        # Get parameters
        tickers = kwargs.get('tickers', self.config['tickers'])
        start_date = kwargs.get('start_date', self.config['start_date'])
        end_date = kwargs.get('end_date', self.config['end_date'])
        
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        logger.info(f"📋 Collecting data for {len(tickers)} tickers")
        
        all_sequences = []
        failed_tickers = []
        
        # Collect data for each ticker with progress bar
        for ticker in tqdm(tickers, desc="Collecting data"):
            
            # Collect raw data
            ticker_data = self.collect_ticker_data(ticker, start_date, end_date)
            
            if ticker_data.empty:
                failed_tickers.append(ticker)
                continue
            
            # Calculate technical indicators
            ticker_data = self.calculate_technical_indicators(ticker_data)
            
            # Create sequences
            ticker_sequences = self.create_sequences(ticker_data, self.config['sequence_length'])
            
            if not ticker_sequences.empty:
                all_sequences.append(ticker_sequences)
        
        if not all_sequences:
            logger.error("❌ No data collected successfully")
            return False
        
        # Combine all sequences
        logger.info("🔗 Combining sequences...")
        final_dataset = pd.concat(all_sequences, ignore_index=True)
        
        # Remove rows with NaN values
        initial_count = len(final_dataset)
        final_dataset = final_dataset.dropna()
        final_count = len(final_dataset)
        
        logger.info(f"🧹 Removed {initial_count - final_count:,} rows with NaN values")
        logger.info(f"✅ Final dataset: {final_count:,} samples")
        
        # Save dataset to correct location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"professional_dataset_{timestamp}.csv"
        dataset_path = self.data_dir / dataset_filename
        
        final_dataset.to_csv(dataset_path, index=False)
        logger.info(f"💾 Dataset saved: {dataset_path}")
        
        # Create latest dataset symlink
        latest_path = self.data_dir / "latest_dataset.csv"
        if latest_path.exists():
            latest_path.unlink()
        
        # For Windows, just copy the file
        final_dataset.to_csv(latest_path, index=False)
        logger.info(f"🔗 Latest dataset: {latest_path}")
        
        # Generate quality report
        self._generate_quality_report(final_dataset, dataset_path, failed_tickers)
        
        logger.info("✅ Data collection completed successfully!")
        return True
    
    def _generate_quality_report(self, dataset: pd.DataFrame, dataset_path: Path, failed_tickers: List[str]):
        """Generate data quality report"""
        
        logger.info("📋 Generating quality report...")
        
        # Calculate statistics
        total_samples = len(dataset)
        feature_count = len([col for col in dataset.columns if col.startswith('feature_')])
        unique_tickers = dataset['Ticker'].nunique()
        label_distribution = dataset['Label'].value_counts().sort_index()
        
        # Create report
        report = {
            "collection_timestamp": datetime.now().isoformat(),
            "dataset_path": str(dataset_path),
            "statistics": {
                "total_samples": total_samples,
                "feature_count": feature_count,
                "unique_tickers": unique_tickers,
                "failed_tickers": len(failed_tickers),
                "success_rate": f"{((unique_tickers) / (unique_tickers + len(failed_tickers)) * 100):.1f}%"
            },
            "label_distribution": label_distribution.to_dict(),
            "failed_tickers": failed_tickers
        }
        
        # Save report
        report_path = dataset_path.with_suffix('.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Quality report saved: {report_path}")
        
        # Print summary
        logger.info(f"📈 Collection Summary:")
        logger.info(f"  📊 Total samples: {total_samples:,}")
        logger.info(f"  🎯 Features: {feature_count}")
        logger.info(f"  📋 Tickers: {unique_tickers}")
        logger.info(f"  ✅ Success rate: {report['statistics']['success_rate']}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="🚀 Professional Data Collection")
    parser.add_argument('--mode', default='smart_update', choices=['full', 'smart_update', 'incremental'])
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to collect')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = ProfessionalDataCollector()
    
    # Prepare kwargs
    kwargs = {}
    if args.start_date:
        kwargs['start_date'] = args.start_date
    if args.end_date:
        kwargs['end_date'] = args.end_date
    if args.tickers:
        kwargs['tickers'] = args.tickers
    
    # Run collection
    success = collector.collect_dataset(mode=args.mode, **kwargs)
    
    if success:
        print("✅ Data collection completed successfully!")
        print("📁 Data saved to: price model/data/")
        print("🔗 Latest dataset: price model/data/latest_dataset.csv")
    else:
        print("❌ Data collection failed!")
        sys.exit(1)


# For backwards compatibility
def build_dataset(*args, **kwargs):
    """Legacy function for backwards compatibility"""
    collector = ProfessionalDataCollector()
    return collector.collect_dataset(**kwargs)


def load_recent_data(*args, **kwargs):
    """Legacy function for backwards compatibility"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    latest_file = data_dir / "latest_dataset.csv"
    
    if latest_file.exists():
        return pd.read_csv(latest_file)
    else:
        # Find most recent dataset
        datasets = list(data_dir.glob("*dataset*.csv"))
        if datasets:
            latest = max(datasets, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest)
        else:
            logger.error("No dataset found!")
            return pd.DataFrame()


if __name__ == "__main__":
    main()