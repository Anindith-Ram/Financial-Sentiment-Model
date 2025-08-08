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
import time

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

# --- Ensure Windows-safe ASCII logging (strip non-ASCII to avoid UnicodeEncodeError) ---
def _strip_non_ascii(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.encode('ascii', 'ignore').decode('ascii')

class _AsciiAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return _strip_non_ascii(msg), kwargs

logger = _AsciiAdapter(logger, {})


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
            "tickers": self._get_default_tickers(),
            # Labeling configuration
            # label_mode: 'rolling_quantiles' (original) or 'absolute_bands'
            "label_mode": "rolling_quantiles",
            # For absolute_bands: define neutral band and up/down thresholds (daily returns)
            # Example: neutral if |ret| <= 0.0015 (0.15%), up if ret >= 0.003 (0.30%), down if ret <= -0.003
            "neutral_band": 0.0015,
            "up_down_threshold": 0.003
        }
        
        logger.info(f"🏗️ Data Collector initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"🤖 Models directory: {self.models_dir}")
    
    def _get_default_tickers(self) -> List[str]:
        """Get default list of tickers"""
        
        # First priority: Try to load from constituents.csv
        constituents_file = self.project_root / "data" / "constituents.csv"
        if constituents_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(constituents_file)
                col = next((c for c in df.columns if c.lower() in ["ticker","symbol","Ticker","Symbol"]), "Symbol")
                tickers = df[col].astype(str).str.upper().str.replace("\.", "-")  # BRK.B → BRK-B
                tickers = tickers.dropna().unique().tolist()
                logger.info(f"📋 Loaded {len(tickers)} S&P 500 tickers from constituents.csv")
                return tickers
            except Exception as e:
                logger.warning(f"Could not load S&P 500 tickers from constituents.csv: {e}")
        
        # Second priority: Try to fetch current S&P500 membership from Wikipedia
        try:
            import pandas as pd
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp = tables[0]
            sym_col = next((c for c in sp.columns if str(c).lower().startswith('symbol')), sp.columns[0])
            tickers = (
                sp[sym_col]
                .astype(str)
                .str.upper()
                .str.replace('\.', '-', regex=False)  # BRK.B -> BRK-B
                .dropna()
                .unique()
                .tolist()
            )
            if len(tickers) >= 450:
                logger.info(f"📋 Loaded {len(tickers)} S&P 500 tickers from Wikipedia (backup)")
                return tickers
        except Exception as e:
            logger.info(f"Could not fetch S&P 500 from Wikipedia; will try other sources. Reason: {e}")

        # Third priority: Try other local CSV files
        sp500_files = [
            self.project_root / "data" / "tickers_sp500.csv",
            self.project_root / "data" / "sp500_tickers.csv",
            self.project_root / "data" / "sp500_constituents.csv",
        ]
        for f in sp500_files:
            if f.exists():
                try:
                    df = pd.read_csv(f)
                    col = next((c for c in df.columns if c.lower() in ["ticker","symbol","Ticker","Symbol"]), "Ticker")
                    tickers = df[col].astype(str).str.upper().str.replace("\.", "-")  # BRK.B → BRK-B
                    tickers = tickers.dropna().unique().tolist()
                    logger.info(f"📋 Loaded {len(tickers)} S&P 500 tickers from {f.name}")
                    return tickers
                except Exception as e:
                    logger.warning(f"Could not load S&P 500 tickers from {f.name}: {e}")
        
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
            
            # Prefer adjusted OHLC if available
            try:
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                if not hist.empty and 'Close' in hist.columns:
                    df = hist.reset_index()[['Date','Open','High','Low','Close','Volume']]
                    df['Ticker'] = ticker
            except Exception:
                pass
            # Ensure necessary columns
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
            # Cast arrays to float64 for ta-lib compatibility and retry key indicators
            try:
                open_price = open_price.astype('float64')
                high_price = high_price.astype('float64')
                low_price = low_price.astype('float64')
                close_price = close_price.astype('float64')
                volume = volume.astype('float64')
                if len(data) >= 20:
                    data['obv'] = talib.OBV(close_price, volume)
            except Exception as e2:
                logger.error(f"Indicator fallback failed: {e2}")
        
        return data
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 5) -> pd.DataFrame:
        """Create sequences and 3-class labels per ticker.

        Supports two labeling modes:
        - 'rolling_quantiles' (default): next-day return vs rolling 33/66% quantiles
        - 'absolute_bands': fixed return bands using config neutral_band and up_down_threshold
        """
        
        if len(df) < sequence_length + 1:
            logger.warning(f"⚠️ Insufficient data for sequences: {len(df)} < {sequence_length + 1}")
            return pd.DataFrame()
        
        sequences = []
        
        # Compute next-day return and helper series
        df = df.sort_values('Date').copy()
        df['ret1'] = df['Close'].pct_change().shift(-1)
        past_ret = df['ret1'].shift(1)

        label_mode = self.config.get('label_mode', 'rolling_quantiles')
        if label_mode == 'absolute_bands':
            neutral_band = float(self.config.get('neutral_band', 0.0015))
            thr = float(self.config.get('up_down_threshold', 0.003))
            r = df['ret1']
            label3 = np.where(r <= -thr, 0, np.where(np.abs(r) <= neutral_band, 1, 2)).astype(int)
            label3 = np.where(r.isna(), 1, label3)
        else:
            # Rolling 33/66% quantiles (original)
            q33 = past_ret.rolling(252, min_periods=60).quantile(0.33)
            q66 = past_ret.rolling(252, min_periods=60).quantile(0.66)
            label3 = np.where(df['ret1'] < q33, 0, np.where(df['ret1'] > q66, 2, 1)).astype(int)
            # Early region fallback
            med = past_ret.rolling(60, min_periods=20).median()
            mask = q33.isna() | q66.isna()
            label3 = np.array(label3)
            if mask.any():
                label3[mask.values] = np.where(df.loc[mask, 'ret1'] < med[mask], 0, 2)
            label3 = np.where(df['ret1'].isna(), 1, label3)

        # Add richer features (momentum, volatility, gaps, simple regime flags)
        df['ret1_abs'] = df['ret1'].abs()
        df['mom_5'] = df['Close'].pct_change(5)
        df['mom_10'] = df['Close'].pct_change(10)
        df['mom_20'] = df['Close'].pct_change(20)
        df['vol_5'] = df['Close'].pct_change().rolling(5).std()
        df['vol_10'] = df['Close'].pct_change().rolling(10).std()
        df['vol_20'] = df['Close'].pct_change().rolling(20).std()
        df['turnover'] = (df['Volume'] / (df['Volume'].rolling(20).mean().replace(0, np.nan))).fillna(0)
        df['overnight_gap'] = (df['Open'] / df['Close'].shift(1) - 1)
        # Simple regime flags
        vol20 = df['vol_20'].fillna(method='ffill')
        df['regime_high_vol'] = (vol20 > vol20.rolling(60, min_periods=20).median()).astype(int)
        df['regime_trending'] = (df['mom_20'].abs() > df['mom_20'].rolling(60, min_periods=20).median().abs()).astype(int)

        # Get feature columns (exclude metadata & helper columns not needed at inference-time)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'ret1']]
        
        for i in range(len(df) - sequence_length):
            # Get sequence data
            sequence_data = df.iloc[i:i+sequence_length][feature_cols].values.flatten()
            
            # 3-class label from rolling quantiles
            label = int(label3[i+sequence_length])
            
            # Create sequence record
            sequence_record = {f'feature_{j}': val for j, val in enumerate(sequence_data)}
            sequence_record['Label'] = label  # 0/1/2
            # Provide continuous target for optional regression head
            sequence_record['TargetRet'] = float(df.iloc[i+sequence_length]['ret1']) if not pd.isna(df.iloc[i+sequence_length]['ret1']) else 0.0
            # Chronological key for Purged CV
            end_dt = pd.to_datetime(df.iloc[i+sequence_length]['Date'])
            sequence_record['EndDate'] = end_dt.strftime('%Y-%m-%d')
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
        
        # Smart NaN handling: drop only rows missing core OHLCV, keep rows with missing secondary indicators
        core_cols = [c for c in final_dataset.columns if c.startswith('feature_')]  # sequence features only
        initial_count = len(final_dataset)
        # Identify OHLCV-derived slots if available (robust fallback: require at least 80% of features present)
        non_na_ratio = final_dataset[core_cols].notna().mean(axis=1)
        final_dataset = final_dataset[non_na_ratio >= 0.8].copy()
        final_dataset = final_dataset.dropna(subset=['Label','Ticker'])
        final_count = len(final_dataset)
        logger.info(f"🧹 Removed {initial_count - final_count:,} rows after 80% feature coverage filter")
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