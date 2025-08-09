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
            "start_date": "2010-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "sequence_length": 5,
            "tickers": ["AAPL"],  # start with one highly liquid stock for rapid iteration
            # Labeling configuration
            # Options: 'rolling_quantiles', 'absolute_bands', 'triple_barrier'
            "label_mode": "triple_barrier",
            # Absolute bands (fallback)
            "neutral_band": 0.002,
            "up_down_threshold": 0.0035,
            # Triple-barrier params
            "tb_atr_window": 20,
            "tb_up_mult": 1.0,
            "tb_dn_mult": 1.0,
            "tb_t_limit": 5,
            # Context features
            "use_spy_context": True,
            "use_sector_context": True,
            # Cross-sectional ranks
            "compute_cross_sectional": True,
            # Liquidity/quality filters
            "adv_window": 20,
            "min_dollar_vol": 5_000_000.0,
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

        # Always compute basic pandas-based features FIRST so downstream steps don't break
        # Price action features (decomposed returns and shape)
        data['ret_close_close'] = data['Close'].pct_change()
        data['ret_open_open'] = data['Open'].pct_change()
        data['ret_overnight'] = data['Open'] / data['Close'].shift(1) - 1.0
        data['ret_intraday'] = data['Close'] / data['Open'] - 1.0
        data['high_low_ratio'] = data['High'] / data['Low']
        data['close_open_ratio'] = data['Close'] / data['Open']
        # Candle shape metrics
        body = (data['Close'] - data['Open']).abs()
        range_ = (data['High'] - data['Low']).replace(0, np.nan)
        upper_wick = (data['High'] - data[['Open','Close']].max(axis=1)).clip(lower=0)
        lower_wick = (data[['Open','Close']].min(axis=1) - data['Low']).clip(lower=0)
        data['body_frac'] = (body / range_).fillna(0)
        data['upper_wick_frac'] = (upper_wick / range_).fillna(0)
        data['lower_wick_frac'] = (lower_wick / range_).fillna(0)
        # High-Low realized variance estimators (robust even without talib)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_hl = np.log(data['High'] / data['Low']).replace([np.inf, -np.inf], np.nan)
            log_co = np.log((data['Close'] / data['Open']).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        data['rv_parkinson'] = (1.0 / (4.0 * np.log(2))) * (log_hl ** 2)
        data['rv_garman_klass'] = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        data['rv_parkinson'] = data['rv_parkinson'].fillna(0)
        data['rv_garman_klass'] = data['rv_garman_klass'].fillna(0)
        
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
                # Keltner Channels (EMA + ATR)
                ema = talib.EMA(close_price, timeperiod=20)
                data['kc_upper'] = ema + 2.0 * data['atr']
                data['kc_lower'] = ema - 2.0 * data['atr']
                kc_width = (data['kc_upper'] - data['kc_lower']).replace(0, np.nan)
                data['kc_position'] = ((data['Close'] - data['kc_lower']) / kc_width).fillna(0)
                # Squeeze conditions and slopes
                data['bb_squeeze_on'] = ((data['bb_upper'] < data['kc_upper']) & (data['bb_lower'] > data['kc_lower'])).astype(int)
                data['kc_width'] = kc_width.fillna(0)
                data['kc_squeeze_on'] = (data['kc_width'] <= data['kc_width'].rolling(20, min_periods=10).quantile(0.2)).astype(int)
                data['kc_slope'] = (ema - pd.Series(ema).shift(1)).fillna(0)
            
            # Volume indicators
            if len(data) >= 20:
                data['obv'] = talib.OBV(close_price, volume)
                data['ad_line'] = talib.AD(high_price, low_price, close_price, volume)
            
            # Candlestick patterns
            data['pattern_hammer'] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
            data['pattern_engulfing'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
            data['pattern_morning_star'] = talib.CDLMORNINGSTAR(open_price, high_price, low_price, close_price)
            data['pattern_evening_star'] = talib.CDLEVENINGSTAR(open_price, high_price, low_price, close_price)
            
            # (pandas-based features already computed above)
            
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

        # Liquidity/quality filters
        adv_win = int(self.config.get('adv_window', 20))
        min_dv = float(self.config.get('min_dollar_vol', 5_000_000.0))
        df['dollar_vol'] = (df['Close'] * df['Volume']).fillna(0)
        df['adv'] = df['dollar_vol'].rolling(adv_win, min_periods=adv_win//2).mean().fillna(method='bfill')
        df = df[df['adv'] >= min_dv].copy()

        label_mode = self.config.get('label_mode', 'rolling_quantiles')
        if label_mode == 'absolute_bands':
            neutral_band = float(self.config.get('neutral_band', 0.0015))
            thr = float(self.config.get('up_down_threshold', 0.003))
            r = df['ret1']
            label3 = np.where(r <= -thr, 0, np.where(np.abs(r) <= neutral_band, 1, 2)).astype(int)
            label3 = np.where(r.isna(), 1, label3)
        elif label_mode == 'triple_barrier':
            # Volatility-scaled triple-barrier
            atr_win = int(self.config.get('tb_atr_window', 20))
            up_mult = float(self.config.get('tb_up_mult', 1.0))
            dn_mult = float(self.config.get('tb_dn_mult', 1.0))
            t_limit = int(self.config.get('tb_t_limit', 5))
            # ATR as volatility proxy
            atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_win)
            atr = pd.Series(atr, index=df.index).fillna(method='bfill')
            labels = []
            for i in range(len(df)):
                if i + 1 >= len(df):
                    labels.append(1)
                    continue
                entry = df.iloc[i]['Close']
                up = entry + up_mult * atr.iloc[i]
                dn = entry - dn_mult * atr.iloc[i]
                end = min(i + t_limit, len(df) - 1)
                path = df.iloc[i+1:end+1]
                hit_up = (path['High'] >= up).any()
                hit_dn = (path['Low'] <= dn).any()
                if hit_up and not hit_dn:
                    labels.append(2)
                elif hit_dn and not hit_up:
                    labels.append(0)
                else:
                    # time limit: compare end close vs entry (neutral band around 0)
                    ret_tl = (df.iloc[end]['Close'] / entry) - 1.0
                    labels.append(2 if ret_tl > 0 else (0 if ret_tl < 0 else 1))
            label3 = np.array(labels, dtype=int)
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
        # Volatility: realized and EWMA
        ret = df['Close'].pct_change()
        df['vol_5'] = ret.rolling(5).std()
        df['vol_10'] = ret.rolling(10).std()
        df['vol_20'] = ret.rolling(20).std()
        lam = 0.94
        ewma_var = ret.ewm(alpha=(1-lam), adjust=False).var()
        df['ewma_vol'] = np.sqrt(ewma_var).fillna(method='bfill')
        # Realized skew/kurtosis over 20 days
        df['rv_skew_20'] = ret.rolling(20).skew().fillna(0)
        df['rv_kurt_20'] = ret.rolling(20).kurt().fillna(0)
        df['turnover'] = (df['Volume'] / (df['Volume'].rolling(20).mean().replace(0, np.nan))).fillna(0)
        df['overnight_gap'] = (df['Open'] / df['Close'].shift(1) - 1)
        # Simple regime flags
        vol20 = df['vol_20'].fillna(method='ffill')
        df['regime_high_vol'] = (vol20 > vol20.rolling(60, min_periods=20).median()).astype(int)
        df['regime_trending'] = (df['mom_20'].abs() > df['mom_20'].rolling(60, min_periods=20).median().abs()).astype(int)

        # Context: market/sector returns and residualization; earnings proximity
        if bool(self.config.get('use_spy_context', True)):
            try:
                spy = yf.download('SPY', start=df['Date'].min(), end=df['Date'].max(), auto_adjust=True, progress=False)
                spy = spy.rename(columns={'Close': 'SPY_Close'})[['SPY_Close']]
                spy['Date'] = spy.index.tz_localize(None).date
                spy['SPY_ret'] = spy['SPY_Close'].pct_change()
                df = df.merge(spy[['Date','SPY_ret']], on='Date', how='left')
            except Exception:
                df['SPY_ret'] = 0.0
        if bool(self.config.get('use_sector_context', True)):
            # Map Ticker -> Sector ETF (coarse GICS mapping; customize as needed)
            sector_map = {
                'XLB': ['LIN','APD','SHW','ECL','ALB','DD'],
                'XLE': ['XOM','CVX','SLB','COP','EOG','PSX'],
                'XLF': ['JPM','BAC','WFC','C','GS','MS'],
                'XLI': ['HON','UPS','CAT','UNP','RTX','BA'],
                'XLK': ['AAPL','MSFT','NVDA','AVGO','ADBE','CSCO'],
                'XLP': ['PG','KO','PEP','WMT','COST','MDLZ'],
                'XLU': ['NEE','DUK','SO','D','AEP','EXC'],
                'XLV': ['UNH','JNJ','LLY','ABBV','PFE','TMO'],
                'XLY': ['AMZN','HD','MCD','NKE','SBUX','TJX'],
                'XLRE': ['AMT','PLD','CCI','EQIX','PSA','SPG'],
                'XLC': ['GOOGL','META','NFLX','TMUS','VZ','CMCSA']
            }
            tkr = str(df['Ticker'].iloc[0]).upper()
            # Find sector ETF by membership list (fallback to XLK for AAPL)
            sector_etf = 'XLK'
            for etf, members in sector_map.items():
                if tkr in members:
                    sector_etf = etf
                    break
            try:
                sex = yf.download(sector_etf, start=df['Date'].min(), end=df['Date'].max(), auto_adjust=True, progress=False)
                sex = sex.rename(columns={'Close': f'{sector_etf}_Close'})[[f'{sector_etf}_Close']]
                sex['Date'] = sex.index.tz_localize(None).date
                sex[f'{sector_etf}_ret'] = sex[f'{sector_etf}_Close'].pct_change()
                df = df.merge(sex[['Date',f'{sector_etf}_ret']], on='Date', how='left')
                df['SECTOR_ret'] = df[f'{sector_etf}_ret']
            except Exception:
                df['SECTOR_ret'] = 0.0

        # Earnings proximity flags (best-effort via yfinance calendar; may be sparse)
        try:
            cal = yf.Ticker(df['Ticker'].iloc[0]).calendar
            if cal is not None and 'Earnings Date' in cal.index:
                ed = cal.loc['Earnings Date'].values
                ed_dates = pd.to_datetime(pd.Series(ed).astype(str), errors='coerce').dt.date.dropna().unique()
                if len(ed_dates) > 0:
                    ed_df = pd.DataFrame({'Date': ed_dates, 'earn_flag': 1})
                    df = df.merge(ed_df, on='Date', how='left')
                    df['earn_flag'] = df['earn_flag'].fillna(0)
                    # Proximity windows: simple rolling window around earnings
                    df['earn_soon_5d'] = df['earn_flag'].rolling(5, min_periods=1).max().fillna(0)
            else:
                df['earn_flag'] = 0
                df['earn_soon_5d'] = 0
        except Exception:
            df['earn_flag'] = 0
            df['earn_soon_5d'] = 0

        # Residualized returns (market and sector)
        if 'SPY_ret' in df.columns:
            roll = 60
            # simple rolling beta via covariance/variance
            cov = df['ret_close_close'].rolling(roll).cov(df['SPY_ret'])
            var = df['SPY_ret'].rolling(roll).var().replace(0, np.nan)
            beta = (cov / var).fillna(0)
            df['residual_ret'] = (df['ret_close_close'] - beta * df['SPY_ret']).fillna(0)
        else:
            df['residual_ret'] = df['ret_close_close']
        if 'SECTOR_ret' in df.columns:
            roll = 60
            cov_s = df['ret_close_close'].rolling(roll).cov(df['SECTOR_ret'])
            var_s = df['SECTOR_ret'].rolling(roll).var().replace(0, np.nan)
            beta_s = (cov_s / var_s).fillna(0)
            df['residual_ret_sector'] = (df['ret_close_close'] - beta_s * df['SECTOR_ret']).fillna(0)
        else:
            df['residual_ret_sector'] = df['ret_close_close']

        # Multiple-horizon residuals
        for h in [5]:
            ret_h = df['Close'].pct_change(h)
            if 'SPY_ret' in df.columns:
                spy_h = df['SPY_ret'].rolling(h).sum().fillna(0)
                cov_h = ret_h.rolling(60).cov(spy_h)
                var_h = spy_h.rolling(60).var().replace(0, np.nan)
                beta_h = (cov_h / var_h).fillna(0)
                df[f'residual_ret_{h}d'] = (ret_h - beta_h * spy_h).fillna(0)
            else:
                df[f'residual_ret_{h}d'] = ret_h.fillna(0)

        # Cross-sectional z-scores/ranks per day (only meaningful with multi-stock; still computed safely)
        if bool(self.config.get('compute_cross_sectional', True)):
            for col in ['ret_close_close','ret_intraday','ret_overnight','vol_20','turnover','dollar_vol']:
                if col in df.columns:
                    g = df.groupby('Date')[col]
                    df[f'{col}_z_cs'] = (g.transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))).fillna(0)
                    df[f'{col}_rank_cs'] = g.rank(pct=True).fillna(0)

        # Calendar features
        dt = pd.to_datetime(df['Date'])
        df['dow'] = dt.dt.weekday
        df['dom'] = dt.dt.day
        df['month'] = dt.dt.month
        # Simple U.S. holiday feature via pandas (best-effort)
        try:
            import pandas.tseries.holiday as ph
            cal = ph.USFederalHolidayCalendar()
            hol = cal.holidays(start=dt.min(), end=dt.max())
            df['is_holiday'] = dt.isin(hol).astype(int)
        except Exception:
            df['is_holiday'] = 0

        # Get feature columns (exclude metadata & helper columns not needed at inference-time)
        exclude_cols = ['Date', 'Ticker', 'ret1', 'adv']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
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
        coverage_threshold = 1.0
        non_na_ratio = final_dataset[core_cols].notna().mean(axis=1)
        final_dataset = final_dataset[non_na_ratio >= coverage_threshold].copy()
        final_dataset = final_dataset.dropna(subset=['Label','Ticker'])
        final_count = len(final_dataset)
        logger.info(f"🧹 Removed {initial_count - final_count:,} rows after {coverage_threshold*100}% feature coverage filter")
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