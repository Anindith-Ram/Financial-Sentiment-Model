"""
Trading Signals Module
Provides optimal entry and exit recommendations based on model predictions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.inference.predict import CandlestickPredictor
from src.data.data_collection import load_recent_data
import talib


class TradingSignalGenerator:
    """
    Generates trading signals with optimal entry/exit points
    """
    
    def __init__(self, model_path):
        """
        Initialize with trained model
        
        Args:
            model_path (str): Path to trained model file
        """
        self.predictor = CandlestickPredictor(model_path)
        self.confidence_threshold = 0.7  # Minimum confidence for signals
        
    def get_trading_signal(self, ticker, days_back=10):
        """
        Get comprehensive trading signal for a ticker
        
        Args:
            ticker (str): Stock ticker
            days_back (int): Days of historical data to analyze
            
        Returns:
            dict: Trading signal with entry/exit recommendations
        """
        try:
            # Get recent data with features
            df = load_recent_data(ticker, days_back)
            
            # Get model prediction
            prediction_result = self.predictor.predict_single(ticker)
            
            # Calculate technical indicators for confirmation
            technical_signals = self._calculate_technical_signals(df)
            
            # Generate comprehensive trading signal
            signal = self._generate_comprehensive_signal(
                ticker, prediction_result, technical_signals, df
            )
            
            return signal
            
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'signal': 'ERROR',
                'recommendation': 'Unable to generate signal'
            }
    
    def _calculate_technical_signals(self, df):
        """
        Calculate technical indicators for signal confirmation
        
        Args:
            df (pd.DataFrame): Recent data with features
            
        Returns:
            dict: Technical signals
        """
        close = df['Close_adj'].values.astype(np.float64)
        high = df['High_adj'].values.astype(np.float64)
        low = df['Low_adj'].values.astype(np.float64)
        volume = df['Volume_adj'].values.astype(np.float64)
        
        signals = {}
        
        # RSI
        signals['rsi'] = talib.RSI(close, timeperiod=14)[-1]
        signals['rsi_signal'] = 'OVERSOLD' if signals['rsi'] < 30 else 'OVERBOUGHT' if signals['rsi'] > 70 else 'NEUTRAL'
        
        # MACD
        macd, macd_signal, _ = talib.MACD(close)
        signals['macd'] = macd[-1]
        signals['macd_signal'] = macd_signal[-1]
        signals['macd_bullish'] = macd[-1] > macd_signal[-1]
        
        # Moving Averages
        signals['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
        signals['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
        signals['above_sma_20'] = close[-1] > signals['sma_20']
        signals['above_sma_50'] = close[-1] > signals['sma_50']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        signals['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        signals['near_bb_upper'] = close[-1] > bb_upper[-1] * 0.98
        signals['near_bb_lower'] = close[-1] < bb_lower[-1] * 1.02
        
        # Volume
        signals['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
        signals['volume_ratio'] = volume[-1] / signals['volume_sma']
        signals['high_volume'] = signals['volume_ratio'] > 1.5
        
        return signals
    
    def _generate_comprehensive_signal(self, ticker, prediction, technical, df):
        """
        Generate comprehensive trading signal
        
        Args:
            ticker (str): Stock ticker
            prediction (dict): Model prediction result
            technical (dict): Technical signals
            df (pd.DataFrame): Recent data
            
        Returns:
            dict: Comprehensive trading signal
        """
        current_price = df['Close_adj'].iloc[-1]
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Base signal from model
        model_signal = 'BUY' if prediction['prediction'] == 1 else 'SELL'
        confidence = prediction.get('confidence', 0.5)
        
        # Technical confirmation
        technical_score = 0
        confirmations = []
        warnings = []
        
        # RSI confirmation
        if model_signal == 'BUY' and technical['rsi_signal'] == 'OVERSOLD':
            technical_score += 1
            confirmations.append('RSI oversold - good entry point')
        elif model_signal == 'SELL' and technical['rsi_signal'] == 'OVERBOUGHT':
            technical_score += 1
            confirmations.append('RSI overbought - good exit point')
        
        # MACD confirmation
        if model_signal == 'BUY' and technical['macd_bullish']:
            technical_score += 1
            confirmations.append('MACD bullish - trend confirmation')
        elif model_signal == 'SELL' and not technical['macd_bullish']:
            technical_score += 1
            confirmations.append('MACD bearish - trend confirmation')
        
        # Moving average confirmation
        if model_signal == 'BUY' and technical['above_sma_20']:
            technical_score += 1
            confirmations.append('Above 20-day SMA - bullish trend')
        elif model_signal == 'SELL' and not technical['above_sma_20']:
            technical_score += 1
            confirmations.append('Below 20-day SMA - bearish trend')
        
        # Volume confirmation
        if technical['high_volume']:
            technical_score += 1
            confirmations.append('High volume - strong signal')
        
        # Bollinger Band position
        if model_signal == 'BUY' and technical['near_bb_lower']:
            technical_score += 1
            confirmations.append('Near Bollinger Band lower - potential bounce')
        elif model_signal == 'SELL' and technical['near_bb_upper']:
            technical_score += 1
            confirmations.append('Near Bollinger Band upper - potential reversal')
        
        # Calculate overall signal strength
        signal_strength = (confidence + technical_score / 5) / 2
        
        # Generate recommendation
        if signal_strength >= 0.8:
            recommendation = f"STRONG {model_signal}"
        elif signal_strength >= 0.6:
            recommendation = f"MODERATE {model_signal}"
        else:
            recommendation = f"WEAK {model_signal}"
        
        # Entry/Exit points
        if model_signal == 'BUY':
            entry_price = current_price
            stop_loss = entry_price * 0.95  # 5% stop loss
            take_profit = entry_price * 1.15  # 15% take profit
            exit_strategy = f"Exit at ${take_profit:.2f} or stop loss at ${stop_loss:.2f}"
        else:
            exit_price = current_price
            entry_strategy = "Consider selling current position"
            exit_strategy = f"Exit at current price: ${exit_price:.2f}"
        
        return {
            'ticker': ticker,
            'date': current_date,
            'current_price': current_price,
            'model_prediction': model_signal,
            'model_confidence': confidence,
            'technical_score': technical_score,
            'signal_strength': signal_strength,
            'recommendation': recommendation,
            'confirmations': confirmations,
            'warnings': warnings,
            'entry_price': entry_price if model_signal == 'BUY' else None,
            'stop_loss': stop_loss if model_signal == 'BUY' else None,
            'take_profit': take_profit if model_signal == 'BUY' else None,
            'exit_strategy': exit_strategy,
            'technical_indicators': technical
        }
    
    def get_portfolio_signals(self, tickers, days_back=10):
        """
        Get trading signals for multiple tickers
        
        Args:
            tickers (list): List of ticker symbols
            days_back (int): Days of historical data
            
        Returns:
            list: List of trading signals
        """
        signals = []
        
        for ticker in tickers:
            signal = self.get_trading_signal(ticker, days_back)
            signals.append(signal)
        
        # Sort by signal strength
        signals.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
        
        return signals
    
    def print_signal_report(self, signal):
        """
        Print formatted trading signal report
        
        Args:
            signal (dict): Trading signal
        """
        print(f"\n{'='*60}")
        print(f"📊 TRADING SIGNAL REPORT - {signal['ticker']}")
        print(f"{'='*60}")
        print(f"📅 Date: {signal['date']}")
        print(f"💰 Current Price: ${signal['current_price']:.2f}")
        print(f"🎯 Model Prediction: {signal['model_prediction']}")
        print(f"📈 Model Confidence: {signal['model_confidence']:.1%}")
        print(f"🔧 Technical Score: {signal['technical_score']}/5")
        print(f"💪 Signal Strength: {signal['signal_strength']:.1%}")
        print(f"✅ Recommendation: {signal['recommendation']}")
        
        if signal['confirmations']:
            print(f"\n✅ Confirmations:")
            for conf in signal['confirmations']:
                print(f"   • {conf}")
        
        if signal['warnings']:
            print(f"\n⚠️  Warnings:")
            for warn in signal['warnings']:
                print(f"   • {warn}")
        
        if signal['model_prediction'] == 'BUY':
            print(f"\n📈 Entry Strategy:")
            print(f"   • Entry Price: ${signal['entry_price']:.2f}")
            print(f"   • Stop Loss: ${signal['stop_loss']:.2f} (-5%)")
            print(f"   • Take Profit: ${signal['take_profit']:.2f} (+15%)")
            print(f"   • Risk/Reward: 1:3")
        else:
            print(f"\n📉 Exit Strategy:")
            print(f"   • {signal['exit_strategy']}")
        
        print(f"{'='*60}")


def demo_trading_signals():
    """
    Demo function to show trading signals in action
    """
    print("🎯 Trading Signal Demo")
    print("=" * 40)
    
    # Initialize signal generator (you'll need to train the model first)
    try:
        generator = TradingSignalGenerator('models/cnn5cls.pth.best')
        
        # Get signals for popular stocks
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        print(f"📊 Getting signals for {len(tickers)} stocks...")
        signals = generator.get_portfolio_signals(tickers)
        
        # Print top signals
        print(f"\n🏆 TOP TRADING OPPORTUNITIES:")
        for i, signal in enumerate(signals[:3], 1):
            if 'error' not in signal:
                print(f"{i}. {signal['ticker']}: {signal['recommendation']} "
                      f"(Strength: {signal['signal_strength']:.1%})")
        
        # Detailed report for best signal
        if signals and 'error' not in signals[0]:
            generator.print_signal_report(signals[0])
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to train the model first with: python main.py --mode train")


if __name__ == "__main__":
    demo_trading_signals() 