#!/usr/bin/env python3
"""
🎯 STOCK ADVISOR SYSTEM
======================

Ask: "What stocks should I buy today?"
Get: Predictions with confidence, probabilities, and risk metrics!
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import json

warnings.filterwarnings('ignore')
sys.path.append('.')

# Import your models
try:
    from src.training.simple_cnn_trainer import HighPerformanceCNN
    from src.models.advanced_time_series_integration import GPT2EnhancedCNN, create_gpt2_enhanced_cnn
    print("✅ Model imports successful")
except ImportError as e:
    print(f"⚠️ Model import error: {e}")


class StockAdvisor:
    """AI Stock Advisor that provides buy/sell recommendations"""
    
    def __init__(self, model_path: str, model_type: str = "simple_cnn"):
        """
        Initialize the stock advisor
        
        Args:
            model_path: Path to trained model
            model_type: "simple_cnn" or "gpt2_enhanced"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.features_per_day = 62  # Default, will be updated
        
        # Class labels
        self.class_labels = {
            0: "Strong Sell",
            1: "Sell", 
            2: "Hold",
            3: "Buy",
            4: "Strong Buy"
        }
        
        # Risk thresholds
        self.confidence_thresholds = {
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3
        }
        
        self._load_model(model_path)
        print(f"🤖 Stock Advisor initialized with {model_type} model")
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if self.model_type == "simple_cnn":
                self.model = HighPerformanceCNN(features_per_day=self.features_per_day, num_classes=5)
            elif self.model_type == "gpt2_enhanced":
                self.model = create_gpt2_enhanced_cnn(
                    features_per_day=self.features_per_day,
                    hidden_size=768,
                    num_classes=5
                )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded from {model_path}")
            if 'best_val_acc' in checkpoint:
                print(f"📊 Model accuracy: {checkpoint['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def get_live_data(self, tickers: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get live stock data for analysis"""
        
        print(f"📊 Fetching live data for {len(tickers)} stocks...")
        
        stock_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if not df.empty:
                    df.reset_index(inplace=True)
                    stock_data[ticker] = df
                    print(f"  ✅ {ticker}: {len(df)} days of data")
                else:
                    print(f"  ❌ {ticker}: No data available")
                    
            except Exception as e:
                print(f"  ⚠️ {ticker}: Error - {e}")
        
        return stock_data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a stock"""
        
        # Basic indicators that don't require talib
        data = df.copy()
        
        # Simple Moving Averages
        data['sma_5'] = data['Close'].rolling(window=5).mean()
        data['sma_10'] = data['Close'].rolling(window=10).mean()
        data['sma_20'] = data['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        data['ema_5'] = data['Close'].ewm(span=5).mean()
        data['ema_10'] = data['Close'].ewm(span=10).mean()
        
        # Returns and volatility
        data['daily_return'] = data['Close'].pct_change()
        data['volatility'] = data['daily_return'].rolling(window=10).std()
        
        # Price ratios
        data['high_low_ratio'] = data['High'] / data['Low']
        data['close_open_ratio'] = data['Close'] / data['Open']
        
        # Volume indicators
        data['volume_sma'] = data['Volume'].rolling(window=10).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        
        # RSI (simplified)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (simplified)
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        return data
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model prediction"""
        
        # Calculate indicators
        df_with_indicators = self._calculate_technical_indicators(df)
        
        # Select feature columns (simplified set)
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10',
            'daily_return', 'volatility', 'high_low_ratio', 'close_open_ratio',
            'volume_ratio', 'rsi', 'macd', 'macd_signal',
            'bb_position'
        ]
        
        # Get last 5 days for sequence
        df_recent = df_with_indicators.tail(5).copy()
        
        # Fill missing values
        for col in feature_cols:
            if col in df_recent.columns:
                df_recent[col] = df_recent[col].fillna(method='ffill').fillna(0)
            else:
                df_recent[col] = 0
        
        # Extract features
        features = df_recent[feature_cols].values.astype(np.float32)
        
        # Reshape to match model input
        if features.shape[0] == 5 and features.shape[1] == len(feature_cols):
            # Pad features to match expected input size (62 features per day)
            if features.shape[1] < self.features_per_day:
                padding = np.zeros((5, self.features_per_day - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            elif features.shape[1] > self.features_per_day:
                features = features[:, :self.features_per_day]
            
            return features.flatten()  # Flatten for FinancialDataset
        else:
            print(f"⚠️ Feature shape mismatch: {features.shape}")
            return np.zeros(5 * self.features_per_day, dtype=np.float32)
    
    def predict_stock(self, ticker: str, df: pd.DataFrame) -> Dict:
        """Predict recommendation for a single stock"""
        
        try:
            # Prepare features
            features = self._prepare_features(df)
            
            # Convert to tensor
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Reshape for sequence model
            X = X.view(1, 5, self.features_per_day)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get all class probabilities
                class_probs = probabilities[0].cpu().numpy()
            
            # Current price info
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Calculate additional metrics
            volatility = df['Close'].pct_change().tail(10).std() * 100
            volume_avg = df['Volume'].tail(10).mean()
            volume_current = df['Volume'].iloc[-1]
            volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1
            
            # Determine confidence level
            conf_value = confidence.item()
            if conf_value >= self.confidence_thresholds["high"]:
                conf_level = "High"
            elif conf_value >= self.confidence_thresholds["medium"]:
                conf_level = "Medium"
            else:
                conf_level = "Low"
            
            # Create recommendation
            prediction_class = predicted.item()
            recommendation = self.class_labels[prediction_class]
            
            return {
                'ticker': ticker,
                'recommendation': recommendation,
                'confidence': conf_value,
                'confidence_level': conf_level,
                'class_probabilities': {
                    self.class_labels[i]: float(prob) for i, prob in enumerate(class_probs)
                },
                'current_price': float(current_price),
                'price_change_pct': float(price_change),
                'volatility_pct': float(volatility),
                'volume_ratio': float(volume_ratio),
                'risk_level': self._calculate_risk_level(volatility, conf_value),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error predicting {ticker}: {e}")
            return {
                'ticker': ticker,
                'recommendation': "Error",
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_risk_level(self, volatility: float, confidence: float) -> str:
        """Calculate risk level based on volatility and confidence"""
        
        if volatility > 5 or confidence < 0.5:
            return "High Risk"
        elif volatility > 2 or confidence < 0.7:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def analyze_portfolio(self, tickers: List[str]) -> Dict:
        """Analyze multiple stocks and provide portfolio recommendations"""
        
        print(f"\n🎯 ANALYZING PORTFOLIO")
        print("=" * 50)
        
        # Get live data
        stock_data = self.get_live_data(tickers, days=30)
        
        if not stock_data:
            return {'error': 'No data available for any stocks'}
        
        # Analyze each stock
        predictions = []
        for ticker, df in stock_data.items():
            prediction = self.predict_stock(ticker, df)
            predictions.append(prediction)
            
            # Print individual recommendation
            if 'error' not in prediction:
                rec = prediction['recommendation']
                conf = prediction['confidence']
                price = prediction['current_price']
                change = prediction['price_change_pct']
                risk = prediction['risk_level']
                
                emoji = "🚀" if "Buy" in rec else "📉" if "Sell" in rec else "📊"
                print(f"{emoji} {ticker}: {rec} ({conf:.2f} confidence)")
                print(f"    💰 ${price:.2f} ({change:+.1f}%) | {risk}")
        
        # Sort by recommendation strength
        buy_stocks = [p for p in predictions if "Buy" in p.get('recommendation', '')]
        sell_stocks = [p for p in predictions if "Sell" in p.get('recommendation', '')]
        hold_stocks = [p for p in predictions if p.get('recommendation') == 'Hold']
        
        # Sort by confidence
        buy_stocks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        sell_stocks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return {
            'analysis_time': datetime.now().isoformat(),
            'total_stocks': len(predictions),
            'buy_recommendations': buy_stocks,
            'sell_recommendations': sell_stocks,
            'hold_recommendations': hold_stocks,
            'top_buy': buy_stocks[0] if buy_stocks else None,
            'top_sell': sell_stocks[0] if sell_stocks else None,
            'summary': {
                'strong_buys': len([p for p in buy_stocks if p.get('recommendation') == 'Strong Buy']),
                'buys': len([p for p in buy_stocks if p.get('recommendation') == 'Buy']),
                'holds': len(hold_stocks),
                'sells': len([p for p in sell_stocks if p.get('recommendation') == 'Sell']),
                'strong_sells': len([p for p in sell_stocks if p.get('recommendation') == 'Strong Sell'])
            }
        }
    
    def what_to_buy_today(self, tickers: List[str] = None) -> str:
        """Answer: 'What stocks should I buy today?'"""
        
        if tickers is None:
            # Default popular stocks
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        results = self.analyze_portfolio(tickers)
        
        if 'error' in results:
            return f"❌ Error: {results['error']}"
        
        # Generate human-readable response
        response = f"\n🎯 STOCK RECOMMENDATIONS FOR TODAY\n"
        response += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        response += "=" * 50 + "\n\n"
        
        # Top recommendations
        if results['buy_recommendations']:
            response += "🚀 TOP BUY RECOMMENDATIONS:\n"
            for i, stock in enumerate(results['buy_recommendations'][:3], 1):
                ticker = stock['ticker']
                rec = stock['recommendation']
                conf = stock['confidence']
                price = stock['current_price']
                risk = stock['risk_level']
                response += f"{i}. {ticker}: {rec} (Confidence: {conf:.2f})\n"
                response += f"   💰 ${price:.2f} | {risk}\n"
        else:
            response += "📊 No strong buy signals detected today.\n"
        
        # Avoid these
        if results['sell_recommendations']:
            response += f"\n⚠️ AVOID THESE STOCKS:\n"
            for stock in results['sell_recommendations'][:2]:
                ticker = stock['ticker']
                rec = stock['recommendation']
                response += f"❌ {ticker}: {rec}\n"
        
        # Summary
        summary = results['summary']
        response += f"\n📊 PORTFOLIO SUMMARY:\n"
        response += f"🚀 Strong Buys: {summary['strong_buys']}\n"
        response += f"📈 Buys: {summary['buys']}\n"
        response += f"📊 Holds: {summary['holds']}\n"
        response += f"📉 Sells: {summary['sells']}\n"
        response += f"💥 Strong Sells: {summary['strong_sells']}\n"
        
        response += "\n💡 Remember: This is AI analysis, not financial advice!"
        
        return response


def main():
    """Demo the stock advisor"""
    
    # Try to find a trained model
    project_root = Path(__file__).parent.parent.parent
    
    # Look for models
    simple_model = project_root / "models" / "simple_cnn" / "simple_cnn_best.pth"
    research_model = project_root / "models" / "research" / "enhanced_cnn_best.pth"
    
    model_path = None
    model_type = None
    
    if simple_model.exists():
        model_path = simple_model
        model_type = "simple_cnn"
        print(f"📱 Using Simple CNN model: {model_path}")
    elif research_model.exists():
        model_path = research_model
        model_type = "gpt2_enhanced"
        print(f"📱 Using GPT-2 Enhanced model: {model_path}")
    else:
        print("❌ No trained model found!")
        print("Please train a model first using:")
        print("  python src/training/simple_cnn_trainer.py")
        return
    
    try:
        # Initialize advisor
        advisor = StockAdvisor(model_path, model_type)
        
        # Demo query
        print("\n" + "="*60)
        print("🤖 AI STOCK ADVISOR DEMO")
        print("="*60)
        
        response = advisor.what_to_buy_today()
        print(response)
        
    except Exception as e:
        print(f"❌ Error running advisor: {e}")


if __name__ == "__main__":
    main()