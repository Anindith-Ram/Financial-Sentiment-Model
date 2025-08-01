"""
Inference and prediction functions for the candlestick pattern model
"""
import torch
import numpy as np
import talib
from datetime import datetime, timedelta

from config.config import (
    SEQ_LEN, DEVICE, MODEL_OUTPUT_PATH, USE_RAW_COLS, USE_ADJ_COLS, PATTERN_FLAGS
)
from src.models.cnn_model import CandleCNN
from src.data.data_collection import load_recent_data
from src.utils.helpers import trade_signal, get_signal_description


class CandlestickPredictor:
    """
    Predictor class for making candlestick pattern predictions
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor with professional data pipeline
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path or MODEL_OUTPUT_PATH
        self.device = DEVICE
        self.model = None
        self.features_per_day = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"
            print("CUDA not available, using CPU")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model dimensions
            self.features_per_day = checkpoint.get('features_per_day')
            if self.features_per_day is None:
                # Fallback: will be determined from data
                print("Warning: features_per_day not found in model, will determine from data")
            
            # Initialize and load model
            if self.features_per_day:
                self.model = CandleCNN(self.features_per_day)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                print(f"Model loaded successfully from: {self.model_path}")
                print(f"Features per day: {self.features_per_day}")
                print("Using professional data pipeline (explicit raw/adjusted columns)")
            else:
                print("Model will be initialized when first prediction is made")
            
        except Exception as e:
            print(f"Warning: Could not load model from {self.model_path}: {e}")
            print("Model will need to be trained first")
    
    def preprocess_data(self, df):
        """
        Preprocess dataframe for prediction
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and features
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        if len(df) < SEQ_LEN:
            raise ValueError(f"Need at least {SEQ_LEN} days of data, got {len(df)}")
        
        # Take the last SEQ_LEN days
        window = df.tail(SEQ_LEN)
        
        # Organize features by source (professional approach)
        raw_features = [col for col in window.columns if col in USE_RAW_COLS]
        adj_features = [col for col in window.columns if col in USE_ADJ_COLS]
        pattern_features = [col for col in window.columns if col in PATTERN_FLAGS]
        other_features = [col for col in window.columns 
                         if col not in ['Date'] + raw_features + adj_features + pattern_features]
        
        # Combine in logical order: raw + patterns + adjusted + indicators
        feature_columns = raw_features + pattern_features + adj_features + other_features
        
        # Initialize model if not already done
        if self.model is None and self.features_per_day is None:
            self.features_per_day = len(feature_columns)
            self.model = CandleCNN(self.features_per_day)
            
            # Try to load state dict if available
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model initialized with {self.features_per_day} features per day")
            except:
                print(f"Warning: Could not load model weights, using untrained model")
            
            self.model.to(self.device)
            self.model.eval()
        
        # Extract features in the correct order
        features = window[feature_columns].values.astype(np.float32)
        
        # Handle NaN values
        if np.isnan(features).any():
            print("Warning: NaN values detected, filling with zeros")
            features = np.nan_to_num(features)
        
        # Reshape to (1, seq_len, features_per_day) for batch processing
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        return features_tensor
    
    def predict_single(self, ticker, days_history=10):
        """
        Make a prediction for a single ticker
        
        Args:
            ticker (str): Stock ticker symbol
            days_history (int): Number of days of history to use
            
        Returns:
            dict: Prediction results with signal, confidence, etc.
        """
        try:
            # Load recent data using professional pipeline
            df = load_recent_data(ticker, days=days_history)
            
            # Preprocess
            features = self.preprocess_data(df)
            
            # Make prediction
            if self.model is None:
                return {
                    'ticker': ticker,
                    'error': 'Model not loaded. Please train the model first.',
                    'timestamp': datetime.now().isoformat()
                }
            
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).cpu().item()
                confidence = probabilities.max().cpu().item()
            
            # Get signal description
            signal_desc = get_signal_description(predicted_class)
            
            return {
                'ticker': ticker,
                'signal': predicted_class,
                'signal_description': signal_desc,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten().tolist(),
                'timestamp': datetime.now().isoformat(),
                'data_pipeline': 'professional_raw_adjusted_explicit'
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, tickers, days_history=10):
        """
        Make predictions for multiple tickers
        
        Args:
            tickers (list): List of ticker symbols
            days_history (int): Number of days of history to use
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for ticker in tickers:
            print(f"Predicting for {ticker}...")
            result = self.predict_single(ticker, days_history)
            results.append(result)
        
        return results
    
    def get_trading_recommendations(self, tickers, min_confidence=0.6):
        """
        Get trading recommendations based on predictions
        
        Args:
            tickers (list): List of ticker symbols
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            dict: Trading recommendations categorized by action
        """
        predictions = self.predict_batch(tickers)
        
        recommendations = {
            'strong_buy': [],
            'buy': [],
            'hold': [],
            'sell': [],
            'strong_sell': [],
            'errors': []
        }
        
        for pred in predictions:
            if 'error' in pred:
                recommendations['errors'].append(pred)
                continue
            
            if pred['confidence'] < min_confidence:
                continue  # Skip low confidence predictions
            
            signal = pred['signal']
            ticker_info = {
                'ticker': pred['ticker'],
                'confidence': pred['confidence'],
                'signal_description': pred['signal_description']
            }
            
            if signal == 4:  # Strong Buy
                recommendations['strong_buy'].append(ticker_info)
            elif signal == 3:  # Buy
                recommendations['buy'].append(ticker_info)
            elif signal == 2:  # Hold
                recommendations['hold'].append(ticker_info)
            elif signal == 1:  # Sell
                recommendations['sell'].append(ticker_info)
            elif signal == 0:  # Strong Sell
                recommendations['strong_sell'].append(ticker_info)
        
        return recommendations


def predict_next_open(ticker="AAPL", model_path=None):
    """
    Convenience function to predict next day's movement for a single ticker
    
    Args:
        ticker (str): Stock ticker symbol
        model_path (str): Path to model file (optional)
        
    Returns:
        int: Predicted signal (0-4)
    """
    predictor = CandlestickPredictor(model_path)
    result = predictor.predict_single(ticker)
    
    if 'error' in result:
        print(f"Error predicting {ticker}: {result['error']}")
        return None
    
    return result['signal']


def batch_predict(tickers, model_path=None, output_file=None):
    """
    Make batch predictions and optionally save to file
    
    Args:
        tickers (list): List of ticker symbols
        model_path (str): Path to model file (optional)
        output_file (str): Path to save results (optional)
        
    Returns:
        list: Prediction results
    """
    predictor = CandlestickPredictor(model_path)
    results = predictor.predict_batch(tickers)
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results


# Example usage functions
def demo_prediction():
    """Demo function showing how to use the predictor"""
    print("Running candlestick pattern prediction demo...")
    
    # Test with a few popular stocks
    test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    predictor = CandlestickPredictor()
    
    print("\nUsing professional data pipeline (explicit raw/adjusted columns)")
    print("\nSingle predictions:")
    for ticker in test_tickers:
        result = predictor.predict_single(ticker)
        if 'error' not in result:
            print(f"{ticker}: {result['signal_description']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"{ticker}: Error - {result['error']}")
    
    print("\nTrading recommendations:")
    recommendations = predictor.get_trading_recommendations(test_tickers)
    
    for action, stocks in recommendations.items():
        if stocks and action != 'errors':
            print(f"\n{action.upper()}:")
            for stock in stocks:
                print(f"  {stock['ticker']}: {stock['signal_description']} ({stock['confidence']:.3f})")


if __name__ == "__main__":
    demo_prediction() 