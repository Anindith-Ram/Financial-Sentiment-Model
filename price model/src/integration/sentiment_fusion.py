"""
Integration utilities for combining Price Model with Sentiment Model
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json


class SignalFusion:
    """
    Combine price model and sentiment model signals for enhanced predictions
    """
    
    def __init__(self, price_weight: float = 0.6, sentiment_weight: float = 0.4):
        """
        Initialize signal fusion
        
        Args:
            price_weight (float): Weight for price model signals
            sentiment_weight (float): Weight for sentiment model signals
        """
        self.price_weight = price_weight
        self.sentiment_weight = sentiment_weight
        
        # Ensure weights sum to 1
        total_weight = price_weight + sentiment_weight
        self.price_weight = price_weight / total_weight
        self.sentiment_weight = sentiment_weight / total_weight
        
        # Signal mapping
        self.price_signal_map = {
            0: -2,  # Strong Sell
            1: -1,  # Mild Sell  
            2: 0,   # Hold
            3: 1,   # Mild Buy
            4: 2    # Strong Buy
        }
        
        self.sentiment_signal_map = {
            'very_negative': -2,
            'negative': -1,
            'neutral': 0,
            'positive': 1,
            'very_positive': 2
        }
    
    def normalize_price_signal(self, price_signal: int, confidence: float = 1.0) -> float:
        """
        Normalize price signal to [-2, 2] range with confidence weighting
        
        Args:
            price_signal (int): Price signal (0-4)
            confidence (float): Confidence score (0-1)
            
        Returns:
            float: Normalized signal
        """
        base_signal = self.price_signal_map.get(price_signal, 0)
        return base_signal * confidence
    
    def normalize_sentiment_signal(self, sentiment_signal: Union[str, float], 
                                 confidence: float = 1.0) -> float:
        """
        Normalize sentiment signal to [-2, 2] range
        
        Args:
            sentiment_signal (Union[str, float]): Sentiment signal or score
            confidence (float): Confidence score (0-1)
            
        Returns:
            float: Normalized signal
        """
        if isinstance(sentiment_signal, str):
            base_signal = self.sentiment_signal_map.get(sentiment_signal.lower(), 0)
        else:
            # Assume sentiment_signal is a float in [-1, 1] or [0, 1]
            if -1 <= sentiment_signal <= 1:
                base_signal = sentiment_signal * 2  # Scale to [-2, 2]
            else:
                # Assume [0, 1] scale, convert to [-2, 2]
                base_signal = (sentiment_signal - 0.5) * 4
        
        return base_signal * confidence
    
    def combine_signals(self, price_signal: int, sentiment_signal: Union[str, float],
                       price_confidence: float = 1.0, sentiment_confidence: float = 1.0) -> Dict:
        """
        Combine price and sentiment signals
        
        Args:
            price_signal (int): Price model signal (0-4)
            sentiment_signal (Union[str, float]): Sentiment signal
            price_confidence (float): Price model confidence
            sentiment_confidence (float): Sentiment model confidence
            
        Returns:
            dict: Combined signal analysis
        """
        # Normalize signals
        norm_price = self.normalize_price_signal(price_signal, price_confidence)
        norm_sentiment = self.normalize_sentiment_signal(sentiment_signal, sentiment_confidence)
        
        # Weighted combination
        combined_signal = (norm_price * self.price_weight + 
                          norm_sentiment * self.sentiment_weight)
        
        # Convert back to action categories
        if combined_signal <= -1.5:
            action = "STRONG_SELL"
            action_code = 0
        elif combined_signal <= -0.5:
            action = "SELL"
            action_code = 1
        elif combined_signal <= 0.5:
            action = "HOLD"
            action_code = 2
        elif combined_signal <= 1.5:
            action = "BUY"
            action_code = 3
        else:
            action = "STRONG_BUY"
            action_code = 4
        
        # Calculate overall confidence
        overall_confidence = (price_confidence * self.price_weight + 
                            sentiment_confidence * self.sentiment_weight)
        
        # Agreement score (how well signals align)
        agreement = 1 - abs(norm_price - norm_sentiment) / 4.0
        
        return {
            'combined_signal': combined_signal,
            'action': action,
            'action_code': action_code,
            'overall_confidence': overall_confidence,
            'agreement_score': agreement,
            'price_component': norm_price,
            'sentiment_component': norm_sentiment,
            'price_weight': self.price_weight,
            'sentiment_weight': self.sentiment_weight
        }
    
    def batch_combine_signals(self, price_signals: List[int], sentiment_signals: List,
                             price_confidences: Optional[List[float]] = None,
                             sentiment_confidences: Optional[List[float]] = None) -> List[Dict]:
        """
        Combine signals for multiple predictions
        
        Args:
            price_signals (List[int]): List of price signals
            sentiment_signals (List): List of sentiment signals
            price_confidences (List[float], optional): Price confidences
            sentiment_confidences (List[float], optional): Sentiment confidences
            
        Returns:
            List[Dict]: List of combined signal results
        """
        if price_confidences is None:
            price_confidences = [1.0] * len(price_signals)
        if sentiment_confidences is None:
            sentiment_confidences = [1.0] * len(sentiment_signals)
        
        results = []
        for i in range(len(price_signals)):
            result = self.combine_signals(
                price_signals[i], sentiment_signals[i],
                price_confidences[i], sentiment_confidences[i]
            )
            results.append(result)
        
        return results


class TradingRecommendationEngine:
    """
    Generate trading recommendations using combined signals
    """
    
    def __init__(self, min_confidence: float = 0.6, min_agreement: float = 0.5):
        """
        Initialize recommendation engine
        
        Args:
            min_confidence (float): Minimum confidence for recommendations
            min_agreement (float): Minimum agreement between models
        """
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.fusion = SignalFusion()
    
    def generate_recommendation(self, ticker: str, price_prediction: Dict, 
                              sentiment_prediction: Dict) -> Dict:
        """
        Generate trading recommendation for a single ticker
        
        Args:
            ticker (str): Stock ticker
            price_prediction (dict): Price model prediction
            sentiment_prediction (dict): Sentiment model prediction
            
        Returns:
            dict: Trading recommendation
        """
        # Extract signals and confidences
        price_signal = price_prediction.get('signal', 2)
        price_confidence = price_prediction.get('confidence', 0.5)
        
        sentiment_signal = sentiment_prediction.get('sentiment', 'neutral')
        sentiment_confidence = sentiment_prediction.get('confidence', 0.5)
        
        # Combine signals
        combined = self.fusion.combine_signals(
            price_signal, sentiment_signal, price_confidence, sentiment_confidence
        )
        
        # Determine recommendation quality
        high_confidence = combined['overall_confidence'] >= self.min_confidence
        high_agreement = combined['agreement_score'] >= self.min_agreement
        
        if high_confidence and high_agreement:
            recommendation_quality = "HIGH"
        elif high_confidence or high_agreement:
            recommendation_quality = "MEDIUM"
        else:
            recommendation_quality = "LOW"
        
        return {
            'ticker': ticker,
            'recommendation': combined['action'],
            'recommendation_code': combined['action_code'],
            'quality': recommendation_quality,
            'confidence': combined['overall_confidence'],
            'agreement': combined['agreement_score'],
            'price_signal': {
                'signal': price_signal,
                'confidence': price_confidence,
                'description': price_prediction.get('signal_description', 'Unknown')
            },
            'sentiment_signal': {
                'signal': sentiment_signal,
                'confidence': sentiment_confidence,
                'description': sentiment_prediction.get('description', 'Unknown')
            },
            'combined_signal': combined['combined_signal'],
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_portfolio_recommendations(self, tickers: List[str], 
                                         price_predictions: List[Dict],
                                         sentiment_predictions: List[Dict]) -> Dict:
        """
        Generate recommendations for a portfolio of stocks
        
        Args:
            tickers (List[str]): List of tickers
            price_predictions (List[Dict]): Price model predictions
            sentiment_predictions (List[Dict]): Sentiment model predictions
            
        Returns:
            dict: Portfolio recommendations
        """
        recommendations = []
        
        for i, ticker in enumerate(tickers):
            if i < len(price_predictions) and i < len(sentiment_predictions):
                rec = self.generate_recommendation(
                    ticker, price_predictions[i], sentiment_predictions[i]
                )
                recommendations.append(rec)
        
        # Categorize recommendations
        portfolio_analysis = {
            'strong_buy': [],
            'buy': [],
            'hold': [],
            'sell': [],
            'strong_sell': [],
            'high_quality': [],
            'medium_quality': [],
            'low_quality': []
        }
        
        for rec in recommendations:
            action = rec['recommendation'].lower()
            quality = rec['quality'].lower()
            
            # By action
            if action == 'strong_buy':
                portfolio_analysis['strong_buy'].append(rec)
            elif action == 'buy':
                portfolio_analysis['buy'].append(rec)
            elif action == 'hold':
                portfolio_analysis['hold'].append(rec)
            elif action == 'sell':
                portfolio_analysis['sell'].append(rec)
            elif action == 'strong_sell':
                portfolio_analysis['strong_sell'].append(rec)
            
            # By quality
            portfolio_analysis[f'{quality}_quality'].append(rec)
        
        # Summary statistics
        total_recs = len(recommendations)
        portfolio_analysis['summary'] = {
            'total_stocks': total_recs,
            'high_quality_count': len(portfolio_analysis['high_quality']),
            'buy_signals': len(portfolio_analysis['strong_buy']) + len(portfolio_analysis['buy']),
            'sell_signals': len(portfolio_analysis['strong_sell']) + len(portfolio_analysis['sell']),
            'avg_confidence': np.mean([r['confidence'] for r in recommendations]) if recommendations else 0,
            'avg_agreement': np.mean([r['agreement'] for r in recommendations]) if recommendations else 0
        }
        
        return portfolio_analysis
    
    def create_trading_report(self, portfolio_recommendations: Dict) -> str:
        """
        Create a formatted trading report
        
        Args:
            portfolio_recommendations (dict): Portfolio recommendations
            
        Returns:
            str: Formatted report
        """
        report = ["📈 Trading Recommendations Report", "=" * 50]
        
        summary = portfolio_recommendations['summary']
        report.append(f"\n📊 Portfolio Summary:")
        report.append(f"  Total Stocks Analyzed: {summary['total_stocks']}")
        report.append(f"  High Quality Signals: {summary['high_quality_count']}")
        report.append(f"  Buy Signals: {summary['buy_signals']}")
        report.append(f"  Sell Signals: {summary['sell_signals']}")
        report.append(f"  Average Confidence: {summary['avg_confidence']:.3f}")
        report.append(f"  Average Agreement: {summary['avg_agreement']:.3f}")
        
        # High quality recommendations
        high_quality = portfolio_recommendations['high_quality']
        if high_quality:
            report.append(f"\n⭐ High Quality Recommendations:")
            for rec in high_quality[:10]:  # Top 10
                report.append(f"  {rec['ticker']}: {rec['recommendation']} "
                             f"(Conf: {rec['confidence']:.3f}, Agree: {rec['agreement']:.3f})")
        
        # Strong buy recommendations
        strong_buys = portfolio_recommendations['strong_buy']
        if strong_buys:
            report.append(f"\n🚀 Strong Buy Signals:")
            for rec in strong_buys[:5]:  # Top 5
                report.append(f"  {rec['ticker']}: Confidence {rec['confidence']:.3f}")
        
        # Strong sell recommendations
        strong_sells = portfolio_recommendations['strong_sell']
        if strong_sells:
            report.append(f"\n⚠️ Strong Sell Signals:")
            for rec in strong_sells[:5]:  # Top 5
                report.append(f"  {rec['ticker']}: Confidence {rec['confidence']:.3f}")
        
        return "\n".join(report)


def save_recommendations(recommendations: Dict, filepath: str):
    """
    Save recommendations to JSON file
    
    Args:
        recommendations (dict): Recommendations dictionary
        filepath (str): Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)


def load_recommendations(filepath: str) -> Dict:
    """
    Load recommendations from JSON file
    
    Args:
        filepath (str): Input file path
        
    Returns:
        dict: Loaded recommendations
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# Example integration demonstration
def demo_integration():
    """
    Demonstrate integration between price and sentiment models
    """
    print("🔗 Price + Sentiment Model Integration Demo")
    print("=" * 50)
    
    # Example data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    # Mock price model predictions
    price_predictions = [
        {'signal': 4, 'confidence': 0.85, 'signal_description': 'Strong Buy'},
        {'signal': 3, 'confidence': 0.72, 'signal_description': 'Mild Buy'},
        {'signal': 2, 'confidence': 0.65, 'signal_description': 'Hold'},
        {'signal': 1, 'confidence': 0.78, 'signal_description': 'Mild Sell'}
    ]
    
    # Mock sentiment model predictions
    sentiment_predictions = [
        {'sentiment': 'positive', 'confidence': 0.80, 'description': 'Positive news sentiment'},
        {'sentiment': 'very_positive', 'confidence': 0.90, 'description': 'Very positive outlook'},
        {'sentiment': 'neutral', 'confidence': 0.60, 'description': 'Mixed signals'},
        {'sentiment': 'negative', 'confidence': 0.75, 'description': 'Negative market sentiment'}
    ]
    
    # Generate recommendations
    engine = TradingRecommendationEngine(min_confidence=0.6, min_agreement=0.5)
    portfolio_recs = engine.generate_portfolio_recommendations(
        tickers, price_predictions, sentiment_predictions
    )
    
    # Create and print report
    report = engine.create_trading_report(portfolio_recs)
    print(report)
    
    # Show individual recommendations
    print(f"\n🔍 Individual Analysis:")
    for ticker in tickers:
        for rec in portfolio_recs['high_quality'] + portfolio_recs['medium_quality']:
            if rec['ticker'] == ticker:
                print(f"{ticker}:")
                print(f"  Combined Action: {rec['recommendation']}")
                print(f"  Quality: {rec['quality']}")
                print(f"  Price Signal: {rec['price_signal']['description']}")
                print(f"  Sentiment Signal: {rec['sentiment_signal']['description']}")
                print(f"  Agreement Score: {rec['agreement']:.3f}")
                break


if __name__ == "__main__":
    demo_integration() 