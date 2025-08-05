"""
Price Model Package
Candlestick Pattern Recognition and Stock Price Prediction
"""

__version__ = "1.0.0"
__author__ = "Financial Analysis Model Suite"

# Core imports for easy access
try:
    from .data.data_collection import build_dataset, load_recent_data
    from .models.cnn_model import CandleCNN
    # Standard training removed - using progressive training only
    from .inference.predict import CandlestickPredictor, predict_next_open
    from .utils.helpers import label_class, trade_signal, get_signal_description
    
    __all__ = [
        'build_dataset',
        'load_recent_data', 
        'CandleCNN',
        'CandlestickPredictor',
        'predict_next_open',
        'label_class',
        'trade_signal',
        'get_signal_description'
    ]
except ImportError:
    # Handle cases where dependencies might not be installed
    __all__ = [] 