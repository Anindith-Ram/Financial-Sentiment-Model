"""
Model architectures for financial time series prediction
"""

# Import available models
try:
    from .cnn_model import CandleCNN, EnhancedCandleCNN
    __all__ = ['CandleCNN', 'EnhancedCandleCNN']
except ImportError:
    __all__ = []

try:
    from .enhanced_cnn_integration import create_gpt2_enhanced_cnn
    __all__.append('create_gpt2_enhanced_cnn')
except ImportError:
    pass 