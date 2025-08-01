"""
Helper functions for the price model
"""
from config.config import (
    STRONG_UP_THRESHOLD, 
    MILD_UP_THRESHOLD, 
    STRONG_DOWN_THRESHOLD, 
    MILD_DOWN_THRESHOLD
)


def label_class(pct_change):
    """
    Convert percentage change to 5-class label
    
    Args:
        pct_change (float): Percentage change in price
        
    Returns:
        int: Class label (0-4)
            0: Strong Down (<= -2%)
            1: Mild Down (<= -0.5%)  
            2: Flat (-0.5% to 0.5%)
            3: Mild Up (>= 0.5%)
            4: Strong Up (>= 2%)
    """
    if pct_change >= STRONG_UP_THRESHOLD:
        return 4          # Strong Up
    elif pct_change >= MILD_UP_THRESHOLD:
        return 3          # Mild Up
    elif pct_change <= STRONG_DOWN_THRESHOLD:
        return 0          # Strong Down
    elif pct_change <= MILD_DOWN_THRESHOLD:
        return 1          # Mild Down
    else:
        return 2          # Flat


def trade_signal(logits):
    """
    Convert model logits to trade signals
    
    Args:
        logits (torch.Tensor): Model output logits (B, 5)
        
    Returns:
        numpy.ndarray: Trade signals (0-4)
    """
    return logits.argmax(1).cpu().numpy()


def get_signal_description(signal):
    """
    Get human-readable description of trade signal
    
    Args:
        signal (int): Signal value (0-4)
        
    Returns:
        str: Description of the signal
    """
    descriptions = {
        0: "Strong Sell",
        1: "Mild Sell", 
        2: "Hold",
        3: "Mild Buy",
        4: "Strong Buy"
    }
    return descriptions.get(signal, "Unknown") 