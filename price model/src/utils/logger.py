#!/usr/bin/env python3
"""
📋 STRUCTURED LOGGING SYSTEM
============================

Professional logging system with structured output and performance tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured logger with professional formatting
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "finmodel.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class StructuredLogger:
    """Structured logger for JSON-formatted logs"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(name)
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: int = logging.INFO):
        """Log structured event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        self.logger.log(level, json.dumps(event))