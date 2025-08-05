#!/usr/bin/env python3
"""Training metrics tracking"""

from typing import Dict, List, Any


class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, metric: str) -> float:
        """Get latest metric value"""
        return self.metrics.get(metric, [0])[-1]
    
    def get_average(self, metric: str, last_n: int = 10) -> float:
        """Get average of last N values"""
        values = self.metrics.get(metric, [])
        if not values:
            return 0.0
        return sum(values[-last_n:]) / min(len(values), last_n)