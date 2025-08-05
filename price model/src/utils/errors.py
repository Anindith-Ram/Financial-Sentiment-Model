#!/usr/bin/env python3
"""
🛡️ COMPREHENSIVE ERROR HANDLING SYSTEM
====================================

Professional error handling system for financial modeling pipeline with:
- Structured exception hierarchy
- Graceful error recovery
- Detailed logging and monitoring
- Performance tracking
- User-friendly error messages

Author: AI Assistant
Date: 2024
"""

import sys
import traceback
import logging
import functools
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    SYSTEM = "system"
    NETWORK = "network"
    VALIDATION = "validation"


@dataclass
class ErrorInfo:
    """Structured error information"""
    timestamp: str
    category: ErrorCategory
    severity: ErrorSeverity
    error_type: str
    message: str
    context: Dict[str, Any]
    stacktrace: Optional[str] = None
    suggestions: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return data


# Custom Exception Hierarchy
class FinancialPipelineError(Exception):
    """Base exception for financial pipeline"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}


class DataCollectionError(FinancialPipelineError):
    """Data collection specific errors"""
    def __init__(self, message: str, ticker: str = None, **kwargs):
        super().__init__(message, ErrorCategory.DATA_COLLECTION, **kwargs)
        if ticker:
            self.context['ticker'] = ticker


class DataProcessingError(FinancialPipelineError):
    """Data processing specific errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.DATA_PROCESSING, **kwargs)


class DataError(FinancialPipelineError):
    """Data-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.DATA_PROCESSING, **kwargs)


class PipelineError(FinancialPipelineError):
    """General pipeline errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.SYSTEM, **kwargs)


class TrainingError(FinancialPipelineError):
    """Training-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.MODEL_TRAINING, **kwargs)


class ModelTrainingError(FinancialPipelineError):
    """Model training specific errors"""
    def __init__(self, message: str, epoch: int = None, **kwargs):
        super().__init__(message, ErrorCategory.MODEL_TRAINING, **kwargs)
        if epoch is not None:
            self.context['epoch'] = epoch


class ModelInferenceError(FinancialPipelineError):
    """Model inference specific errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.MODEL_INFERENCE, **kwargs)


class ValidationError(FinancialPipelineError):
    """Data/model validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)


class NetworkError(FinancialPipelineError):
    """Network/API related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.HIGH, **kwargs)


class ErrorHandler:
    """
    Comprehensive error handling system with logging, recovery, and monitoring
    """
    
    def __init__(self, log_dir: str = "logs", enable_recovery: bool = True):
        """
        Initialize error handler
        
        Args:
            log_dir: Directory for error logs
            enable_recovery: Enable automatic error recovery
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_recovery = enable_recovery
        self.error_count = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Main logger
        self.logger = logging.getLogger('FinancialPipeline')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = logging.FileHandler(self.log_dir / 'pipeline.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Error-specific file handler
        error_handler = logging.FileHandler(self.log_dir / 'errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """
        Handle and log error with structured information
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            ErrorInfo object with structured error details
        """
        # Extract error information
        if isinstance(error, FinancialPipelineError):
            category = error.category
            severity = error.severity
            error_context = error.context
        else:
            category = ErrorCategory.SYSTEM
            severity = self._infer_severity(error)
            error_context = {}
        
        # Merge context
        if context:
            error_context.update(context)
        
        # Create error info
        error_info = ErrorInfo(
            timestamp=datetime.now().isoformat(),
            category=category,
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            context=error_context,
            stacktrace=traceback.format_exc(),
            suggestions=self._generate_suggestions(error, category)
        )
        
        # Log error
        self._log_error(error_info)
        
        # Track error count
        error_key = f"{category.value}_{type(error).__name__}"
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
        
        # Save error details
        self._save_error_details(error_info)
        
        return error_info
    
    def _infer_severity(self, error: Exception) -> ErrorSeverity:
        """Infer error severity based on exception type"""
        critical_errors = (MemoryError, SystemExit, KeyboardInterrupt)
        high_errors = (ValueError, TypeError, AttributeError, ImportError)
        medium_errors = (FileNotFoundError, PermissionError, ConnectionError)
        
        if isinstance(error, critical_errors):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, high_errors):
            return ErrorSeverity.HIGH
        elif isinstance(error, medium_errors):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate helpful suggestions based on error type and category"""
        suggestions = []
        
        # Category-specific suggestions
        if category == ErrorCategory.DATA_COLLECTION:
            suggestions.extend([
                "Check internet connection and API limits",
                "Verify ticker symbols are valid",
                "Try reducing the date range",
                "Check if market was open on specified dates"
            ])
        
        elif category == ErrorCategory.DATA_PROCESSING:
            suggestions.extend([
                "Check data file format and columns",
                "Verify data types are correct",
                "Check for missing or corrupted data",
                "Ensure sufficient memory is available"
            ])
        
        elif category == ErrorCategory.MODEL_TRAINING:
            suggestions.extend([
                "Check if dataset has sufficient samples",
                "Verify model architecture compatibility",
                "Try reducing batch size or learning rate",
                "Check for GPU memory issues"
            ])
        
        elif category == ErrorCategory.MODEL_INFERENCE:
            suggestions.extend([
                "Verify model file exists and is not corrupted",
                "Check input data format matches training data",
                "Ensure model was properly saved",
                "Try loading model with CPU instead of GPU"
            ])
        
        # Error type-specific suggestions
        error_type = type(error).__name__
        
        if error_type == "FileNotFoundError":
            suggestions.append("Check if file path exists and is accessible")
        elif error_type == "MemoryError":
            suggestions.extend([
                "Reduce batch size",
                "Free up system memory",
                "Use data streaming instead of loading all at once"
            ])
        elif error_type == "ValueError":
            suggestions.append("Check input data format and value ranges")
        elif error_type == "ConnectionError":
            suggestions.extend([
                "Check internet connection",
                "Try again after a few minutes",
                "Use VPN if API is geo-restricted"
            ])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _save_error_details(self, error_info: ErrorInfo):
        """Save detailed error information to JSON file"""
        error_file = self.log_dir / f"error_details_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing errors if file exists
        if error_file.exists():
            try:
                with open(error_file, 'r') as f:
                    errors = json.load(f)
            except:
                errors = []
        else:
            errors = []
        
        # Add new error
        errors.append(error_info.to_dict())
        
        # Save updated errors
        try:
            with open(error_file, 'w') as f:
                json.dump(errors, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save error details: {e}")
    
    def get_error_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get error summary for specified number of days"""
        summary = {
            'total_errors': sum(self.error_count.values()),
            'error_categories': {},
            'top_errors': [],
            'recent_critical_errors': []
        }
        
        # Process error logs from recent days
        for day in range(days):
            date = datetime.now() - pd.Timedelta(days=day)
            error_file = self.log_dir / f"error_details_{date.strftime('%Y%m%d')}.json"
            
            if error_file.exists():
                try:
                    with open(error_file, 'r') as f:
                        daily_errors = json.load(f)
                    
                    for error in daily_errors:
                        category = error.get('category', 'unknown')
                        severity = error.get('severity', 'unknown')
                        
                        # Count by category
                        if category not in summary['error_categories']:
                            summary['error_categories'][category] = 0
                        summary['error_categories'][category] += 1
                        
                        # Track critical errors
                        if severity == 'critical':
                            summary['recent_critical_errors'].append({
                                'timestamp': error.get('timestamp'),
                                'message': error.get('message'),
                                'error_type': error.get('error_type')
                            })
                
                except Exception:
                    continue
        
        # Get top error types
        summary['top_errors'] = sorted(
            self.error_count.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return summary


def error_handler_decorator(category: ErrorCategory = ErrorCategory.SYSTEM, 
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          retry_count: int = 0,
                          retry_delay: float = 1.0):
    """
    Decorator for automatic error handling with optional retry logic
    
    Args:
        category: Error category
        severity: Default error severity
        retry_count: Number of retries (0 = no retry)
        retry_delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            last_error = None
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # Handle error
                    error_info = handler.handle_error(e, {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_attempts': retry_count + 1
                    })
                    
                    # If this is the last attempt or critical error, re-raise
                    if attempt == retry_count or error_info.severity == ErrorSeverity.CRITICAL:
                        raise
                    
                    # Wait before retry
                    if retry_delay > 0:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
            # This should never be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return=None, 
                log_errors: bool = True, **kwargs) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Return value if function fails
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            handler = ErrorHandler()
            handler.handle_error(e, {
                'function': func.__name__,
                'safe_execution': True
            })
        return default_return


class PerformanceMonitor:
    """Monitor performance and detect potential issues"""
    
    def __init__(self):
        self.metrics = {}
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self._record_metric(func_name, execution_time, success)
                return result
            return wrapper
        return decorator
    
    def _record_metric(self, func_name: str, execution_time: float, success: bool):
        """Record performance metric"""
        if func_name not in self.metrics:
            self.metrics[func_name] = {
                'total_calls': 0,
                'total_time': 0,
                'successful_calls': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
        
        metric = self.metrics[func_name]
        metric['total_calls'] += 1
        metric['total_time'] += execution_time
        metric['min_time'] = min(metric['min_time'], execution_time)
        metric['max_time'] = max(metric['max_time'], execution_time)
        
        if success:
            metric['successful_calls'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        report = {}
        for func_name, metric in self.metrics.items():
            avg_time = metric['total_time'] / metric['total_calls']
            success_rate = metric['successful_calls'] / metric['total_calls']
            
            report[func_name] = {
                'average_time': avg_time,
                'min_time': metric['min_time'],
                'max_time': metric['max_time'],
                'total_calls': metric['total_calls'],
                'success_rate': success_rate,
                'total_time': metric['total_time']
            }
        
        return report


# Global instances
error_handler = ErrorHandler()
performance_monitor = PerformanceMonitor()


def main():
    """Demo of error handling system"""
    print("🛡️ Error Handling System Demo")
    
    # Test different error types
    try:
        raise DataCollectionError("Failed to download AAPL data", ticker="AAPL")
    except Exception as e:
        error_info = error_handler.handle_error(e)
        print(f"Handled error: {error_info.message}")
        print(f"Suggestions: {error_info.suggestions}")
    
    # Test safe execution
    def risky_function():
        raise ValueError("Something went wrong")
    
    result = safe_execute(risky_function, default_return="Safe default")
    print(f"Safe execution result: {result}")
    
    # Test performance monitoring
    @performance_monitor.time_function("demo_function")
    def demo_function():
        time.sleep(0.1)
        return "Demo complete"
    
    demo_function()
    report = performance_monitor.get_performance_report()
    print(f"Performance report: {report}")


if __name__ == "__main__":
    main()