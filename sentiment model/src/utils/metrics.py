import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score
)
from typing import Dict, List, Tuple, Optional

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted')
    }

def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str] = None) -> Dict[str, float]:
    """Compute per-class metrics."""
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name.lower()}'] = precision[i]
        metrics[f'recall_{class_name.lower()}'] = recall[i]
        metrics[f'f1_{class_name.lower()}'] = f1[i]
        metrics[f'support_{class_name.lower()}'] = support[i]
    
    return metrics

def compute_roc_auc_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC metrics for multi-class classification."""
    try:
        roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr')
        roc_auc_ovo = roc_auc_score(y_true, y_proba, multi_class='ovo')
        return {
            'roc_auc_ovr': roc_auc_ovr,
            'roc_auc_ovo': roc_auc_ovo
        }
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC metrics: {e}")
        return {}

def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics derived from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics from confusion matrix
    tn = cm[0, 0]  # True negatives
    fp = cm[0, 1] + cm[0, 2]  # False positives
    fn = cm[1, 0] + cm[2, 0]  # False negatives
    tp = cm[1, 1] + cm[2, 2]  # True positives
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'specificity': specificity,
        'sensitivity': sensitivity,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def compute_financial_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            confidence_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute financial-specific metrics."""
    metrics = {}
    
    # Sentiment distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    
    for label, count in zip(unique, counts):
        metrics[f'sentiment_distribution_class_{label}'] = count / total
    
    # High confidence predictions
    if confidence_scores is not None:
        high_conf_mask = confidence_scores > 0.8
        if np.any(high_conf_mask):
            high_conf_accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
            metrics['high_confidence_accuracy'] = high_conf_accuracy
            metrics['high_confidence_ratio'] = np.sum(high_conf_mask) / len(confidence_scores)
    
    return metrics

def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_proba: Optional[np.ndarray] = None,
                                confidence_scores: Optional[np.ndarray] = None,
                                class_names: List[str] = None) -> Dict[str, float]:
    """Compute all comprehensive metrics."""
    metrics = {}
    
    # Basic metrics
    metrics.update(compute_basic_metrics(y_true, y_pred))
    
    # Per-class metrics
    metrics.update(compute_per_class_metrics(y_true, y_pred, class_names))
    
    # ROC-AUC metrics
    if y_proba is not None:
        metrics.update(compute_roc_auc_metrics(y_true, y_proba))
    
    # Confusion matrix metrics
    metrics.update(compute_confusion_matrix_metrics(y_true, y_pred))
    
    # Financial metrics
    metrics.update(compute_financial_metrics(y_true, y_pred, confidence_scores))
    
    return metrics

def calculate_confidence_metrics(predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """Calculate confidence-related metrics."""
    confidence_scores = np.max(probabilities, axis=1)
    
    return {
        'mean_confidence': np.mean(confidence_scores),
        'std_confidence': np.std(confidence_scores),
        'min_confidence': np.min(confidence_scores),
        'max_confidence': np.max(confidence_scores),
        'high_confidence_ratio': np.sum(confidence_scores > 0.8) / len(confidence_scores),
        'low_confidence_ratio': np.sum(confidence_scores < 0.5) / len(confidence_scores)
    }

def find_error_patterns(y_true: np.ndarray, y_pred: np.ndarray, 
                       texts: List[str], confidence_scores: Optional[np.ndarray] = None) -> Dict:
    """Analyze error patterns in predictions."""
    errors = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            error = {
                'text': texts[i],
                'true_label': int(true),
                'predicted_label': int(pred),
                'error_type': f'{int(true)}_to_{int(pred)}'
            }
            if confidence_scores is not None:
                error['confidence'] = float(confidence_scores[i])
            errors.append(error)
    
    # Analyze error patterns
    error_types = {}
    for error in errors:
        error_type = error['error_type']
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    return {
        'total_errors': len(errors),
        'error_types': error_types,
        'error_details': errors
    }

def generate_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, 
                                     class_names: List[str] = None) -> Dict:
    """Generate a detailed classification report as a dictionary."""
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                output_dict=True, zero_division=0)
    return report 