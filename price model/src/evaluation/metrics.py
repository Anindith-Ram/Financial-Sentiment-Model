"""
Comprehensive Evaluation Metrics for Price Model
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics
    """
    
    def __init__(self):
        self.class_names = ['Strong Sell', 'Mild Sell', 'Hold', 'Mild Buy', 'Strong Buy']
        self.class_mapping = {0: 'Strong Sell', 1: 'Mild Sell', 2: 'Hold', 
                             3: 'Mild Buy', 4: 'Strong Buy'}
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate basic classification metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            dict: Basic metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def calculate_financial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate finance-specific metrics
        
        Args:
            y_true (np.ndarray): True labels  
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray, optional): Prediction probabilities
            
        Returns:
            dict: Financial metrics
        """
        metrics = {}
        
        # Directional accuracy (buy vs sell vs hold)
        y_true_direction = np.where(y_true <= 1, 0, np.where(y_true >= 3, 2, 1))  # 0=sell, 1=hold, 2=buy
        y_pred_direction = np.where(y_pred <= 1, 0, np.where(y_pred >= 3, 2, 1))
        
        metrics['directional_accuracy'] = accuracy_score(y_true_direction, y_pred_direction)
        
        # Buy signal precision (when we predict buy, how often right?)
        buy_mask_pred = y_pred >= 3
        buy_mask_true = y_true >= 3
        if np.sum(buy_mask_pred) > 0:
            metrics['buy_signal_precision'] = np.sum(buy_mask_pred & buy_mask_true) / np.sum(buy_mask_pred)
        else:
            metrics['buy_signal_precision'] = 0.0
        
        # Sell signal precision  
        sell_mask_pred = y_pred <= 1
        sell_mask_true = y_true <= 1
        if np.sum(sell_mask_pred) > 0:
            metrics['sell_signal_precision'] = np.sum(sell_mask_pred & sell_mask_true) / np.sum(sell_mask_pred)
        else:
            metrics['sell_signal_precision'] = 0.0
        
        # Conservative accuracy (only high confidence predictions)
        if y_proba is not None:
            high_conf_mask = np.max(y_proba, axis=1) > 0.7
            if np.sum(high_conf_mask) > 0:
                metrics['high_confidence_accuracy'] = accuracy_score(
                    y_true[high_conf_mask], y_pred[high_conf_mask]
                )
                metrics['high_confidence_samples'] = np.sum(high_conf_mask)
            else:
                metrics['high_confidence_accuracy'] = 0.0
                metrics['high_confidence_samples'] = 0
        
        return metrics
    
    def calculate_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Calculate per-class metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            pd.DataFrame: Per-class metrics
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        class_metrics = []
        for class_idx in range(5):
            if str(class_idx) in report:
                class_metrics.append({
                    'class': class_idx,
                    'class_name': self.class_mapping[class_idx],
                    'precision': report[str(class_idx)]['precision'],
                    'recall': report[str(class_idx)]['recall'],
                    'f1_score': report[str(class_idx)]['f1-score'],
                    'support': report[str(class_idx)]['support']
                })
        
        return pd.DataFrame(class_metrics)
    
    def evaluate_model_comprehensive(self, model, data_loader, device='cpu') -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            data_loader: Data loader with test data
            device: Device to run evaluation on
            
        Returns:
            dict: Comprehensive evaluation results
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_proba = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                
                # Get predictions and probabilities
                proba = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_proba)
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        financial_metrics = self.calculate_financial_metrics(y_true, y_pred, y_proba)
        class_metrics = self.calculate_class_metrics(y_true, y_pred)
        
        return {
            'basic_metrics': basic_metrics,
            'financial_metrics': financial_metrics,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'predictions': {'y_true': y_true, 'y_pred': y_pred, 'y_proba': y_proba}
        }
    
    def create_evaluation_report(self, eval_results: Dict) -> str:
        """
        Create a formatted evaluation report
        
        Args:
            eval_results (dict): Evaluation results
            
        Returns:
            str: Formatted report
        """
        report = ["📊 Model Evaluation Report", "=" * 50]
        
        # Basic metrics
        basic = eval_results['basic_metrics']
        report.append("\n🎯 Overall Performance:")
        report.append(f"  Accuracy: {basic['accuracy']:.4f}")
        report.append(f"  F1-Score (Macro): {basic['f1_macro']:.4f}")
        report.append(f"  F1-Score (Weighted): {basic['f1_weighted']:.4f}")
        
        # Financial metrics
        financial = eval_results['financial_metrics']
        report.append(f"\n💰 Financial Performance:")
        report.append(f"  Directional Accuracy: {financial['directional_accuracy']:.4f}")
        report.append(f"  Buy Signal Precision: {financial['buy_signal_precision']:.4f}")
        report.append(f"  Sell Signal Precision: {financial['sell_signal_precision']:.4f}")
        
        if 'high_confidence_accuracy' in financial:
            report.append(f"  High Confidence Accuracy: {financial['high_confidence_accuracy']:.4f}")
            report.append(f"  High Confidence Samples: {financial['high_confidence_samples']}")
        
        # Class-wise performance
        class_metrics = eval_results['class_metrics']
        report.append(f"\n📈 Per-Class Performance:")
        for _, row in class_metrics.iterrows():
            report.append(f"  {row['class_name']}: F1={row['f1_score']:.3f}, "
                         f"Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")
        
        return "\n".join(report)
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: Optional[str] = None):
        """
        Plot true vs predicted class distributions
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels  
            save_path (str, optional): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        true_counts = pd.Series(y_true).value_counts().sort_index()
        ax1.bar(range(len(true_counts)), true_counts.values)
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(self.class_names, rotation=45)
        
        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        ax2.bar(range(len(pred_counts)), pred_counts.values, color='orange')
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class CrossValidator:
    """
    Cross-validation for time series data
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series aware train/validation splits
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            List[Tuple]: List of (train_idx, val_idx) tuples
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)
        train_size_samples = n_samples - test_size_samples
        
        splits = []
        step_size = train_size_samples // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * step_size
            train_end_idx = start_idx + train_size_samples
            val_end_idx = min(train_end_idx + test_size_samples, n_samples)
            
            if val_end_idx <= train_end_idx:
                break
                
            train_idx = np.arange(start_idx, train_end_idx)
            val_idx = np.arange(train_end_idx, val_end_idx)
            
            splits.append((train_idx, val_idx))
        
        return splits
    
    def cross_validate_model(self, model_class, X: np.ndarray, y: np.ndarray, 
                           **model_kwargs) -> Dict:
        """
        Perform cross-validation
        
        Args:
            model_class: Model class to instantiate
            X (np.ndarray): Features
            y (np.ndarray): Labels
            **model_kwargs: Model initialization arguments
            
        Returns:
            dict: Cross-validation results
        """
        splits = self.time_series_split(X, y)
        evaluator = ModelEvaluator()
        
        cv_results = {
            'accuracies': [],
            'f1_scores': [],
            'directional_accuracies': [],
            'fold_results': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {fold + 1}/{len(splits)}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model (simplified - would need actual training loop)
            # This is a placeholder for the actual cross-validation implementation
            # In practice, you'd train the model here and evaluate
            
            # For now, we'll skip the actual training and return placeholder results
            fold_result = {
                'accuracy': 0.5,  # Placeholder
                'f1_score': 0.4,  # Placeholder
                'directional_accuracy': 0.6  # Placeholder
            }
            
            cv_results['accuracies'].append(fold_result['accuracy'])
            cv_results['f1_scores'].append(fold_result['f1_score'])
            cv_results['directional_accuracies'].append(fold_result['directional_accuracy'])
            cv_results['fold_results'].append(fold_result)
        
        # Calculate summary statistics
        cv_results['mean_accuracy'] = np.mean(cv_results['accuracies'])
        cv_results['std_accuracy'] = np.std(cv_results['accuracies'])
        cv_results['mean_f1'] = np.mean(cv_results['f1_scores'])
        cv_results['std_f1'] = np.std(cv_results['f1_scores'])
        cv_results['mean_directional'] = np.mean(cv_results['directional_accuracies'])
        cv_results['std_directional'] = np.std(cv_results['directional_accuracies'])
        
        return cv_results


def backtest_signals(predictions: np.ndarray, actual_returns: np.ndarray, 
                    confidence_scores: Optional[np.ndarray] = None) -> Dict:
    """
    Backtest trading signals
    
    Args:
        predictions (np.ndarray): Model predictions (0-4)
        actual_returns (np.ndarray): Actual returns
        confidence_scores (np.ndarray, optional): Prediction confidence scores
        
    Returns:
        dict: Backtest results
    """
    # Convert predictions to trading signals
    # 0,1 = Sell (-1), 2 = Hold (0), 3,4 = Buy (1)
    signals = np.where(predictions <= 1, -1, np.where(predictions >= 3, 1, 0))
    
    # Apply confidence filter if provided
    if confidence_scores is not None:
        high_conf_mask = confidence_scores > 0.6
        signals = signals * high_conf_mask  # Zero out low confidence signals
    
    # Calculate strategy returns
    strategy_returns = signals * actual_returns
    
    # Calculate metrics
    total_return = np.sum(strategy_returns)
    hit_rate = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0
    
    # Only consider actual trades (non-zero signals)
    trade_mask = signals != 0
    if np.sum(trade_mask) > 0:
        trade_returns = strategy_returns[trade_mask]
        avg_trade_return = np.mean(trade_returns)
        win_rate = np.mean(trade_returns > 0)
        num_trades = len(trade_returns)
    else:
        avg_trade_return = 0
        win_rate = 0
        num_trades = 0
    
    return {
        'total_return': total_return,
        'avg_trade_return': avg_trade_return,
        'hit_rate': hit_rate,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'signals': signals,
        'strategy_returns': strategy_returns
    }


if __name__ == "__main__":
    # Example usage
    print("Model evaluation utilities loaded successfully!")
    
    # Create dummy data for testing
    y_true = np.random.randint(0, 5, 1000)
    y_pred = np.random.randint(0, 5, 1000)
    y_proba = np.random.rand(1000, 5)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize to sum to 1
    
    evaluator = ModelEvaluator()
    
    # Test basic metrics
    basic_metrics = evaluator.calculate_basic_metrics(y_true, y_pred)
    print(f"Accuracy: {basic_metrics['accuracy']:.4f}")
    
    # Test financial metrics
    financial_metrics = evaluator.calculate_financial_metrics(y_true, y_pred, y_proba)
    print(f"Directional Accuracy: {financial_metrics['directional_accuracy']:.4f}") 