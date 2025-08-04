"""
Advanced Training Utilities
Implements all modern ML training improvements for the price model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for reducing overconfidence
    """
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, inputs, targets):
        log_probs = nn.LogSoftmax(dim=1)(inputs)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
            
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)
            
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class EarlyStopping:
    """
    Early stopping with patience and overfitting detection
    """
    def __init__(self, patience=7, min_delta=0.001, overfit_threshold=5):
        self.patience = patience
        self.min_delta = min_delta
        self.overfit_threshold = overfit_threshold
        
        self.best_score = None
        self.counter = 0
        self.overfit_counter = 0
        self.early_stop = False
        
    def __call__(self, val_score, train_loss=None, val_loss=None):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        # Check for overfitting
        if train_loss is not None and val_loss is not None:
            if val_loss > train_loss * 1.1:  # 10% tolerance
                self.overfit_counter += 1
            else:
                self.overfit_counter = 0
                
            if self.overfit_counter >= self.overfit_threshold:
                self.early_stop = True
                print(f"Early stopping due to overfitting (val_loss > train_loss for {self.overfit_threshold} epochs)")


class WarmupScheduler:
    """
    Learning rate warmup scheduler
    """
    def __init__(self, optimizer, warmup_epochs, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.start_lr + (self.target_lr - self.start_lr) * self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1


class AdvancedMetrics:
    """
    Track advanced metrics during training
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.confidences = []
        
    def update(self, preds, targets, confidences=None):
        self.predictions.extend(preds.cpu().detach().numpy())
        self.targets.extend(targets.cpu().detach().numpy())
        if confidences is not None:
            self.confidences.extend(confidences.cpu().detach().numpy())
            
    def compute(self):
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = (preds == targets).mean()
        
        # F1 scores
        f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # Directional accuracy (Buy vs Sell vs Hold)
        pred_direction = np.where(preds <= 1, 0, np.where(preds >= 3, 2, 1))
        target_direction = np.where(targets <= 1, 0, np.where(targets >= 3, 2, 1))
        directional_accuracy = (pred_direction == target_direction).mean()
        
        # Signal precision
        buy_mask_pred = preds >= 3
        buy_mask_target = targets >= 3
        buy_precision = (buy_mask_pred & buy_mask_target).sum() / buy_mask_pred.sum() if buy_mask_pred.sum() > 0 else 0
        
        sell_mask_pred = preds <= 1
        sell_mask_target = targets <= 1
        sell_precision = (sell_mask_pred & sell_mask_target).sum() / sell_mask_pred.sum() if sell_mask_pred.sum() > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'directional_accuracy': directional_accuracy,
            'buy_signal_precision': buy_precision,
            'sell_signal_precision': sell_precision
        }
        
        # High confidence metrics
        if self.confidences:
            confidences = np.array(self.confidences)
            high_conf_mask = confidences > 0.7
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = (preds[high_conf_mask] == targets[high_conf_mask]).mean()
                metrics['high_confidence_accuracy'] = high_conf_accuracy
                metrics['high_confidence_samples'] = high_conf_mask.sum()
                
        return metrics


class TrainingVisualizer:
    """
    Real-time training visualization
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def update(self, train_loss, val_loss, val_accuracy, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        self.learning_rates.append(lr)
        
    def plot(self, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate schedule
        ax3.plot(epochs, self.learning_rates, 'm-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
        ax4.plot(epochs, loss_diff, 'orange', label='Val Loss - Train Loss')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Indicator')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()


def get_optimizer(model_parameters, optimizer_name, lr, weight_decay=0):
    """
    Get optimizer by name
    """
    # Ensure model_parameters is a list
    if not isinstance(model_parameters, list):
        model_parameters = list(model_parameters)
    
    if not model_parameters:
        raise ValueError("Empty parameter list provided to optimizer!")
    
    print(f"Creating {optimizer_name} optimizer with {len(model_parameters)} parameter groups")
    
    optimizers = {
        'Adam': optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay),
        'AdamW': optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay),
        'SGD': optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    }
    
    if optimizer_name not in optimizers:
        warnings.warn(f"Optimizer {optimizer_name} not found, using AdamW")
        optimizer_name = 'AdamW'
        
    return optimizers[optimizer_name]


def get_scheduler(optimizer, scheduler_name, **kwargs):
    """
    Get learning rate scheduler by name
    """
    if scheduler_name == "StepLR":
        return StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == "CosineAnnealing":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode='max', patience=kwargs.get('patience', 3), 
                               factor=kwargs.get('factor', 0.5))
    elif scheduler_name == "OneCycle":
        return OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 0.01), 
                         total_steps=kwargs.get('total_steps', 100))
    else:
        warnings.warn(f"Scheduler {scheduler_name} not found, using ReduceLROnPlateau")
        return ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


def get_loss_function(loss_name, num_classes=5, class_weights=None, device='cpu', **kwargs):
    """
    Get loss function by name
    """
    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    if loss_name == "CrossEntropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "FocalLoss":
        return FocalLoss(alpha=kwargs.get('alpha', 1.0), gamma=kwargs.get('gamma', 2.0), weight=class_weights)
    elif loss_name == "LabelSmoothing":
        return LabelSmoothingLoss(num_classes, smoothing=kwargs.get('smoothing', 0.1), weight=class_weights)
    else:
        warnings.warn(f"Loss function {loss_name} not found, using CrossEntropy")
        return nn.CrossEntropyLoss(weight=class_weights) 