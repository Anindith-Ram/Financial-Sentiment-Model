"""
Advanced training utilities for improved model performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from typing import Dict, Any, Optional
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for better generalization
    """
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class CosineAnnealingWarmRestartsWithWarmup:
    """
    Cosine annealing with warm restarts and warmup
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_epochs=0, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
            
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing phase
            T_cur = self.current_epoch - self.warmup_epochs
            T_i = self.T_0
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= self.T_mult
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1


def get_optimizer(parameters, optimizer_name, **kwargs):
    """Get optimizer with enhanced settings"""
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            parameters,
            lr=kwargs.get('lr', 1e-4),
            weight_decay=kwargs.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(
            parameters,
            lr=kwargs.get('lr', 1e-4),
            weight_decay=kwargs.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(
            parameters,
            lr=kwargs.get('lr', 1e-3),
            momentum=0.9,
            weight_decay=kwargs.get('weight_decay', 1e-4),
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_loss_function(loss_name, num_classes=5, device='cuda', **kwargs):
    """Get loss function with enhanced options"""
    if loss_name.lower() == 'crossentropyloss':
        class_weights = kwargs.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=kwargs.get('label_smoothing', 0.05)
        ).to(device)
    elif loss_name.lower() == 'focalloss':
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0)
        ).to(device)
    elif loss_name.lower() == 'labelsmoothingloss':
        return LabelSmoothingLoss(
            classes=num_classes,
            smoothing=kwargs.get('label_smoothing', 0.1)
        ).to(device)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_scheduler(optimizer, scheduler_name, epochs, **kwargs):
    """Get learning rate scheduler with enhanced options"""
    if scheduler_name.lower() == 'reducelronplateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('lr_factor', 0.5),
            patience=kwargs.get('lr_patience', 3),
            min_lr=kwargs.get('lr_min', 1e-6),
            verbose=True
        )
    elif scheduler_name.lower() == 'cosineannealingwarmrestarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs // 4,
            T_mult=2,
            eta_min=kwargs.get('lr_min', 1e-6)
        )
    elif scheduler_name.lower() == 'cosineannealingwarmrestartswithwarmup':
        return CosineAnnealingWarmRestartsWithWarmup(
            optimizer,
            T_0=epochs // 4,
            T_mult=2,
            eta_min=kwargs.get('lr_min', 1e-6),
            warmup_epochs=kwargs.get('warmup_epochs', 3),
            warmup_start_lr=kwargs.get('warmup_start_lr', 1e-6)
        )
    elif scheduler_name.lower() == 'onecyclelr':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('lr', 1e-4),
            epochs=epochs,
            steps_per_epoch=kwargs.get('steps_per_epoch', 100),
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class EarlyStopping:
    """Enhanced early stopping with multiple criteria"""
    def __init__(self, patience=10, min_delta=0.001, mode='min', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        if self.best_score is None:
            self.best_score = val_loss
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif (self.mode == 'min' and val_loss < self.best_score - self.min_delta) or \
             (self.mode == 'max' and val_loss > self.best_score + self.min_delta):
            self.best_score = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class AdvancedMetrics:
    """Advanced metrics tracking"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
    def get_summary(self):
        return {
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'best_train_acc': max(self.train_accuracies) if self.train_accuracies else None,
            'best_val_acc': max(self.val_accuracies) if self.val_accuracies else None,
            'final_lr': self.learning_rates[-1] if self.learning_rates else None
        }


class TrainingVisualizer:
    """Enhanced training visualization"""
    def __init__(self):
        pass
        
    def plot_training_curves(self, metrics):
        """Plot training curves (placeholder for matplotlib integration)"""
        # This would integrate with matplotlib for visualization
        pass


def add_noise_to_data(data, noise_strength=0.01):
    """Add Gaussian noise to training data for regularization"""
    noise = torch.randn_like(data) * noise_strength
    return data + noise


def mixup_data(data, targets, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(data.device)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    targets_a, targets_b = targets, targets[index]
    
    return mixed_data, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b) 