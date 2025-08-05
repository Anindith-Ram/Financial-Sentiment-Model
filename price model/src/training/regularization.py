#!/usr/bin/env python3
"""Advanced regularization techniques"""

import torch
import torch.nn as nn
import numpy as np


class MixUp:
    """MixUp data augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, data, target):
        """Apply MixUp"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)
        
        mixed_data = lam * data + (1 - lam) * data[index, :]
        target_a, target_b = target, target[index]
        
        return mixed_data, target_a, target_b, lam


class CutMix:
    """CutMix data augmentation"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, data, target):
        """Apply CutMix"""
        # Simplified implementation
        return self.mixup_fallback(data, target)
    
    def mixup_fallback(self, data, target):
        """Fallback to MixUp for time series data"""
        mixup = MixUp(self.alpha)
        return mixup(data, target)


class LabelSmoothing(nn.Module):
    """Label smoothing loss"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """Forward pass with label smoothing"""
        num_classes = pred.size(-1)
        
        # One-hot encode targets
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1))