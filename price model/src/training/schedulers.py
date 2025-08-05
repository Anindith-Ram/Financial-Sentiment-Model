#!/usr/bin/env python3
"""Learning rate schedulers"""

import torch.optim as optim


def get_scheduler(optimizer, scheduler_type: str = "cosine_annealing", 
                 epochs: int = 100, warmup_epochs: int = 5):
    """Get learning rate scheduler"""
    
    if scheduler_type == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return scheduler