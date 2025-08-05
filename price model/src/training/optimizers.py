#!/usr/bin/env python3
"""Advanced optimizers"""

import torch.optim as optim
from typing import Dict, Any


def get_optimizer(model, optimizer_type: str = "adamw", learning_rate: float = 0.001,
                 weight_decay: float = 1e-4, component_lrs: Dict[str, float] = None):
    """Get optimizer with component-specific learning rates"""
    
    if component_lrs:
        # Create parameter groups with different learning rates
        param_groups = []
        
        # Default group
        param_groups.append({
            'params': [p for p in model.parameters()],
            'lr': learning_rate
        })
        
        optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    return optimizer