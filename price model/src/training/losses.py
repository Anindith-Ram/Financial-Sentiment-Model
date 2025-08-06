import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Weighted Focal Loss for multi-class classification.

    Args:
        gamma (float): focusing parameter.
        weight (Tensor, optional): class weights.
    """

    def __init__(self, gamma: float = 1.5, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(input, dim=1)
        p = torch.exp(logp)
        focal_term = (1 - p) ** self.gamma
        loss = F.nll_loss(focal_term * logp, target, weight=self.weight, reduction=self.reduction)
        return loss
