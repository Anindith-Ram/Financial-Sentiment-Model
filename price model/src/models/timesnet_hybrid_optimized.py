#!/usr/bin/env python3
"""
Optimized TimesNet-Hybrid Model for Overfitting Prevention
=========================================================
Reduced model size with adaptive dropout scheduling to prevent overfitting
while maintaining performance. Based on training curve analysis showing
validation loss increasing after epoch ~30.

Key Changes:
- Reduced CNN channels: 96 → 64 (33% reduction)
- Reduced TimesNet embedding: 192 → 128 (33% reduction) 
- Reduced TimesNet depth: 3 → 2 layers (faster training)
- Adaptive dropout: 0.5 → 0.2 during training
- Added residual connections for better gradients
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    """Helper: 1-D Convolution → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int, p: int):
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Optimized Multi-scale CNN branch
# ---------------------------------------------------------------------------
class OptimizedMultiScaleCNN(nn.Module):
    """Reduced-size multi-scale CNN with better regularization."""

    def __init__(self, features_per_day: int = 62, out_channels: int = 64):  # Reduced from 96
        super().__init__()
        self.branch1 = ConvBNReLU(features_per_day, out_channels, k=1, p=0)
        self.branch3 = ConvBNReLU(features_per_day, out_channels, k=3, p=1)
        self.branch5 = ConvBNReLU(features_per_day, out_channels, k=5, p=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T=5, F) ➜ convert to (B, F, T)
        x = x.transpose(1, 2)
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)  # (B, 3*out_channels=192, T)
        # Global pooling over time dim (T=5) ➜ (B, C)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        return out  # (B, 192)


# ---------------------------------------------------------------------------
# Optimized TimesNet-style encoder
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Patchify the sequence – for 5-day window we take patch length 1 (day)."""

    def __init__(self, in_features: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F)
        return self.proj(x)  # (B, T, D)


class OptimizedTimesBlock(nn.Module):
    """Optimized transformer encoder block with better regularization."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2):  # Reduced heads from 8
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),  # Added dropout in MLP
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class OptimizedTimesNetEncoder(nn.Module):
    """Optimized TimesNet encoder with reduced size."""

    def __init__(self, in_features: int = 62, emb_dim: int = 128, depth: int = 2):  # Reduced depth from 3
        super().__init__()
        self.patch = PatchEmbedding(in_features, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 6, emb_dim))  # 5 days + CLS
        self.blocks = nn.ModuleList([OptimizedTimesBlock(emb_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T=5, F)
        x = self.patch(x)  # (B, 5, D)
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)  # (B,1,D)
        x = torch.cat([cls_tokens, x], dim=1) + self.pos_emb  # (B,6,D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_feat = x[:, 0]  # (B, D)
        return cls_feat  # (B, D)


# ---------------------------------------------------------------------------
# Adaptive Dropout Module
# ---------------------------------------------------------------------------
class AdaptiveDropout(nn.Module):
    """Dropout that adapts its rate during training."""
    
    def __init__(self, initial_p: float = 0.5, final_p: float = 0.2):
        super().__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.current_p = initial_p
        
    def set_dropout_rate(self, epoch: int, total_epochs: int):
        """Update dropout rate based on training progress."""
        # Linear decay from initial_p to final_p
        progress = epoch / total_epochs
        self.current_p = self.initial_p * (1 - progress) + self.final_p * progress
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.current_p, training=self.training)


# ---------------------------------------------------------------------------
# Optimized Hybrid Model
# ---------------------------------------------------------------------------
class OptimizedTimesNetHybrid(nn.Module):
    """Optimized hybrid model with reduced overfitting capacity."""
    
    def __init__(self, features_per_day: int = 62, num_classes: int = 5):
        super().__init__()
        # Reduced capacity components
        self.cnn = OptimizedMultiScaleCNN(features_per_day, out_channels=64)  # ➜ 192-d
        self.timesnet = OptimizedTimesNetEncoder(in_features=features_per_day, emb_dim=128, depth=2)  # 128-d
        
        # Improved fusion with residual connection
        cnn_dim = 192  # 3 * 64
        times_dim = 128
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim + times_dim, cnn_dim, bias=True),
            nn.Sigmoid()
        )
        
        # Optimized classifier with adaptive dropout
        self.adaptive_dropout1 = AdaptiveDropout(initial_p=0.5, final_p=0.2)
        self.adaptive_dropout2 = AdaptiveDropout(initial_p=0.4, final_p=0.1)
        
        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim, 128),  # Reduced from 256
            nn.ReLU(),
            self.adaptive_dropout1,
            nn.Linear(128, 64),       # Reduced from 128
            nn.ReLU(), 
            self.adaptive_dropout2,
            nn.Linear(64, num_classes)
        )
        
        print(f"🏗️ Optimized Model Architecture:")
        print(f"  📉 CNN channels: 96 → 64 (-33%)")
        print(f"  📉 TimesNet emb: 192 → 128 (-33%)")
        print(f"  📉 TimesNet depth: 3 → 2 (-33%)")
        print(f"  📉 Classifier: 256→128→5 → 128→64→5 (-50%)")
        print(f"  🎛️ Adaptive dropout: 0.5→0.2 during training")

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update adaptive dropout rates based on training progress."""
        self.adaptive_dropout1.set_dropout_rate(epoch, total_epochs)
        self.adaptive_dropout2.set_dropout_rate(epoch, total_epochs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 5, 62)
        cnn_feat = self.cnn(x)          # (B, 192)
        t_feat = self.timesnet(x)       # (B, 128)
        
        # Improved gated fusion with residual connection
        combined = torch.cat([cnn_feat, t_feat], dim=1)  # (B, 320)
        gate = self.gate(combined)      # (B, 192)
        
        # Residual connection: original CNN + gated combination
        fused = cnn_feat + gate * cnn_feat  # Enhanced with skip connection
        
        out = self.classifier(fused)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_optimized_timesnet_hybrid(features_per_day: int = 62, num_classes: int = 5):
    """Factory for optimized model to prevent overfitting."""
    return OptimizedTimesNetHybrid(features_per_day=features_per_day, num_classes=num_classes)


if __name__ == "__main__":
    # Quick test
    model = create_optimized_timesnet_hybrid()
    dummy = torch.randn(8, 5, 62)
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    out = model(dummy)
    print("Output shape:", out.shape)
    
    # Test adaptive dropout
    model.set_epoch(0, 50)  # Start of training
    print(f"Initial dropout rates: {model.adaptive_dropout1.current_p:.3f}, {model.adaptive_dropout2.current_p:.3f}")
    
    model.set_epoch(25, 50)  # Mid training
    print(f"Mid dropout rates: {model.adaptive_dropout1.current_p:.3f}, {model.adaptive_dropout2.current_p:.3f}")
    
    model.set_epoch(50, 50)  # End of training
    print(f"Final dropout rates: {model.adaptive_dropout1.current_p:.3f}, {model.adaptive_dropout2.current_p:.3f}")