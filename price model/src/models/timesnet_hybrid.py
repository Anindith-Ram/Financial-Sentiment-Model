#!/usr/bin/env python3
"""
TimesNet-Hybrid Model
=====================
Hybrid model that pairs a light multi-scale CNN (for local candle/indicator patterns)
with a compact "TimesNet" style temporal encoder (patch-based transformer) and a
small classifier.  Designed for 5-day context ➜ 1-day horizon stock movement.

This stays <4 M parameters so it trains quickly on consumer GPUs.
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
# Multi-scale CNN branch
# ---------------------------------------------------------------------------
class MultiScaleCNN(nn.Module):
    """Three parallel convolutions with kernel sizes 1, 3, 5."""

    def __init__(self, features_per_day: int = 62, out_channels: int = 96):
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
        out = torch.cat([b1, b3, b5], dim=1)  # (B, 3*out_channels, T)
        # Global pooling over time dim (T=5) ➜ (B, C)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        out = F.dropout(out, p=0.3, training=self.training)  # Added dropout for regularisation
        return out  # (B, C)


# ---------------------------------------------------------------------------
# Minimal TimesNet-style encoder
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Patchify the sequence – for 5-day window we take patch length 1 (day)."""

    def __init__(self, in_features: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F)
        return self.proj(x)  # (B, T, D)


class TimesBlock(nn.Module):
    """Single transformer encoder block (similar to PatchTST/TimesNet)."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TimesNetEncoder(nn.Module):
    """Stack of TimesBlocks with learnable [CLS] token."""

    def __init__(self, in_features: int = 62, emb_dim: int = 192, depth: int = 3):
        super().__init__()
        self.patch = PatchEmbedding(in_features, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 6, emb_dim))  # 5 days + CLS
        self.blocks = nn.ModuleList([TimesBlock(emb_dim) for _ in range(depth)])
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
# Gated Residual Fusion + Classifier
# ---------------------------------------------------------------------------
class TimesNetHybrid(nn.Module):
    def __init__(self, features_per_day: int = 62, num_classes: int = 5,
                 cnn_channels: int = 512, timesnet_emb: int = 512, timesnet_depth: int = 5):
        super().__init__()
        self.cnn = MultiScaleCNN(features_per_day, out_channels=cnn_channels)  # ➜ 3*cnn_channels-d (bigger)
        self.timesnet = TimesNetEncoder(in_features=features_per_day, emb_dim=timesnet_emb, depth=timesnet_depth)
        fused_dim = cnn_channels * 3  # from MultiScaleCNN concat
        times_dim = timesnet_emb
        self.gate = nn.Sequential(
            nn.Linear(fused_dim + times_dim, fused_dim, bias=True),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 5, 62)
        cnn_feat = self.cnn(x)          # (B, 288)
        t_feat = self.timesnet(x)       # (B, 192)
        gate = self.gate(torch.cat([cnn_feat, t_feat], dim=1))  # (B, 288)
        fused = cnn_feat + gate * cnn_feat + (1 - gate) * t_feat.mean(dim=1, keepdim=False).unsqueeze(1).repeat(1, cnn_feat.size(1))
        out = self.classifier(fused)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_timesnet_hybrid(features_per_day: int = 62, num_classes: int = 5,
                            cnn_channels: int = 512, timesnet_emb: int = 512, timesnet_depth: int = 5):
    """Factory that allows passing bigger model hyperparameters."""
    return TimesNetHybrid(features_per_day=features_per_day, num_classes=num_classes,
                          cnn_channels=cnn_channels, timesnet_emb=timesnet_emb, timesnet_depth=timesnet_depth)


if __name__ == "__main__":
    # Quick forward-pass test
    model = create_timesnet_hybrid()
    dummy = torch.randn(8, 5, 62)
    out = model(dummy)
    print("Output:", out.shape)
