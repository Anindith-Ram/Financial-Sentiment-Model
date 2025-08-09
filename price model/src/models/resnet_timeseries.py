#!/usr/bin/env python3
"""
ResNet for Short-Horizon Time Series (5-day context)
====================================================
Processes inputs of shape (B, T=5, F=features_per_day) using 1D residual blocks
across the time axis. Produces dual-head outputs: classification logits and an
auxiliary regression prediction (next-day return), matching the current pipeline.
"""

from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck1D(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        width = out_channels
        # 1x1 reduce
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        # 3x1 conv
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        # 1x1 expand
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNetTimeSeries(nn.Module):
    def __init__(self, features_per_day: int, num_classes: int = 3,
                 base_channels: int = 64, layers: List[int] = None,
                 block: nn.Module = Bottleneck1D, dropout: float = 0.2):
        super().__init__()
        self.features_per_day = features_per_day
        self.model_type = "resnet_ts"

        # Default to ResNet-50 layout if not provided
        if layers is None:
            layers = [3, 4, 6, 3]

        # Stem: (B, T, F) -> (B, F, T), then conv on F channels
        self.stem = nn.Sequential(
            nn.Conv1d(features_per_day, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layers_list = nn.ModuleList()
        in_channels = base_channels
        stage_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        for idx, (num_blocks, out_ch) in enumerate(zip(layers, stage_channels)):
            stage = self._make_stage(block, in_channels, out_ch, num_blocks, dropout)
            setattr(self, f"layer{idx+1}", stage)  # register with conventional name for grouping
            self.layers_list.append(stage)
            in_channels = out_ch * block.expansion

        self.out_channels = in_channels

        self.head_trunk = nn.Sequential(
            nn.Linear(self.out_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.head_class = nn.Linear(512, num_classes)
        self.head_reg = nn.Linear(512, 1)

    def _make_stage(self, block, in_channels: int, out_channels: int, blocks: int,
                    dropout: float) -> nn.Sequential:
        layers = []
        # Keep stride=1 to preserve T=5 resolution throughout
        layers.append(block(in_channels, out_channels, stride=1, dropout=dropout))
        in_ch = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_ch, out_channels, stride=1, dropout=dropout))
            in_ch = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.stem(x)
        # Iterate layers via a loop as requested
        for stage in self.layers_list:
            x = stage(x)
        # Global average pooling over time dimension (T)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, C)
        x = self.head_trunk(x)
        logits = self.head_class(x)
        ret_pred = self.head_reg(x).squeeze(-1)
        return logits, ret_pred


def create_resnet_time_series(features_per_day: int, num_classes: int = 3,
                              base_channels: int = 64, depth: str = "50") -> ResNetTimeSeries:
    """Factory for ResNetTimeSeries.

    depth:
      - "50" or "resnet50": Bottleneck blocks with [3,4,6,3]
      - "34": Basic blocks [3,4,6,3] but BasicBlock (optional future)
      - "small"|"medium"|"large": legacy settings
    """
    if depth in ("50", "resnet50"):
        return ResNetTimeSeries(features_per_day=features_per_day, num_classes=num_classes,
                                base_channels=base_channels, layers=[3, 4, 6, 3], block=Bottleneck1D)
    layers_map = {
        "small": [2, 2, 1],
        "medium": [2, 2, 2],
        "large": [3, 3, 3],
    }
    layers = layers_map.get(depth, [3, 4, 6, 3])
    block = Bottleneck1D
    return ResNetTimeSeries(features_per_day=features_per_day, num_classes=num_classes,
                            base_channels=base_channels, layers=layers, block=block)


if __name__ == "__main__":
    model = create_resnet_time_series(features_per_day=62, num_classes=3)
    dummy = torch.randn(8, 5, 62)
    out = model(dummy)
    print(out[0].shape, out[1].shape)


