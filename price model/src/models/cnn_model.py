"""
Enhanced CNN Model for Candlestick Pattern Recognition
Includes attention mechanism and residual connections for better accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionModule(nn.Module):
    """Attention mechanism for sequence modeling"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.output(attended)
        
        return output + x  # Residual connection


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Residual connection
        out = F.relu(out)
        return out


class EnhancedCandleCNN(nn.Module):
    """
    Enhanced CNN model with attention and residual connections
    Designed for 50%+ accuracy target
    """
    
    def __init__(self, features_per_day: int = 65, num_classes: int = 5, 
                 hidden_size: int = 128, use_attention: bool = True):
        super().__init__()
        self.features_per_day = features_per_day
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        # Feature extraction layers
        self.feature_conv1 = nn.Conv1d(features_per_day, hidden_size, kernel_size=3, padding=1)
        self.feature_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.feature_bn1 = nn.BatchNorm1d(hidden_size)
        self.feature_bn2 = nn.BatchNorm1d(hidden_size)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(3)
        ])
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionModule(hidden_size)
        
        # Temporal convolution
        self.temporal_conv1 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.temporal_bn2 = nn.BatchNorm1d(hidden_size * 2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, features_per_day)
        batch_size, seq_len, features = x.shape
        
        # Transpose for 1D convolution: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Feature extraction
        x = F.relu(self.feature_bn1(self.feature_conv1(x)))
        x = F.relu(self.feature_bn2(self.feature_conv2(x)))
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Attention mechanism
        if self.use_attention:
            # Transpose back for attention: (batch_size, seq_len, hidden_size)
            x_attn = x.transpose(1, 2)
            x_attn = self.attention(x_attn)
            x = x_attn.transpose(1, 2)  # Back to (batch_size, hidden_size, seq_len)
        
        # Temporal convolution
        x = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        x = F.relu(self.temporal_bn2(self.temporal_conv2(x)))
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, hidden_size * 2, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_size * 2)
        
        # Classification
        x = self.classifier(x)
        
        return x


# Backward compatibility
class CandleCNN(EnhancedCandleCNN):
    """Simplified CNN model for candlestick pattern recognition"""
    
    def __init__(self, features_per_day: int = 65, num_classes: int = 5, 
                 hidden_size: int = 128, use_attention: bool = False):
        super().__init__(features_per_day, num_classes, hidden_size, use_attention) 