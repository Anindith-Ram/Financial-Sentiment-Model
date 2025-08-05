"""
GPT-2 Enhanced CNN Integration
==============================

Advanced CNN architecture with multi-scale convolutions, pattern attention,
and residual connections for superior candlestick pattern recognition.
Keeps GPT-2 for sequence understanding while enhancing CNN for pattern detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
import warnings
warnings.filterwarnings("ignore")


class GPT2TemporalExtractor(nn.Module):
    """
    Enhanced GPT-2 based temporal feature extractor
    Extracts high-level temporal patterns from financial sequences
    """
    
    def __init__(self, features_per_day: int = 71, hidden_size: int = 768):
        super().__init__()
        self.features_per_day = features_per_day
        self.hidden_size = hidden_size
        
        # Prioritized model candidates (free models)
        self.model_candidates = [
            "gpt2",                           # General language patterns (proven for sequences)
            "distilgpt2",                     # Lightweight GPT-2 (faster inference)
            "microsoft/DialoGPT-medium",      # Conversational patterns (fallback)
            "microsoft/DialoGPT-small"        # Lightweight fallback
        ]
        
        # Initialize pretrained model
        self.pretrained_model = None
        self.tokenizer = None
        self._load_pretrained_model()
        
        # Feature projection for numerical data
        self.feature_projection = nn.Sequential(
            nn.Linear(features_per_day, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Enhanced temporal processing with attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        self.temporal_norm = nn.LayerNorm(hidden_size)
        
    def _load_pretrained_model(self):
        """Load the best available pretrained model"""
        for model_name in self.model_candidates:
            try:
                print(f"Loading {model_name} for temporal feature extraction...")
                self.pretrained_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Freeze pretrained model for feature extraction
                for param in self.pretrained_model.parameters():
                    param.requires_grad = False
                    
                print(f"✓ Successfully loaded {model_name}")
                break
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                continue
        
        if self.pretrained_model is None:
            print("⚠️ All pretrained models failed. Using enhanced fallback.")
            self._create_enhanced_fallback()
    
    def _create_enhanced_fallback(self):
        """Enhanced fallback with temporal convolutions and attention"""
        self.pretrained_model = nn.Sequential(
            nn.Linear(self.features_per_day, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal convolution layers
        self.temporal_conv1 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=2)
        self.temporal_bn1 = nn.BatchNorm1d(self.hidden_size)
        self.temporal_bn2 = nn.BatchNorm1d(self.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal features from financial sequences"""
        return self._extract_gpt2_features(x)
    
    def _extract_gpt2_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using GPT-2 or enhanced fallback
        x: (batch_size, seq_len, features_per_day)
        """
        batch_size, seq_len, features = x.shape
        
        if hasattr(self.pretrained_model, 'config'):  # Real GPT-2 model
            # Project numerical features to GPT-2 embedding space
            x_projected = self.feature_projection(x)  # (batch_size, seq_len, hidden_size)
            
            # Create attention mask for the sequence
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=x.device)
            
            # Extract features using GPT-2
            with torch.no_grad():
                outputs = self.pretrained_model(
                    inputs_embeds=x_projected,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Use the last hidden state
            features = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            
        else:  # Enhanced fallback
            # Process through enhanced temporal network
            x_reshaped = x.view(-1, features)  # (batch_size * seq_len, features)
            features = self.pretrained_model(x_reshaped)  # (batch_size * seq_len, hidden_size)
            features = features.view(batch_size, seq_len, self.hidden_size)
            
            # Apply temporal convolutions
            features = features.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
            features = F.relu(self.temporal_bn1(self.temporal_conv1(features)))
            features = F.relu(self.temporal_bn2(self.temporal_conv2(features)))
            features = features.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        
        # Apply temporal attention for enhanced sequence understanding
        features_attended, _ = self.temporal_attention(features, features, features)
        features = self.temporal_norm(features + features_attended)  # Residual connection
        
        return features


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN with multi-scale convolutions, pattern attention,
    and residual connections for superior candlestick pattern recognition
    """
    
    def __init__(self, features_per_day: int = 71, hidden_size: int = 768):
        super().__init__()
        self.features_per_day = features_per_day
        self.hidden_size = hidden_size
        
        # Multi-scale convolutions for different pattern sizes
        self.conv_3 = nn.Conv1d(features_per_day, 64, kernel_size=3, padding=1)   # Short patterns
        self.conv_5 = nn.Conv1d(features_per_day, 64, kernel_size=5, padding=2)   # Medium patterns  
        self.conv_7 = nn.Conv1d(features_per_day, 64, kernel_size=7, padding=3)   # Long patterns
        
        # Batch normalization for each scale
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_5 = nn.BatchNorm1d(64)
        self.bn_7 = nn.BatchNorm1d(64)
        
        # Pattern attention mechanism
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=192,  # 64 * 3 scales
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Residual connections
        self.residual_conv = nn.Conv1d(192, 192, kernel_size=1)
        self.residual_bn = nn.BatchNorm1d(192)
        
        # Enhanced feature extraction
        self.conv1 = nn.Conv1d(192, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced CNN forward pass with multi-scale pattern detection
        x: (batch_size, seq_len, features_per_day)
        """
        # Reshape for convolutions
        x = x.transpose(1, 2)  # (batch_size, features_per_day, seq_len)
        
        # Multi-scale pattern detection
        x_3 = F.relu(self.bn_3(self.conv_3(x)))  # Short patterns
        x_5 = F.relu(self.bn_5(self.conv_5(x)))  # Medium patterns
        x_7 = F.relu(self.bn_7(self.conv_7(x)))  # Long patterns
        
        # Concatenate multi-scale features
        x_multi = torch.cat([x_3, x_5, x_7], dim=1)  # (batch_size, 192, seq_len)
        
        # Residual connection
        x_residual = self.residual_conv(x_multi)
        x_multi = F.relu(self.residual_bn(x_multi + x_residual))
        
        # Pattern attention mechanism
        x_attn = x_multi.transpose(1, 2)  # (batch_size, seq_len, 192)
        x_attended, _ = self.pattern_attention(x_attn, x_attn, x_attn)
        x_attn = x_attn + x_attended  # Residual connection
        x_multi = x_attn.transpose(1, 2)  # (batch_size, 192, seq_len)
        
        # Enhanced feature extraction
        x = F.relu(self.bn1(self.conv1(x_multi)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, 512, 1)
        x = x.squeeze(-1)  # (batch_size, 512)
        
        return x


class GPT2EnhancedCNN(nn.Module):
    """
    GPT-2 Enhanced CNN with advanced pattern recognition
    Combines GPT-2's sequence understanding with enhanced CNN pattern detection
    """
    
    def __init__(self, features_per_day: int = 71, hidden_size: int = 768,
                 num_classes: int = 5, fusion_method: str = "attention"):
        super().__init__()
        self.features_per_day = features_per_day
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # GPT-2 temporal extractor
        self.gpt2_extractor = GPT2TemporalExtractor(
            features_per_day=features_per_day,
            hidden_size=hidden_size
        )
        
        # Enhanced CNN for pattern recognition
        self.enhanced_cnn = EnhancedCNN(
            features_per_day=features_per_day,
            hidden_size=hidden_size
        )
        
        # Fusion layer
        if fusion_method == "attention":
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        elif fusion_method == "concatenation":
            self.fusion_layer = None
        elif fusion_method == "weighted":
            self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512 + self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GPT-2 + Enhanced CNN fusion
        x: (batch_size, seq_len, features_per_day)
        """
        # Extract GPT-2 temporal features
        gpt2_features = self.gpt2_extractor(x)  # (batch_size, seq_len, hidden_size)
        
        # Extract enhanced CNN pattern features
        cnn_features = self.enhanced_cnn(x)  # (batch_size, 512)
        
        # Global pooling for GPT-2 features
        gpt2_features = gpt2_features.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        gpt2_features = F.adaptive_avg_pool1d(gpt2_features, 1)  # (batch_size, hidden_size, 1)
        gpt2_features = gpt2_features.squeeze(-1)  # (batch_size, hidden_size)
        
        # Fusion methods
        if self.fusion_method == "attention":
            # Attention-based fusion
            gpt2_features_attn = gpt2_features.unsqueeze(1)  # (batch_size, 1, hidden_size)
            cnn_features_attn = cnn_features.unsqueeze(1)  # (batch_size, 1, 512)
            
            # Project CNN features to same dimension
            cnn_projection = nn.Linear(512, self.hidden_size).to(x.device)
            cnn_features_attn = cnn_projection(cnn_features_attn)
            
            # Apply attention
            fused_features, _ = self.fusion_layer(gpt2_features_attn, cnn_features_attn, cnn_features_attn)
            fused_features = fused_features.squeeze(1)  # (batch_size, hidden_size)
            
            # Concatenate for final classification
            final_features = torch.cat([cnn_features, gpt2_features], dim=1)
            
        elif self.fusion_method == "concatenation":
            # Simple concatenation
            final_features = torch.cat([cnn_features, gpt2_features], dim=1)
            
        elif self.fusion_method == "weighted":
            # Weighted fusion
            weights = F.softmax(self.fusion_weights, dim=0)
            weighted_cnn = weights[0] * cnn_features
            weighted_gpt2 = weights[1] * gpt2_features
            
            # Keep original dimensions for concatenation
            final_features = torch.cat([weighted_cnn, weighted_gpt2], dim=1)
            
        else:
            # Default to concatenation
            final_features = torch.cat([cnn_features, gpt2_features], dim=1)
        
        # Final classification
        output = self.classifier(final_features)
        
        return output


def create_gpt2_enhanced_cnn(features_per_day: int = 71, hidden_size: int = 768,
                            num_classes: int = 5, fusion_method: str = "attention") -> GPT2EnhancedCNN:
    """
    Create an enhanced GPT-2 CNN model with advanced pattern recognition
    
    Args:
        features_per_day: Number of features per time step
        hidden_size: Hidden size for GPT-2 features
        num_classes: Number of output classes
        fusion_method: Method to fuse GPT-2 and CNN features ("attention", "concatenation", "weighted")
    
    Returns:
        GPT2EnhancedCNN model
    """
    return GPT2EnhancedCNN(
        features_per_day=features_per_day,
        hidden_size=hidden_size,
        num_classes=num_classes,
        fusion_method=fusion_method
    )


# Backward compatibility
def create_enhanced_gpt2_cnn(*args, **kwargs) -> GPT2EnhancedCNN:
    """Backward compatibility function"""
    return create_gpt2_enhanced_cnn(*args, **kwargs) 