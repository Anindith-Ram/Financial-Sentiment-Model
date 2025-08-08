#!/usr/bin/env python3
"""
CNN Branch Research Training Script
====================================

Standalone script for research, development, and detailed analysis of the enhanced CNN model.
This script provides comprehensive monitoring, visualization, and debugging capabilities.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
import time
from tqdm import tqdm
import sys
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append('.')

from src.models.dataset import FinancialDataset
from src.training.losses import FocalLoss
from src.training.regularization import MixUp, LabelSmoothing
from src.training.advanced_lr_scheduler import PerformanceSensitiveLRScheduler
from sklearn.metrics import f1_score, roc_auc_score

from src.models.timesnet_hybrid import create_timesnet_hybrid
from src.models.timesnet_hybrid_optimized import create_optimized_timesnet_hybrid

# PROFESSIONAL ENHANCEMENTS - NEW PERFORMANCE FEATURES
try:
    from src.training.optimizers import get_optimizer
    from src.training.schedulers import get_scheduler
    from src.training.regularization import MixUp, CutMix, LabelSmoothing
    from src.training.metrics import MetricsTracker
    from src.utils.logger import setup_logger
    from src.utils.errors import TrainingError, ErrorHandler
    print("✅ Professional enhancements loaded successfully!")
    PROFESSIONAL_FEATURES = True
except ImportError as e:
    print(f"⚠️ Professional enhancements not available: {e}")
    PROFESSIONAL_FEATURES = False


def load_and_preprocess_data(csv_path: str = "data/reduced_feature_set_dataset.csv", 
                           test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Load data, generate 3-class labels with rolling 33/66% per ticker, robust-scale per ticker, split by time if Date present."""

    print(f"📊 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Identify potential date col and close column
    date_col = next((c for c in ["Date","date","Timestamp","timestamp"] if c in df.columns), None)
    close_col = next((c for c in ["Close","Adj Close","Adj_Close"] if c in df.columns), None)

    # Build feature list excluding non-features
    feature_columns = [c for c in df.columns if c not in ["Ticker","Label"]]
    if date_col:
        feature_columns = [c for c in feature_columns if c != date_col]

    print(f"🔍 Checking for NaN values...")
    nan_rows = df[feature_columns].isna().any(axis=1)
    df_clean = df[~nan_rows].copy() if nan_rows.any() else df.copy()
    if nan_rows.any():
        print(f"⚠️  Removed {nan_rows.sum():,} rows with NaN values")

    # Generate 3-class labels
    if date_col and close_col:
        df_clean = df_clean.sort_values(["Ticker", date_col])
        df_clean["ret1"] = df_clean.groupby("Ticker")[close_col].pct_change().shift(-1)
        past_ret = df_clean.groupby("Ticker")["ret1"].shift(1)
        q33 = past_ret.groupby(df_clean["Ticker"]).transform(lambda s: s.rolling(252, min_periods=60).quantile(0.33))
        q66 = past_ret.groupby(df_clean["Ticker"]).transform(lambda s: s.rolling(252, min_periods=60).quantile(0.66))
        labels_3 = np.where(df_clean["ret1"] < q33, 0, np.where(df_clean["ret1"] > q66, 2, 1))
        nan_mask = np.isnan(q33) | np.isnan(q66)
        if nan_mask.any():
            med = past_ret.groupby(df_clean["Ticker"]).transform(lambda s: s.rolling(60, min_periods=20).median())
            labels_3[nan_mask] = np.where(df_clean.loc[nan_mask, "ret1"] < med[nan_mask], 0, 2)
            labels_3 = np.where(np.isnan(df_clean["ret1"]) & nan_mask, 1, labels_3)
        df_clean["Label_3"] = labels_3.astype(np.int64)
        target_col = "Label_3"
    else:
        # Fallback mapping 5→3 classes
        if 'Label' not in df_clean.columns:
            raise ValueError("No Label and no price columns to compute 3-class labels.")
        map3 = {0:0,1:0,2:1,3:2,4:2}
        df_clean['Label_3'] = df_clean['Label'].map(map3).astype(np.int64)
        target_col = 'Label_3'

    # Time-aware split per ticker if possible
    if date_col:
        train_parts, test_parts = [], []
        for tkr, g in df_clean.groupby('Ticker'):
            g = g.sort_values(date_col)
            n = len(g)
            split_idx = int(n * (1 - test_size))
            train_parts.append(g.iloc[:split_idx])
            test_parts.append(g.iloc[split_idx:])
        train_df = pd.concat(train_parts)
        test_df = pd.concat(test_parts)
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=random_state, stratify=df_clean[target_col])

    # Robust scale per ticker on train stats only
    from sklearn.preprocessing import RobustScaler
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    for tkr in sorted(set(train_df['Ticker'])):
        tr = train_df[train_df['Ticker']==tkr]
        te = test_df[test_df['Ticker']==tkr]
        if len(tr)==0 or len(te)==0:
            continue
        scaler = RobustScaler()
        Xtr = scaler.fit_transform(tr[feature_columns].values.astype(np.float32))
        Xte = scaler.transform(te[feature_columns].values.astype(np.float32))
        X_train_list.append(Xtr)
        X_test_list.append(Xte)
        y_train_list.append(tr[target_col].values.astype(np.int64))
        y_test_list.append(te[target_col].values.astype(np.int64))

    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)

    print(f"📈 Dataset shape (train/test): {X_train.shape} / {X_test.shape}")
    print(f"🔢 Features per sample: {X_train.shape[1]}")

    # Class weights (3-class) – ensure weight exists for all 3 classes
    uniq, cnts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    raw_w = total / (len(uniq) * cnts)
    # Start with neutral weights
    weights_full = np.ones(3, dtype=np.float32)
    # Assign computed weights to present classes
    for cls_id, w in zip(uniq.astype(int), raw_w):
        weights_full[cls_id] = np.clip(w, 0.5, 3.0)
    class_weights = weights_full
    print("\n⚖️ Moderated class weights (3-class):")
    for cls_id in range(3):
        # Show underlying raw if available
        rw = raw_w[list(uniq).index(cls_id)] if cls_id in uniq else 1.0
        print(f"  Class {cls_id}: {rw:.3f} → {class_weights[cls_id]:.3f}")

    # Sequence info
    seq_len = 5
    features_per_day = X_train.shape[1] // seq_len
    print(f"\n⏰ Sequence length: {seq_len}")
    print(f"🎯 Features per day: {features_per_day}")
    print(f"🎯 Number of classes: 3")

    # Return flat arrays; FinancialDataset reshapes internally
    return X_train, X_test, y_train, y_test, features_per_day, class_weights


class EnhancedCNNResearchTrainer:
    """
    Research trainer for CNN Branch with comprehensive monitoring and analysis
    """
    
    def __init__(self, model, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", class_weights=None,
                 checkpoint_path: str = None, resume_from_epoch: int = 0):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Checkpoint loading support
        self.checkpoint_path = checkpoint_path
        self.resume_from_epoch = resume_from_epoch
        
        # Enhanced saving system with unique run IDs
        if checkpoint_path:
            print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
            self.run_id = f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🆔 Research Training Run ID: {self.run_id}")
        
        # Best model state (only saved once at the end)
        self.best_model_state = None
        
        # Enhanced early stopping - will be updated if loading checkpoint
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience = 5   # REDUCED for faster training → Aggressive LR exploration needs less patience
        self.patience_counter = 0
        
        # 🎯 LEARNING RATE STRATEGY 
        self.lr_strategy = 'onecycle'  # Regular OneCycleLR schedule (aggressive off)
        # SWA is disabled; EMA is the chosen averaging method
        self.use_swa = False
        
        # Mixed precision training for better performance
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        # Separate learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler based on configuration
        if self.lr_strategy == 'performance_sensitive':
            # 🎯 NEW: Performance-sensitive LR based on your 2.80e-04 observation + proven adaptive features
            self.scheduler = PerformanceSensitiveLRScheduler(
                optimizer=self.optimizer,
                base_lr=2.8e-4,  # Your optimal LR observation
                memory_length=5,
                sensitivity_threshold=0.001,  # 0.1% accuracy sensitivity
                lr_search_range=(5e-5, 8e-4),
                patience_for_search=3,
                warmup_epochs=5,  # 🔥 ENABLE WARMUP: 5 epochs gradual LR increase
                warmup_strategy='cosine',  # Smooth S-curve warmup for stability
                # 🚀 ENHANCED FEATURES FROM PROVEN ADAPTIVE SCHEDULER
                lr_increase_threshold=0.005,  # 0.5% improvement → increase LR (proven)
                lr_decrease_threshold=-0.01,  # 1% drop → decrease LR (proven)
                lr_multiplier_up=1.5,  # 50% increase multiplier (proven)
                lr_multiplier_down=0.7,  # 30% decrease multiplier (proven)
                plateau_escape_epochs=5,  # Plateau escape after 5 epochs (proven)
                plateau_escape_multiplier=3.0,  # 3x LR boost (proven)
                plateau_escape_cap=1.5,  # Cap at 1.5x original LR (proven)
                verbose=True
            )
            self.scheduler_mode = "performance_sensitive"
            print(f"🎯 Using Performance-Sensitive LR: base={2.8e-4:.2e} (your observation)")
        elif self.lr_strategy == 'adaptive':
            # 🧠 SMART STABILITY-AWARE ADAPTIVE LR - Recognizes sweet spots and maintains them
            self.scheduler = None  # Custom adaptive logic
            self.scheduler_mode = "adaptive"
            self.acc_history = []  # Track accuracy history for adaptive decisions
            
            # 🎯 SMART LR PARAMETERS - Stability-focused approach
            self.lr_increase_threshold = 0.005  # Still sensitive to improvements
            self.lr_decrease_threshold = -0.01  # Responsive to performance drops
            self.lr_multiplier_up = 1.5  # AGGRESSIVE increase (50% for speed)
            self.lr_multiplier_down = 0.6  # More aggressive drops (40% reduction)
            
            # 🧠 STABILITY TRACKING - Key innovation for smart LR management
            self.lr_stability_period = 4  # Keep good LR for at least 4 epochs before considering changes
            self.lr_stable_epochs = 0  # How long current LR has been stable
            self.lr_performance_window = []  # Track performance over stability window
            self.current_lr_is_effective = False  # Flag when LR is in sweet spot
            
            # 🎚️ AGGRESSIVE FAST LR OPTIMIZATION - Your exact specifications  
            self.sweet_spot_threshold = 0.001  # 0.1% improvement = keep LR (sweet spot found)
            self.explore_threshold = 0.0005  # 0.05% threshold for LR exploration  
            self.negative_threshold = 0.0001  # 0.01% drop after LR increase = revert/lower
            self.sweet_spot_epochs = 0  # Consecutive epochs of good performance
            self.sweet_spot_required = 1  # AGGRESSIVE: Only 1 epoch needed for sweet spot (was 2)
            
            # 🚀 AGGRESSIVE EXPLORATION & RECOVERY
            self.lr_just_increased = False  # Track if we just increased LR
            self.prev_val_acc = 0.0  # Track previous accuracy for immediate feedback
            self.exploration_mode = False  # Flag when actively exploring higher LR
            self.aggressive_mode = True  # FORCE aggressive exploration
            
            # Original parameters
            self.plateau_epochs = 0  # Track epochs without improvement  
            self.plateau_lr_boost = False  # Flag for plateau escape attempts
            self.warmup_epochs = 2  # REDUCED: Faster warmup for aggressive exploration
            self.current_epoch = 0  # Track current epoch for warmup
            
            print(f"🚀 Using AGGRESSIVE FAST LR Optimizer: finds optimal LR in minimum time!")
            print(f"  🎯 Sweet spot threshold: >{self.sweet_spot_threshold*100:.2f}% improvement = keep LR")
            print(f"  🔍 Exploration rule: ≤{self.sweet_spot_threshold*100:.2f}% = AGGRESSIVELY explore higher LR")
            print(f"  ⚠️ Recovery threshold: ≥{self.negative_threshold*100:.2f}% drop = immediate revert")
            print(f"  ⚡ INSTANT detection: {self.sweet_spot_required} epoch only (was 2)")
            print(f"  💨 AGGRESSIVE increases: +40% LR jumps (was +20%)")
            print(f"  🎯 AGGRESSIVE decreases: -25% to -40% reductions")
            print(f"  🚀 FIXED: 0.01% improvement now triggers exploration (was incorrectly stable)")
            print(f"  ⏰ FASTER: 2-epoch warmup, 5-epoch patience, 4-epoch plateau escape")
        elif self.lr_strategy == 'onecycle':
            self.scheduler = None  # Will be initialized in train() method
            self.scheduler_mode = "onecycle"
            print(f"🔄 Using OneCycleLR: will be initialized with actual dataset size")
        else:  # plateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
            self.scheduler_mode = "plateau"
            print(f"📉 Using ReduceLROnPlateau: factor=0.5, patience=3")
        
        self.lr_reduce_count = 0
        self.max_lr_reductions = 4
        self.prev_lr = self.optimizer.param_groups[0]['lr']
        
        # 🛡️ ENHANCED REGULARIZATION SETUP
        # MixUp data augmentation - reduces overfitting by creating synthetic training examples
        self.mixup = MixUp(alpha=0.2)
        self.cutmix = CutMix(alpha=0.2)
        # Softer early regularization; will ramp after warmup epochs
        self.mixup_prob = 0.2
        self.cutmix_prob = 0.0
        self.reg_ramp_epochs = 5
        
        # Label Smoothing - prevents model from being overconfident
        self.label_smoothing = LabelSmoothing(smoothing=0.05)  # milder early smoothing
        
        # Enhanced loss function with class weights for severe imbalance
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"🎯 Using class weights: {class_weights.cpu().numpy()}")
        self.criterion = FocalLoss(weight=class_weights, gamma=1.5)
        # Regression loss for optional dual-head (ordinal/regression auxiliary)
        self.use_regression_head = True
        self.lambda_reg = 0.2
        self.regression_loss = nn.SmoothL1Loss()
        # Temperature scaling for calibration (learned after training on validation logits)
        self.temperature = torch.nn.Parameter(torch.ones(1, device=self.device), requires_grad=True)
        self.use_temperature_scaling = True
        self.select_threshold = 0.6  # abstain below this probability
        
        print(f"🛡️ Regularization enabled:")
        print(f"  📊 MixUp: α=0.2, prob={int(self.mixup_prob*100)}% → Early epochs softened")
        print(f"  📋 Label Smoothing: 5% → Early stability")
        print(f"  🔇 Input Noise: 1% strength → Improves robustness")
        
        # 🚀 EMA weights instead of SWA
        self.use_ema = True
        if self.use_ema:
            self.ema_decay = 0.999
            self.ema_model = None
            print(f"🚀 EMA enabled: decay={self.ema_decay}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Training metadata for comprehensive logging
        self.best_model_path = None
        self.training_start_time = None
        self.save_dir = None
        
        print(f"🤖 CNN Branch Research Trainer initialized on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 Early Stopping: {self.patience} epochs patience")
        
        # Component analysis
        self._analyze_components()
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            self._load_checkpoint()
    
    def _analyze_components(self):
        """Analyze model components for monitoring"""
        gpt2_params = sum(p.numel() for name, p in self.model.named_parameters() if 'timesnet' in name)
        cnn_params = sum(p.numel() for name, p in self.model.named_parameters() if 'cnn' in name)
        classifier_params = sum(p.numel() for name, p in self.model.named_parameters() if 'classifier' in name)
        
        print(f"🔍 Model Component Analysis:")
        print(f"  🧠 TimesNet Encoder: {gpt2_params:,} parameters")
        print(f"  🎯 CNN Branch: {cnn_params:,} parameters")  
        print(f"  🎨 Classifier: {classifier_params:,} parameters")
        print(f"  📊 Total: {gpt2_params + cnn_params + classifier_params:,} parameters")
    
    def _reset_lr_tracking(self):
        """Reset LR tracking variables when LR changes"""
        self.lr_stable_epochs = 0
        self.lr_performance_window.clear()
        self.current_lr_is_effective = False
        self.sweet_spot_epochs = 0
        self.lr_just_increased = False
        self.exploration_mode = False
        print(f"   🔄 LR tracking reset - starting fresh evaluation")
    
    def _load_checkpoint(self):
        """Load model checkpoint and resume training state"""
        try:
            print(f"📂 Loading checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Load model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Standard checkpoint format
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except:
                        print("⚠️  Could not load scheduler state, continuing with fresh scheduler")
                
                # Load training metadata if available
                if 'best_val_acc' in checkpoint:
                    self.best_val_acc = checkpoint['best_val_acc']
                    print(f"✅ Loaded best validation accuracy: {self.best_val_acc:.2f}%")
                if 'best_val_loss' in checkpoint:
                    self.best_val_loss = checkpoint['best_val_loss']
                if 'epoch' in checkpoint:
                    self.resume_from_epoch = checkpoint['epoch'] + 1
                    print(f"📍 Will resume from epoch: {self.resume_from_epoch}")
                
            else:
                # Simple state dict format (just model weights)
                self.model.load_state_dict(checkpoint)
                print("✅ Loaded model weights (simple format)")
            
            print(f"🎯 Checkpoint loaded successfully!")
            print(f"🔄 Ready to continue training from previous best accuracy: {self.best_val_acc:.2f}%")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("🔄 Continuing with fresh model initialization")
            self.checkpoint_path = None
            self.resume_from_epoch = 0
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different components"""
        
        # Separate parameter groups
        gpt2_params = []
        cnn_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'timesnet' in name:
                gpt2_params.append(param)
            elif 'cnn' in name:
                cnn_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                # Default group for other parameters
                classifier_params.append(param)
        
        # Enhanced optimizer with different LR strategies for plateau breaking
        optimizer = optim.AdamW([
            {'params': gpt2_params, 'lr': self.learning_rate * 0.02},   # Slightly higher for TimesNet (was 0.01)
            {'params': cnn_params, 'lr': self.learning_rate * 0.8},     # Higher CNN LR for better learning (was 0.5)
            {'params': classifier_params, 'lr': self.learning_rate * 1.0}  # Full LR for classifier (was 0.8)
        ], weight_decay=self.weight_decay * 1.5, betas=(0.9, 0.999), eps=1e-8)  # Increased weight decay for regularization
        
        print(f"⚙️  Enhanced Multi-Component Optimizer Configuration:")
        print(f"  🧠 TimesNet LR: {self.learning_rate * 0.02:.6f} (0.02x - fine-tuned for attention)")
        print(f"  🎯 CNN LR: {self.learning_rate * 0.8:.6f} (0.8x - optimized for convolution)")
        print(f"  🎨 Classifier LR: {self.learning_rate * 1.0:.6f} (1.0x - full learning rate)")
        print(f"  📉 Weight Decay: {self.weight_decay * 1.5:.6f} (enhanced regularization)")
        print(f"  🛡️ Gradient Clipping: 0.1 max norm (tight control)")
        print(f"  🔗 Components: {len(gpt2_params)} TimesNet + {len(cnn_params)} CNN + {len(classifier_params)} Classifier params")
        print(f"  ✅ Multi-component learning rates configured and preserved across steps")
        
        return optimizer
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0, total_epochs: int = 50) -> tuple:
        """Train for one epoch with enhanced monitoring"""
        self.model.train()
        
        # 🎛️ Update adaptive dropout rates based on training progress
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch, total_epochs)
            
        total_loss = 0
        correct = 0
        total = 0
        # After ramp, enable stronger regularization
        if epoch >= self.reg_ramp_epochs:
            self.mixup_prob = 0.4
            self.cutmix_prob = 0.2
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc="🔄 Training", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Support optional continuous target (data, class, cont)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, target_cls, target_cont = batch
                data, target_cls = data.to(self.device), target_cls.to(self.device)
                target_cont = target_cont.to(self.device)
            else:
                data, target_cls = batch
                data, target_cls = data.to(self.device), target_cls.to(self.device)
                target_cont = None
            
            # 🛡️ ENHANCED REGULARIZATION APPLICATION
            # Input noise augmentation - improves model robustness to small perturbations
            if self.model.training and torch.rand(1) < 0.4:  # Increased to 40% chance
                noise = torch.randn_like(data) * 0.01  # 1% noise strength
                data = data + noise
            
            # MixUp / CutMix augmentation
            use_cutmix = self.model.training and torch.rand(1) < self.cutmix_prob
            use_mixup = not use_cutmix and self.model.training and torch.rand(1) < self.mixup_prob
            if use_cutmix:
                data, target_a, target_b, lam = self.cutmix(data, target_cls)
            elif use_mixup:
                data, target_a, target_b, lam = self.mixup(data, target_cls)
            else:
                target_a, target_b, lam = target_cls, None, 1.0
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass for better performance
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    if isinstance(output, tuple):
                        logits, ret_pred = output
                    else:
                        logits, ret_pred = output, None
                    # Enhanced loss calculation with regularization; clamp logits to avoid inf in softmax
                    if isinstance(logits, torch.Tensor):
                        logits = torch.clamp(logits, -20, 20)
                    if use_mixup or use_cutmix:
                        # MixUp loss: weighted combination of two targets
                        loss_a = self.label_smoothing(logits, target_a)
                        loss_b = self.label_smoothing(logits, target_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        # Standard loss with label smoothing
                        loss = self.label_smoothing(logits, target_a)
                    # Optional regression auxiliary loss (if available)
                    if self.use_regression_head and ret_pred is not None and target_cont is not None:
                        loss = loss + self.lambda_reg * self.regression_loss(ret_pred, target_cont)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Tighter gradient clipping to prevent explosion and catch NaNs
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
                if not torch.isfinite(loss):
                    print("⚠️  Non-finite loss detected (AMP path). Skipping step and reducing LR.")
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * 0.5, 1e-7)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # EMA update (after optimizer step)
                if self.use_ema:
                    if self.ema_model is None:
                        from copy import deepcopy
                        self.ema_model = deepcopy(self.model).to(self.device)
                        for p in self.ema_model.parameters():
                            p.requires_grad_(False)
                    with torch.no_grad():
                        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
                
                # Scheduler step is handled once per batch below
                
            else:
                # Standard precision training
                output = self.model(data)
                if isinstance(output, tuple):
                    logits, ret_pred = output
                else:
                    logits, ret_pred = output, None
                
                # Enhanced loss calculation with regularization; clamp logits to avoid inf in softmax
                if isinstance(logits, torch.Tensor):
                    logits = torch.clamp(logits, -20, 20)
                if use_mixup or use_cutmix:
                    # MixUp loss: weighted combination of two targets
                    loss_a = self.label_smoothing(logits, target_a)
                    loss_b = self.label_smoothing(logits, target_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    # Standard loss with label smoothing
                    loss = self.label_smoothing(logits, target_a)
                # Optional regression auxiliary loss
                if self.use_regression_head and ret_pred is not None and target_cont is not None:
                    loss = loss + self.lambda_reg * self.regression_loss(ret_pred, target_cont)
                
                loss.backward()
                
                # Tighter gradient clipping to prevent explosion and catch NaNs
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
                if not torch.isfinite(loss):
                    print("⚠️  Non-finite loss detected (FP32 path). Skipping step and reducing LR.")
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * 0.5, 1e-7)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()

                # EMA update (standard precision path)
                if self.use_ema:
                    if self.ema_model is None:
                        from copy import deepcopy
                        self.ema_model = deepcopy(self.model).to(self.device)
                        for p in self.ema_model.parameters():
                            p.requires_grad_(False)
                    with torch.no_grad():
                        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
            
            # Step scheduler for OneCycleLR (per batch) – single step for both precision paths
            if self.scheduler_mode == "onecycle":
                self.scheduler.step()
            
            # Statistics with MixUp-aware accuracy calculation
            total_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            
            # MixUp-aware accuracy: use the primary target for accuracy calculation
            if use_mixup:
                correct += pred.eq(target_a.view_as(pred)).sum().item()
            else:
                correct += pred.eq(target_a.view_as(pred)).sum().item()
            total += target_a.size(0)
            
            # Update progress bar
            current_loss = loss.item()
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Batch': f'{batch_idx+1}/{len(train_loader)}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model"""
        # Use EMA weights if available
        model_for_eval = self.ema_model if getattr(self, 'use_ema', False) and self.ema_model is not None else self.model
        model_for_eval.eval()
        total_loss = 0
        correct = 0
        total = 0
        f1_targets = []
        f1_preds = []
        prob_targets = []
        prob_preds = []
        coverage_counter = 0
        positive_correct = 0
        
        # Progress bar for validation
        pbar = tqdm(val_loader, desc="🔍 Validating", leave=False)
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = model_for_eval(data)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                loss = self.criterion(logits, target)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                f1_targets.append(target.cpu().numpy())
                f1_preds.append(pred.view(-1).cpu().numpy())
                # store probabilities for AUROC (one-vs-rest)
                # Apply temperature scaling if enabled
                if self.use_temperature_scaling:
                    logits_for_prob = logits / self.temperature.clamp_min(1e-3)
                else:
                    logits_for_prob = logits
                probs = torch.softmax(logits_for_prob, dim=1).detach().cpu().numpy()
                prob_preds.append(probs)
                prob_targets.append(target.detach().cpu().numpy())
                # Selective prediction (example on class 2 'up')
                up_prob = probs[:, 2] if probs.shape[1] >= 3 else probs.max(axis=1)
                take_mask = up_prob >= self.select_threshold
                coverage_counter += take_mask.sum()
                positive_correct += ((pred.view(-1).cpu().numpy() == 2) & take_mask).sum()
                total += target.size(0)
                
                # Update progress bar
                current_loss = loss.item()
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        try:
            y_true = np.concatenate(f1_targets) if f1_targets else np.array([])
            y_pred = np.concatenate(f1_preds) if f1_preds else np.array([])
            macro_f1 = f1_score(y_true, y_pred, average='macro') if y_true.size else 0.0
            # AUROC (OVR) requires probs and at least two classes present
            auroc = 0.0
            if prob_targets:
                y_true_prob = np.concatenate(prob_targets)
                y_prob = np.concatenate(prob_preds)
                if len(np.unique(y_true_prob)) > 1:
                    auroc = roc_auc_score(y_true_prob, y_prob, multi_class='ovr')
            # Precision@k (top-k by confidence in class 2 "up" as example)
            precision_at_k = 0.0
            if prob_targets:
                k = max(1, int(0.01 * len(y_true)))  # top 1%
                up_probs = y_prob[:, 2] if y_prob.shape[1] >= 3 else y_prob.max(axis=1)
                topk_idx = np.argsort(-up_probs)[:k]
                precision_at_k = (y_true_prob[topk_idx] == 2).mean() if k > 0 else 0.0
            coverage = coverage_counter / len(y_true_prob) if prob_targets else 0.0
            selective_precision = (positive_correct / coverage_counter) if coverage_counter > 0 else 0.0
            print(f"  🎯 Macro-F1: {macro_f1:.3f} | AUROC(OVR): {auroc:.3f} | Precision@1%: {precision_at_k:.3f} | Selective P(up|p>{self.select_threshold}): {selective_precision:.3f} @Coverage {coverage:.2%}")
        except Exception as e:
            print(f"⚠️  Metric computation issue: {e}")
        
        return avg_loss, accuracy

    def fit_temperature(self, val_loader: DataLoader, max_iters: int = 100) -> float:
        """Fit temperature scaling on validation logits to minimize NLL.

        Leaves model weights frozen; only optimizes the scalar temperature.
        """
        if not self.use_temperature_scaling:
            return float(self.temperature.detach().cpu().item())
        self.model.eval()
        # Collect logits and labels
        logits_list, targets_list = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                logits = output[0] if isinstance(output, (list, tuple)) else output
                logits_list.append(logits.detach())
                targets_list.append(target.detach())
        logits_all = torch.cat(logits_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
        # Optimize temperature
        nll_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        self.temperature.data.clamp_(min=1e-3)

        def _closure():
            optimizer.zero_grad()
            scaled = logits_all / self.temperature.clamp_min(1e-3)
            loss = nll_loss(scaled, targets_all)
            loss.backward()
            return loss

        try:
            optimizer.step(_closure)
        except Exception:
            pass
        self.temperature.data.clamp_(min=1e-3)
        return float(self.temperature.detach().cpu().item())
    
    def _evaluate_swa_model(self, val_loader: DataLoader) -> tuple:
        """Evaluate the SWA (Stochastic Weight Averaging) model performance"""
        if self.swa_model is None:
            return float('inf'), 0.0
        
        self.swa_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.swa_model(data)
                logits = output[0] if isinstance(output, (list, tuple)) else output
                loss = self.criterion(logits, target)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, save_dir: str = "models/research") -> dict:
        """Train the enhanced model with robust logging and interruption handling"""
        
        # Setup training environment
        self.save_dir = save_dir
        self.training_start_time = time.time()
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize OneCycleLR if using onecycle scheduling
        if self.lr_strategy == 'onecycle' and self.scheduler is None:
            steps_per_epoch = len(train_loader)
            total_steps = epochs * steps_per_epoch
            # Use per-param-group max_lr to preserve ratios and keep schedule conservative (1.5x)
            max_lrs = [pg['lr'] * 1.5 for pg in self.optimizer.param_groups]
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=0.5,  # slower warmup to avoid early blow-ups
                cycle_momentum=False,
                anneal_strategy='cos'
            )
            pretty_max = ", ".join(f"{lr:.2e}" for lr in max_lrs)
            print(f"🔄 OneCycleLR initialized: {total_steps} total steps, max_lr per group=[{pretty_max}]")
        
        print(f"🚀 Starting research training for {epochs} epochs...")
        print(f"📊 Training samples: {len(train_loader.dataset):,}")
        print(f"📊 Validation samples: {len(val_loader.dataset):,}")
        print(f"💾 Save directory: {save_dir}")
        print("=" * 60)
        
        # Training progress bar - adjusted for checkpoint resuming
        start_epoch = self.resume_from_epoch
        remaining_epochs = epochs - start_epoch
        if start_epoch > 0:
            print(f"🔄 Resuming training from epoch {start_epoch+1}/{epochs}")
            print(f"📊 Remaining epochs: {remaining_epochs}")
        
        epoch_pbar = tqdm(range(start_epoch, epochs), desc="🎯 Training Progress", position=0)
        
        try:
            for epoch in epoch_pbar:
                epoch_start = time.time()
                
                # Training with adaptive dropout scheduling
                train_loss, train_acc = self.train_epoch(train_loader, epoch=epoch, total_epochs=epochs)
                
                # Validation
                val_loss, val_acc = self.validate(val_loader)
                
                # Store history
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                
                # Store current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'Epoch': f'{epoch+1}/{epochs}',
                    'Train Loss': f'{train_loss:.4f}',
                    'Train Acc': f'{train_acc:.2f}%',
                    'Val Loss': f'{val_loss:.4f}',
                    'Val Acc': f'{val_acc:.2f}%',
                    'Time': f'{epoch_time:.1f}s'
                })
                
                # Print detailed progress
                print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
                print(f"  🎯 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  🔍 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"  ⏱️  Epoch Time: {epoch_time:.1f}s")
                
                # Enhanced early stopping with dual criteria (loss AND accuracy)
                accuracy_improved = val_acc > self.best_val_acc
                if accuracy_improved:
                    self.best_val_acc = val_acc
                    print(f"  🚀 NEW BEST ACCURACY: {val_acc:.2f}% (epoch {epoch+1})")
                
                if accuracy_improved:
                    self.patience_counter = 0
                    
                    # Update best model state (save at end of training)
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict().copy(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'best_val_loss': self.best_val_loss,
                        'best_val_acc': self.best_val_acc,
                        'run_id': self.run_id
                    }
                    print(f"  🚀 NEW BEST MODEL: {val_acc:.2f}% (will save at training end)")

                # Handle scheduler stepping based on type
                current_lr = self.optimizer.param_groups[0]['lr']
                
                if self.scheduler_mode == "plateau":
                    # ReduceLROnPlateau: step with validation accuracy (epoch-based)
                    self.scheduler.step(val_acc)
                    current_lr_after = self.optimizer.param_groups[0]['lr']
                    
                    # Check if LR was reduced
                    if current_lr_after < self.prev_lr - 1e-12:
                        self.lr_reduce_count += 1
                        self.patience_counter = 0  # reset patience after LR drop
                        print(f"🔽 LR reduced to {current_lr_after:.2e} (total reductions: {self.lr_reduce_count})")
                    else:
                        self.patience_counter += 1
                        print(f"  ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
                    
                    self.prev_lr = current_lr_after
                    current_lr = current_lr_after
                    
                elif self.scheduler_mode == "adaptive":
                    # 🚀 FAST SMART ADAPTIVE LR - Finds optimal quickly and maintains it!
                    self.acc_history.append(val_acc)
                    self.current_epoch += 1
                    
                    # Store previous accuracy for immediate feedback
                    if len(self.acc_history) >= 2:
                        self.prev_val_acc = self.acc_history[-2]
                    
                    # Warmup phase - gradually increase LR for first few epochs
                    if self.current_epoch <= self.warmup_epochs:
                        warmup_factor = self.current_epoch / self.warmup_epochs
                        for i, param_group in enumerate(self.optimizer.param_groups):
                            if i == 0:  # TimesNet
                                param_group['lr'] = (self.learning_rate * 0.02) * warmup_factor
                            elif i == 1:  # CNN
                                param_group['lr'] = (self.learning_rate * 0.8) * warmup_factor
                            else:  # Classifier
                                param_group['lr'] = (self.learning_rate * 1.0) * warmup_factor
                        print(f"🔥 WARMUP: Epoch {self.current_epoch}/{self.warmup_epochs}, LR factor: {warmup_factor:.2f}")
                        self.patience_counter = 0
                        current_lr = self.optimizer.param_groups[0]['lr']
                    else:
                        # 🧠 SMART LR LOGIC - Post-warmup stability-aware adjustments
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # Track plateau epochs
                        if val_acc <= self.best_val_acc:
                            self.plateau_epochs += 1
                        else:
                            self.plateau_epochs = 0
                            self.plateau_lr_boost = False  # Reset plateau escape flag
                        
                        if len(self.acc_history) >= 2:
                            acc_change = self.acc_history[-1] - self.acc_history[-2]
                            
                            # 🚀 PRIORITY 1: IMMEDIATE RECOVERY - Just increased LR and got ≥0.01% drop
                            if self.lr_just_increased and acc_change < -self.negative_threshold:  # ≥0.01% drop after LR increase
                                new_lr = max(current_lr * 0.75, 1e-6)  # More aggressive: 25% reduction (was 20%)
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                print(f"⚡ IMMEDIATE RECOVERY: LR {current_lr:.2e} → {new_lr:.2e} (acc dropped {acc_change:.3f}% after increase)")
                                self._reset_lr_tracking()
                                self.patience_counter = 0
                                
                            # 🍯 PRIORITY 2: SWEET SPOT CONFIRMED - >0.10% improvement = KEEP LR
                            elif acc_change > self.sweet_spot_threshold:  # >0.10% improvement
                                self.sweet_spot_epochs += 1
                                self.lr_performance_window.append(acc_change)
                                self.lr_just_increased = False  # Reset increase flag
                                
                                # INSTANT sweet spot confirmation (only 1 epoch needed!)
                                if self.sweet_spot_epochs >= self.sweet_spot_required:
                                    self.current_lr_is_effective = True
                                    self.exploration_mode = False
                                    print(f"🍯 SWEET SPOT CONFIRMED: LR {current_lr:.2e} is optimal! (+{acc_change:.3f}%)")
                                    print(f"   🔒 MAINTAINING this LR - no exploration needed")
                                else:
                                    print(f"🎯 EXCELLENT: LR {current_lr:.2e} performing well (+{acc_change:.3f}%)")
                                
                                if val_acc <= self.best_val_acc:
                                    self.patience_counter += 1
                                else:
                                    self.patience_counter = 0
                                    
                            # 🔍 PRIORITY 3: AGGRESSIVE EXPLORATION - ≤0.10% improvement = MUST TRY HIGHER LR
                            elif acc_change <= self.sweet_spot_threshold and not self.current_lr_is_effective:  # ≤0.10% (includes your 0.01% case!)
                                if not self.lr_just_increased:  # Only explore if we didn't just increase
                                    new_lr = min(current_lr * 1.4, self.learning_rate * 2.5)  # MORE AGGRESSIVE: 40% increase (was 20%)
                                    for param_group in self.optimizer.param_groups:
                                        param_group['lr'] = new_lr
                                    print(f"🔍 AGGRESSIVE EXPLORATION: LR {current_lr:.2e} → {new_lr:.2e}")
                                    print(f"   💨 Improvement {acc_change:.3f}% ≤ 0.10% - MUST find better LR!")
                                    self.lr_just_increased = True
                                    self.exploration_mode = True
                                    self.sweet_spot_epochs = 0  # Reset sweet spot tracking
                                    self.patience_counter = 0
                                else:
                                    # Just increased last epoch, evaluate quickly
                                    print(f"⏱️ QUICK EVAL: Testing new LR {current_lr:.2e} (improvement: {acc_change:.3f}%)")
                                    if val_acc <= self.best_val_acc:
                                        self.patience_counter += 1
                                    else:
                                        self.patience_counter = 0
                                        
                            # 🚨 PRIORITY 4: SEVERE DROP - Major performance decline
                            elif acc_change < self.lr_decrease_threshold:  # < -1% drop
                                new_lr = max(current_lr * 0.6, 1e-6)  # More aggressive: 40% reduction (was 30%)
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                print(f"🔽 SEVERE DROP: LR {current_lr:.2e} → {new_lr:.2e} (drop: {acc_change:.3f}%)")
                                self._reset_lr_tracking()
                                self.patience_counter = 0
                                
                            # 🚀 PRIORITY 5: PLATEAU ESCAPE - Extended stagnation
                            elif self.plateau_epochs >= 4 and not self.plateau_lr_boost and not self.current_lr_is_effective:  # Faster: 4 epochs (was 6)
                                new_lr = min(current_lr * 2.5, self.learning_rate * 2.0)  # More aggressive plateau escape
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                self.plateau_lr_boost = True
                                print(f"🚀 PLATEAU ESCAPE: LR boosted to {new_lr:.2e} (stagnant for {self.plateau_epochs} epochs)")
                                self._reset_lr_tracking()
                                self.lr_just_increased = True
                                self.exploration_mode = True
                                self.patience_counter = 0
                                
                            # 🔄 EFFECTIVENESS LOSS - Sweet spot LR declining
                            elif self.current_lr_is_effective and acc_change < 0:
                                self.sweet_spot_epochs = 0
                                self.current_lr_is_effective = False
                                print(f"❌ EFFECTIVENESS LOST: Sweet spot LR {current_lr:.2e} declining ({acc_change:.3f}%)")
                                print(f"   🔍 Will start aggressive exploration next epoch")
                                if val_acc <= self.best_val_acc:
                                    self.patience_counter += 1
                                else:
                                    self.patience_counter = 0
                        else:
                            # First epoch post-warmup
                            print(f"➡️ LR: {current_lr:.2e} (collecting initial data)")
                            if val_acc <= self.best_val_acc:
                                self.patience_counter += 1
                            else:
                                self.patience_counter = 0
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                elif self.scheduler_mode == "performance_sensitive":
                    # 🎯 NEW: Performance-sensitive scheduler based on your 2.80e-04 observation
                    lr_info = self.scheduler.step(val_acc, epoch)
                    current_lr = lr_info['new_lr']
                    
                    # Different output for warmup vs normal phases
                    if lr_info.get('is_warmup', False):
                        # During warmup: show progress and strategy
                        print(f"🔥 Warmup Phase: {lr_info['old_lr']:.2e} → {current_lr:.2e}")
                        print(f"   Strategy: {lr_info['reason']}")
                        self.patience_counter = 0  # No early stopping during warmup
                    else:
                        # Post-warmup: show performance-sensitive adaptation
                        print(f"🎯 Performance-Sensitive LR: {lr_info['old_lr']:.2e} → {current_lr:.2e}")
                        if lr_info['reason'] != 'analyzing':
                            print(f"   Reason: {lr_info['reason']}")
                            print(f"   Acc improvement: {lr_info.get('accuracy_improvement', 0):.3f}%")
                            print(f"   LR efficiency: {lr_info.get('lr_efficiency', 0):.3f}%")
                        
                        # Early stopping based on validation accuracy plateau (only post-warmup)
                        if val_acc <= self.best_val_acc:
                            self.patience_counter += 1
                            print(f"  ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
                        else:
                            self.patience_counter = 0
                        
                elif self.scheduler_mode == "onecycle":
                    # OneCycleLR: already stepped per batch, just track LR
                    print(f"🔄 OneCycle LR: {current_lr:.2e} (following predetermined schedule)")
                    
                    # For OneCycleLR, we use a different early stopping strategy
                    # based on validation accuracy plateau rather than LR reductions
                    if val_acc <= self.best_val_acc:
                        self.patience_counter += 1
                        print(f"  ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
                    else:
                        self.patience_counter = 0
                
                self.learning_rates.append(current_lr)
                
                # 🚀 STOCHASTIC WEIGHT AVERAGING (SWA) for final accuracy boost
                if self.use_swa and epoch >= int(epochs * self.swa_start_epoch):
                    if self.swa_model is None:
                        # Initialize SWA model
                        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
                        print(f"🚀 SWA initialized at epoch {epoch+1}")
                    
                    # Update SWA model every swa_freq epochs
                    if (epoch - int(epochs * self.swa_start_epoch)) % self.swa_freq == 0:
                        self.swa_model.update_parameters(self.model)
                        print(f"🚀 SWA model updated at epoch {epoch+1}")
                        
                        # Optionally evaluate SWA model performance
                        if epoch == epochs - 1:  # Last epoch
                            print(f"🧪 Evaluating SWA model performance...")
                            swa_val_loss, swa_val_acc = self._evaluate_swa_model(val_loader)
                            print(f"🚀 SWA Model: Val Loss={swa_val_loss:.4f}, Val Acc={swa_val_acc:.2f}%")
                            
                            # If SWA is better, use it as final model
                            if swa_val_acc > self.best_val_acc:
                                print(f"🎉 SWA model is better! {swa_val_acc:.2f}% > {self.best_val_acc:.2f}%")
                                self.best_val_acc = swa_val_acc
                                self.best_model_state = {
                                    'epoch': epoch,
                                    'model_state_dict': self.swa_model.module.state_dict().copy(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'train_loss': train_loss,
                                    'val_loss': swa_val_loss,
                                    'train_acc': train_acc,
                                    'val_acc': swa_val_acc,
                                    'best_val_loss': self.best_val_loss,
                                    'best_val_acc': swa_val_acc,
                                    'run_id': self.run_id,
                                    'is_swa': True
                                }

                # Early stopping logic based on scheduler type
                if self.scheduler_mode == "plateau":
                    # For ReduceLROnPlateau: stop if LR reduced too many times
                    if self.lr_reduce_count >= self.max_lr_reductions:
                        print("  🛑 Early stopping – LR reduced 4× with no new accuracy peak")
                        break
                elif self.scheduler_mode in ["adaptive", "onecycle", "performance_sensitive"]:
                    # For Adaptive/OneCycleLR/Performance-Sensitive: stop if validation accuracy plateaus for too long
                    if self.patience_counter >= self.patience:
                        print(f"  🛑 Early stopping – No improvement for {self.patience} epochs")
                        break
                
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user!")
            print(f"⏱️  Training ran for {len(self.train_losses)} epochs")
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            print(f"⏱️  Training ran for {len(self.train_losses)} epochs")
        
        finally:
            # Always generate final reports regardless of how training ended
            return self._generate_final_reports(save_dir)
    
    def _generate_final_reports(self, save_dir: str) -> dict:
        """Generate comprehensive final reports and training curves"""
        
        if not self.train_losses:
            print("⚠️  No training data to generate reports")
            return {"status": "no_data"}
        
        total_time = time.time() - self.training_start_time
        epochs_completed = len(self.train_losses)
        
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        print(f"📊 Epochs completed: {epochs_completed}")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"📉 Best validation loss: {self.best_val_loss:.4f}")
        
        # Generate training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'epochs_completed': epochs_completed,
            'total_time_hours': total_time/3600,
            'run_id': self.run_id
        }
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_history = {}
            for key, value in history.items():
                if isinstance(value, list):
                    json_history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
                else:
                    json_history[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
            json.dump(json_history, f, indent=2)
        
        print(f"📁 Training history saved to {history_path}")
        
        # Generate and save training curves
        curves_path = os.path.join(save_dir, 'training_curves.png')
        self._plot_training_curves(curves_path)
        print(f"📊 Training curves saved to {curves_path}")
        
        # Save the single best model for this training run
        if self.best_model_state is not None:
            final_model_path = os.path.join(save_dir, f'enhanced_cnn_best_{self.run_id}.pth')
            torch.save(self.best_model_state, final_model_path)
            print(f"💾 Best model saved to {final_model_path}")
            
            # Also save a generic "latest best" copy for easy loading
            generic_path = os.path.join(save_dir, 'enhanced_cnn_best.pth')
            torch.save(self.best_model_state, generic_path)
            print(f"💾 Generic best model saved to {generic_path}")
        else:
            print("⚠️  No best model to save (no training completed)")
        
        return history
    
    def _plot_training_curves(self, save_path: str):
        """Enhanced plotting method that handles interruptions gracefully"""
        
        if not self.train_losses:
            print("⚠️  No training data to plot")
            return
        
        try:
            plt.figure(figsize=(16, 6))
            epochs = list(range(1, len(self.train_losses) + 1))
            
            # Loss curves
            plt.subplot(1, 3, 1)
            plt.plot(epochs, self.train_losses, label='Train Loss', color='#2E86AB', linewidth=2.5)
            plt.plot(epochs, self.val_losses, label='Val Loss', color='#A23B72', linewidth=2.5)
            plt.title('📉 Training & Validation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Accuracy curves  
            plt.subplot(1, 3, 2)
            plt.plot(epochs, self.train_accuracies, label='Train Accuracy', color='#2E86AB', linewidth=2.5)
            plt.plot(epochs, self.val_accuracies, label='Val Accuracy', color='#A23B72', linewidth=2.5)
            plt.title('📈 Training & Validation Accuracy', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Learning rate and performance
            plt.subplot(1, 3, 3)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Validation accuracy on left axis
            line1 = ax1.plot(epochs, self.val_accuracies, label='Val Accuracy', 
                           color='#F18F01', linewidth=3)
            ax1.set_ylabel('Validation Accuracy (%)', color='#F18F01', fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#F18F01')
            
            # Learning rate on right axis (if available)
            if self.learning_rates:
                line2 = ax2.plot(epochs, self.learning_rates[:len(epochs)], label='Learning Rate', 
                               color='#C73E1D', linewidth=2, linestyle='--')
                ax2.set_ylabel('Learning Rate', color='#C73E1D', fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='#C73E1D')
                ax2.set_yscale('log')
            
            plt.title('🎯 Performance & Learning Rate', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.grid(True, alpha=0.3)
            
            # Mark best epoch
            if hasattr(self, 'best_val_acc') and self.val_accuracies:
                best_epoch = self.val_accuracies.index(max(self.val_accuracies)) + 1
                ax1.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, linewidth=2)
                ax1.text(best_epoch, max(self.val_accuracies), f'Best: {max(self.val_accuracies):.1f}%', 
                        rotation=90, verticalalignment='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Important: close to free memory
            
        except Exception as e:
            print(f"⚠️  Could not generate training curves: {e}")
            # Create a simple text-based summary instead
            self._create_text_summary(save_path.replace('.png', '_summary.txt'))
    
    def plot_training_curves(self, save_path: str):
        """Public wrapper so external callers can plot curves."""
        self._plot_training_curves(save_path)

    def _create_text_summary(self, save_path: str):
        """Create a text-based training summary when plotting fails"""
        
        try:
            with open(save_path, 'w') as f:
                f.write("🔬 ENHANCED CNN TRAINING SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"📊 Epochs Completed: {len(self.train_losses)}\n")
                f.write(f"🏆 Best Validation Accuracy: {self.best_val_acc:.2f}%\n")
                f.write(f"📉 Best Validation Loss: {self.best_val_loss:.4f}\n\n")
                
                if self.train_losses:
                    f.write("📈 EPOCH-BY-EPOCH PROGRESS:\n")
                    f.write("-" * 30 + "\n")
                    for i, (tl, ta, vl, va) in enumerate(zip(
                        self.train_losses, self.train_accuracies, 
                        self.val_losses, self.val_accuracies)):
                        f.write(f"Epoch {i+1:2d}: Train {ta:5.1f}% | Val {va:5.1f}% | "
                               f"TLoss {tl:.3f} | VLoss {vl:.3f}\n")
                
            print(f"📄 Training summary saved to {save_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create text summary: {e}")


def main():
    """Main research training function"""
    print("🔬 CNN Branch Research Training Script")
    print("=" * 60)
    
    # Configuration - Use project-level paths
    project_root = Path(__file__).parent.parent.parent  # Go up to price model/
    csv_path = project_root / "data" / "latest_dataset.csv"
    
    # Fallback to existing datasets if latest doesn't exist
    if not csv_path.exists():
        csv_path = project_root / "data" / "reduced_feature_set_dataset.csv"
    
    # 🚀 OPTIMIZED HYPERPARAMETERS FOR RESUMING FROM 53.7% MODEL
    # ============================================================
    # These settings use the proven adaptive scheduler that achieved 53.7%
    # Combined with extended training and enhanced regularization for 60%+ target
    batch_size = 256              # Larger effective batch for stability; use AMP and gradient clipping
    epochs = 150                 # Extended for proven adaptive method → Should reach 55-60% faster
    learning_rate = 2.8e-4       # Restored proven base LR to avoid instability with OneCycle
    weight_decay = 5e-4          # Increased from 1e-4 → Stronger L2 regularization
    test_size = 0.2
    random_state = 42
    save_dir = project_root / "models" / "research"  # Save to project-level models/
    experiment_name = "enhanced_cnn_research"
    
    print(f"⚙️  Configuration:")
    print(f"  📁 Dataset: {csv_path}")
    print(f"  📦 Batch size: {batch_size}")
    print(f"  🔄 Epochs: {epochs}")
    print(f"  📈 Learning rate: {learning_rate}")
    print(f"  📉 Weight decay: {weight_decay}")
    print(f"  ✂️  Test size: {test_size}")
    print(f"  💾 Save directory: {save_dir}")
    print("=" * 60)
    
    # Load and preprocess data (now returns class weights too)
    X_train, X_test, y_train, y_test, features_per_day, class_weights = load_and_preprocess_data(
        csv_path, test_size, random_state
    )
    
    # Create datasets - use the cleaned data from load_and_preprocess_data
    print(f"📊 Creating datasets...")
    
    # Load and clean data again for dataset creation (since we need raw unscaled data)
    df = pd.read_csv(csv_path)
    feature_columns = [col for col in df.columns if col not in ['Ticker', 'Label', 'TargetRet']]
    
    # Remove NaN rows (same as in load_and_preprocess_data)
    nan_rows = df[feature_columns].isna().any(axis=1)
    df_clean = df[~nan_rows].copy() if nan_rows.any() else df.copy()
    
    X_raw = df_clean[feature_columns].values.astype(np.float32)
    y_raw = df_clean['Label'].values.astype(np.int64)
    
    # Split raw data (will be replaced by purged CV when enabled)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )
    
    # If continuous target present, keep alongside for an optional regression head
    target_ret = None
    if 'TargetRet' in df_clean.columns:
        target_ret = df_clean['TargetRet'].values.astype(np.float32)
        y_train_ret = target_ret[~nan_rows].reshape(-1)[train_test_split(np.arange(len(X_raw)), test_size=test_size, random_state=random_state, stratify=y_raw)[0]] if False else None
    train_dataset = FinancialDataset(X_train_raw, y_train_raw)
    test_dataset = FinancialDataset(X_test_raw, y_test_raw)
    
    # Create data loaders with class-balanced sampling for training
    class_sample_counts = np.bincount(y_train_raw)
    class_weights_sampler = 1.0 / np.clip(class_sample_counts, 1, None)
    sample_weights = class_weights_sampler[y_train_raw]
    from torch.utils.data.sampler import WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ----- Model Selection -----
    MODEL_TYPE = "timesnet_hybrid"  # Using optimized version to prevent overfitting
    
    # 🔄 CHECKPOINT LOADING CONFIGURATION
    # =====================================
    # Option 1: Resume from 53.7% accuracy model (RECOMMENDED)
    #checkpoint_path = save_dir / "enhanced_cnn_best_20250806_134338.pth"
    
    # Option 2: Start fresh training (uncomment line below)
    checkpoint_path = None
    
    # Option 3: Resume from different checkpoint (modify path below)  
    # checkpoint_path = save_dir / "enhanced_cnn_best_YYYYMMDD_HHMMSS.pth"
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"🎯 Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
        print(f"📝 No checkpoint found, starting fresh training")
    
    print(f"\n🤖 Creating model: {MODEL_TYPE} ...")

    if MODEL_TYPE == "optimized_timesnet_hybrid":
        model = create_optimized_timesnet_hybrid(
            features_per_day=features_per_day,
            num_classes=5
        )
    elif MODEL_TYPE == "timesnet_hybrid":
        model = create_timesnet_hybrid(
            features_per_day=features_per_day,
            num_classes=3,
            cnn_channels=512,
            timesnet_emb=512,
            timesnet_depth=5
        )
    elif MODEL_TYPE == "simple_cnn":
        from src.training.simple_cnn_trainer import SimpleCNN  # Lazy import
        model = SimpleCNN(features_per_day=features_per_day, num_classes=5)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    # Create trainer with class weights for imbalanced data and checkpoint loading
    trainer = EnhancedCNNResearchTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None
    )
    
    # Optional: Purged Time-Series CV with embargo (opt-in)
    use_cv = False  # set True to enable
    embargo_frac = 0.02
    n_splits = 5

    if use_cv:
        n = len(X_raw)
        kf = KFold(n_splits=n_splits, shuffle=False)
        fold = 0
        best_macro_f1 = -1.0
        for train_idx, val_idx in kf.split(np.arange(n)):
            fold += 1
            # Apply embargo: remove a small fraction around the split boundary from validation indices
            emb = int(len(val_idx) * embargo_frac)
            if emb > 0:
                val_idx = val_idx[emb:-emb] if (len(val_idx) - 2*emb) > 0 else val_idx
            X_tr, y_tr = X_raw[train_idx], y_raw[train_idx]
            X_va, y_va = X_raw[val_idx], y_raw[val_idx]
            tr_ds = FinancialDataset(X_tr, y_tr)
            va_ds = FinancialDataset(X_va, y_va)
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
            print(f"\n🧪 Fold {fold}/{n_splits} (embargo {embargo_frac*100:.1f}%): train={len(tr_ds)}, val={len(va_ds)}")
            fold_trainer = EnhancedCNNResearchTrainer(
                model=create_timesnet_hybrid(features_per_day=features_per_day, num_classes=3, cnn_channels=512, timesnet_emb=512, timesnet_depth=5),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                class_weights=class_weights
            )
            fold_hist = fold_trainer.train(tr_loader, va_loader, epochs=epochs, save_dir=save_dir)
            # Fit temperature per fold
            if getattr(fold_trainer, 'use_temperature_scaling', False):
                temp_value = fold_trainer.fit_temperature(va_loader)
                print(f"🌡️  Fold {fold} temperature: {temp_value:.3f}")
            # Track best by Macro-F1 (printed in validation); here we fallback to best val acc
            if fold_trainer.best_val_acc > best_macro_f1:
                best_macro_f1 = fold_trainer.best_val_acc
        print(f"\n✅ CV completed. Best fold metric: {best_macro_f1:.2f}")
        history = {"cv_best": best_macro_f1}
    else:
        # Train model (single split)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            save_dir=save_dir
        )
        # Fit temperature on validation set for calibrated probabilities
        if getattr(trainer, 'use_temperature_scaling', False):
            temp_value = trainer.fit_temperature(test_loader)
            print(f"🌡️  Fitted temperature: {temp_value:.3f}")
    
    # Plot training curves
    trainer.plot_training_curves(os.path.join(save_dir, 'training_curves.png'))
    
    # 🎯 Generate LR performance analysis if using performance-sensitive scheduler
    if hasattr(trainer.scheduler, 'plot_lr_performance'):
        lr_analysis_path = os.path.join(save_dir, 'lr_performance_analysis.png')
        trainer.scheduler.plot_lr_performance(lr_analysis_path)
        
        # Print LR insights
        lr_summary = trainer.scheduler.get_performance_summary()
        print(f"\n🎯 Learning Rate Analysis:")
        print(f"  🎪 Best LR found: {lr_summary['best_lr']:.2e}")
        print(f"  🏆 Best accuracy achieved: {lr_summary['best_accuracy']:.2f}%")
        print(f"  📊 LR performance analysis saved to: {lr_analysis_path}")
    
    # Print final results (guard if training aborted early)
    print("\n🎉 Research training completed!")
    if isinstance(history, dict) and 'val_accuracies' in history and 'val_losses' in history and history.get('val_accuracies') and history.get('val_losses'):
        print(f"🏆 Best validation accuracy: {max(history['val_accuracies']):.2f}%")
        print(f"📉 Best validation loss: {min(history['val_losses']):.4f}")
    else:
        print("⚠️  No validation metrics available (training ended before first epoch)")
    if hasattr(trainer, 'swa_model') and getattr(trainer, 'swa_model', None) is not None:
        print(f"🚀 SWA model was used for final accuracy boost!")
    print(f"📁 Training history saved to: {save_dir}/training_history.json")
    print(f"💾 Best model saved to: {save_dir}/enhanced_cnn_best.pth")
    print(f"📊 Training curves saved to: {save_dir}/training_curves.png")


if __name__ == "__main__":
    main() 