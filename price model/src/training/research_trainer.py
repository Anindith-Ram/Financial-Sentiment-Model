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
    """Load and preprocess data for enhanced training"""
    
    print(f"📊 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Separate features and labels - exclude Ticker and Label columns
    feature_columns = [col for col in df.columns if col not in ['Ticker', 'Label']]
    
    # Handle NaN values before training
    print(f"🔍 Checking for NaN values...")
    nan_rows = df[feature_columns].isna().any(axis=1)
    if nan_rows.any():
        nan_count = nan_rows.sum()
        print(f"⚠️  Found {nan_count:,} rows with NaN values ({nan_count/len(df)*100:.2f}%)")
        print(f"🧹 Removing NaN rows for training...")
        df_clean = df[~nan_rows].copy()
        print(f"✅ Clean dataset: {len(df_clean):,} rows")
    else:
        df_clean = df.copy()
        print(f"✅ No NaN values found")
    
    X = df_clean[feature_columns].values.astype(np.float32)
    y = df_clean['Label'].values.astype(np.int64)
    
    print(f"📈 Dataset shape: {X.shape}")
    print(f"🔢 Features per sample: {X.shape[1]}")
    
    # Analyze class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n🎯 Class Distribution Analysis:")
    total_samples = len(y)
    for cls, count in zip(unique_classes, class_counts):
        percentage = count / total_samples * 100
        print(f"  Class {cls}: {count:,} samples ({percentage:.1f}%)")
    
    # Calculate moderated class weights to prevent gradient explosion
    raw_weights = len(y) / (len(unique_classes) * class_counts)
    
    # Cap maximum weight to prevent extreme values (max 3x, min 0.5x)
    max_weight = 3.0
    min_weight = 0.5
    class_weights = np.clip(raw_weights, min_weight, max_weight)
    
    print(f"\n⚖️ Moderated class weights (capped for stability):")
    for cls, raw_w, mod_w in zip(unique_classes, raw_weights, class_weights):
        print(f"  Class {cls}: {raw_w:.3f} → {mod_w:.3f} (moderated)")
    
    # Handle extreme feature scaling - CRITICAL FIX
    print(f"\n🔧 Feature Scaling Analysis:")
    print(f"  Before scaling - Min: {X.min():.2e}, Max: {X.max():.2e}")
    print(f"  Range: {X.max() - X.min():.2e}")
    
    # Use RobustScaler to handle outliers and extreme values
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  After scaling - Min: {X_scaled.min():.3f}, Max: {X_scaled.max():.3f}")
    print(f"  ✅ Feature scaling completed!")
    
    # Calculate features per day based on actual data
    seq_len = 5
    features_per_day = X.shape[1] // seq_len
    print(f"\n⏰ Sequence length: {seq_len}")
    print(f"🎯 Features per day: {features_per_day}")
    print(f"🎯 Number of classes: {len(unique_classes)}")
    
    # Split data with stratification
    print(f"\n✂️  Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✂️  Split completed:")
    print(f"  🎯 Train: {X_train.shape}")
    print(f"  🎯 Test: {X_test.shape}")
    
    # Reshape for sequences (data is already scaled)
    X_train_reshaped = X_train.reshape(-1, seq_len, features_per_day)
    X_test_reshaped = X_test.reshape(-1, seq_len, features_per_day)
    
    print(f"\n🔄 Reshaped data for sequences:")
    print(f"  🎯 Train: {X_train_reshaped.shape}")
    print(f"  🎯 Test: {X_test_reshaped.shape}")
    print(f"  🎯 Features per day: {features_per_day}")
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, features_per_day, class_weights


class EnhancedCNNResearchTrainer:
    """
    Research trainer for CNN Branch with comprehensive monitoring and analysis
    """
    
    def __init__(self, model, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Enhanced saving system with unique run IDs
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🆔 Research Training Run ID: {self.run_id}")
        
        # Best model state (only saved once at the end)
        self.best_model_state = None
        
        # Enhanced early stopping (3 epochs as requested)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience = 5   # Reduced from 10 → More aggressive early stopping prevents overfitting
        self.patience_counter = 0
        
        # Dynamic Learning Rate Configuration
        # Options: 'adaptive', 'onecycle', 'plateau'
        self.lr_strategy = 'adaptive'  # Adaptive LR that responds to performance changes
        
        # Mixed precision training for better performance
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        # Separate learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler based on configuration
        if self.lr_strategy == 'adaptive':
            # Adaptive LR that responds to performance changes
            self.scheduler = None  # Custom adaptive logic
            self.scheduler_mode = "adaptive"
            self.acc_history = []  # Track accuracy history for adaptive decisions
            self.lr_increase_threshold = 0.005  # Increase LR if improving by >0.5% (more sensitive)
            self.lr_decrease_threshold = -0.01  # Decrease LR if dropping by >1% (less sensitive to small drops)
            self.lr_multiplier_up = 1.5  # Increase by 50% (more aggressive)
            self.lr_multiplier_down = 0.7  # Decrease by 30% (more aggressive)
            self.plateau_epochs = 0  # Track epochs without improvement
            self.plateau_lr_boost = False  # Flag for plateau escape attempts
            self.warmup_epochs = 3  # Number of epochs for LR warmup
            self.current_epoch = 0  # Track current epoch for warmup
            print(f"🧠 Using Adaptive LR: responds to accuracy changes")
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
        self.mixup = MixUp(alpha=0.2)  # Conservative alpha to maintain data integrity
        self.mixup_prob = 0.5  # Apply MixUp to 50% of batches
        
        # Label Smoothing - prevents model from being overconfident
        self.label_smoothing = LabelSmoothing(smoothing=0.1)  # 10% smoothing
        
        # Enhanced loss function with class weights for severe imbalance
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"🎯 Using class weights: {class_weights.cpu().numpy()}")
        self.criterion = FocalLoss(weight=class_weights, gamma=1.5)
        
        print(f"🛡️ Regularization enabled:")
        print(f"  📊 MixUp: α=0.2, prob=50% → Creates diverse training examples")
        print(f"  📋 Label Smoothing: 10% → Prevents overconfidence")
        print(f"  🔇 Input Noise: 1% strength → Improves robustness")
        
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
        
        print(f"⚙️  Enhanced Optimizer Configuration (Plateau-Breaking):")
        print(f"  🧠 TimesNet LR: {self.learning_rate * 0.02:.6f} (0.02x - slightly higher)")
        print(f"  🎯 CNN LR: {self.learning_rate * 0.8:.6f} (0.8x - enhanced learning)")
        print(f"  🎨 Classifier LR: {self.learning_rate * 1.0:.6f} (1.0x - full learning rate)")
        print(f"  📉 Weight Decay: {self.weight_decay * 1.5:.6f} (enhanced regularization)")
        print(f"  🛡️ Gradient Clipping: 0.3 max norm (controlled gradients)")
        
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
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc="🔄 Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 🛡️ ENHANCED REGULARIZATION APPLICATION
            # Input noise augmentation - improves model robustness to small perturbations
            if self.model.training and torch.rand(1) < 0.4:  # Increased to 40% chance
                noise = torch.randn_like(data) * 0.01  # 1% noise strength
                data = data + noise
            
            # MixUp augmentation - creates synthetic training examples between classes
            use_mixup = self.model.training and torch.rand(1) < self.mixup_prob
            if use_mixup:
                data, target_a, target_b, lam = self.mixup(data, target)
            else:
                target_a, target_b, lam = target, None, 1.0
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass for better performance
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    
                    # Enhanced loss calculation with regularization
                    if use_mixup:
                        # MixUp loss: weighted combination of two targets
                        loss_a = self.label_smoothing(output, target_a)
                        loss_b = self.label_smoothing(output, target_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        # Standard loss with label smoothing
                        loss = self.label_smoothing(output, target_a)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Tighter gradient clipping to prevent explosion
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # Reduced from 0.3 → Tighter gradient control
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Step scheduler for OneCycleLR (per batch) - mixed precision path
                if self.scheduler_mode == "onecycle":
                    self.scheduler.step()
                
            else:
                # Standard precision training
                output = self.model(data)
                
                # Enhanced loss calculation with regularization
                if use_mixup:
                    # MixUp loss: weighted combination of two targets
                    loss_a = self.label_smoothing(output, target_a)
                    loss_b = self.label_smoothing(output, target_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    # Standard loss with label smoothing
                    loss = self.label_smoothing(output, target_a)
                
                loss.backward()
                
                # Tighter gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # Reduced from 0.3 → Tighter gradient control
                self.optimizer.step()
            
            # Step scheduler for OneCycleLR (per batch)
            if self.scheduler_mode == "onecycle":
                self.scheduler.step()
            
            # Statistics with MixUp-aware accuracy calculation
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            
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
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for validation
        pbar = tqdm(val_loader, desc="🔍 Validating", leave=False)
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
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
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=self.learning_rate * 3,  # Peak LR 3x higher than base
                total_steps=total_steps,
                pct_start=0.3,  # 30% warmup, 70% cooldown
                cycle_momentum=False,  # For AdamW
                anneal_strategy='cos'  # Cosine annealing
            )
            print(f"🔄 OneCycleLR initialized: {total_steps} total steps, max_lr={self.learning_rate * 3:.2e}")
        
        print(f"🚀 Starting research training for {epochs} epochs...")
        print(f"📊 Training samples: {len(train_loader.dataset):,}")
        print(f"📊 Validation samples: {len(val_loader.dataset):,}")
        print(f"💾 Save directory: {save_dir}")
        print("=" * 60)
        
        # Training progress bar
        epoch_pbar = tqdm(range(epochs), desc="🎯 Training Progress", position=0)
        
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
                    # Enhanced Adaptive LR with plateau detection and escape mechanisms
                    self.acc_history.append(val_acc)
                    self.current_epoch += 1
                    
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
                        # Track plateau epochs (only after warmup)
                        if val_acc <= self.best_val_acc:
                            self.plateau_epochs += 1
                        else:
                            self.plateau_epochs = 0
                            self.plateau_lr_boost = False  # Reset plateau escape flag
                    
                    if len(self.acc_history) >= 2 and self.current_epoch > self.warmup_epochs:
                        acc_change = self.acc_history[-1] - self.acc_history[-2]
                        
                        # Plateau escape mechanism - aggressive LR boost every 5 epochs of no improvement
                        if self.plateau_epochs >= 5 and not self.plateau_lr_boost:
                            new_lr = min(current_lr * 3.0, self.learning_rate * 1.5)  # 3x boost, cap at 1.5x original
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = new_lr
                            self.plateau_lr_boost = True
                            print(f"🚀 PLATEAU ESCAPE: LR boosted to {new_lr:.2e} (plateau for {self.plateau_epochs} epochs)")
                            self.patience_counter = 0
                            
                        elif acc_change > self.lr_increase_threshold:
                            # Accuracy improving significantly -> increase LR
                            new_lr = min(current_lr * self.lr_multiplier_up, self.learning_rate * 2)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = new_lr
                            print(f"🔼 LR increased to {new_lr:.2e} (acc improved by {acc_change:.2f}%)")
                            self.patience_counter = 0
                            
                        elif acc_change < self.lr_decrease_threshold:
                            # Accuracy dropping significantly -> decrease LR 
                            new_lr = max(current_lr * self.lr_multiplier_down, 1e-6)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = new_lr
                            print(f"🔽 LR decreased to {new_lr:.2e} (acc dropped by {acc_change:.2f}%)")
                            self.patience_counter = 0
                            
                        else:
                            # Stable performance
                            print(f"➡️ LR stable at {current_lr:.2e} (acc change: {acc_change:.2f}%, plateau: {self.plateau_epochs})")
                            if val_acc <= self.best_val_acc:
                                self.patience_counter += 1
                                print(f"  ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
                            else:
                                self.patience_counter = 0
                    elif self.current_epoch > self.warmup_epochs:
                        print(f"➡️ LR: {current_lr:.2e} (collecting history, plateau: {self.plateau_epochs})")
                        if val_acc <= self.best_val_acc:
                            self.patience_counter += 1
                        else:
                            self.patience_counter = 0
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
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

                # Early stopping logic based on scheduler type
                if self.scheduler_mode == "plateau":
                    # For ReduceLROnPlateau: stop if LR reduced too many times
                    if self.lr_reduce_count >= self.max_lr_reductions:
                        print("  🛑 Early stopping – LR reduced 4× with no new accuracy peak")
                        break
                elif self.scheduler_mode in ["adaptive", "onecycle"]:
                    # For Adaptive/OneCycleLR: stop if validation accuracy plateaus for too long
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
    
    # 🔧 ENHANCED HYPERPARAMETERS FOR OVERFITTING PREVENTION
    batch_size = 32              # Reduced from 64 → Better generalization with smaller batches
    epochs = 50
    learning_rate = 0.0002       # Reduced from 0.0005 → More conservative learning prevents overfitting
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
    feature_columns = [col for col in df.columns if col not in ['Ticker', 'Label']]
    
    # Remove NaN rows (same as in load_and_preprocess_data)
    nan_rows = df[feature_columns].isna().any(axis=1)
    df_clean = df[~nan_rows].copy() if nan_rows.any() else df.copy()
    
    X_raw = df_clean[feature_columns].values.astype(np.float32)
    y_raw = df_clean['Label'].values.astype(np.int64)
    
    # Split raw data
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )
    
    train_dataset = FinancialDataset(X_train_raw, y_train_raw)
    test_dataset = FinancialDataset(X_test_raw, y_test_raw)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ----- Model Selection -----
    MODEL_TYPE = "optimized_timesnet_hybrid"  # Using optimized version to prevent overfitting
    print(f"\n🤖 Creating model: {MODEL_TYPE} ...")

    if MODEL_TYPE == "optimized_timesnet_hybrid":
        model = create_optimized_timesnet_hybrid(
            features_per_day=features_per_day,
            num_classes=5
        )
    elif MODEL_TYPE == "timesnet_hybrid":
        model = create_timesnet_hybrid(
            features_per_day=features_per_day,
            num_classes=5
        )
    elif MODEL_TYPE == "simple_cnn":
        from src.training.simple_cnn_trainer import SimpleCNN  # Lazy import
        model = SimpleCNN(features_per_day=features_per_day, num_classes=5)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    # Create trainer with class weights for imbalanced data
    trainer = EnhancedCNNResearchTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        save_dir=save_dir
    )
    
    # Plot training curves
    trainer.plot_training_curves(os.path.join(save_dir, 'training_curves.png'))
    
    # Print final results
    print("\n🎉 Research training completed!")
    print(f"🏆 Best validation accuracy: {max(history['val_accuracies']):.2f}%")
    print(f"📉 Best validation loss: {min(history['val_losses']):.4f}")
    print(f"📁 Training history saved to: {save_dir}/training_history.json")
    print(f"💾 Best model saved to: {save_dir}/enhanced_cnn_best.pth")
    print(f"📊 Training curves saved to: {save_dir}/training_curves.png")


if __name__ == "__main__":
    main() 