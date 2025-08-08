"""
Progressive Training Implementation
Two-stage training approach for better performance with larger datasets
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG
from src.training.advanced_utils import (
    get_optimizer, get_loss_function, get_scheduler,
    EarlyStopping, AdvancedMetrics, TrainingVisualizer
)
from src.data.data_collection import build_dataset
from src.models.dataset import CandlestickDataLoader


class ProgressiveTrainer:
    """
    Progressive trainer that implements two-stage training:
    Stage 1: Train on 300 high-quality tickers
    Stage 2: Fine-tune on 400 tickers with lower learning rate
    """
    
    def __init__(self, model, device='cuda', experiment_name=None):
        """
        Initialize progressive trainer
        
        Args:
            model: CNN model
            device: Training device
            experiment_name: Name for this training experiment
        """
        self.model = model
        self.device = device
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Progressive training settings
        self.stage1_epochs = TRAINING_CONFIG.get('STAGE_1_EPOCHS', 25)
        self.stage2_epochs = TRAINING_CONFIG.get('STAGE_2_EPOCHS', 20)
        self.stage3_epochs = TRAINING_CONFIG.get('STAGE_3_EPOCHS', 15)
        self.stage4_epochs = TRAINING_CONFIG.get('STAGE_4_EPOCHS', 10)
        self.stage1_tickers = TRAINING_CONFIG.get('STAGE_1_TICKERS', 300)
        self.stage2_tickers = TRAINING_CONFIG.get('STAGE_2_TICKERS', 400)
        self.stage3_tickers = TRAINING_CONFIG.get('STAGE_3_TICKERS', 500)
        self.stage4_tickers = TRAINING_CONFIG.get('STAGE_4_TICKERS', 500)
        self.stage1_lr = TRAINING_CONFIG.get('STAGE_1_LR', 2e-4)
        self.stage2_lr = TRAINING_CONFIG.get('STAGE_2_LR', 1e-4)
        self.stage3_lr = TRAINING_CONFIG.get('STAGE_3_LR', 5e-5)
        self.stage4_lr = TRAINING_CONFIG.get('STAGE_4_LR', 1e-5)
        self.stage1_batch_size = TRAINING_CONFIG.get('STAGE_1_BATCH_SIZE', 256)
        self.stage2_batch_size = TRAINING_CONFIG.get('STAGE_2_BATCH_SIZE', 64)
        self.stage3_batch_size = TRAINING_CONFIG.get('STAGE_3_BATCH_SIZE', 32)
        self.stage4_batch_size = TRAINING_CONFIG.get('STAGE_4_BATCH_SIZE', 16)
        
        # Training logs
        self.stage1_logs = []
        self.stage2_logs = []
        self.stage3_logs = []
        self.stage4_logs = []
        self.combined_logs = []
        
        # Cache for data loaders to prevent multiple instances
        self._data_loaders_cache = {}
        
        print(f"Progressive Trainer initialized for experiment: {self.experiment_name}")
        print(f"  Stage 1: {self.stage1_tickers} tickers, {self.stage1_epochs} epochs, LR={self.stage1_lr}")
        print(f"  Stage 2: {self.stage2_tickers} tickers, {self.stage2_epochs} epochs, LR={self.stage2_lr}")
        print(f"  Stage 3: {self.stage3_tickers} tickers, {self.stage3_epochs} epochs, LR={self.stage3_lr}")
        print(f"  Stage 4: {self.stage4_tickers} tickers, {self.stage4_epochs} epochs, LR={self.stage4_lr}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Train for one epoch with enhanced regularization
        
        Args:
            train_loader: Training data loader
            
        Returns:
            tuple: (train_loss, train_acc, metrics)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Enhanced regularization: Add noise to data
            if TRAINING_CONFIG.get('ADD_NOISE', False):
                noise_strength = TRAINING_CONFIG.get('NOISE_STRENGTH', 0.02)
                noise = torch.randn_like(data) * noise_strength
                data = data + noise
            
            # Enhanced regularization: Mixup augmentation
            if TRAINING_CONFIG.get('MIXUP_AUGMENTATION', False):
                mixup_alpha = TRAINING_CONFIG.get('MIXUP_ALPHA', 0.3)
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    batch_size = data.size(0)
                    index = torch.randperm(batch_size).to(self.device)
                    data = lam * data + (1 - lam) * data[index, :]
                    target_a, target_b = target, target[index]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            
            # Enhanced loss calculation
            if TRAINING_CONFIG.get('MIXUP_AUGMENTATION', False) and mixup_alpha > 0:
                loss = lam * self.criterion(logits, target_a) + (1 - lam) * self.criterion(logits, target_b)
            else:
                loss = self.criterion(logits, target)
            
            # Backward pass with gradient clipping
            loss.backward()
            
            # Enhanced regularization: Gradient clipping
            if TRAINING_CONFIG.get('GRADIENT_CLIPPING', False):
                max_grad_norm = TRAINING_CONFIG.get('MAX_GRAD_NORM', 0.5)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Calculate accuracy
            pred = logits.argmax(dim=1, keepdim=True)
            if TRAINING_CONFIG.get('MIXUP_AUGMENTATION', False) and mixup_alpha > 0:
                # For mixup, use original target for accuracy
                correct += pred.eq(target_a.view_as(pred)).sum().item()
            else:
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            total += target.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.1f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (val_loss, val_acc, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for validation
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                logits = output[0] if isinstance(output, (list, tuple)) else output
                loss = self.criterion(logits, target)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, {}
    
    def prepare_stage1_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 1 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 1
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 1 DATA (using existing dataset)")
        print("="*60)
        
        return self._get_cached_data_loaders("Stage 1", self.stage1_batch_size)
    
    def prepare_stage2_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 2 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 2
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 2 DATA (using existing dataset)")
        print("="*60)
        
        return self._get_cached_data_loaders("Stage 2", self.stage2_batch_size)
    
    def prepare_stage3_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 3 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 3
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 3 DATA (using existing dataset)")
        print("="*60)
        
        return self._get_cached_data_loaders("Stage 3", self.stage3_batch_size)
    
    def prepare_stage4_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 4 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 4
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 4 DATA (using existing dataset)")
        print("="*60)
        
        return self._get_cached_data_loaders("Stage 4", self.stage4_batch_size)
    
    def train_stage1(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train stage 1: 300 tickers with higher learning rate
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            dict: Stage 1 training results
        """
        print("\n" + "="*60)
        print("🚀 STAGE 1 TRAINING (300 tickers)")
        print("="*60)
        
        # Initialize stage 1 components
        self.optimizer = get_optimizer(
            self.model.parameters(),
            TRAINING_CONFIG['OPTIMIZER'],
            lr=self.stage1_lr,
            weight_decay=TRAINING_CONFIG['WEIGHT_DECAY']
        )
        
        self.criterion = get_loss_function(
            TRAINING_CONFIG['LOSS_FUNCTION'],
            num_classes=5,
            device=self.device,
            class_weights=TRAINING_CONFIG.get('CLASS_WEIGHTS', None)
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            TRAINING_CONFIG['LR_SCHEDULER'],
            epochs=self.stage1_epochs,
            **{k: v for k, v in TRAINING_CONFIG.items() if k.startswith('LR_')}
        )
        
        # Initialize monitoring
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG['PATIENCE'],
            min_delta=TRAINING_CONFIG['MIN_DELTA']
        )
        
        # Enhanced overfitting detection
        self.overfit_threshold = TRAINING_CONFIG.get('OVERFIT_THRESHOLD', 3)
        self.train_val_gaps = []
        self.best_train_val_gap = float('inf')
        
        self.advanced_metrics = AdvancedMetrics()
        self.visualizer = TrainingVisualizer()
        
        # Training loop
        best_val_loss = float('inf')
        stage1_logs = []
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.stage1_epochs), desc="Stage 1 Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Train epoch
            train_loss, train_acc, train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(val_loss)
            
            # Log epoch
            epoch_log = {
                'stage': 1,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            
            stage1_logs.append(epoch_log)
            self.combined_logs.append(epoch_log)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Acc': f'{train_acc*100:.1f}%',
                'Val Acc': f'{val_acc*100:.1f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Enhanced overfitting detection
            train_val_gap = abs(train_acc - val_acc)
            self.train_val_gaps.append(train_val_gap)
            
            # Check for overfitting
            if train_val_gap > self.overfit_threshold:
                epoch_pbar.write(f"   ⚠️  Overfitting detected! Train/Val gap: {train_val_gap:.2f}% > {self.overfit_threshold}%")
                if len(self.train_val_gaps) >= 3 and all(gap > self.overfit_threshold for gap in self.train_val_gaps[-3:]):
                    epoch_pbar.write(f"   🛑 Stopping due to persistent overfitting")
                    break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'stage': 1,
                    'features_per_day': self.model.features_per_day if hasattr(self.model, 'features_per_day') else 340
                }, f'models/{self.experiment_name}/stage1_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 1 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 1 logs
        self.stage1_logs = stage1_logs
        with open(f'logs/{self.experiment_name}/stage1_logs.json', 'w') as f:
            json.dump(stage1_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 1 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/{self.experiment_name}/stage1_logs.json")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': len(stage1_logs),
            'logs': stage1_logs
        }
    
    def train_stage2(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train stage 2: 400 tickers with lower learning rate
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            dict: Stage 2 training results
        """
        print("\n" + "="*60)
        print("🚀 STAGE 2 TRAINING (400 tickers)")
        print("="*60)
        
        # Load best model from stage 1
        if os.path.exists(f'models/{self.experiment_name}/stage1_best.pth'):
            checkpoint = torch.load(f'models/{self.experiment_name}/stage1_best.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded stage 1 best model (val_loss: {checkpoint['val_loss']:.4f})")
        else:
            print("⚠️  No stage 1 model found, starting fresh")
        
        # Initialize stage 2 components with lower learning rate
        self.optimizer = get_optimizer(
            self.model.parameters(),
            TRAINING_CONFIG['OPTIMIZER'],
            lr=self.stage2_lr,
            weight_decay=TRAINING_CONFIG['WEIGHT_DECAY']
        )
        
        self.criterion = get_loss_function(
            TRAINING_CONFIG['LOSS_FUNCTION'],
            num_classes=5,
            device=self.device,
            class_weights=TRAINING_CONFIG.get('CLASS_WEIGHTS', None)
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            TRAINING_CONFIG['LR_SCHEDULER'],
            epochs=self.stage2_epochs,
            **{k: v for k, v in TRAINING_CONFIG.items() if k.startswith('LR_')}
        )
        
        # Reset early stopping for stage 2
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG['PATIENCE'],
            min_delta=TRAINING_CONFIG['MIN_DELTA']
        )
        
        # Training loop
        best_val_loss = float('inf')
        stage2_logs = []
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.stage2_epochs), desc="Stage 2 Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Train epoch
            train_loss, train_acc, train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(val_loss)
            
            # Log epoch
            epoch_log = {
                'stage': 2,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            
            stage2_logs.append(epoch_log)
            self.combined_logs.append(epoch_log)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Acc': f'{train_acc*100:.1f}%',
                'Val Acc': f'{val_acc*100:.1f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'stage': 2,
                    'features_per_day': self.model.features_per_day if hasattr(self.model, 'features_per_day') else 340
                }, f'models/{self.experiment_name}/stage2_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 2 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 2 logs
        self.stage2_logs = stage2_logs
        with open(f'logs/{self.experiment_name}/stage2_logs.json', 'w') as f:
            json.dump(stage2_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 2 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/{self.experiment_name}/stage2_logs.json")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': len(stage2_logs),
            'logs': stage2_logs
        }
    
    def train_stage3(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train stage 3: Advanced optimization with very low learning rate
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            dict: Stage 3 training results
        """
        print("\n" + "="*60)
        print("🚀 STAGE 3 TRAINING (Advanced Optimization)")
        print("="*60)
        
        # Load best model from stage 2
        if os.path.exists(f'models/{self.experiment_name}/stage2_best.pth'):
            checkpoint = torch.load(f'models/{self.experiment_name}/stage2_best.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded stage 2 best model (val_loss: {checkpoint['val_loss']:.4f})")
        else:
            print("⚠️  No stage 2 model found, starting fresh")
        
        # Initialize stage 3 components with very low learning rate
        self.optimizer = get_optimizer(
            self.model.parameters(),
            TRAINING_CONFIG['OPTIMIZER'],
            lr=self.stage3_lr,
            weight_decay=TRAINING_CONFIG['WEIGHT_DECAY']
        )
        
        self.criterion = get_loss_function(
            TRAINING_CONFIG['LOSS_FUNCTION'],
            num_classes=5,
            device=self.device,
            class_weights=TRAINING_CONFIG.get('CLASS_WEIGHTS', None)
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            TRAINING_CONFIG['LR_SCHEDULER'],
            epochs=self.stage3_epochs,
            **{k: v for k, v in TRAINING_CONFIG.items() if k.startswith('LR_')}
        )
        
        # Reset early stopping for stage 3
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG['PATIENCE'],
            min_delta=TRAINING_CONFIG['MIN_DELTA']
        )
        
        # Training loop
        best_val_loss = float('inf')
        stage3_logs = []
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.stage3_epochs), desc="Stage 3 Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Train epoch
            train_loss, train_acc, train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(val_loss)
            
            # Log epoch
            epoch_log = {
                'stage': 3,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            
            stage3_logs.append(epoch_log)
            self.combined_logs.append(epoch_log)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Acc': f'{train_acc*100:.1f}%',
                'Val Acc': f'{val_acc*100:.1f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'stage': 3,
                    'features_per_day': self.model.features_per_day if hasattr(self.model, 'features_per_day') else 340
                }, f'models/{self.experiment_name}/stage3_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 3 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 3 logs
        self.stage3_logs = stage3_logs
        with open(f'logs/{self.experiment_name}/stage3_logs.json', 'w') as f:
            json.dump(stage3_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 3 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/{self.experiment_name}/stage3_logs.json")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': len(stage3_logs),
            'logs': stage3_logs
        }
    
    def train_stage4(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train stage 4: Final precision tuning for 1-day predictions
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            dict: Stage 4 training results
        """
        print("\n" + "="*60)
        print("🚀 STAGE 4 TRAINING (Final Precision Tuning)")
        print("="*60)
        
        # Load best model from stage 3
        if os.path.exists(f'models/{self.experiment_name}/stage3_best.pth'):
            checkpoint = torch.load(f'models/{self.experiment_name}/stage3_best.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded stage 3 best model (val_loss: {checkpoint['val_loss']:.4f})")
        else:
            print("⚠️  No stage 3 model found, starting fresh")
        
        # Initialize stage 4 components with minimal learning rate
        self.optimizer = get_optimizer(
            self.model.parameters(),
            TRAINING_CONFIG['OPTIMIZER'],
            lr=self.stage4_lr,
            weight_decay=TRAINING_CONFIG['WEIGHT_DECAY']
        )
        
        self.criterion = get_loss_function(
            TRAINING_CONFIG['LOSS_FUNCTION'],
            num_classes=5,
            device=self.device,
            class_weights=TRAINING_CONFIG.get('CLASS_WEIGHTS', None)
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            TRAINING_CONFIG['LR_SCHEDULER'],
            epochs=self.stage4_epochs,
            **{k: v for k, v in TRAINING_CONFIG.items() if k.startswith('LR_')}
        )
        
        # Reset early stopping for stage 4
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG['PATIENCE'],
            min_delta=TRAINING_CONFIG['MIN_DELTA']
        )
        
        # Training loop
        best_val_loss = float('inf')
        stage4_logs = []
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.stage4_epochs), desc="Stage 4 Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Train epoch
            train_loss, train_acc, train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(val_loss)
            
            # Log epoch
            epoch_log = {
                'stage': 4,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            
            stage4_logs.append(epoch_log)
            self.combined_logs.append(epoch_log)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Acc': f'{train_acc*100:.1f}%',
                'Val Acc': f'{val_acc*100:.1f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'stage': 4,
                    'features_per_day': self.model.features_per_day if hasattr(self.model, 'features_per_day') else 340
                }, f'models/{self.experiment_name}/stage4_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 4 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 4 logs
        self.stage4_logs = stage4_logs
        with open(f'logs/{self.experiment_name}/stage4_logs.json', 'w') as f:
            json.dump(stage4_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 4 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/{self.experiment_name}/stage4_logs.json")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': len(stage4_logs),
            'logs': stage4_logs
        }
    
    def train_progressive(self) -> Dict:
        """
        Run complete progressive training
        
        Returns:
            dict: Complete training results
        """
        print("\n" + "="*80)
        print("🎯 PROGRESSIVE TRAINING STARTED")
        print("="*80)
        
        # Create experiment-specific directories
        os.makedirs(f'logs/{self.experiment_name}', exist_ok=True)
        os.makedirs(f'models/{self.experiment_name}', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        start_time = datetime.now()
        
        # Overall progress bar for stages
        stages = [
            ("Stage 1: Foundation", self.prepare_stage1_data, self.train_stage1),
            ("Stage 2: Refinement", self.prepare_stage2_data, self.train_stage2),
            ("Stage 3: Advanced", self.prepare_stage3_data, self.train_stage3),
            ("Stage 4: Precision", self.prepare_stage4_data, self.train_stage4)
        ]
        
        stage_results = {}
        stage_pbar = tqdm(stages, desc="Progressive Training", unit="stage")
        
        for stage_name, prepare_func, train_func in stage_pbar:
            stage_pbar.set_description(f"Running {stage_name}")
            
            # Prepare data for this stage
            train_loader, val_loader = prepare_func()
            
            # Train this stage
            if stage_name.startswith("Stage 1"):
                stage_results['stage1'] = train_func(train_loader, val_loader)
            elif stage_name.startswith("Stage 2"):
                stage_results['stage2'] = train_func(train_loader, val_loader)
            elif stage_name.startswith("Stage 3"):
                stage_results['stage3'] = train_func(train_loader, val_loader)
            elif stage_name.startswith("Stage 4"):
                stage_results['stage4'] = train_func(train_loader, val_loader)
            
            # Update stage progress
            stage_pbar.set_postfix({
                'Current Stage': stage_name.split(':')[0],
                'Best Val Loss': f"{stage_results.get(f'stage{stage_name.split()[1][0]}', {}).get('best_val_loss', 0):.4f}"
            })
        
        # Combine results
        total_time = datetime.now() - start_time
        
        # Save final results
        final_results = {
            'total_time': total_time,
            'stages': stage_results,
            'final_model_path': f'models/{self.experiment_name}/stage4_best.pth',
            'experiment_name': self.experiment_name
        }
        
        # Generate training curves
        self.generate_training_curves(stage_results)
        
        # Save results to JSON
        results_path = f'logs/{self.experiment_name}/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n📊 Training results saved to: {results_path}")
        print(f"🎯 Final model saved to: {final_results['final_model_path']}")
        
        return final_results
    
    def _print_progressive_summary(self, results: Dict):
        """Print progressive training summary"""
        print("\n" + "="*80)
        print("📊 PROGRESSIVE TRAINING SUMMARY")
        print("="*80)
        
        stage1 = results['stage1']
        stage2 = results['stage2']
        stage3 = results['stage3']
        stage4 = results['stage4']
        
        print(f"Stage 1 Results (Foundation):")
        print(f"   Epochs trained: {stage1['epochs_trained']}")
        print(f"   Best validation loss: {stage1['best_val_loss']:.4f}")
        
        print(f"\nStage 2 Results (Refinement):")
        print(f"   Epochs trained: {stage2['epochs_trained']}")
        print(f"   Best validation loss: {stage2['best_val_loss']:.4f}")
        
        print(f"\nStage 3 Results (Advanced):")
        print(f"   Epochs trained: {stage3['epochs_trained']}")
        print(f"   Best validation loss: {stage3['best_val_loss']:.4f}")
        
        print(f"\nStage 4 Results (Precision):")
        print(f"   Epochs trained: {stage4['epochs_trained']}")
        print(f"   Best validation loss: {stage4['best_val_loss']:.4f}")
        
        print(f"\nOverall Results:")
        print(f"   Total training time: {results['total_training_time']}")
        print(f"   Total improvement: {stage1['best_val_loss'] - stage4['best_val_loss']:.4f}")
        
        if stage4['best_val_loss'] < stage1['best_val_loss']:
            print(f"   ✅ 4-Stage Progressive training successful!")
        else:
            print(f"   ⚠️  No overall improvement")
        
        print(f"\n📁 Files saved:")
        print(f"   - models/{self.experiment_name}/stage1_best.pth")
        print(f"   - models/{self.experiment_name}/stage2_best.pth")
        print(f"   - models/{self.experiment_name}/stage3_best.pth")
        print(f"   - models/{self.experiment_name}/stage4_best.pth")
        print(f"   - logs/{self.experiment_name}/stage1_logs.json")
        print(f"   - logs/{self.experiment_name}/stage2_logs.json")
        print(f"   - logs/{self.experiment_name}/stage3_logs.json")
        print(f"   - logs/{self.experiment_name}/stage4_logs.json")
        print(f"   - logs/{self.experiment_name}/progressive_training_results.json")

    def _get_cached_data_loaders(self, stage_name, batch_size):
        """
        Get cached data loaders to prevent multiple dataset instances
        
        Args:
            stage_name (str): Name of the stage
            batch_size (int): Batch size for this stage
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        cache_key = f"{stage_name}_{batch_size}"
        
        if cache_key not in self._data_loaders_cache:
            print(f"Creating new data loaders for {stage_name} with batch size {batch_size}")
            
            # Clear memory before creating new loaders
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Create data loaders
            from config.config import DATA_OUTPUT_PATH
            from src.models.dataset import CandlestickDataLoader
            
            data_loader = CandlestickDataLoader(
                csv_file=DATA_OUTPUT_PATH,
                batch_size=batch_size,
                train_split=0.85
            )
            
            train_loader = data_loader.get_train_loader()
            val_loader = data_loader.get_val_loader()
            
            # Cache the loaders
            self._data_loaders_cache[cache_key] = (train_loader, val_loader)
            
            print(f"✅ {stage_name} data prepared:")
            print(f"   Training samples: {len(train_loader.dataset)}")
            print(f"   Validation samples: {len(val_loader.dataset)}")
            print(f"   Batch size: {batch_size}")
        else:
            print(f"Using cached data loaders for {stage_name}")
        
        return self._data_loaders_cache[cache_key]

    def generate_training_curves(self, stage_results: Dict) -> None:
        """
        Generate comprehensive training curves for all stages
        
        Args:
            stage_results: Dictionary containing results from all stages
        """
        print("\n📊 Generating Training Curves...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Progressive Training Curves - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Colors for different stages
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        
        for stage_idx, (stage_name, stage_data) in enumerate(stage_results.items()):
            if stage_data is None or 'history' not in stage_data:
                continue
                
            history = stage_data['history']
            color = colors[stage_idx]
            
            # Plot 1: Loss Curves
            axes[0, 0].plot(history['train_loss'], label=f'{stage_name} Train', 
                           color=color, linestyle='-', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label=f'{stage_name} Val', 
                           color=color, linestyle='--', linewidth=2)
            
            # Plot 2: Accuracy Curves
            axes[0, 1].plot(history['train_acc'], label=f'{stage_name} Train', 
                           color=color, linestyle='-', linewidth=2)
            axes[0, 1].plot(history['val_acc'], label=f'{stage_name} Val', 
                           color=color, linestyle='--', linewidth=2)
            
            # Plot 3: Learning Rate
            if 'lr' in history:
                axes[1, 0].plot(history['lr'], label=stage_name, 
                               color=color, linewidth=2)
            
            # Plot 4: Train/Val Gap (Overfitting Detection)
            train_val_gap = [abs(t - v) for t, v in zip(history['train_acc'], history['val_acc'])]
            axes[1, 1].plot(train_val_gap, label=stage_name, 
                           color=color, linewidth=2)
        
        # Customize plots
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Train/Val Gap (Overfitting Detection)', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add threshold line for overfitting
        threshold = TRAINING_CONFIG.get('OVERFIT_THRESHOLD', 3)
        axes[1, 1].axhline(y=threshold, color='red', linestyle='--', 
                           alpha=0.7, label=f'Threshold ({threshold}%)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = f'logs/{self.experiment_name}/training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to: {plot_path}")
        
        # Also save as interactive HTML
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create interactive plot
            fig_interactive = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Loss Curves', 'Accuracy Curves', 
                              'Learning Rate Schedule', 'Train/Val Gap'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for stage_idx, (stage_name, stage_data) in enumerate(stage_results.items()):
                if stage_data is None or 'history' not in stage_data:
                    continue
                    
                history = stage_data['history']
                color = colors[stage_idx]
                
                # Add traces for each subplot
                fig_interactive.add_trace(
                    go.Scatter(y=history['train_loss'], name=f'{stage_name} Train Loss',
                             line=dict(color=color, width=2)), row=1, col=1)
                fig_interactive.add_trace(
                    go.Scatter(y=history['val_loss'], name=f'{stage_name} Val Loss',
                             line=dict(color=color, width=2, dash='dash')), row=1, col=1)
                
                fig_interactive.add_trace(
                    go.Scatter(y=history['train_acc'], name=f'{stage_name} Train Acc',
                             line=dict(color=color, width=2)), row=1, col=2)
                fig_interactive.add_trace(
                    go.Scatter(y=history['val_acc'], name=f'{stage_name} Val Acc',
                             line=dict(color=color, width=2, dash='dash')), row=1, col=2)
                
                if 'lr' in history:
                    fig_interactive.add_trace(
                        go.Scatter(y=history['lr'], name=f'{stage_name} LR',
                                 line=dict(color=color, width=2)), row=2, col=1)
                
                train_val_gap = [abs(t - v) for t, v in zip(history['train_acc'], history['val_acc'])]
                fig_interactive.add_trace(
                    go.Scatter(y=train_val_gap, name=f'{stage_name} Gap',
                             line=dict(color=color, width=2)), row=2, col=2)
            
            # Update layout
            fig_interactive.update_layout(
                title=f'Progressive Training Curves - {self.experiment_name}',
                height=800,
                showlegend=True
            )
            
            # Save interactive plot
            html_path = f'logs/{self.experiment_name}/training_curves_interactive.html'
            fig_interactive.write_html(html_path)
            print(f"📊 Interactive training curves saved to: {html_path}")
            
        except ImportError:
            print("⚠️  Plotly not available, skipping interactive plot generation")
        
        plt.close()


def run_progressive_training(experiment_name=None):
    """Run progressive training with enhanced logging"""
    from src.models.cnn_model import CandleCNN
    from config.config import MODEL_CONFIG
    
    # Initialize model
    model = CandleCNN(
        features_per_day=MODEL_CONFIG['FEATURES_PER_DAY'],
        num_classes=MODEL_CONFIG['NUM_CLASSES'],
        hidden_size=MODEL_CONFIG['HIDDEN_SIZE']
    )
    
    # Move model to device
    device = MODEL_CONFIG['DEVICE']
    model = model.to(device)
    
    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Initialize progressive trainer with experiment name
    trainer = ProgressiveTrainer(model, device=device, experiment_name=experiment_name)
    
    # Verify trainer model is on correct device
    print(f"Trainer model device: {next(trainer.model.parameters()).device}")
    
    # Run progressive training
    results = trainer.train_progressive()
    
    return results


if __name__ == "__main__":
    # Run progressive training
    results = run_progressive_training()
    print("Progressive training completed!") 