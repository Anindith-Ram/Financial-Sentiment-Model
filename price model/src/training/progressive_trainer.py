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

from config.config import TRAINING_CONFIG, DATA_CONFIG
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
    
    def __init__(self, model, device='cuda'):
        """
        Initialize progressive trainer
        
        Args:
            model: CNN model
            device: Training device
        """
        self.model = model
        self.device = device
        
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
        self.stage2_batch_size = TRAINING_CONFIG.get('STAGE_2_BATCH_SIZE', 128)
        self.stage3_batch_size = TRAINING_CONFIG.get('STAGE_3_BATCH_SIZE', 64)
        self.stage4_batch_size = TRAINING_CONFIG.get('STAGE_4_BATCH_SIZE', 32)
        
        # Training logs
        self.stage1_logs = []
        self.stage2_logs = []
        self.stage3_logs = []
        self.stage4_logs = []
        self.combined_logs = []
        
        print(f"Progressive Trainer initialized:")
        print(f"  Stage 1: {self.stage1_tickers} tickers, {self.stage1_epochs} epochs, LR={self.stage1_lr}")
        print(f"  Stage 2: {self.stage2_tickers} tickers, {self.stage2_epochs} epochs, LR={self.stage2_lr}")
        print(f"  Stage 3: {self.stage3_tickers} tickers, {self.stage3_epochs} epochs, LR={self.stage3_lr}")
        print(f"  Stage 4: {self.stage4_tickers} tickers, {self.stage4_epochs} epochs, LR={self.stage4_lr}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Train for one epoch
        
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
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, {}
    
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
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                current_loss = total_loss / (batch_idx + 1)
                current_acc = correct / total
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.3f}'
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
        
        # Use existing dataset from config
        from config.config import DATA_OUTPUT_PATH
        dataset_path = DATA_OUTPUT_PATH
        
        print(f"[INFO] Using existing dataset: {dataset_path}")
        
        # Progress bar for data loading
        with tqdm(total=1, desc="Loading Stage 1 Data", leave=False) as pbar:
            # Create data loaders
            data_loader = CandlestickDataLoader(
                csv_file=dataset_path,
                batch_size=self.stage1_batch_size,
                train_split=0.8
            )
            
            train_loader, val_loader = data_loader.get_loaders()
            pbar.update(1)
        
        print(f"✅ Stage 1 data prepared:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Batch size: {self.stage1_batch_size}")
        
        return train_loader, val_loader
    
    def prepare_stage2_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 2 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 2
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 2 DATA (using existing dataset)")
        print("="*60)
        
        # Use existing dataset from config
        from config.config import DATA_OUTPUT_PATH
        dataset_path = DATA_OUTPUT_PATH
        
        print(f"[INFO] Using existing dataset: {dataset_path}")
        
        # Progress bar for data loading
        with tqdm(total=1, desc="Loading Stage 2 Data", leave=False) as pbar:
            # Create data loaders
            data_loader = CandlestickDataLoader(
                csv_file=dataset_path,
                batch_size=self.stage2_batch_size,
                train_split=0.8
            )
            
            train_loader, val_loader = data_loader.get_loaders()
            pbar.update(1)
        
        print(f"✅ Stage 2 data prepared:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Batch size: {self.stage2_batch_size}")
        
        return train_loader, val_loader
    
    def prepare_stage3_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 3 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 3
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 3 DATA (using existing dataset)")
        print("="*60)
        
        # Use existing dataset from config
        from config.config import DATA_OUTPUT_PATH
        dataset_path = DATA_OUTPUT_PATH
        
        print(f"[INFO] Using existing dataset: {dataset_path}")
        
        # Progress bar for data loading
        with tqdm(total=1, desc="Loading Stage 3 Data", leave=False) as pbar:
            # Create data loaders
            data_loader = CandlestickDataLoader(
                csv_file=dataset_path,
                batch_size=self.stage3_batch_size,
                train_split=0.8
            )
            
            train_loader, val_loader = data_loader.get_loaders()
            pbar.update(1)
        
        print(f"✅ Stage 3 data prepared:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Batch size: {self.stage3_batch_size}")
        
        return train_loader, val_loader
    
    def prepare_stage4_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 4 using existing dataset
        
        Returns:
            tuple: (train_loader, val_loader) for stage 4
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 4 DATA (using existing dataset)")
        print("="*60)
        
        # Use existing dataset from config
        from config.config import DATA_OUTPUT_PATH
        dataset_path = DATA_OUTPUT_PATH
        
        print(f"[INFO] Using existing dataset: {dataset_path}")
        
        # Progress bar for data loading
        with tqdm(total=1, desc="Loading Stage 4 Data", leave=False) as pbar:
            # Create data loaders
            data_loader = CandlestickDataLoader(
                csv_file=dataset_path,
                batch_size=self.stage4_batch_size,
                train_split=0.8
            )
            
            train_loader, val_loader = data_loader.get_loaders()
            pbar.update(1)
        
        print(f"✅ Stage 4 data prepared:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Batch size: {self.stage4_batch_size}")
        
        return train_loader, val_loader
    
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
            device=self.device
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
                'Train Acc': f'{train_acc:.3f}',
                'Val Acc': f'{val_acc:.3f}',
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
                    'stage': 1
                }, 'models/stage1_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 1 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 1 logs
        self.stage1_logs = stage1_logs
        with open('logs/stage1_logs.json', 'w') as f:
            json.dump(stage1_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 1 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/stage1_logs.json")
        
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
        if os.path.exists('models/stage1_best.pth'):
            checkpoint = torch.load('models/stage1_best.pth', map_location=self.device)
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
            device=self.device
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
                'Train Acc': f'{train_acc:.3f}',
                'Val Acc': f'{val_acc:.3f}',
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
                    'stage': 2
                }, 'models/stage2_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 2 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 2 logs
        self.stage2_logs = stage2_logs
        with open('logs/stage2_logs.json', 'w') as f:
            json.dump(stage2_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 2 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/stage2_logs.json")
        
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
        if os.path.exists('models/stage2_best.pth'):
            checkpoint = torch.load('models/stage2_best.pth', map_location=self.device)
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
            device=self.device
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
                'Train Acc': f'{train_acc:.3f}',
                'Val Acc': f'{val_acc:.3f}',
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
                    'stage': 3
                }, 'models/stage3_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 3 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 3 logs
        self.stage3_logs = stage3_logs
        with open('logs/stage3_logs.json', 'w') as f:
            json.dump(stage3_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 3 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/stage3_logs.json")
        
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
        if os.path.exists('models/stage3_best.pth'):
            checkpoint = torch.load('models/stage3_best.pth', map_location=self.device)
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
            device=self.device
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
                'Train Acc': f'{train_acc:.3f}',
                'Val Acc': f'{val_acc:.3f}',
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
                    'stage': 4
                }, 'models/stage4_best.pth')
                epoch_pbar.write(f"   💾 Saved stage 4 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                epoch_pbar.write(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save stage 4 logs
        self.stage4_logs = stage4_logs
        with open('logs/stage4_logs.json', 'w') as f:
            json.dump(stage4_logs, f, indent=2, default=str)
        
        print(f"\n✅ Stage 4 completed:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training logs saved to: logs/stage4_logs.json")
        
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
        
        # Create logs directory
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
        
        combined_results = {
            'stage1': stage_results['stage1'],
            'stage2': stage_results['stage2'],
            'stage3': stage_results['stage3'],
            'stage4': stage_results['stage4'],
            'total_training_time': str(total_time),
            'combined_logs': self.combined_logs
        }
        
        # Save combined results
        with open('logs/progressive_training_results.json', 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Generate summary
        self._print_progressive_summary(combined_results)
        
        return combined_results
    
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
        print(f"   - models/stage1_best.pth")
        print(f"   - models/stage2_best.pth")
        print(f"   - models/stage3_best.pth")
        print(f"   - models/stage4_best.pth")
        print(f"   - logs/stage1_logs.json")
        print(f"   - logs/stage2_logs.json")
        print(f"   - logs/stage3_logs.json")
        print(f"   - logs/stage4_logs.json")
        print(f"   - logs/progressive_training_results.json")


def run_progressive_training():
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
    
    # Initialize progressive trainer
    trainer = ProgressiveTrainer(model, device=device)
    
    # Run progressive training
    results = trainer.train_progressive()
    
    return results


if __name__ == "__main__":
    # Run progressive training
    results = run_progressive_training()
    print("Progressive training completed!") 