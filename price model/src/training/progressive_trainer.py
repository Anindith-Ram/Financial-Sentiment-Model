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
        
        # Progressive training settings
        self.stage1_epochs = TRAINING_CONFIG.get('STAGE_1_EPOCHS', 15)
        self.stage2_epochs = TRAINING_CONFIG.get('STAGE_2_EPOCHS', 20)
        self.stage1_tickers = TRAINING_CONFIG.get('STAGE_1_TICKERS', 300)
        self.stage2_tickers = TRAINING_CONFIG.get('STAGE_2_TICKERS', 400)
        self.stage1_lr = TRAINING_CONFIG.get('STAGE_1_LR', 1e-3)
        self.stage2_lr = TRAINING_CONFIG.get('STAGE_2_LR', 5e-4)
        self.stage1_batch_size = TRAINING_CONFIG.get('STAGE_1_BATCH_SIZE', 256)
        self.stage2_batch_size = TRAINING_CONFIG.get('STAGE_2_BATCH_SIZE', 128)
        
        # Training logs
        self.stage1_logs = []
        self.stage2_logs = []
        self.combined_logs = []
        
        print(f"Progressive Trainer initialized:")
        print(f"  Stage 1: {self.stage1_tickers} tickers, {self.stage1_epochs} epochs, LR={self.stage1_lr}")
        print(f"  Stage 2: {self.stage2_tickers} tickers, {self.stage2_epochs} epochs, LR={self.stage2_lr}")
    
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
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, {}
    
    def prepare_stage1_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 1 (300 tickers)
        
        Returns:
            tuple: (train_loader, val_loader) for stage 1
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 1 DATA (300 tickers)")
        print("="*60)
        
        # Temporarily set N_TICKERS to 300
        original_n_tickers = DATA_CONFIG['N_TICKERS']
        DATA_CONFIG['N_TICKERS'] = self.stage1_tickers
        
        try:
            # Build dataset with 300 tickers
            dataset_path = build_dataset()
            
            # Create data loaders
            data_loader = CandlestickDataLoader(
                csv_file=dataset_path,
                batch_size=self.stage1_batch_size,
                train_split=0.8
            )
            
            train_loader, val_loader = data_loader.get_loaders()
            
            print(f"✅ Stage 1 data prepared:")
            print(f"   Training samples: {len(train_loader.dataset)}")
            print(f"   Validation samples: {len(val_loader.dataset)}")
            print(f"   Batch size: {self.stage1_batch_size}")
            
            return train_loader, val_loader
            
        finally:
            # Restore original N_TICKERS
            DATA_CONFIG['N_TICKERS'] = original_n_tickers
    
    def prepare_stage2_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for stage 2 (400 tickers)
        
        Returns:
            tuple: (train_loader, val_loader) for stage 2
        """
        print("\n" + "="*60)
        print("📊 PREPARING STAGE 2 DATA (400 tickers)")
        print("="*60)
        
        # Build dataset with 400 tickers
        dataset_path = build_dataset()
        
        # Create data loaders
        data_loader = CandlestickDataLoader(
            csv_file=dataset_path,
            batch_size=self.stage2_batch_size,
            train_split=0.8
        )
        
        train_loader, val_loader = data_loader.get_loaders()
        
        print(f"✅ Stage 2 data prepared:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Batch size: {self.stage2_batch_size}")
        
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
        
        for epoch in range(self.stage1_epochs):
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
            
            # Print progress
            print(f"Stage 1 Epoch {epoch+1:2d}/{self.stage1_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
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
                print(f"   💾 Saved stage 1 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
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
            checkpoint = torch.load('models/stage1_best.pth')
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
        
        for epoch in range(self.stage2_epochs):
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
            
            # Print progress
            print(f"Stage 2 Epoch {epoch+1:2d}/{self.stage2_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
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
                print(f"   💾 Saved stage 2 best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
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
        
        # Stage 1: Train on 300 tickers
        train_loader_1, val_loader_1 = self.prepare_stage1_data()
        stage1_results = self.train_stage1(train_loader_1, val_loader_1)
        
        # Stage 2: Fine-tune on 400 tickers
        train_loader_2, val_loader_2 = self.prepare_stage2_data()
        stage2_results = self.train_stage2(train_loader_2, val_loader_2)
        
        # Combine results
        total_time = datetime.now() - start_time
        
        combined_results = {
            'stage1': stage1_results,
            'stage2': stage2_results,
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
        
        print(f"Stage 1 Results:")
        print(f"   Epochs trained: {stage1['epochs_trained']}")
        print(f"   Best validation loss: {stage1['best_val_loss']:.4f}")
        
        print(f"\nStage 2 Results:")
        print(f"   Epochs trained: {stage2['epochs_trained']}")
        print(f"   Best validation loss: {stage2['best_val_loss']:.4f}")
        
        print(f"\nOverall Results:")
        print(f"   Total training time: {results['total_training_time']}")
        print(f"   Improvement: {stage1['best_val_loss'] - stage2['best_val_loss']:.4f}")
        
        if stage2['best_val_loss'] < stage1['best_val_loss']:
            print(f"   ✅ Progressive training successful!")
        else:
            print(f"   ⚠️  No improvement in stage 2")
        
        print(f"\n📁 Files saved:")
        print(f"   - models/stage1_best.pth")
        print(f"   - models/stage2_best.pth")
        print(f"   - logs/stage1_logs.json")
        print(f"   - logs/stage2_logs.json")
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
    
    # Initialize progressive trainer
    trainer = ProgressiveTrainer(model, device=MODEL_CONFIG['DEVICE'])
    
    # Run progressive training
    results = trainer.train_progressive()
    
    return results


if __name__ == "__main__":
    # Run progressive training
    results = run_progressive_training()
    print("Progressive training completed!") 