"""
Enhanced CNN Production Training Script
=====================================

Integrated training script for production deployment and automated pipelines.
This script provides streamlined training with essential monitoring only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
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
from src.models.timesnet_hybrid import create_timesnet_hybrid

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


class EnhancedCNNProductionTrainer:
    """
    Production trainer for Enhanced CNN with streamlined monitoring
    """
    
    def __init__(self, model, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Separate learning rates for different components
        self.optimizer = self._create_optimizer()
        # Cosine scheduler with 10% warm-up
        self.total_epochs = None
        self.scheduler = None
        self.cosine_scheduler = None
        
        # Loss function
        self.criterion = FocalLoss(gamma=1.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping on accuracy
        self.best_val_acc = 0.0
        self.patience = 3
        self.patience_counter = 0
        
        print(f"🤖 Enhanced CNN Production Trainer initialized on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Component analysis
        self._analyze_components()
    
    def _analyze_components(self):
        """Analyze model components for monitoring"""
        gpt2_params = sum(p.numel() for name, p in self.model.named_parameters() if 'timesnet' in name)
        cnn_params = sum(p.numel() for name, p in self.model.named_parameters() if 'enhanced_cnn' in name)
        classifier_params = sum(p.numel() for name, p in self.model.named_parameters() if 'classifier' in name)
        
        print(f"🔍 Model Component Analysis:")
        print(f"  🧠 TimesNet Encoder: {gpt2_params:,} parameters")
        print(f"  🎯 Enhanced CNN: {cnn_params:,} parameters")
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
        
        # Different learning rates for different components
        optimizer = optim.AdamW([
            {'params': gpt2_params, 'lr': self.learning_rate * 0.1},  # Lower LR for TimesNet
            {'params': cnn_params, 'lr': self.learning_rate * 1.5},    # Higher LR for CNN
            {'params': classifier_params, 'lr': self.learning_rate}     # Standard LR for classifier
        ], weight_decay=self.weight_decay)
        
        print(f"⚙️  Optimizer Configuration:")
        print(f"  🧠 TimesNet LR: {self.learning_rate * 0.1:.6f} (0.1x)")
        print(f"  🎯 CNN LR: {self.learning_rate * 1.5:.6f} (1.5x)")
        print(f"  🎨 Classifier LR: {self.learning_rate:.6f} (1.0x)")
        print(f"  📉 Weight Decay: {self.weight_decay}")
        
        return optimizer
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch with streamlined monitoring"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc="🔄 Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            loss = self.criterion(logits, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                self.cosine_scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_loss = loss.item()
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
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
                logits = output[0] if isinstance(output, (list, tuple)) else output
                loss = self.criterion(logits, target)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
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
              epochs: int = 50, save_dir: str = "models/production") -> dict:
        """Train the enhanced model with early stopping and checkpointing"""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🚀 Starting production training for {epochs} epochs...")
        # Setup cosine scheduler with warm-up
        self.total_epochs = epochs
        steps_per_epoch = len(train_loader)
        warmup_steps = max(1, int(0.1 * epochs * steps_per_epoch))
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        cosine = CosineAnnealingLR(self.optimizer, T_max=epochs*steps_per_epoch)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))
        self.cosine_scheduler = cosine
        print(f"📊 Training samples: {len(train_loader.dataset):,}")
        print(f"📊 Validation samples: {len(val_loader.dataset):,}")
        print(f"💾 Save directory: {save_dir}")
        print("=" * 50)
        
        # Training progress bar
        epoch_pbar = tqdm(range(epochs), desc="🎯 Training Progress", position=0)
        
        start_time = time.time()
        
        for epoch in epoch_pbar:
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Update epoch progress bar (streamlined)
            epoch_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{epochs}',
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_acc:.2f}%',
                'Time': f'{epoch_time:.1f}s'
            })
            
            # Print streamlined progress
            print(f"📊 Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
                    # Early stopping based on accuracy only
        if val_acc > getattr(self, 'best_val_acc', 0):
                self.best_val_acc = val_acc
                self.patience_counter = 0

                best_model_path = os.path.join(save_dir, 'best_acc_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, best_model_path)
                print(f"  ✅ Saved new best-accuracy model: {val_acc:.2f}%")
        else:
                self.patience_counter += 1
                print(f"  ⚠️  No improvement for {self.patience_counter} epochs")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'enhanced_cnn_checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"  💾 Saved checkpoint")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'epochs_trained': len(self.train_losses),
            'total_training_time': total_time,
            'average_epoch_time': total_time / len(self.train_losses)
        }
        
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n🎉 Production training completed!")
        print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        print(f"📊 Average epoch time: {total_time/len(self.train_losses):.1f}s")
        print(f"📁 History saved to {history_path}")
        
        return history


def load_data_and_create_loaders(csv_path: str = "data/reduced_feature_set_dataset.csv", 
                                batch_size: int = 64, test_size: float = 0.2):
    """Load data and create data loaders for production training"""
    
    print(f"📊 Loading data from {csv_path}...")
    
    # Load dataset
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
    
    # Create dataset
    dataset = FinancialDataset(X, y)
    
    # Rolling-window split (70:15:15) no shuffling
    print("✂️  Rolling-window split 70:15:15 (train:val:test)…")
    total_len = len(dataset)
    train_end = int(0.70 * total_len)
    val_end = int(0.85 * total_len)
    indices = list(range(total_len))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(val_dataset):,}")
    
    return train_loader, val_loader


def run_production_training(csv_path: str = "data/reduced_feature_set_dataset.csv",
                          batch_size: int = 64,
                          epochs: int = 50,
                          learning_rate: float = 0.001,
                          weight_decay: float = 1e-4,
                          save_dir: str = "models/production",
                          fusion_method: str = "attention") -> tuple:
    """
    Run production training for enhanced CNN
    
    Args:
        csv_path: Path to the dataset CSV file
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        save_dir: Directory to save models and checkpoints
        fusion_method: Fusion method ("attention", "concatenation", "weighted")
    
    Returns:
        tuple: (trainer, history)
    """
    
    print("🚀 Enhanced CNN Production Training")
    print("=" * 40)
    
    # Load data
    train_loader, val_loader = load_data_and_create_loaders(csv_path, batch_size)
    
    # ----- Model Selection -----
    MODEL_TYPE = "timesnet_hybrid"  # Change if needed
    print(f"🤖 Creating model: {MODEL_TYPE} ...")

    # Calculate features per day and num_classes from dataset
    df_temp = pd.read_csv(csv_path)
    label_col = 'Label' if 'Label' in df_temp.columns else ('Label_3' if 'Label_3' in df_temp.columns else None)
    if label_col is None:
        raise ValueError("Dataset must contain 'Label' or 'Label_3' column for supervised training")
    feature_cols_temp = [col for col in df_temp.columns if col not in ['Ticker', label_col]]
    features_per_day = len(feature_cols_temp) // 5
    num_classes = int(df_temp[label_col].nunique())
    print(f"📊 Auto-detected features per day: {features_per_day}")
    print(f"🎯 Auto-detected num classes: {num_classes}")

    if MODEL_TYPE == "timesnet_hybrid":
        model = create_timesnet_hybrid(features_per_day=features_per_day, num_classes=num_classes)
    elif MODEL_TYPE == "simple_cnn":
        from src.training.simple_cnn_trainer import SimpleCNN
        model = SimpleCNN(features_per_day=features_per_day, num_classes=5)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    # Create trainer
    trainer = EnhancedCNNProductionTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir
    )
    
    # Print final results
    print("\n🎉 Production training completed!")
    if isinstance(history, dict) and history.get('val_accuracies') and history.get('val_losses'):
        print(f"🏆 Best validation accuracy: {max(history['val_accuracies']):.2f}%")
        print(f"📉 Best validation loss: {min(history['val_losses']):.4f}")
    else:
        print("⚠️  No validation metrics available (training ended before first epoch)")
    print(f"📁 Training history saved to: {save_dir}/training_history.json")
    print(f"💾 Best model saved to: {save_dir}/enhanced_cnn_best.pth")
    
    return trainer, history


def main():
    """Main production training function"""
    
    # Configuration - Use project-level paths
    project_root = Path(__file__).parent.parent.parent  # Go up to price model/
    csv_path = project_root / "data" / "latest_dataset.csv"
    
    # Fallback to existing datasets if latest doesn't exist
    if not csv_path.exists():
        csv_path = project_root / "data" / "reduced_feature_set_dataset.csv"
    
    batch_size = 64
    epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-4
    save_dir = project_root / "models" / "production"  # Save to project-level models/
    fusion_method = "attention"
    
    print(f"⚙️  Configuration:")
    print(f"  📁 Dataset: {csv_path}")
    print(f"  📦 Batch size: {batch_size}")
    print(f"  🔄 Epochs: {epochs}")
    print(f"  📈 Learning rate: {learning_rate}")
    print(f"  📉 Weight decay: {weight_decay}")
    print(f"  💾 Save directory: {save_dir}")
    print(f"  🔗 Fusion method: {fusion_method}")
    print("=" * 40)
    
    # Run training
    trainer, history = run_production_training(
        csv_path=csv_path,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=save_dir,
        fusion_method=fusion_method
    )


if __name__ == "__main__":
    main() 