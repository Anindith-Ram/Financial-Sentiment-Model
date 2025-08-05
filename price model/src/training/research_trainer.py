#!/usr/bin/env python3
"""
Enhanced CNN Research Training Script
====================================

Standalone script for research, development, and detailed analysis of the enhanced CNN model.
This script provides comprehensive monitoring, visualization, and debugging capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

from src.models.advanced_time_series_integration import create_gpt2_enhanced_cnn
from src.models.dataset import FinancialDataset

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
    
    # Calculate features per day based on actual data
    seq_len = 5
    features_per_day = X.shape[1] // seq_len
    print(f"⏰ Sequence length: {seq_len}")
    print(f"🎯 Features per day: {features_per_day}")
    print(f"🎯 Number of classes: {len(np.unique(y))}")
    
    # Split data
    print(f"✂️  Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    print(f"📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for sequences
    X_train_reshaped = X_train_scaled.reshape(-1, seq_len, features_per_day)
    X_test_reshaped = X_test_scaled.reshape(-1, seq_len, features_per_day)
    
    print(f"🔄 Reshaped data:")
    print(f"  🎯 Train: {X_train_reshaped.shape}")
    print(f"  🎯 Test: {X_test_reshaped.shape}")
    print(f"  🎯 Features per day: {features_per_day}")
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, features_per_day


class EnhancedCNNResearchTrainer:
    """
    Research trainer for Enhanced CNN with comprehensive monitoring and analysis
    """
    
    def __init__(self, model, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Separate learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        print(f"🤖 Enhanced CNN Research Trainer initialized on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Component analysis
        self._analyze_components()
    
    def _analyze_components(self):
        """Analyze model components for monitoring"""
        gpt2_params = sum(p.numel() for name, p in self.model.named_parameters() if 'gpt2_extractor' in name)
        cnn_params = sum(p.numel() for name, p in self.model.named_parameters() if 'enhanced_cnn' in name)
        classifier_params = sum(p.numel() for name, p in self.model.named_parameters() if 'classifier' in name)
        
        print(f"🔍 Model Component Analysis:")
        print(f"  🧠 GPT-2 Extractor: {gpt2_params:,} parameters")
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
            if 'gpt2_extractor' in name:
                gpt2_params.append(param)
            elif 'enhanced_cnn' in name:
                cnn_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                # Default group for other parameters
                classifier_params.append(param)
        
        # Different learning rates for different components
        optimizer = optim.AdamW([
            {'params': gpt2_params, 'lr': self.learning_rate * 0.1},  # Lower LR for GPT-2 (frozen)
            {'params': cnn_params, 'lr': self.learning_rate * 1.5},    # Higher LR for CNN (pattern learning)
            {'params': classifier_params, 'lr': self.learning_rate}     # Standard LR for classifier
        ], weight_decay=self.weight_decay)
        
        print(f"⚙️  Optimizer Configuration:")
        print(f"  🧠 GPT-2 LR: {self.learning_rate * 0.1:.6f} (0.1x)")
        print(f"  🎯 CNN LR: {self.learning_rate * 1.5:.6f} (1.5x)")
        print(f"  🎨 Classifier LR: {self.learning_rate:.6f} (1.0x)")
        print(f"  📉 Weight Decay: {self.weight_decay}")
        
        return optimizer
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch with enhanced monitoring"""
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
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
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
        """Train the enhanced model with early stopping and checkpointing"""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🚀 Starting research training for {epochs} epochs...")
        print(f"📊 Training samples: {len(train_loader.dataset):,}")
        print(f"📊 Validation samples: {len(val_loader.dataset):,}")
        print(f"💾 Save directory: {save_dir}")
        print("=" * 60)
        
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
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(save_dir, 'enhanced_cnn_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, best_model_path)
                print(f"  ✅ Saved best model to {best_model_path}")
                
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
                print(f"  💾 Saved checkpoint to {checkpoint_path}")
            
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
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
            'total_training_time': total_time,
            'average_epoch_time': total_time / len(self.train_losses)
        }
        
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n🎉 Research training completed!")
        print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        print(f"📊 Average epoch time: {total_time/len(self.train_losses):.1f}s")
        print(f"📁 History saved to {history_path}")
        
        return history
    
    def plot_training_curves(self, save_path: str = "models/research/training_curves.png"):
        """Plot training curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', color='blue', linewidth=2)
        plt.plot(self.val_accuracies, label='Val Accuracy', color='red', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined view
        plt.subplot(1, 3, 3)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        line2 = ax1.plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        line3 = ax2.plot(self.train_accuracies, label='Train Acc', color='green', linewidth=2, linestyle='--')
        line4 = ax2.plot(self.val_accuracies, label='Val Acc', color='orange', linewidth=2, linestyle='--')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='black')
        ax2.set_ylabel('Accuracy (%)', color='black')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        plt.title('Combined Training Metrics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training curves saved to {save_path}")


def main():
    """Main research training function"""
    print("🔬 Enhanced CNN Research Training Script")
    print("=" * 60)
    
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
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, features_per_day = load_and_preprocess_data(
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
    
    # Create enhanced model
    print(f"\n🤖 Creating enhanced GPT-2 CNN model...")
    model = create_gpt2_enhanced_cnn(
        features_per_day=features_per_day,
        hidden_size=768,
        num_classes=5,
        fusion_method="attention"
    )
    
    # Create trainer
    trainer = EnhancedCNNResearchTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
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