#!/usr/bin/env python3
"""
🚀 SIMPLIFIED HIGH-PERFORMANCE CNN TRAINER
==========================================

Back to basics - pure CNN that should beat 55% accuracy.
No GPT-2 complexity, just optimized CNN with your working features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
import time
from tqdm import tqdm
import sys

warnings.filterwarnings('ignore')
sys.path.append('.')

try:
    from src.models.dataset import FinancialDataset
    print("✅ Professional enhancements loaded successfully!")
except ImportError:
    print("⚠️ Using basic dataset handling")


class HighPerformanceCNN(nn.Module):
    """Optimized CNN for financial prediction - back to what works!"""
    
    def __init__(self, features_per_day: int = 62, num_classes: int = 5):
        super().__init__()
        
        # Multi-scale feature extraction (what made your original CNN work)
        self.conv1 = nn.Conv1d(features_per_day, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        # Pattern detection layers
        self.pattern_conv = nn.Conv1d(512, 256, kernel_size=2)
        
        # Attention for important features
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        
        print(f"🎯 High-Performance CNN created")
        print(f"📊 Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        # x shape: (batch, seq_len, features_per_day)
        x = x.transpose(1, 2)  # (batch, features_per_day, seq_len)
        
        # Multi-scale convolutions
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Pattern detection
        x = torch.relu(self.bn4(self.pattern_conv(x)))
        
        # Attention mechanism
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.transpose(1, 2)  # Back to (batch, features, seq_len)
        
        # Classification
        return self.classifier(x)


class SimpleCNNTrainer:
    """Simplified trainer focused on performance"""
    
    def __init__(self, model, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Create unique run ID for this training session
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🆔 Training Run ID: {self.run_id}")
        
        # Simple, effective optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=learning_rate, weight_decay=weight_decay)
        
        # Class-weighted loss
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"⚖️ Using class weights: {class_weights.cpu().numpy()}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Mixed precision for speed
        self.scaler = GradScaler() if device == "cuda" else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Enhanced model tracking per run
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience = 4
        self.patience_counter = 0
        self.best_model_path = None
        self.best_epoch = 0
        
        print(f"🚀 Simple CNN Trainer initialized on {device}")
        print(f"📈 Target: Beat 55% accuracy!")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="🔄 Training", leave=False)
        
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            current_acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.1f}%'})
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs: int = 50, save_dir: str = "models/simple_cnn"):
        """Train the model"""
        
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
        
        print(f"🚀 Starting simplified CNN training for {epochs} epochs...")
        print(f"🎯 Goal: Beat 55% validation accuracy!")
        print("=" * 60)
        
        try:
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Training
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)
                
                # Store history
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                
                epoch_time = time.time() - epoch_start
                
                # Print progress
                print(f"\n📊 Epoch {epoch+1}/{epochs}:")
                print(f"  🎯 Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
                print(f"  📉 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  ⏱️  Time: {epoch_time:.1f}s")
                
                # Track best model with enhanced saving strategy
                accuracy_improved = val_acc > self.best_val_acc
                loss_improved = val_loss < self.best_val_loss
                
                if accuracy_improved:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch + 1
                    print(f"  🚀 NEW BEST ACCURACY: {val_acc:.2f}% (Epoch {self.best_epoch})!")
                    
                    # Save new best model with run ID - overwrite previous best for this run
                    self.best_model_path = os.path.join(save_dir, f'best_run_{self.run_id}_acc_{val_acc:.2f}.pth')
                    
                    # Remove previous best accuracy model for this run
                    for file in os.listdir(save_dir):
                        if file.startswith(f'best_run_{self.run_id}_acc_') and file != os.path.basename(self.best_model_path):
                            try:
                                os.remove(os.path.join(save_dir, file))
                                print(f"  🗑️ Removed previous best: {file}")
                            except:
                                pass
                    
                    torch.save({
                        'run_id': self.run_id,
                        'epoch': epoch,
                        'best_epoch': self.best_epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_acc': self.best_val_acc,
                        'best_val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'hyperparameters': {
                            'learning_rate': self.learning_rate,
                            'batch_size': len(next(iter(train_loader))[0]),
                            'model_params': sum(p.numel() for p in self.model.parameters())
                        }
                    }, self.best_model_path)
                    print(f"  ✅ Best model saved: {os.path.basename(self.best_model_path)}")
                
                if loss_improved:
                    self.best_val_loss = val_loss
                    print(f"  📉 New best loss: {val_loss:.4f}")
                
                if accuracy_improved or loss_improved:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"  ⚠️  No improvement for {self.patience_counter}/{self.patience} epochs")
                    
                    if self.patience_counter >= self.patience:
                        print(f"  🛑 Early stopping! Best accuracy: {self.best_val_acc:.2f}%")
                        break
                
                # Show progress toward goal
                if val_acc > 55:
                    print(f"  🎉 GOAL ACHIEVED! {val_acc:.2f}% > 55%")
                else:
                    print(f"  📈 Progress: {val_acc:.2f}% / 55% target")
        
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted!")
        
        finally:
            # Generate final report
            total_time = time.time() - start_time
            
            print(f"\n🎉 Training completed!")
            print(f"🏆 Best accuracy: {self.best_val_acc:.2f}%")
            print(f"📉 Best loss: {self.best_val_loss:.4f}")
            print(f"⏱️  Total time: {total_time/3600:.2f} hours")
            
            # Save training curves
            self._plot_curves(save_dir)
            
            # Save final model
            if self.best_model_path:
                final_path = os.path.join(save_dir, 'simple_cnn_best.pth')
                if os.path.exists(self.best_model_path):
                    import shutil
                    shutil.copy2(self.best_model_path, final_path)
                    print(f"💾 Final model: {final_path}")
            
            return {
                'best_val_acc': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'epochs_completed': len(self.train_losses),
                'total_time_hours': total_time/3600,
                'best_model_path': self.best_model_path
            }
    
    def _plot_curves(self, save_dir):
        """Plot training curves"""
        
        try:
            plt.figure(figsize=(12, 4))
            epochs = list(range(1, len(self.train_losses) + 1))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.train_accuracies, label='Train', color='blue', linewidth=2)
            plt.plot(epochs, self.val_accuracies, label='Validation', color='red', linewidth=2)
            plt.axhline(y=55, color='green', linestyle='--', alpha=0.7, label='Target (55%)')
            plt.title('📈 Model Accuracy', fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.train_losses, label='Train', color='blue', linewidth=2)
            plt.plot(epochs, self.val_losses, label='Validation', color='red', linewidth=2)
            plt.title('📉 Model Loss', fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            curves_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Curves saved: {curves_path}")
            
        except Exception as e:
            print(f"⚠️  Could not save curves: {e}")


def load_and_preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load and preprocess data for simplified training"""
    
    print(f"📊 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean data
    feature_columns = [col for col in df.columns if col not in ['Ticker', 'Label']]
    nan_rows = df[feature_columns].isna().any(axis=1)
    if nan_rows.any():
        nan_count = nan_rows.sum()
        print(f"🧹 Removing {nan_count:,} NaN rows ({nan_count/len(df)*100:.1f}%)")
        df_clean = df[~nan_rows].copy()
    else:
        df_clean = df.copy()
    
    X = df_clean[feature_columns].values.astype(np.float32)
    y = df_clean['Label'].values.astype(np.int64)
    
    # Class analysis
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n🎯 Class Distribution:")
    for cls, count in zip(unique_classes, class_counts):
        percentage = count / len(y) * 100
        print(f"  Class {cls}: {count:,} samples ({percentage:.1f}%)")
    
    # Moderate class weights
    raw_weights = len(y) / (len(unique_classes) * class_counts)
    class_weights = np.clip(raw_weights, 0.5, 2.5)  # More conservative weights
    
    print(f"\n⚖️ Class weights:")
    for cls, weight in zip(unique_classes, class_weights):
        print(f"  Class {cls}: {weight:.2f}")
    
    # Scale features
    print(f"\n🔧 Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Range: {X_scaled.min():.2f} to {X_scaled.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Reshape for sequences
    seq_len = 5
    features_per_day = X.shape[1] // seq_len
    
    X_train_seq = X_train.reshape(-1, seq_len, features_per_day)
    X_test_seq = X_test.reshape(-1, seq_len, features_per_day)
    
    print(f"📦 Final shapes:")
    print(f"  Train: {X_train_seq.shape}")
    print(f"  Test: {X_test_seq.shape}")
    print(f"  Features per day: {features_per_day}")
    
    return X_train_seq, X_test_seq, y_train, y_test, features_per_day, class_weights


def main():
    """Main training function"""
    print("🚀 SIMPLIFIED HIGH-PERFORMANCE CNN")
    print("Goal: Beat 55% accuracy without GPT-2 complexity!")
    print("=" * 60)
    
    # Configuration
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data" / "reduced_feature_set_dataset.csv"
    save_dir = project_root / "models" / "simple_cnn"
    
    # Load data
    X_train, X_test, y_train, y_test, features_per_day, class_weights = load_and_preprocess_data(csv_path)
    
    # Create datasets
    train_dataset = FinancialDataset(X_train.reshape(X_train.shape[0], -1), y_train)
    test_dataset = FinancialDataset(X_test.reshape(X_test.shape[0], -1), y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create model
    model = HighPerformanceCNN(features_per_day=features_per_day, num_classes=5)
    
    # Create trainer
    trainer = SimpleCNNTrainer(model, learning_rate=0.001, class_weights=class_weights)
    
    # Train
    results = trainer.train(train_loader, test_loader, epochs=30, save_dir=save_dir)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  🏆 Best accuracy: {results['best_val_acc']:.2f}%")
    print(f"  📉 Best loss: {results['best_val_loss']:.4f}")
    print(f"  {'🎉 SUCCESS!' if results['best_val_acc'] > 55 else '📈 Keep improving!'}")


if __name__ == "__main__":
    main()