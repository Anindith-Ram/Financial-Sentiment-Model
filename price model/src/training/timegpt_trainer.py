"""
Time Series Enhanced CNN Trainer
Handles transfer learning from time series models to financial forecasting
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
from typing import Dict, List, Tuple, Optional
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.timegpt_integration import create_time_series_enhanced_cnn
from src.models.dataset import FinancialDataset
from config.config import MODEL_CONFIG, TRAINING_CONFIG


class TimeSeriesTrainer:
    """
    Trainer for time series-enhanced CNN with transfer learning capabilities
    """
    
    def __init__(self, model, device: str = 'cuda', 
                 experiment_name: str = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4):
        self.model = model
        self.device = device
        self.experiment_name = experiment_name or f"time_series_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize optimizer with different learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Create experiment directory
        self.experiment_dir = f"models/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"🚀 Time Series Trainer initialized")
        print(f"📁 Experiment directory: {self.experiment_dir}")
        print(f"🔧 Device: {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for different components"""
        
        # Separate parameters by component
        time_series_params = []
        cnn_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'time_series_extractor' in name:
                time_series_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                cnn_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': time_series_params, 'lr': self.learning_rate * 0.1},  # Lower LR for time series (frozen)
            {'params': cnn_params, 'lr': self.learning_rate},  # Normal LR for CNN
            {'params': classifier_params, 'lr': self.learning_rate * 2.0}  # Higher LR for classifier
        ]
        
        return optim.AdamW(param_groups, weight_decay=self.weight_decay)
    
    def load_data(self, csv_path: str, test_size: float = 0.2, 
                  val_size: float = 0.1, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare data for training
        
        Args:
            csv_path: Path to the CSV dataset
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        print(f"📊 Loading data from: {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📊 Dataset shape: {df.shape}")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['Label', 'Date', 'Ticker']]
        X = df[feature_columns].values
        y = df['Label'].values
        
        print(f"📊 Features: {len(feature_columns)}")
        print(f"📊 Samples: {len(X)}")
        print(f"📊 Label distribution: {np.bincount(y)}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Train samples: {len(X_train)}")
        print(f"📊 Validation samples: {len(X_val)}")
        print(f"📊 Test samples: {len(X_test)}")
        
        # Create datasets
        train_dataset = FinancialDataset(X_train, y_train, seq_len=5)
        val_dataset = FinancialDataset(X_val, y_val, seq_len=5)
        test_dataset = FinancialDataset(X_test, y_test, seq_len=5)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f"📊 Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
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
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, patience: int = 10) -> Dict:
        """
        Train the time series-enhanced model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"🚀 Starting time series-enhanced training for {epochs} epochs")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f"📊 Epoch {epoch+1}/{epochs}:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(f"{self.experiment_dir}/best_model.pth")
                print(f"💾 Saved best model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping after {epoch+1} epochs")
                    break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, path: str):
        """Save model to path"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.experiment_dir}/checkpoint_epoch_{epoch+1}.pth"
        self.save_model(checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to: {self.experiment_dir}/training_curves.png")


def run_time_series_training(csv_path: str = "data/reduced_feature_set_dataset.csv",
                           features_per_day: int = 71,
                           num_classes: int = 5,
                           hidden_size: int = 128,
                           epochs: int = 50,
                           batch_size: int = 32,
                           experiment_name: str = None):
    """
    Run time series-enhanced training
    
    Args:
        csv_path: Path to the dataset CSV
        features_per_day: Number of features per day
        num_classes: Number of output classes
        hidden_size: Hidden layer size
        epochs: Number of training epochs
        batch_size: Batch size for training
        experiment_name: Name for the experiment
    """
    
    # Create model
    print("🔧 Creating time series-enhanced CNN...")
    model = create_time_series_enhanced_cnn(
        features_per_day=features_per_day,
        num_classes=num_classes,
        hidden_size=hidden_size,
        use_attention=True,
        fusion_method="attention"
    )
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = TimeSeriesTrainer(
        model=model,
        device=device,
        experiment_name=experiment_name,
        learning_rate=1e-4,
        weight_decay=1e-4
    )
    
    # Load data
    train_loader, val_loader, test_loader = trainer.load_data(
        csv_path=csv_path,
        batch_size=batch_size
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Test on test set
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"🎯 Final Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    return trainer, history


# Backward compatibility
def run_timegpt_training(*args, **kwargs):
    """Backward compatibility function"""
    return run_time_series_training(*args, **kwargs)


if __name__ == "__main__":
    # Run time series-enhanced training
    trainer, history = run_time_series_training(
        csv_path="data/reduced_feature_set_dataset.csv",
        features_per_day=71,
        num_classes=5,
        hidden_size=128,
        epochs=50,
        batch_size=32,
        experiment_name="time_series_enhanced_v1"
    ) 