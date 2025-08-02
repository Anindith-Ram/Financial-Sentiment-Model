"""
Advanced Training script for the candlestick pattern CNN model
Includes all modern ML training improvements and enhanced training modes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
from datetime import datetime
import warnings

# Add logs directory to path for enhanced logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))

# Add src and config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, 
    DATA_OUTPUT_PATH, MODEL_OUTPUT_PATH,
    # Advanced training configs
    EARLY_STOPPING, PATIENCE, MIN_DELTA, OVERFIT_THRESHOLD,
    LR_SCHEDULER, LR_PATIENCE, LR_FACTOR, LR_WARMUP, WARMUP_EPOCHS, WARMUP_START_LR,
    OPTIMIZER, WEIGHT_DECAY, GRADIENT_CLIPPING, MAX_GRAD_NORM,
    LOSS_FUNCTION, FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING,
    MIXED_PRECISION, GRADIENT_ACCUMULATION, ACCUMULATION_STEPS,
    ADVANCED_METRICS, SAVE_CHECKPOINTS, CHECKPOINT_EVERY, VISUALIZATION,
    DROPOUT_RATE, ADD_NOISE, NOISE_STRENGTH, SEQ_LEN
)
from src.models.cnn_model import CandleCNN, EnhancedCandleCNN
from src.models.dataset import CandlestickDataLoader
from src.training.advanced_utils import (
    EarlyStopping, WarmupScheduler, AdvancedMetrics, TrainingVisualizer,
    get_optimizer, get_scheduler, get_loss_function
)

# Import enhanced training components
try:
    from logs.training_logs import create_training_logger
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    print("⚠️  Enhanced logging not available. Install training_logs.py in logs/ directory.")

try:
    from src.training.progressive_trainer import ProgressiveTrainer, run_progressive_training
    PROGRESSIVE_TRAINING_AVAILABLE = True
except ImportError:
    PROGRESSIVE_TRAINING_AVAILABLE = False
    print("⚠️  Progressive training not available. Install progressive_trainer.py in src/training/ directory.")


class EnhancedAdvancedTrainer:
    """
    Enhanced Advanced Trainer with comprehensive logging and multiple training modes
    Optimized for swing trading (1-3 day predictions)
    """
    
    def __init__(self, csv_file=None, model_save_path=None, enable_logging=True, experiment_name=None):
        """
        Initialize the enhanced advanced trainer for swing trading
        
        Args:
            csv_file (str): Path to the dataset CSV file
            model_save_path (str): Path to save the trained model
            enable_logging (bool): Enable comprehensive logging
            experiment_name (str): Name for the experiment
        """
        self.csv_file = csv_file or DATA_OUTPUT_PATH
        self.model_save_path = model_save_path or MODEL_OUTPUT_PATH
        self.device = DEVICE
        self.enable_logging = enable_logging and ENHANCED_LOGGING_AVAILABLE
        self.experiment_name = experiment_name or f"swing_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("🚀 Initializing Swing Trading Training Pipeline...")
        print("=" * 60)
        print("📈 Focus: 1-3 day return predictions")
        print("🎯 Target: Swing trading signals (1-day horizon)")
        print("📊 Features: 35 swing trading optimized indicators")
        
        # Initialize data loaders
        self.data_loader = CandlestickDataLoader(
            csv_file=self.csv_file,
            batch_size=BATCH_SIZE,
            train_split=0.85  # Swing trading uses more training data
        )
        
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        
        # Initialize model optimized for swing trading
        features_per_day = self.data_loader.get_feature_dim()
        print(f"📊 Model Configuration:")
        print(f"   Features per day: {features_per_day}")
        print(f"   Sequence length: {SEQ_LEN} days")
        print(f"   Prediction horizon: 1 day")
        print(f"   Classes: 5 (Strong Sell to Strong Buy)")
        
        self.model = EnhancedCandleCNN(
            features_per_day=features_per_day,
            num_classes=5,
            hidden_size=128,
            use_attention=True
        ).to(self.device)
        
        # Initialize enhanced logging
        if self.enable_logging:
            self.logger = create_training_logger(self.experiment_name)
            print(f"📊 Enhanced logging enabled: {self.experiment_name}")
            
            # Log data quality
            feature_names = self.data_loader.get_feature_names()
            self.logger.log_data_quality(self.train_loader, self.val_loader, feature_names)
        else:
            self.logger = None
        
        # Advanced optimizer
        model_parameters = list(self.model.parameters())
        if not model_parameters:
            raise ValueError("Model has no parameters to optimize!")
            
        self.optimizer = get_optimizer(
            model_parameters, 
            OPTIMIZER, 
            LEARNING_RATE, 
            WEIGHT_DECAY
        )
        
        # Advanced loss function
        class_weights = self.data_loader.get_class_weights().to(self.device)
        self.criterion = get_loss_function(
            LOSS_FUNCTION,
            num_classes=5,
            class_weights=class_weights,
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            smoothing=LABEL_SMOOTHING
        )
        
        # Learning rate scheduler
        scheduler_kwargs = {
            'patience': LR_PATIENCE,
            'factor': LR_FACTOR,
            'T_max': EPOCHS,
            'max_lr': LEARNING_RATE * 10,
            'total_steps': len(self.train_loader) * EPOCHS
        }
        self.scheduler = get_scheduler(self.optimizer, LR_SCHEDULER, **scheduler_kwargs)
        
        # Warmup scheduler
        if LR_WARMUP:
            self.warmup_scheduler = WarmupScheduler(
                self.optimizer, 
                WARMUP_EPOCHS, 
                WARMUP_START_LR, 
                LEARNING_RATE
            )
        else:
            self.warmup_scheduler = None
        
        # Early stopping
        if EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                overfit_threshold=OVERFIT_THRESHOLD
            )
        else:
            self.early_stopping = None
        
        # Advanced metrics
        if ADVANCED_METRICS:
            self.advanced_metrics = AdvancedMetrics()
        else:
            self.advanced_metrics = None
        
        # Training visualization
        if VISUALIZATION:
            self.visualizer = TrainingVisualizer()
        else:
            self.visualizer = None
        
        # Mixed precision training
        if MIXED_PRECISION:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training history
        self.train_losses = []
        self.val_accuracies = []
        self.epoch = 0
        
        # Print training configuration
        self._print_training_config()
    
    def _print_training_config(self):
        """Print training configuration optimized for swing trading"""
        print("\n📊 SWING TRADING TRAINING CONFIGURATION:")
        print("=" * 50)
        print(f"🎯 Prediction Target: 1-day returns")
        print(f"📈 Sequence Length: {SEQ_LEN} days (swing trading lookback)")
        print(f"📊 Features per day: {self.data_loader.get_feature_dim()}")
        print(f"🔄 Batch Size: {BATCH_SIZE}")
        print(f"📚 Learning Rate: {LEARNING_RATE}")
        print(f"⏱️  Epochs: {EPOCHS}")
        print(f"🎯 Early Stopping: {PATIENCE} epochs patience")
        print(f"📊 Train/Val Split: 85/15 (swing trading optimized)")
        print(f"⚡ Optimizer: {OPTIMIZER}")
        print(f"📈 Loss Function: {LOSS_FUNCTION}")
        print(f"🎯 Class Weights: Enabled for imbalanced swing signals")
        print(f"📊 Mixed Precision: {MIXED_PRECISION}")
        print(f"🔄 Gradient Clipping: {MAX_GRAD_NORM}")
        print("=" * 50)
    
    def train_epoch(self):
        """
        Train for one epoch with all enhancements
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Reset advanced metrics
        if self.advanced_metrics:
            self.advanced_metrics.reset()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Add noise if enabled
            if ADD_NOISE:
                noise = torch.randn_like(data) * NOISE_STRENGTH
                data = data + noise
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if MIXED_PRECISION and self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if GRADIENT_CLIPPING:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update advanced metrics
            if self.advanced_metrics:
                self.advanced_metrics.update(output, target)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        Validate the model with enhanced metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Reset advanced metrics
        if self.advanced_metrics:
            self.advanced_metrics.reset()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Mixed precision validation
                if MIXED_PRECISION and self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update advanced metrics
                if self.advanced_metrics:
                    self.advanced_metrics.update(output, target)
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        # Get advanced metrics
        advanced_metrics = None
        if self.advanced_metrics:
            advanced_metrics = self.advanced_metrics.get_metrics()
        
        return accuracy, avg_loss, advanced_metrics
    
    def train(self, epochs=None, mode="standard"):
        """
        Enhanced training with multiple modes
        
        Args:
            epochs (int): Number of epochs to train
            mode (str): Training mode ("standard", "progressive", "enhanced")
        """
        if mode == "progressive" and PROGRESSIVE_TRAINING_AVAILABLE:
            return self._train_progressive()
        elif mode == "enhanced":
            return self._train_enhanced(epochs)
        else:
            return self._train_standard(epochs)
    
    def _train_standard(self, epochs=None):
        """Standard training mode"""
        epochs = epochs or EPOCHS
        
        print(f"\n🎯 Starting Standard Training for {epochs} epochs...")
        print("=" * 60)
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            print(f"\n📈 Epoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Warmup scheduler
            if self.warmup_scheduler and epoch < WARMUP_EPOCHS:
                self.warmup_scheduler.step()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_accuracy, val_loss, advanced_metrics = self.validate()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Enhanced logging
            if self.enable_logging and self.logger:
                train_metrics = {'loss': train_loss, 'accuracy': train_acc}
                val_metrics = {'loss': val_loss, 'accuracy': val_accuracy}
                model_state = {'total_params': sum(p.numel() for p in self.model.parameters())}
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.logger.log_epoch(epoch, train_metrics, val_metrics, model_state, current_lr)
            
            # Print results
            print(f"📊 Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            
            if advanced_metrics:
                print(f"  📈 Advanced Metrics:")
                print(f"    F1 Macro: {advanced_metrics.get('f1_macro', 0):.4f}")
                print(f"    F1 Weighted: {advanced_metrics.get('f1_weighted', 0):.4f}")
                print(f"    Directional Accuracy: {advanced_metrics.get('directional_accuracy', 0):.4f}")
                print(f"    Buy Signal Precision: {advanced_metrics.get('buy_signal_precision', 0):.4f}")
                print(f"    Sell Signal Precision: {advanced_metrics.get('sell_signal_precision', 0):.4f}")
                if 'high_confidence_accuracy' in advanced_metrics:
                    print(f"    High Confidence Accuracy: {advanced_metrics['high_confidence_accuracy']:.4f}")
                    print(f"    High Confidence Samples: {advanced_metrics['high_confidence_samples']}")
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if LR_SCHEDULER == "ReduceLROnPlateau":
                self.scheduler.step(val_accuracy)
            elif LR_SCHEDULER in ["StepLR", "CosineAnnealing"]:
                self.scheduler.step()
            
            # Update visualization
            if self.visualizer:
                self.visualizer.update(train_loss, val_loss, val_accuracy, current_lr)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(f"{self.model_save_path}.best")
                print(f"✅ New best model saved! Accuracy: {best_accuracy:.4f}")
            
            # Save checkpoints
            if SAVE_CHECKPOINTS and (epoch + 1) % CHECKPOINT_EVERY == 0:
                checkpoint_path = f"{self.model_save_path}.checkpoint_epoch_{epoch + 1}"
                self.save_model(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if self.early_stopping:
                self.early_stopping(val_accuracy, train_loss, val_loss)
                if self.early_stopping.early_stop:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    print(f"   Best accuracy: {best_accuracy:.4f}")
                    break
        
        # Generate final report if logging enabled
        if self.enable_logging and self.logger:
            report = self.logger.generate_training_report()
            print("\n" + "="*60)
            print("📊 ENHANCED TRAINING REPORT")
            print("="*60)
            print(report)
        
        print("\n" + "=" * 60)
        print(f"🏁 Training completed!")
        print(f"📊 Final Results:")
        print(f"   Best validation accuracy: {best_accuracy:.4f}")
        print(f"   Total epochs: {epoch + 1}")
        
        # Save final model
        self.save_model()
        
        # Generate final visualization
        if self.visualizer:
            plot_path = f"{self.model_save_path}_training_curves.png"
            self.visualizer.plot(save_path=plot_path)
            print(f"📈 Training curves saved: {plot_path}")
        
        return best_accuracy
    
    def _train_enhanced(self, epochs=None):
        """Enhanced training mode with comprehensive logging"""
        epochs = epochs or EPOCHS
        
        print(f"\n🎯 Starting Enhanced Training for {epochs} epochs...")
        print("=" * 60)
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_accuracy, val_loss, advanced_metrics = self.validate()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Enhanced logging
            if self.enable_logging and self.logger:
                train_metrics = {'loss': train_loss, 'accuracy': train_acc}
                val_metrics = {'loss': val_loss, 'accuracy': val_accuracy}
                model_state = {'total_params': sum(p.numel() for p in self.model.parameters())}
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.logger.log_epoch(epoch, train_metrics, val_metrics, model_state, current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_accuracy:.3f} | "
                  f"LR: {current_lr:.2e}")
            
            # Learning rate scheduling
            if LR_SCHEDULER == "ReduceLROnPlateau":
                self.scheduler.step(val_accuracy)
            elif LR_SCHEDULER in ["StepLR", "CosineAnnealing"]:
                self.scheduler.step()
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(f"{self.model_save_path}.best")
                print(f"   💾 Saved best model (val_acc: {val_accuracy:.4f})")
            
            # Early stopping
            if self.early_stopping:
                self.early_stopping(val_accuracy, train_loss, val_loss)
                if self.early_stopping.early_stop:
                    print(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Generate final report
        if self.enable_logging and self.logger:
            report = self.logger.generate_training_report()
            print("\n" + "="*60)
            print("📊 ENHANCED TRAINING REPORT")
            print("="*60)
            print(report)
        
        print(f"\n✅ Enhanced training completed!")
        print(f"📊 Best validation accuracy: {best_accuracy:.4f}")
        
        return best_accuracy
    
    def _train_progressive(self):
        """Progressive training mode"""
        if not PROGRESSIVE_TRAINING_AVAILABLE:
            print("❌ Progressive training not available. Falling back to standard training.")
            return self._train_standard()
        
        print(f"\n🎯 Starting Progressive Training...")
        print("=" * 60)
        
        # Run progressive training
        results = run_progressive_training()
        
        return results.get('stage2', {}).get('best_val_loss', 0.0)
    
    def save_model(self, path=None):
        """
        Save the model state with enhanced metadata
        
        Args:
            path (str): Path to save the model (uses default if None)
        """
        save_path = path or self.model_save_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save comprehensive model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'features_per_day': self.model.features_per_day,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'epoch': self.epoch,
            'config': {
                'optimizer': OPTIMIZER,
                'loss_function': LOSS_FUNCTION,
                'lr_scheduler': LR_SCHEDULER,
                'early_stopping': EARLY_STOPPING,
                'mixed_precision': MIXED_PRECISION,
                'gradient_clipping': GRADIENT_CLIPPING
            },
            'timestamp': datetime.now().isoformat()
        }, save_path)
        
        print(f"💾 Model saved to: {save_path}")


# Backward compatibility - keep the old Trainer class name
class AdvancedTrainer(EnhancedAdvancedTrainer):
    """
    Backward compatibility wrapper for the enhanced trainer
    """
    def __init__(self, csv_file=None, model_save_path=None):
        super().__init__(csv_file=csv_file, model_save_path=model_save_path, enable_logging=False)

class Trainer(AdvancedTrainer):
    """
    Backward compatibility wrapper for the advanced trainer
    """
    pass


def main():
    """Main training script with enhanced modes"""
    parser = argparse.ArgumentParser(description='Enhanced Training Script')
    parser.add_argument('--mode', choices=['standard', 'enhanced', 'progressive'], 
                       default='standard', help='Training mode')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--csv-file', type=str, default=None, help='Dataset CSV file')
    parser.add_argument('--model-save-path', type=str, default=None, help='Model save path')
    parser.add_argument('--enable-logging', action='store_true', help='Enable enhanced logging')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    print(f"🎯 Enhanced Training Started")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs or EPOCHS}")
    print(f"Enhanced Logging: {args.enable_logging}")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = EnhancedAdvancedTrainer(
            csv_file=args.csv_file,
            model_save_path=args.model_save_path,
            enable_logging=args.enable_logging,
            experiment_name=args.experiment_name
        )
        
        # Train
        best_accuracy = trainer.train(epochs=args.epochs, mode=args.mode)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Check logs/ directory for detailed results")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 