#!/usr/bin/env python3
"""
Advanced Learning Rate Scheduler with Performance Sensitivity
============================================================
Implements sophisticated LR adaptation based on accuracy patterns and
performance tracking over multiple epochs. Designed to be more sensitive
to optimal LR ranges like the observed 2.80e-04 sweet spot.
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class PerformanceSensitiveLRScheduler:
    """
    Learning rate scheduler that tracks performance across multiple epochs
    and adapts more sensitively to accuracy improvements.
    """
    
    def __init__(self, optimizer, 
                 base_lr: float = 2e-4,
                 memory_length: int = 5,
                 sensitivity_threshold: float = 0.001,  # 0.1% accuracy change
                 lr_search_range: Tuple[float, float] = (1e-5, 1e-3),
                 patience_for_search: int = 3,
                 warmup_epochs: int = 5,  # NEW: Learning rate warmup
                 warmup_strategy: str = 'cosine',  # 'linear', 'cosine', or 'exponential'
                 # 🚀 ENHANCED FEATURES FROM PROVEN ADAPTIVE SCHEDULER
                 lr_increase_threshold: float = 0.005,  # 0.5% improvement threshold
                 lr_decrease_threshold: float = -0.01,  # 1% drop threshold  
                 lr_multiplier_up: float = 1.5,  # 50% increase multiplier
                 lr_multiplier_down: float = 0.7,  # 30% decrease multiplier
                 plateau_escape_epochs: int = 5,  # Epochs before plateau escape
                 plateau_escape_multiplier: float = 3.0,  # 3x LR boost
                 plateau_escape_cap: float = 1.5,  # Cap at 1.5x original LR
                 verbose: bool = True):
        """
        Args:
            optimizer: PyTorch optimizer
            base_lr: Starting learning rate (based on your 2.80e-04 observation)
            memory_length: How many epochs to track for LR performance analysis
            sensitivity_threshold: Minimum accuracy change to consider significant
            lr_search_range: (min_lr, max_lr) for automatic LR range testing
            patience_for_search: Epochs without improvement before triggering LR search
            warmup_epochs: Number of epochs for LR warmup (0 to disable)
            warmup_strategy: Warmup curve type ('linear', 'cosine', 'exponential')
            lr_increase_threshold: Accuracy improvement threshold for LR increase (0.5%)
            lr_decrease_threshold: Accuracy drop threshold for LR decrease (-1%)
            lr_multiplier_up: Multiplier for LR increases (1.5x)
            lr_multiplier_down: Multiplier for LR decreases (0.7x)
            plateau_escape_epochs: Epochs without improvement before plateau escape (5)
            plateau_escape_multiplier: LR boost multiplier for plateau escape (3x)
            plateau_escape_cap: Maximum LR relative to base_lr for plateau escape (1.5x)
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.memory_length = memory_length
        self.sensitivity_threshold = sensitivity_threshold
        self.lr_min, self.lr_max = lr_search_range
        self.patience_for_search = patience_for_search
        self.warmup_epochs = warmup_epochs
        self.warmup_strategy = warmup_strategy
        
        # 🚀 ENHANCED ADAPTIVE SCHEDULER FEATURES (from proven system)
        self.lr_increase_threshold = lr_increase_threshold
        self.lr_decrease_threshold = lr_decrease_threshold
        self.lr_multiplier_up = lr_multiplier_up
        self.lr_multiplier_down = lr_multiplier_down
        self.plateau_escape_epochs = plateau_escape_epochs
        self.plateau_escape_multiplier = plateau_escape_multiplier
        self.plateau_escape_cap = plateau_escape_cap
        
        self.verbose = verbose
        
        # Performance tracking
        self.accuracy_history = deque(maxlen=memory_length)
        self.lr_history = deque(maxlen=memory_length)
        self.lr_performance_map = {}  # LR -> average accuracy improvement
        
        # LR search state
        self.epochs_without_improvement = 0
        self.best_accuracy = 0.0
        self.best_lr = base_lr
        self.search_active = False
        self.search_lrs = []
        self.search_index = 0
        
        # 🚀 PLATEAU ESCAPE STATE (from proven adaptive scheduler)
        self.plateau_epochs = 0  # Track epochs without improvement
        self.plateau_lr_boost = False  # Flag for plateau escape attempts
        
        # 🎯 MULTI-COMPONENT OPTIMIZER SUPPORT
        self.is_multi_component = self._detect_multi_component_optimizer()
        self.component_lr_ratios = self._get_component_lr_ratios() if self.is_multi_component else None
        
        # Set initial LR - start with warmup LR if warmup is enabled
        if self.warmup_epochs > 0:
            initial_lr = self.base_lr * 0.1  # Start at 10% of base LR for warmup
            self._set_lr(initial_lr)
        else:
            self._set_lr(base_lr)
        
        if verbose:
            print(f"🎯 Advanced LR Scheduler initialized:")
            print(f"  📍 Base LR: {base_lr:.2e} (from your 2.80e-04 observation)")
            print(f"  🧠 Memory length: {memory_length} epochs")
            print(f"  🎚️ Sensitivity: {sensitivity_threshold*100:.1f}% accuracy change")
            print(f"  🔍 Search range: {self.lr_min:.2e} - {self.lr_max:.2e}")
            if warmup_epochs > 0:
                print(f"  🔥 Warmup: {warmup_epochs} epochs using {warmup_strategy} strategy")
            else:
                print(f"  ❄️ No warmup: Starting directly at base LR")
            
            # Print enhanced adaptive features
            print(f"  🚀 Enhanced Adaptive Features:")
            print(f"    📈 LR increase threshold: {lr_increase_threshold*100:.1f}% accuracy improvement")
            print(f"    📉 LR decrease threshold: {abs(lr_decrease_threshold)*100:.1f}% accuracy drop")
            print(f"    ⬆️ LR multiplier up: {lr_multiplier_up}x")
            print(f"    ⬇️ LR multiplier down: {lr_multiplier_down}x")
            print(f"    🚀 Plateau escape: {plateau_escape_multiplier}x boost after {plateau_escape_epochs} epochs")
            
            if self.is_multi_component:
                print(f"  🎯 Multi-component optimizer detected with {len(self.optimizer.param_groups)} groups")
                for i, ratio in enumerate(self.component_lr_ratios):
                    component_name = ['TimesNet', 'CNN', 'Classifier'][i] if i < 3 else f'Group_{i}'
                    print(f"    {component_name}: {ratio:.3f}x base LR")
            else:
                print(f"  🎯 Single-component optimizer")
    
    def _detect_multi_component_optimizer(self) -> bool:
        """Detect if optimizer has multiple parameter groups (multi-component)."""
        return len(self.optimizer.param_groups) > 1
    
    def _get_component_lr_ratios(self) -> List[float]:
        """Get the LR ratios for multi-component optimizer."""
        if not self.is_multi_component:
            return [1.0]
        
        # 🚨 BUG FIX: Always use the proven ratios from the working adaptive scheduler
        # Don't try to calculate from current LRs as they may be incorrectly set
        return [0.02, 0.8, 1.0]  # TimesNet: 0.02x, CNN: 0.8x, Classifier: 1.0x
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups, respecting component ratios."""
        if self.is_multi_component and self.component_lr_ratios:
            # Multi-component: apply ratios
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(self.component_lr_ratios):
                    component_lr = lr * self.component_lr_ratios[i]
                    param_group['lr'] = component_lr
                else:
                    param_group['lr'] = lr
        else:
            # Single component: set same LR for all groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def _calculate_warmup_lr(self, epoch: int) -> float:
        """Calculate learning rate during warmup phase."""
        if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
            return self.base_lr
        
        # 🚨 BUG FIX: Ensure warmup goes from LOW to HIGH
        progress = epoch / self.warmup_epochs  # 0 to 1
        
        if self.warmup_strategy == 'linear':
            # Linear warmup: start_lr → base_lr  
            start_lr = self.base_lr * 0.1  # Start at 10% of base LR
            warmup_lr = start_lr + (self.base_lr - start_lr) * progress
        elif self.warmup_strategy == 'cosine':
            # Cosine warmup: smooth S-curve from low to high
            start_lr = self.base_lr * 0.1  # Start at 10% of base LR
            warmup_lr = start_lr + (self.base_lr - start_lr) * (1 - np.cos(progress * np.pi)) / 2
        elif self.warmup_strategy == 'exponential':
            # Exponential warmup: slow start, fast finish
            start_lr = self.base_lr * 0.1  # Start at 10% of base LR
            warmup_lr = start_lr + (self.base_lr - start_lr) * (progress ** 0.5)
        else:
            # Default to linear
            start_lr = self.base_lr * 0.1
            warmup_lr = start_lr + (self.base_lr - start_lr) * progress
        
        # Ensure warmup LR stays within bounds
        return max(min(warmup_lr, self.lr_max), self.lr_min)
    
    def step(self, val_accuracy: float, epoch: int) -> Dict[str, float]:
        """
        Update learning rate based on validation accuracy.
        
        Args:
            val_accuracy: Current validation accuracy (0-100 scale)
            epoch: Current epoch number
            
        Returns:
            Dict with LR adjustment information
        """
        current_lr = self._get_lr()
        
        # Track performance
        self.accuracy_history.append(val_accuracy)
        self.lr_history.append(current_lr)
        
        # 🔥 WARMUP PHASE: Use gradual LR increase
        if epoch < self.warmup_epochs:
            warmup_lr = self._calculate_warmup_lr(epoch + 1)  # +1 because epoch is 0-indexed
            self._set_lr(warmup_lr)
            
            warmup_progress = (epoch + 1) / self.warmup_epochs * 100
            if self.verbose:
                print(f"🔥 WARMUP: Epoch {epoch+1}/{self.warmup_epochs} ({warmup_progress:.0f}%)")
                print(f"    LR: {current_lr:.2e} → {warmup_lr:.2e} ({self.warmup_strategy} warmup)")
            
            return {
                'old_lr': current_lr,
                'new_lr': warmup_lr,
                'accuracy_improvement': 0,
                'lr_efficiency': 0,
                'reason': f'warmup_{self.warmup_strategy}_{epoch+1}/{self.warmup_epochs}',
                'is_warmup': True
            }
        
        # POST-WARMUP: Normal performance-sensitive adaptation
        # Calculate performance metrics
        info = self._analyze_performance(val_accuracy, current_lr, epoch)
        
        # Determine LR adjustment strategy
        new_lr = self._determine_new_lr(val_accuracy, current_lr, epoch, info)
        
        if new_lr != current_lr:
            self._set_lr(new_lr)
            if self.verbose:
                print(f"🎚️ LR adjusted: {current_lr:.2e} → {new_lr:.2e} ({info['reason']})")
        
        return {
            'old_lr': current_lr,
            'new_lr': new_lr,
            'accuracy_improvement': info.get('accuracy_improvement', 0),
            'lr_efficiency': info.get('lr_efficiency', 0),
            'reason': info['reason'],
            'is_warmup': False
        }
    
    def _analyze_performance(self, val_accuracy: float, current_lr: float, epoch: int) -> Dict:
        """Analyze recent performance and LR efficiency."""
        info = {'reason': 'analyzing'}
        
        if len(self.accuracy_history) < 2:
            return info
        
        # Calculate accuracy improvement over last epoch
        accuracy_improvement = val_accuracy - self.accuracy_history[-2]
        info['accuracy_improvement'] = accuracy_improvement
        
        # Update LR performance tracking
        if current_lr not in self.lr_performance_map:
            self.lr_performance_map[current_lr] = []
        self.lr_performance_map[current_lr].append(accuracy_improvement)
        
        # Calculate LR efficiency (average improvement at this LR)
        lr_efficiency = np.mean(self.lr_performance_map[current_lr])
        info['lr_efficiency'] = lr_efficiency
        
        # Track best performance
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_lr = current_lr
            self.epochs_without_improvement = 0
            self.plateau_epochs = 0  # Reset plateau counter on improvement
            self.plateau_lr_boost = False  # Reset plateau escape flag
            info['new_best'] = True
        else:
            self.epochs_without_improvement += 1
            self.plateau_epochs += 1  # Track plateau epochs
            info['new_best'] = False
        
        return info
    
    def _determine_new_lr(self, val_accuracy: float, current_lr: float, epoch: int, info: Dict) -> float:
        """Determine optimal learning rate based on performance analysis."""
        
        # Early epochs: conservative exploration around base LR
        if epoch < 5:
            if info.get('accuracy_improvement', 0) > self.sensitivity_threshold:
                # Good improvement: slightly increase LR
                new_lr = min(current_lr * 1.1, self.lr_max)
                info['reason'] = f"early_boost (+{info['accuracy_improvement']:.3f}%)"
                return new_lr
            elif info.get('accuracy_improvement', 0) < -self.sensitivity_threshold:
                # Poor performance: slightly decrease LR
                new_lr = max(current_lr * 0.9, self.lr_min)
                info['reason'] = f"early_reduce ({info['accuracy_improvement']:.3f}%)"
                return new_lr
            else:
                info['reason'] = "early_stable"
                return current_lr
        
        # Main training: sophisticated adaptation
        if len(self.accuracy_history) >= self.memory_length:
            return self._sophisticated_lr_adaptation(current_lr, info)
        
        # Fallback: gradual adaptation
        return self._gradual_lr_adaptation(current_lr, info)
    
    def _sophisticated_lr_adaptation(self, current_lr: float, info: Dict) -> float:
        """Advanced LR adaptation using performance history and proven adaptive features."""
        
        accuracy_improvement = info.get('accuracy_improvement', 0)
        lr_efficiency = info.get('lr_efficiency', 0)
        
        # 🚀 PLATEAU ESCAPE MECHANISM (from proven adaptive scheduler)
        if self.plateau_epochs >= self.plateau_escape_epochs and not self.plateau_lr_boost:
            # Aggressive LR boost every 5 epochs of no improvement
            new_lr = min(current_lr * self.plateau_escape_multiplier, 
                        self.base_lr * self.plateau_escape_cap)
            self.plateau_lr_boost = True
            info['reason'] = f"plateau_escape_{self.plateau_escape_multiplier}x_boost"
            print(f"🚀 PLATEAU ESCAPE: LR boosted to {new_lr:.2e} (plateau for {self.plateau_epochs} epochs)")
            return new_lr
        
        # 🎯 PROVEN THRESHOLD-BASED ADAPTATION
        # Strategy 1: Significant accuracy improvement → increase LR
        if accuracy_improvement > self.lr_increase_threshold:
            new_lr = min(current_lr * self.lr_multiplier_up, self.base_lr * 2.0)
            info['reason'] = f"threshold_increase_+{accuracy_improvement:.3f}%"
            return new_lr
        
        # Strategy 2: Significant accuracy drop → decrease LR
        elif accuracy_improvement < self.lr_decrease_threshold:
            new_lr = max(current_lr * self.lr_multiplier_down, self.lr_min)
            info['reason'] = f"threshold_decrease_{accuracy_improvement:.3f}%"
            return new_lr
        
        # Strategy 3: Performance-sensitive fine-tuning (new enhancement)
        else:
            # Find best performing LR from history
            best_historical_lr = self._find_best_historical_lr()
            
            # If current improvement is moderately good, blend with historical best
            if accuracy_improvement > self.sensitivity_threshold:
                new_lr = current_lr * 0.9 + best_historical_lr * 0.1  # Gentle blend
                info['reason'] = f"performance_blend_+{accuracy_improvement:.3f}%"
                return max(min(new_lr, self.lr_max), self.lr_min)
            
            # If stagnating, trigger LR search
            elif self.epochs_without_improvement >= self.patience_for_search:
                return self._trigger_lr_search(current_lr, info)
            
            # Stable performance - fine-tune based on recent trend
            else:
                if len(self.accuracy_history) >= 5:
                    recent_trend = np.mean(list(self.accuracy_history)[-3:]) - np.mean(list(self.accuracy_history)[-5:-2])
                    if recent_trend > 0:
                        new_lr = current_lr * 1.01  # Tiny increase for uptrend
                        info['reason'] = f"fine_tune_uptrend_{recent_trend:.3f}%"
                    else:
                        new_lr = current_lr * 0.99  # Tiny decrease for downtrend
                        info['reason'] = f"fine_tune_downtrend_{recent_trend:.3f}%"
                else:
                    new_lr = current_lr  # Keep stable if insufficient history
                    info['reason'] = "insufficient_history"
                
                return max(min(new_lr, self.lr_max), self.lr_min)
    
    def _gradual_lr_adaptation(self, current_lr: float, info: Dict) -> float:
        """Simple gradual adaptation for early training phases."""
        accuracy_improvement = info.get('accuracy_improvement', 0)
        
        if accuracy_improvement > self.sensitivity_threshold:
            new_lr = min(current_lr * 1.05, self.lr_max)
            info['reason'] = f"gradual_increase (+{accuracy_improvement:.3f}%)"
        elif accuracy_improvement < -self.sensitivity_threshold:
            new_lr = max(current_lr * 0.95, self.lr_min)
            info['reason'] = f"gradual_decrease ({accuracy_improvement:.3f}%)"
        else:
            new_lr = current_lr
            info['reason'] = "gradual_stable"
        
        return new_lr
    
    def _find_best_historical_lr(self) -> float:
        """Find the LR that historically gave the best average improvement."""
        if not self.lr_performance_map:
            return self.base_lr
        
        best_lr = self.base_lr
        best_avg_improvement = float('-inf')
        
        for lr, improvements in self.lr_performance_map.items():
            if len(improvements) >= 2:  # Need at least 2 data points
                avg_improvement = np.mean(improvements)
                if avg_improvement > best_avg_improvement:
                    best_avg_improvement = avg_improvement
                    best_lr = lr
        
        return best_lr
    
    def _trigger_lr_search(self, current_lr: float, info: Dict) -> float:
        """Implement systematic LR search when stagnating."""
        if not self.search_active:
            # Initialize LR search around current LR
            self.search_lrs = self._generate_search_lrs(current_lr)
            self.search_active = True
            self.search_index = 0
            if self.verbose:
                print(f"🔍 Triggering LR search around {current_lr:.2e}")
        
        # Get next LR in search sequence
        if self.search_index < len(self.search_lrs):
            new_lr = self.search_lrs[self.search_index]
            self.search_index += 1
            info['reason'] = f"lr_search_{self.search_index}/{len(self.search_lrs)}"
        else:
            # Search complete, return to best found LR
            best_lr = self._find_best_historical_lr()
            self.search_active = False
            self.epochs_without_improvement = 0
            new_lr = best_lr
            info['reason'] = f"search_complete_best"
        
        return new_lr
    
    def _generate_search_lrs(self, center_lr: float) -> List[float]:
        """Generate LR search sequence around center LR."""
        # Search in ±50% range around center LR
        multipliers = [0.5, 0.7, 0.85, 1.15, 1.3, 1.5]
        search_lrs = []
        
        for mult in multipliers:
            lr = center_lr * mult
            if self.lr_min <= lr <= self.lr_max:
                search_lrs.append(lr)
        
        # Sort by distance from best known LR
        best_lr = self._find_best_historical_lr()
        search_lrs.sort(key=lambda lr: abs(lr - best_lr))
        
        return search_lrs[:4]  # Limit to 4 search points
    
    def get_performance_summary(self) -> Dict:
        """Get summary of LR performance for analysis."""
        summary = {
            'best_lr': self.best_lr,
            'best_accuracy': self.best_accuracy,
            'lr_performance_map': {
                f"{lr:.2e}": {
                    'avg_improvement': np.mean(improvements),
                    'num_epochs': len(improvements),
                    'total_improvement': sum(improvements)
                }
                for lr, improvements in self.lr_performance_map.items()
                if len(improvements) > 0
            },
            'current_lr': self._get_lr(),
            'epochs_without_improvement': self.epochs_without_improvement
        }
        return summary
    
    def plot_lr_performance(self, save_path: str = None):
        """Plot LR vs performance for analysis."""
        if not self.lr_performance_map:
            print("No LR performance data to plot yet.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: LR vs Average Improvement
        plt.subplot(2, 2, 1)
        lrs = []
        avg_improvements = []
        for lr, improvements in self.lr_performance_map.items():
            if len(improvements) > 0:
                lrs.append(lr)
                avg_improvements.append(np.mean(improvements))
        
        plt.scatter(lrs, avg_improvements, alpha=0.7, s=50)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=self.best_lr, color='green', linestyle='--', alpha=0.7, label=f'Best LR: {self.best_lr:.2e}')
        plt.axvline(x=2.8e-4, color='purple', linestyle='--', alpha=0.7, label='Your observation: 2.80e-04')
        plt.xlabel('Learning Rate')
        plt.ylabel('Average Accuracy Improvement (%)')
        plt.title('LR vs Average Performance')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: LR History with Warmup Indication
        plt.subplot(2, 2, 2)
        lr_values = list(self.lr_history)
        epochs = list(range(len(lr_values)))
        
        # Plot LR history
        plt.plot(epochs, lr_values, marker='o', markersize=3, label='LR History')
        
        # Highlight warmup phase if it exists
        if self.warmup_epochs > 0 and len(lr_values) >= self.warmup_epochs:
            warmup_epochs_range = list(range(min(self.warmup_epochs, len(lr_values))))
            warmup_lrs = lr_values[:len(warmup_epochs_range)]
            plt.plot(warmup_epochs_range, warmup_lrs, marker='s', markersize=4, 
                    color='red', alpha=0.7, label=f'Warmup ({self.warmup_strategy})')
            plt.axvline(x=self.warmup_epochs-1, color='red', linestyle='--', alpha=0.5, 
                       label='Warmup End')
        
        plt.xlabel('Recent Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Recent LR History')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy History
        plt.subplot(2, 2, 3)
        plt.plot(list(self.accuracy_history), marker='o', markersize=3, color='green')
        plt.xlabel('Recent Epochs')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Recent Accuracy History')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: LR Efficiency (epochs used per LR)
        plt.subplot(2, 2, 4)
        lrs = []
        epoch_counts = []
        for lr, improvements in self.lr_performance_map.items():
            lrs.append(lr)
            epoch_counts.append(len(improvements))
        
        plt.bar(range(len(lrs)), epoch_counts, alpha=0.7)
        plt.xlabel('LR Index')
        plt.ylabel('Epochs Used')
        plt.title('LR Usage Distribution')
        plt.xticks(range(len(lrs)), [f"{lr:.2e}" for lr in lrs], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"LR performance plot saved to {save_path}")
        else:
            plt.show()


def create_performance_sensitive_scheduler(optimizer, base_lr: float = 2.8e-4, **kwargs):
    """Factory function to create the advanced scheduler."""
    return PerformanceSensitiveLRScheduler(optimizer, base_lr=base_lr, **kwargs)