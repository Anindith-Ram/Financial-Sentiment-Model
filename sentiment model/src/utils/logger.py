import logging
import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
import wandb
import mlflow
from pathlib import Path

class ExperimentLogger:
    """Comprehensive experiment logging with wandb and mlflow support."""
    
    def __init__(self, 
                 experiment_name: str,
                 project_name: str = "financial-sentiment",
                 log_dir: str = "logs",
                 use_wandb: bool = True,
                 use_mlflow: bool = False,
                 config: Optional[Dict] = None):
        
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self.config = config or {}
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
    
    def _setup_logging(self):
        """Setup file and console logging."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def _setup_experiment_tracking(self):
        """Initialize wandb and mlflow tracking."""
        if self.use_wandb:
            try:
                wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=self.config,
                    dir=self.log_dir
                )
                self.logger.info("Wandb initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        
        if self.use_mlflow:
            try:
                mlflow.set_experiment(self.experiment_name)
                mlflow.start_run()
                mlflow.log_params(self.config)
                self.logger.info("MLflow initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize mlflow: {e}")
                self.use_mlflow = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all active tracking systems."""
        # Log to file
        self.logger.info(f"Metrics at step {step}: {metrics}")
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)
        
        # Log to mlflow
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters."""
        self.logger.info(f"Configuration: {config}")
        
        # Save config to file
        config_file = os.path.join(self.log_dir, f"{self.experiment_name}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.config.update(config)
        
        # Log to mlflow
        if self.use_mlflow:
            mlflow.log_params(config)
    
    def log_model(self, model_path: str, model_name: str = "finbert_sentiment"):
        """Log model artifacts."""
        self.logger.info(f"Logging model from {model_path}")
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.save(model_path)
        
        # Log to mlflow
        if self.use_mlflow:
            mlflow.log_artifact(model_path)
    
    def log_plots(self, plot_paths: Dict[str, str]):
        """Log plot files."""
        for plot_name, plot_path in plot_paths.items():
            if os.path.exists(plot_path):
                self.logger.info(f"Logging plot: {plot_name} from {plot_path}")
                
                # Log to wandb
                if self.use_wandb and wandb.run is not None:
                    wandb.log({plot_name: wandb.Image(plot_path)})
                
                # Log to mlflow
                if self.use_mlflow:
                    mlflow.log_artifact(plot_path)
    
    def log_predictions(self, predictions: Dict[str, Any], filename: str = "predictions.json"):
        """Log prediction results."""
        pred_file = os.path.join(self.log_dir, filename)
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        self.logger.info(f"Predictions saved to {pred_file}")
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.save(pred_file)
        
        # Log to mlflow
        if self.use_mlflow:
            mlflow.log_artifact(pred_file)
    
    def log_error_analysis(self, error_analysis: Dict[str, Any], filename: str = "error_analysis.json"):
        """Log error analysis results."""
        error_file = os.path.join(self.log_dir, filename)
        with open(error_file, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        self.logger.info(f"Error analysis saved to {error_file}")
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.save(error_file)
        
        # Log to mlflow
        if self.use_mlflow:
            mlflow.log_artifact(error_file)
    
    def log_training_curves(self, history: Dict[str, list]):
        """Log training curves."""
        self.logger.info("Logging training curves")
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            for metric_name, values in history.items():
                for step, value in enumerate(values):
                    wandb.log({metric_name: value}, step=step)
        
        # Save to file
        history_file = os.path.join(self.log_dir, "training_history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log experiment summary."""
        self.logger.info(f"Experiment summary: {summary}")
        
        # Save summary to file
        summary_file = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.run.summary.update(summary)
        
        # Log to mlflow
        if self.use_mlflow:
            for key, value in summary.items():
                mlflow.log_metric(key, value)
    
    def finish(self):
        """Finish the experiment and cleanup."""
        self.logger.info("Finishing experiment")
        
        # Finish wandb
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
        
        # Finish mlflow
        if self.use_mlflow:
            mlflow.end_run()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

def create_experiment_logger(experiment_name: str, 
                           config: Dict[str, Any],
                           use_wandb: bool = True,
                           use_mlflow: bool = False) -> ExperimentLogger:
    """Factory function to create an experiment logger."""
    return ExperimentLogger(
        experiment_name=experiment_name,
        config=config,
        use_wandb=use_wandb,
        use_mlflow=use_mlflow
    )

def log_experiment_results(experiment_name: str,
                          metrics: Dict[str, float],
                          config: Dict[str, Any],
                          model_path: Optional[str] = None,
                          plot_paths: Optional[Dict[str, str]] = None,
                          predictions: Optional[Dict[str, Any]] = None,
                          error_analysis: Optional[Dict[str, Any]] = None):
    """Convenience function to log complete experiment results."""
    
    with ExperimentLogger(experiment_name, config=config) as logger:
        # Log configuration
        logger.log_config(config)
        
        # Log metrics
        logger.log_metrics(metrics)
        
        # Log model if provided
        if model_path:
            logger.log_model(model_path)
        
        # Log plots if provided
        if plot_paths:
            logger.log_plots(plot_paths)
        
        # Log predictions if provided
        if predictions:
            logger.log_predictions(predictions)
        
        # Log error analysis if provided
        if error_analysis:
            logger.log_error_analysis(error_analysis)
        
        # Log summary
        summary = {
            "best_accuracy": metrics.get("accuracy", 0),
            "best_f1_weighted": metrics.get("f1_weighted", 0),
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat()
        }
        logger.log_summary(summary) 