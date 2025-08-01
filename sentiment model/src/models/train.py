import argparse
import os
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import ExperimentLogger
from metrics import compute_comprehensive_metrics

MODEL_CHOICES = ['xgboost', 'lightgbm', 'transformer', 'lstm']

class FinancialDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_config(config_path):
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError('Config must be .yaml, .yml, or .json')

def parse_args():
    parser = argparse.ArgumentParser(description='Train ML models on multi-source sentiment features.')
    parser.add_argument('--config', type=str, help='Path to training config (.yaml/.json)')
    parser.add_argument('--data_path', type=str, default='data/processed/fused_sentiment_scores.parquet', help='Path to processed sentiment data')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_CHOICES, help='Model to train')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save trained models')
    return parser.parse_args()

def train_xgboost(X_train, y_train, X_val, y_val, config, w_train, w_val):
    model = xgb.XGBClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 6),
        learning_rate=config.get('learning_rate', 0.1),
        use_label_encoder=False,
        eval_metric='logloss'
    )
    fit_kwargs = {}
    if w_train is not None:
        fit_kwargs['sample_weight'] = w_train
        fit_kwargs['eval_sample_weight'] = [w_val]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True, **fit_kwargs)
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, config, w_train, w_val):
    model = lgb.LGBMClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', -1),
        learning_rate=config.get('learning_rate', 0.1)
    )
    fit_kwargs = {}
    if w_train is not None:
        fit_kwargs['sample_weight'] = w_train
        fit_kwargs['eval_sample_weight'] = [w_val]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True, **fit_kwargs)
    return model

def train_transformer(X_train, y_train, X_val, y_val, config):
    # Placeholder for transformer model training
    return None

def train_lstm(X_train, y_train, X_val, y_val, config):
    # Placeholder for LSTM model training
    return None

def main():
    args = parse_args()
    
    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Load data
    df = pd.read_parquet(args.data_path)
    # Separate weight column if present
    sample_weights = df.pop('sample_weight') if 'sample_weight' in df.columns else None

    # Identify the target column - could be 'composite_score' or 'return_t+1'
    target_col = None
    if 'return_t+1' in df.columns:
        target_col = 'return_t+1'
        labels = (df[target_col] > 0).astype(int).values  # Binary: positive return = 1
    elif 'composite_score' in df.columns:
        target_col = 'composite_score'
        labels = (df[target_col] > 0).astype(int).values  # Binary: positive sentiment = 1
    else:
        raise ValueError("No target column found ('return_t+1' or 'composite_score')")
    
    feature_cols_to_drop = [target_col, 'date', 'ticker', 'age_days']
    features = df.drop(columns=[c for c in feature_cols_to_drop if c in df.columns]).values
    
    # Train/validation split (keep weight alignment)
    if sample_weights is not None:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            features,
            labels,
            sample_weights.values,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        w_train = w_val = None
    
    # Initialize experiment logger
    experiment_name = config.get('experiment_name', 'financial_sentiment_training')
    with ExperimentLogger(experiment_name=experiment_name, config=config) as logger:
        
        # Train model
        if args.model == 'xgboost':
            model = train_xgboost(X_train, y_train, X_val, y_val, config, w_train, w_val)
        elif args.model == 'lightgbm':
            model = train_lightgbm(X_train, y_train, X_val, y_val, config, w_train, w_val)
        elif args.model == 'transformer':
            model = train_transformer(X_train, y_train, X_val, y_val, config)
        elif args.model == 'lstm':
            model = train_lstm(X_train, y_train, X_val, y_val, config)
        else:
            raise ValueError(f'Unknown model: {args.model}')
        
        # Evaluate model
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_comprehensive_metrics(y_val, y_pred, y_proba)
        logger.log_metrics(metrics)
        
        # Save model
        model_path = os.path.join(args.output_dir, f'{args.model}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.log_model(model_path)
        
        # SHAP/LIME interpretability
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_val)
        shap.summary_plot(shap_values, X_val, show=False)
        plt.savefig(os.path.join(args.output_dir, f'{args.model}_shap_summary.png'))
        
        lime_explainer = LimeTabularExplainer(X_train, mode='classification')
        lime_exp = lime_explainer.explain_instance(X_val[0], model.predict_proba)
        lime_exp.save_to_file(os.path.join(args.output_dir, f'{args.model}_lime.html'))

if __name__ == '__main__':
    main() 