import argparse
import os
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import shap
import lime
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import wandb

class FinancialDataset:
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
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

def compute_comprehensive_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    
    # Precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(precision)):
        metrics[f'precision_class_{i}'] = precision[i]
        metrics[f'recall_class_{i}'] = recall[i]
        metrics[f'f1_class_{i}'] = f1[i]
        metrics[f'support_class_{i}'] = support[i]
    
    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
        except:
            pass
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics_curves(history, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['eval_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curves
    ax2.plot(history['eval_accuracy'], label='Validation Accuracy')
    ax2.plot(history['eval_f1_weighted'], label='Validation F1 (Weighted)')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_worst_predictions(model, tokenizer, texts, true_labels, pred_labels, probs, n=10):
    """Find worst false positives and false negatives."""
    errors = []
    for i, (text, true, pred, prob) in enumerate(zip(texts, true_labels, pred_labels, probs)):
        if true != pred:
            confidence = max(prob)
            errors.append({
                'text': text,
                'true': true,
                'pred': pred,
                'confidence': confidence,
                'prob_dist': prob
            })
    
    # Sort by confidence (most confident mistakes first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    return errors[:n]

def shap_explainer(model, tokenizer, texts, n_samples=100):
    """Generate SHAP explanations."""
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(texts[:n_samples])
    return shap_values

def lime_explainer(model, tokenizer, texts, n_samples=10):
    """Generate LIME explanations."""
    explainer = LimeTextExplainer(class_names=['Negative', 'Neutral', 'Positive'])
    
    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        return probs.numpy()
    
    explanations = []
    for text in texts[:n_samples]:
        exp = explainer.explain_instance(text, predict_proba, num_features=20)
        explanations.append(exp)
    
    return explanations

def k_fold_cross_validation(model_class, tokenizer, texts, labels, k=5, **kwargs):
    """Perform K-fold cross validation."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f'Fold {fold + 1}/{k}')
        
        train_texts, val_texts = texts[train_idx], texts[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # Train model
        model = model_class.from_pretrained('ProsusAI/finbert', num_labels=3)
        # ... training code here ...
        
        # Evaluate
        model.eval()
        val_dataset = FinancialDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        metrics = compute_comprehensive_metrics(true_labels, predictions)
        fold_metrics.append(metrics)
    
    return fold_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation with metrics, confusion matrix, and interpretability.')
    parser.add_argument('--config', type=str, help='Path to evaluation config (.yaml/.json)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed/labeled_data.csv', help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='metrics', help='Directory to save evaluation results')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--n_samples_interpret', type=int, default=50, help='Number of samples for interpretability')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    
    # Load data
    df = pd.read_csv(args.data_path)
    texts = df['text'].values
    labels = df['label'].values
    
    # Evaluate on test set
    test_dataset = FinancialDataset(texts, labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    predictions = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    # Compute metrics
    metrics = compute_comprehensive_metrics(true_labels, predictions, np.array(probabilities))
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, 
                        os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Find worst predictions
    worst_preds = find_worst_predictions(model, tokenizer, texts, true_labels, 
                                        predictions, probabilities, n=10)
    
    with open(os.path.join(args.output_dir, 'worst_predictions.json'), 'w') as f:
        json.dump(worst_preds, f, indent=2, default=str)
    
    # SHAP explanations
    print('Generating SHAP explanations...')
    shap_values = shap_explainer(model, tokenizer, texts, n_samples=min(50, len(texts)))
    shap.save_html(os.path.join(args.output_dir, 'shap_explanations.html'), shap_values)
    
    # LIME explanations
    print('Generating LIME explanations...')
    lime_explanations = lime_explainer(model, tokenizer, texts, n_samples=min(10, len(texts)))
    
    # Save LIME results
    lime_results = []
    for i, exp in enumerate(lime_explanations):
        lime_results.append({
            'text': texts[i],
            'explanation': str(exp.as_list())
        })
    
    with open(os.path.join(args.output_dir, 'lime_explanations.json'), 'w') as f:
        json.dump(lime_results, f, indent=2)
    
    print(f'Evaluation results saved to {args.output_dir}')
    print(f'Overall accuracy: {metrics["accuracy"]:.4f}')
    print(f'F1 (weighted): {metrics["f1_weighted"]:.4f}')

if __name__ == '__main__':
    main() 