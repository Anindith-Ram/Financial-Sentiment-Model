import argparse
import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from torch.utils.data import DataLoader
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import ExperimentLogger
from metrics import compute_comprehensive_metrics

class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

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

def calibrate_model(model, tokenizer, val_loader, device):
    """Apply temperature scaling calibration."""
    model.eval()
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits for calibration"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits_list.append(outputs.logits.cpu())
            labels_list.append(batch['labels'].cpu())
    
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Initialize temperature scaling
    temperature_scaling = TemperatureScaling()
    temperature_scaling.to(device)
    
    # Optimize temperature
    optimizer = torch.optim.LBFGS([temperature_scaling.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(temperature_scaling(logits.to(device)), labels.to(device))
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    return temperature_scaling

def distill_model(teacher_model, teacher_tokenizer, student_tokenizer, 
                 train_loader, val_loader, device, config):
    """Distill FinBERT to DistilBERT using soft labels."""
    
    # Initialize student model
    student_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=3
    )
    student_model.to(device)
    
    # Training parameters
    learning_rate = config.get('distillation_lr', 2e-5)
    epochs = config.get('distillation_epochs', 3)
    temperature = config.get('distillation_temperature', 2.0)
    alpha = config.get('distillation_alpha', 0.7)  # Weight for soft vs hard labels
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Distillation epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            # Get teacher predictions (soft labels)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits / temperature
                teacher_probs = torch.softmax(teacher_logits, dim=1)
            
            # Get student predictions
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits / temperature
            student_probs = torch.softmax(student_logits, dim=1)
            
            # Compute distillation loss
            distillation_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(student_probs), teacher_probs
            )
            
            # Compute hard label loss
            hard_loss = criterion(student_logits, labels)
            
            # Combined loss
            loss = alpha * distillation_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average loss: {total_loss/len(train_loader):.4f}")
    
    return student_model

def evaluate_calibration(model, tokenizer, test_loader, temperature_scaling, device):
    """Evaluate model calibration."""
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    calibrated_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating calibration"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            
            # Original predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Calibrated predictions
            calibrated_logits = temperature_scaling(logits)
            calibrated_prob = torch.softmax(calibrated_logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            calibrated_probs.extend(calibrated_prob.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    # Compute metrics
    original_metrics = compute_comprehensive_metrics(
        np.array(true_labels), 
        np.array(predictions), 
        np.array(probabilities)
    )
    
    calibrated_metrics = compute_comprehensive_metrics(
        np.array(true_labels), 
        np.array(predictions), 
        np.array(calibrated_probs)
    )
    
    return original_metrics, calibrated_metrics, temperature_scaling.temperature.item()

def parse_args():
    parser = argparse.ArgumentParser(description='Model calibration and distillation.')
    parser.add_argument('--config', type=str, help='Path to calibration config (.yaml/.json)')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Path to teacher model')
    parser.add_argument('--data_path', type=str, default='data/processed/labeled_data.csv', help='Path to data')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--calibrate', action='store_true', help='Apply temperature scaling calibration')
    parser.add_argument('--distill', action='store_true', help='Distill to DistilBERT')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    df = pd.read_csv(args.data_path)
    texts = df['text'].values
    labels = df['label'].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Load teacher model
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_path)
    teacher_model.to(device)
    
    # Create data loaders
    train_dataset = FinancialDataset(train_texts, train_labels, teacher_tokenizer)
    val_dataset = FinancialDataset(val_texts, val_labels, teacher_tokenizer)
    test_dataset = FinancialDataset(test_texts, test_labels, teacher_tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    experiment_name = config.get('experiment_name', 'model_calibration_distillation')
    
    with ExperimentLogger(
        experiment_name=experiment_name,
        config=config
    ) as logger:
        
        if args.calibrate:
            print("Applying temperature scaling calibration...")
            temperature_scaling = calibrate_model(teacher_model, teacher_tokenizer, val_loader, device)
            
            # Evaluate calibration
            original_metrics, calibrated_metrics, temperature = evaluate_calibration(
                teacher_model, teacher_tokenizer, test_loader, temperature_scaling, device
            )
            
            # Save calibrated model
            calibrated_model_path = os.path.join(args.output_dir, 'finbert_calibrated')
            os.makedirs(calibrated_model_path, exist_ok=True)
            
            # Save temperature scaling
            torch.save(temperature_scaling.state_dict(), os.path.join(calibrated_model_path, 'temperature_scaling.pt'))
            
            # Save tokenizer
            teacher_tokenizer.save_pretrained(calibrated_model_path)
            
            # Log results
            logger.log_metrics(original_metrics, step=0)
            logger.log_metrics(calibrated_metrics, step=1)
            logger.log_metrics({'temperature': temperature})
            
            print(f"Temperature: {temperature:.4f}")
            print(f"Original accuracy: {original_metrics['accuracy']:.4f}")
            print(f"Calibrated accuracy: {calibrated_metrics['accuracy']:.4f}")
        
        if args.distill:
            print("Distilling model to DistilBERT...")
            student_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Create student data loaders
            student_train_dataset = FinancialDataset(train_texts, train_labels, student_tokenizer)
            student_val_dataset = FinancialDataset(val_texts, val_labels, student_tokenizer)
            
            student_train_loader = DataLoader(student_train_dataset, batch_size=16, shuffle=True)
            student_val_loader = DataLoader(student_val_dataset, batch_size=32)
            
            # Distill model
            student_model = distill_model(
                teacher_model, teacher_tokenizer, student_tokenizer,
                student_train_loader, student_val_loader, device, config
            )
            
            # Save distilled model
            distilled_model_path = os.path.join(args.output_dir, 'distilbert_distilled')
            os.makedirs(distilled_model_path, exist_ok=True)
            student_model.save_pretrained(distilled_model_path)
            student_tokenizer.save_pretrained(distilled_model_path)
            
            # Evaluate distilled model
            student_test_dataset = FinancialDataset(test_texts, test_labels, student_tokenizer)
            student_test_loader = DataLoader(student_test_dataset, batch_size=32)
            
            student_model.eval()
            predictions = []
            probabilities = []
            true_labels = []
            
            with torch.no_grad():
                for batch in tqdm(student_test_loader, desc="Evaluating distilled model"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = student_model(**batch)
                    probs = torch.softmax(outputs.logits, dim=1)
                    preds = torch.argmax(outputs.logits, dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
                    true_labels.extend(batch['labels'].cpu().numpy())
            
            distilled_metrics = compute_comprehensive_metrics(
                np.array(true_labels), 
                np.array(predictions), 
                np.array(probabilities)
            )
            
            logger.log_metrics(distilled_metrics)
            logger.log_model(distilled_model_path)
            
            print(f"Distilled model accuracy: {distilled_metrics['accuracy']:.4f}")
            print(f"Distilled model F1: {distilled_metrics['f1_weighted']:.4f}")

if __name__ == '__main__':
    main() 