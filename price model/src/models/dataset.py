"""
PyTorch Dataset for Candlestick Pattern Data
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config.config import SEQ_LEN


class CandlestickDataset(Dataset):
    """
    PyTorch Dataset for candlestick pattern recognition
    """
    
    def __init__(self, csv_file, train_split=0.8, is_training=True):
        """
        Initialize the dataset
        
        Args:
            csv_file (str): Path to the CSV file containing the data
            train_split (float): Proportion of data to use for training
            is_training (bool): Whether this is a training dataset
        """
        self.df = pd.read_csv(csv_file)
        
        # Split data randomly
        np.random.seed(42)  # For reproducible splits
        indices = np.random.permutation(len(self.df))
        split_idx = int(len(self.df) * train_split)
        
        if is_training:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        # Extract features and labels
        feature_columns = [col for col in self.df.columns if col not in ['Ticker', 'Label']]
        self.X = self.df[feature_columns].values.astype(np.float32)[self.indices]
        self.y = self.df['Label'].values.astype(np.int64)[self.indices]
        
        # Normalize features to prevent numerical overflow
        print("🔧 Normalizing features to prevent numerical overflow...")
        self.X = self.normalize_features(self.X)
        print(f"✅ Normalization complete. Feature range: {self.X.min():.4f} to {self.X.max():.4f}")
        
        # Calculate dimensions
        self.seq_len = SEQ_LEN
        self.features_per_day = len(feature_columns) // SEQ_LEN
        
        print(f"Dataset initialized:")
        print(f"  Samples: {len(self.X)}")
        print(f"  Features per day: {self.features_per_day}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Class distribution: {np.bincount(self.y)}")
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (features, label)
                features: torch.Tensor of shape (seq_len, features_per_day)
                label: torch.Tensor scalar
        """
        # Reshape features from flat to (seq_len, features_per_day)
        features = self.X[idx].reshape(self.seq_len, self.features_per_day)
        features = torch.from_numpy(features).float()  # Ensure float32
        
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        return features, label
    
    def normalize_features(self, X):
        """
        Normalize features using robust scaling to handle outliers
        
        Args:
            X (np.ndarray): Raw feature data
            
        Returns:
            np.ndarray: Normalized feature data
        """
        # Use robust scaling (median and IQR) to handle outliers
        median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        
        # Avoid division by zero
        iqr[iqr == 0] = 1.0
        
        # Robust normalization: (x - median) / IQR
        X_normalized = (X - median) / iqr
        
        # Clip extreme values to prevent outliers
        X_normalized = np.clip(X_normalized, -10, 10)
        
        return X_normalized
    
    def get_feature_names(self):
        """Get the names of features in order"""
        feature_columns = [col for col in self.df.columns if col not in ['Ticker', 'Label']]
        return feature_columns
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced dataset
        
        Returns:
            torch.Tensor: Class weights for loss function
        """
        class_counts = np.bincount(self.y)
        total_samples = len(self.y)
        num_classes = len(class_counts)
        
        # Calculate inverse frequency weights
        weights = total_samples / (num_classes * class_counts)
        return torch.FloatTensor(weights)


class CandlestickDataLoader:
    """
    Utility class to create train/validation data loaders
    """
    
    def __init__(self, csv_file, batch_size=256, train_split=0.8, num_workers=0):
        """
        Initialize data loaders
        
        Args:
            csv_file (str): Path to CSV file
            batch_size (int): Batch size for training
            train_split (float): Training split ratio
            num_workers (int): Number of worker processes for data loading
        """
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = CandlestickDataset(
            csv_file, train_split=train_split, is_training=True
        )
        self.val_dataset = CandlestickDataset(
            csv_file, train_split=train_split, is_training=False
        )
    
    def get_train_loader(self):
        """Get training data loader"""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_feature_dim(self):
        """Get the number of features per day"""
        return self.train_dataset.features_per_day
    
    def get_class_weights(self):
        """Get class weights from training dataset"""
        return self.train_dataset.get_class_weights()
    
    def get_feature_names(self):
        """Get the names of features in order"""
        return self.train_dataset.get_feature_names()
    
    def get_loaders(self):
        """Get both train and validation loaders"""
        return self.get_train_loader(), self.get_val_loader() 