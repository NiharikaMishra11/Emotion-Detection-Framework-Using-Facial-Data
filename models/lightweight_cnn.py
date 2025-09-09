"""
Lightweight CNN Model for Emotion Detection

This module implements a lightweight CNN architecture optimized for CPU inference
on emotion detection tasks. The model is designed to be efficient while maintaining
good performance on facial emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Any
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """
    Custom dataset class for emotion detection.
    """
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 transform=None, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of emotion labels
            transform: Image transformations
            target_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Dataset initialized with {len(image_paths)} samples")
        logger.info(f"Classes: {self.label_encoder.classes_}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
        else:
            # Resize image
            image = cv2.resize(image, self.target_size)
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.encoded_labels[idx]
        
        return image, label

class LightweightCNN(nn.Module):
    """
    Lightweight CNN architecture for emotion detection.
    
    Architecture:
    - 3 Convolutional blocks with BatchNorm and ReLU
    - MaxPooling for downsampling
    - Global Average Pooling
    - 2 Fully connected layers with Dropout
    - Optimized for CPU inference
    """
    
    def __init__(self, num_classes: int = 6, input_channels: int = 3):
        """
        Initialize the lightweight CNN.
        
        Args:
            num_classes: Number of emotion classes
            input_channels: Number of input channels (3 for RGB)
        """
        super(LightweightCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

class CNNModelTrainer:
    """
    Trainer for the lightweight CNN model.
    
    Features:
    - Training with early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Training history tracking
    - CPU-optimized training
    """
    
    def __init__(self, 
                 num_classes: int = 6,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 device: str = None):
        """
        Initialize the CNN trainer.
        
        Args:
            num_classes: Number of emotion classes
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use ('cpu' or 'cuda')
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = LightweightCNN(num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"CNN Trainer initialized - Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_data_loaders(self, image_paths: List[str], labels: List[str],
                           test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            test_size: Proportion of data for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = EmotionDataset(train_paths, train_labels, train_transform)
        val_dataset = EmotionDataset(val_paths, val_labels, val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        logger.info(f"Data loaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
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
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
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
    
    def train(self, image_paths: List[str], labels: List[str],
              patience: int = 10) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(image_paths, labels)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.save_model("models/best_cnn_model.pth")
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'num_classes': self.num_classes
        }, filepath)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from: {filepath}")
    
    def predict(self, image_paths: List[str]) -> np.ndarray:
        """
        Make predictions on new images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Predicted labels
        """
        self.model.eval()
        predictions = []
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            for image_path in image_paths:
                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    predictions.append(0)  # Default prediction
                    continue
                
                image = cv2.resize(image, (128, 128))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = transform(image).unsqueeze(0).to(self.device)
                
                # Make prediction
                output = self.model(image)
                pred = output.argmax(dim=1).item()
                predictions.append(pred)
        
        return np.array(predictions)

def main():
    """Example usage of the CNN trainer."""
    # This would be called with actual image data
    # trainer = CNNModelTrainer(num_classes=6, batch_size=16, num_epochs=30)
    # 
    # # Load image paths and labels
    # image_paths, labels = load_image_data("data/processed")
    # 
    # # Train model
    # history = trainer.train(image_paths, labels)
    # 
    # # Plot training history
    # plot_training_history(history)
    pass

if __name__ == "__main__":
    main()
