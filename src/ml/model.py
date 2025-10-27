"""
Machine Learning Model Definition for Defect Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging
from typing import Dict, Any, Optional


class DefectDetectionCNN(nn.Module):
    """
    Convolutional Neural Network for defect detection
    Based on ResNet architecture with custom classifier
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the defect detection model
        
        Args:
            num_classes: Number of output classes (default: 2 for Good/Defect)
            pretrained: Whether to use pretrained weights
        """
        super(DefectDetectionCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output logits
        """
        return self.backbone(x)


class DefectDetectionViT(nn.Module):
    """
    Vision Transformer for defect detection
    Alternative to CNN-based approach
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the Vision Transformer model
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(DefectDetectionViT, self).__init__()
        
        # Use Vision Transformer as backbone
        try:
            from torchvision.models import vision_transformer as vit
            self.backbone = vit.vit_b_16(pretrained=pretrained)
            
            # Replace the classifier head
            self.backbone.heads.head = nn.Linear(
                self.backbone.heads.head.in_features,
                num_classes
            )
        except ImportError:
            raise ImportError("Vision Transformer not available in this PyTorch version")
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass through the Vision Transformer"""
        return self.backbone(x)


class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model based on configuration
        
        Args:
            model_type: Type of model ('cnn', 'vit')
            config: Model configuration
            
        Returns:
            PyTorch model instance
        """
        num_classes = config.get('num_classes', 2)
        pretrained = config.get('pretrained', True)
        
        if model_type.lower() == 'cnn':
            return DefectDetectionCNN(num_classes=num_classes, pretrained=pretrained)
        elif model_type.lower() == 'vit':
            return DefectDetectionViT(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class ModelTrainer:
    """
    Training utilities for defect detection models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize model trainer
        
        Args:
            model: PyTorch model to train
            device: Device to run training on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for model trainer"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """
        Train the model for one epoch
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer for training
            criterion: Loss function
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader, criterion):
        """
        Validate the model
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, filepath: str, epoch: int = None, optimizer_state: Dict = None):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save the model
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'num_classes': getattr(self.model, 'num_classes', 2)
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded from {filepath}")
        return checkpoint