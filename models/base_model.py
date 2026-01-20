"""Base model class for deepfake detection."""

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all deepfake detection models."""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize base model.
        
        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    def save(self, save_path: str):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, load_path: str, device: str = 'cpu'):
        """
        Load model checkpoint.
        
        Args:
            load_path: Path to load the model from
            device: Device to load the model on
        """
        checkpoint = torch.load(load_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_path}")
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for name, param in self.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
