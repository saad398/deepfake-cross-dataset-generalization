"""EfficientNet model for deepfake detection."""

import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet-B0 architecture for deepfake detection."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize EfficientNet model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before final classifier
        """
        super().__init__(num_classes)
        self.dropout = dropout
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(pretrained=True)
        else:
            self.backbone = models.efficientnet_b0(pretrained=False)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        # Extract features using the backbone without classifier
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetB4(BaseModel):
    """EfficientNet-B4 architecture for deepfake detection (higher capacity)."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize EfficientNet-B4 model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before final classifier
        """
        super().__init__(num_classes)
        self.dropout = dropout
        
        # Load pretrained EfficientNet-B4
        if pretrained:
            self.backbone = models.efficientnet_b4(pretrained=True)
        else:
            self.backbone = models.efficientnet_b4(pretrained=False)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet-B4.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
