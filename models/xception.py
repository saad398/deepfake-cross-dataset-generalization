"""XceptionNet model for deepfake detection."""

import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel


class XceptionNet(BaseModel):
    """
    XceptionNet architecture for deepfake detection.
    Based on the Xception architecture with modifications for binary classification.
    
    Note: PyTorch doesn't have a built-in Xception model, so we use a modified
    version based on depthwise separable convolutions similar to Xception's design.
    For production use, you might want to use timm library's Xception implementation.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize XceptionNet model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before final classifier
        """
        super().__init__(num_classes)
        self.dropout = dropout
        
        # For simplicity, we'll use a ResNet-like architecture with depthwise separable convolutions
        # In production, consider using timm library: timm.create_model('xception', pretrained=True)
        
        # Entry flow
        self.entry_flow = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Middle flow - Depthwise Separable Convolutions
        self.middle_flow = nn.Sequential(
            self._make_separable_block(64, 128, stride=2),
            self._make_separable_block(128, 256, stride=2),
            self._make_separable_block(256, 728, stride=2),
            
            # Repeat middle flow blocks
            self._make_separable_block(728, 728, stride=1),
            self._make_separable_block(728, 728, stride=1),
            self._make_separable_block(728, 728, stride=1),
        )
        
        # Exit flow
        self.exit_flow = nn.Sequential(
            self._make_separable_block(728, 1024, stride=2),
            
            # Final separable convolutions
            nn.Conv2d(1024, 1536, kernel_size=3, padding=1, groups=1024, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1536, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout and classifier
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_separable_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> nn.Module:
        """
        Create a depthwise separable convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the convolution
            
        Returns:
            Sequential block with depthwise separable convolution
        """
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through XceptionNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Entry flow
        x = self.entry_flow(x)
        
        # Middle flow
        x = self.middle_flow(x)
        
        # Exit flow
        x = self.exit_flow(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout and classification
        x = self.dropout_layer(x)
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# Alternative: Using a pre-trained ResNet as Xception substitute
class XceptionNetResNet(BaseModel):
    """
    Xception-like model using ResNet50 as backbone.
    This provides better pretrained weights from ImageNet.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize XceptionNet using ResNet50 backbone.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate before final classifier
        """
        super().__init__(num_classes)
        self.dropout = dropout
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout_layer(x)
        x = self.fc(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
