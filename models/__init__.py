"""Model module for deepfake detection."""

from .base_model import BaseModel
from .xception import XceptionNet
from .efficientnet import EfficientNetModel

__all__ = [
    'BaseModel',
    'XceptionNet',
    'EfficientNetModel',
    'get_model',
]


def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True, **kwargs):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_name: Name of the model ('xception', 'efficientnet')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    models = {
        'xception': XceptionNet,
        'efficientnet': EfficientNetModel,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name.lower()](num_classes=num_classes, pretrained=pretrained, **kwargs)
