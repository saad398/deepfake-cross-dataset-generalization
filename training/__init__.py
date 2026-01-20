"""Training module for deepfake detection."""

from .trainer import Trainer
from .train import train_model

__all__ = [
    'Trainer',
    'train_model',
]
