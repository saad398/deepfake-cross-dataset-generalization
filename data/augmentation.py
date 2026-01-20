"""Data augmentation pipelines for deepfake detection."""

from typing import Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class BasicAugmentation:
    """Basic augmentation pipeline for baseline model."""
    
    def __init__(self, image_size: int = 224):
        """
        Initialize basic augmentation pipeline.
        
        Args:
            image_size: Target image size
        """
        self.image_size = image_size
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build basic augmentation pipeline."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def __call__(self, **kwargs):
        """Apply augmentation."""
        return self.transform(**kwargs)


class RobustAugmentation:
    """Robust augmentation pipeline for improved cross-dataset generalization."""
    
    def __init__(
        self,
        image_size: int = 224,
        config: Dict[str, Any] = None
    ):
        """
        Initialize robust augmentation pipeline.
        
        Args:
            image_size: Target image size
            config: Configuration dictionary for augmentation parameters
        """
        self.image_size = image_size
        self.config = config or {}
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build robust augmentation pipeline with multiple augmentation techniques."""
        transforms = []
        
        # Resize first
        transforms.append(A.Resize(self.image_size, self.image_size))
        
        # Random crop with resize
        if self.config.get('random_crop', {}).get('enabled', True):
            scale = self.config.get('random_crop', {}).get('scale', [0.8, 1.0])
            ratio = self.config.get('random_crop', {}).get('ratio', [0.9, 1.1])
            transforms.append(
                A.RandomResizedCrop(
                    height=self.image_size,
                    width=self.image_size,
                    scale=scale,
                    ratio=ratio,
                    p=0.5
                )
            )
        
        # Horizontal flip
        if self.config.get('horizontal_flip', {}).get('enabled', True):
            prob = self.config.get('horizontal_flip', {}).get('probability', 0.5)
            transforms.append(A.HorizontalFlip(p=prob))
        
        # Color jitter - brightness, contrast, saturation, hue
        if self.config.get('color_jitter', {}).get('enabled', True):
            brightness = self.config.get('color_jitter', {}).get('brightness', 0.2)
            contrast = self.config.get('color_jitter', {}).get('contrast', 0.2)
            saturation = self.config.get('color_jitter', {}).get('saturation', 0.2)
            hue = self.config.get('color_jitter', {}).get('hue', 0.1)
            transforms.append(
                A.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    p=0.5
                )
            )
        
        # Gaussian blur - simulate different camera qualities
        if self.config.get('gaussian_blur', {}).get('enabled', True):
            kernel_sizes = self.config.get('gaussian_blur', {}).get('kernel_sizes', [3, 5, 7])
            sigma = self.config.get('gaussian_blur', {}).get('sigma', [0.1, 2.0])
            # Convert kernel_sizes to blur_limit (min, max)
            blur_limit = (min(kernel_sizes), max(kernel_sizes))
            transforms.append(
                A.GaussianBlur(
                    blur_limit=blur_limit,
                    sigma_limit=sigma,
                    p=0.3
                )
            )
        
        # JPEG compression - simulate different compression artifacts
        if self.config.get('jpeg_compression', {}).get('enabled', True):
            quality_lower = self.config.get('jpeg_compression', {}).get('quality_lower', 70)
            quality_upper = self.config.get('jpeg_compression', {}).get('quality_upper', 100)
            transforms.append(
                A.ImageCompression(
                    quality_lower=quality_lower,
                    quality_upper=quality_upper,
                    compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                    p=0.5
                )
            )
        
        # Gaussian noise - improve robustness to noise
        if self.config.get('gaussian_noise', {}).get('enabled', True):
            var_limit = self.config.get('gaussian_noise', {}).get('var_limit', [10.0, 50.0])
            transforms.append(
                A.GaussNoise(
                    var_limit=var_limit,
                    p=0.3
                )
            )
        
        # Resolution scaling - simulate different resolutions
        if self.config.get('resolution_scaling', {}).get('enabled', True):
            scale_limit = self.config.get('resolution_scaling', {}).get('scale_limit', [0.8, 1.2])
            # RandomScale expects scale_limit as a tuple (lower, upper) relative to 1.0
            # e.g., [-0.2, 0.2] for 0.8 to 1.2
            lower_bound = scale_limit[0] - 1.0
            upper_bound = scale_limit[1] - 1.0
            transforms.append(
                A.RandomScale(
                    scale_limit=(lower_bound, upper_bound),
                    interpolation=cv2.INTER_LINEAR,
                    p=0.3
                )
            )
        
        # Normalize with ImageNet statistics
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
        
        # Convert to PyTorch tensor
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def __call__(self, **kwargs):
        """Apply augmentation."""
        return self.transform(**kwargs)


class TestAugmentation:
    """Test-time augmentation (no random operations)."""
    
    def __init__(self, image_size: int = 224):
        """
        Initialize test augmentation pipeline.
        
        Args:
            image_size: Target image size
        """
        self.image_size = image_size
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build test augmentation pipeline."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def __call__(self, **kwargs):
        """Apply augmentation."""
        return self.transform(**kwargs)


def get_augmentation_pipeline(
    augmentation_type: str,
    image_size: int = 224,
    config: Dict[str, Any] = None,
    is_training: bool = True
):
    """
    Factory function to get the appropriate augmentation pipeline.
    
    Args:
        augmentation_type: Type of augmentation ('basic', 'robust')
        image_size: Target image size
        config: Configuration dictionary for augmentation parameters
        is_training: Whether it's for training or testing
        
    Returns:
        Augmentation pipeline
    """
    if not is_training:
        return TestAugmentation(image_size)
    
    if augmentation_type == 'basic':
        return BasicAugmentation(image_size)
    elif augmentation_type == 'robust':
        return RobustAugmentation(image_size, config)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
