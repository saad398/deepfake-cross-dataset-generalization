"""PyTorch Dataset implementations for deepfake detection."""

import os
from typing import Optional, Callable, Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class DeepfakeDataset(Dataset):
    """Base class for deepfake detection datasets."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        """
        Initialize deepfake dataset.
        
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to be applied on images
            image_size: Target image size
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.samples = []
        self.labels = []
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label) where label is 0 for real, 1 for fake
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        else:
            # Default: resize and convert to tensor
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def _load_samples(self):
        """Load samples from the dataset directory. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_samples method")


class FaceForensicsDataset(DeepfakeDataset):
    """FaceForensics++ dataset loader."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        manipulation_types: Optional[List[str]] = None
    ):
        """
        Initialize FaceForensics++ dataset.
        
        Args:
            root_dir: Root directory of FaceForensics++ dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to be applied on images
            image_size: Target image size
            manipulation_types: List of manipulation types to include
                                (e.g., ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'])
        """
        super().__init__(root_dir, split, transform, image_size)
        self.manipulation_types = manipulation_types or ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self._load_samples()
    
    def _load_samples(self):
        """Load FaceForensics++ samples."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            print(f"Warning: FaceForensics++ {self.split} directory not found at {split_dir}")
            return
        
        # Load real samples
        real_dir = split_dir / 'real'
        if real_dir.exists():
            for img_file in real_dir.glob('*.png'):
                self.samples.append(str(img_file))
                self.labels.append(0)  # 0 for real
            for img_file in real_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(0)
        
        # Load fake samples from different manipulation types
        for manip_type in self.manipulation_types:
            fake_dir = split_dir / manip_type
            if fake_dir.exists():
                for img_file in fake_dir.glob('*.png'):
                    self.samples.append(str(img_file))
                    self.labels.append(1)  # 1 for fake
                for img_file in fake_dir.glob('*.jpg'):
                    self.samples.append(str(img_file))
                    self.labels.append(1)


class CelebDFDataset(DeepfakeDataset):
    """Celeb-DF dataset loader."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        """
        Initialize Celeb-DF dataset.
        
        Args:
            root_dir: Root directory of Celeb-DF dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to be applied on images
            image_size: Target image size
        """
        super().__init__(root_dir, split, transform, image_size)
        self._load_samples()
    
    def _load_samples(self):
        """Load Celeb-DF samples."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            print(f"Warning: Celeb-DF {self.split} directory not found at {split_dir}")
            return
        
        # Load real samples (YouTube-real)
        real_dir = split_dir / 'real'
        if real_dir.exists():
            for img_file in real_dir.glob('*.png'):
                self.samples.append(str(img_file))
                self.labels.append(0)
            for img_file in real_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(0)
        
        # Load fake samples (Celeb-synthesis)
        fake_dir = split_dir / 'fake'
        if fake_dir.exists():
            for img_file in fake_dir.glob('*.png'):
                self.samples.append(str(img_file))
                self.labels.append(1)
            for img_file in fake_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(1)


class DFDCDataset(DeepfakeDataset):
    """DFDC (Deepfake Detection Challenge) dataset loader."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        """
        Initialize DFDC dataset.
        
        Args:
            root_dir: Root directory of DFDC dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to be applied on images
            image_size: Target image size
        """
        super().__init__(root_dir, split, transform, image_size)
        self._load_samples()
    
    def _load_samples(self):
        """Load DFDC samples."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            print(f"Warning: DFDC {self.split} directory not found at {split_dir}")
            return
        
        # Load real samples
        real_dir = split_dir / 'real'
        if real_dir.exists():
            for img_file in real_dir.glob('*.png'):
                self.samples.append(str(img_file))
                self.labels.append(0)
            for img_file in real_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(0)
        
        # Load fake samples
        fake_dir = split_dir / 'fake'
        if fake_dir.exists():
            for img_file in fake_dir.glob('*.png'):
                self.samples.append(str(img_file))
                self.labels.append(1)
            for img_file in fake_dir.glob('*.jpg'):
                self.samples.append(str(img_file))
                self.labels.append(1)


def get_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    image_size: int = 224
) -> DeepfakeDataset:
    """
    Factory function to get the appropriate dataset.
    
    Args:
        dataset_name: Name of the dataset ('FaceForensics++', 'Celeb-DF', 'DFDC')
        root_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transform to be applied on images
        image_size: Target image size
        
    Returns:
        Appropriate dataset instance
    """
    datasets = {
        'FaceForensics++': FaceForensicsDataset,
        'Celeb-DF': CelebDFDataset,
        'DFDC': DFDCDataset,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    
    return datasets[dataset_name](root_dir, split, transform, image_size)
