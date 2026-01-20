"""Main training script for deepfake detection models."""

import argparse
import random
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_dataset, get_augmentation_pipeline
from models import get_model
from training import Trainer


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict, augmentation_type: str = 'robust'):
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration dictionary
        augmentation_type: Type of augmentation ('basic' or 'robust')
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config['data']
    aug_config = config['augmentation']
    
    # Get augmentation pipelines
    train_transform = get_augmentation_pipeline(
        augmentation_type=augmentation_type,
        image_size=data_config['image_size'],
        config=aug_config,
        is_training=True
    )
    
    val_transform = get_augmentation_pipeline(
        augmentation_type='basic',  # Always use basic for validation
        image_size=data_config['image_size'],
        config=aug_config,
        is_training=False
    )
    
    # Create datasets
    dataset_name = data_config['train_dataset']
    data_root = Path(data_config['data_root']) / dataset_name.replace('+', '')
    
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        root_dir=str(data_root),
        split='train',
        transform=train_transform,
        image_size=data_config['image_size']
    )
    
    val_dataset = get_dataset(
        dataset_name=dataset_name,
        root_dir=str(data_root),
        split='val',
        transform=val_transform,
        image_size=data_config['image_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def train_model(config_path: str, augmentation_type: str = 'robust', model_name: str = None):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        augmentation_type: Type of augmentation ('basic' or 'robust')
        model_name: Model architecture name (overrides config if provided)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Override model if provided
    if model_name is not None:
        config['model']['architecture'] = model_name
    
    # Create data loaders
    print(f"\nCreating data loaders with {augmentation_type} augmentation...")
    train_loader, val_loader = create_data_loaders(config, augmentation_type)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"\nCreating {config['model']['architecture']} model...")
    model = get_model(
        model_name=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model'].get('dropout', 0.5)
    )
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_name = config['training']['optimizer'].lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler_name = config['training']['scheduler'].lower()
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['scheduler_patience'],
            factor=config['training']['scheduler_factor'],
            verbose=True
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Create save directory
    save_dir = Path(config['training']['save_dir']) / f"{config['model']['architecture']}_{augmentation_type}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = Path(config['logging']['tensorboard_dir']) / f"{config['model']['architecture']}_{augmentation_type}"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=str(save_dir),
        log_dir=str(log_dir),
        early_stopping_patience=config['training']['early_stopping_patience'],
        gradient_clip=config['training'].get('gradient_clip'),
        log_interval=config['logging']['log_interval'],
        mixed_precision=config['training'].get('mixed_precision', False)
    )
    
    # Train model
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {config['model']['architecture']}")
    print(f"  Augmentation: {augmentation_type}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Save directory: {save_dir}")
    print(f"{'='*60}\n")
    
    history = trainer.train(num_epochs=config['training']['epochs'])
    
    # Save training history
    import json
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    print("\nTraining completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['basic', 'robust'],
        default='robust',
        help='Augmentation type (basic or robust)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['xception', 'efficientnet'],
        default=None,
        help='Model architecture (overrides config if provided)'
    )
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        augmentation_type=args.augmentation,
        model_name=args.model
    )


if __name__ == '__main__':
    main()
