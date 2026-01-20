"""Evaluation script for deepfake detection models."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data import get_dataset, get_augmentation_pipeline
from models import get_model
from evaluation.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    compute_performance_drop,
    format_metrics_table,
    PerformanceMetrics
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu',
    compute_probabilities: bool = True
) -> tuple:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        compute_probabilities: Whether to compute class probabilities
        
    Returns:
        Tuple of (y_true, y_pred, y_prob)
    """
    model.eval()
    model.to(device)
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            if compute_probabilities:
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities) if compute_probabilities else None
    
    return y_true, y_pred, y_prob


def evaluate_on_dataset(
    model: nn.Module,
    dataset_name: str,
    data_root: str,
    split: str,
    config: dict,
    device: str = 'cpu'
) -> PerformanceMetrics:
    """
    Evaluate model on a specific dataset.
    
    Args:
        model: Model to evaluate
        dataset_name: Name of the dataset
        data_root: Root directory of datasets
        split: Dataset split ('test' or 'val')
        config: Configuration dictionary
        device: Device to use for evaluation
        
    Returns:
        PerformanceMetrics object
    """
    # Create test transform
    test_transform = get_augmentation_pipeline(
        augmentation_type='basic',
        image_size=config['data']['image_size'],
        is_training=False
    )
    
    # Create dataset
    dataset_path = Path(data_root) / dataset_name.replace('+', '')
    dataset = get_dataset(
        dataset_name=dataset_name,
        root_dir=str(dataset_path),
        split=split,
        transform=test_transform,
        image_size=config['data']['image_size']
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nEvaluating on {dataset_name} ({split} split)...")
    print(f"Total samples: {len(dataset)}")
    
    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, data_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return metrics, y_true, y_pred, y_prob


def cross_dataset_evaluation(
    checkpoint_path: str,
    config_path: str,
    output_dir: str = './results'
):
    """
    Perform cross-dataset evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        output_dir: Directory to save results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = get_model(
        model_name=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model'].get('dropout', 0.5)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded successfully")
    
    # Get datasets to evaluate
    train_dataset = config['data']['train_dataset']
    test_datasets = config['data']['test_datasets']
    data_root = config['data']['data_root']
    
    # Evaluate on all datasets
    all_metrics = {}
    all_predictions = {}
    
    for dataset_name in test_datasets:
        try:
            metrics, y_true, y_pred, y_prob = evaluate_on_dataset(
                model=model,
                dataset_name=dataset_name,
                data_root=data_root,
                split='test',
                config=config,
                device=device
            )
            
            all_metrics[dataset_name] = metrics
            all_predictions[dataset_name] = {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist() if y_prob is not None else None
            }
            
            # Print metrics
            print(f"\n{dataset_name} Results:")
            print(metrics)
            
            # Compute per-class metrics
            per_class = compute_per_class_metrics(y_true, y_pred)
            print("\nPer-class metrics:")
            for class_name, class_metrics in per_class.items():
                print(f"  {class_name}:")
                for metric_name, value in class_metrics.items():
                    print(f"    {metric_name}: {value}")
        
        except Exception as e:
            print(f"\nError evaluating on {dataset_name}: {e}")
            continue
    
    # Print summary table
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("="*80)
    print(format_metrics_table(all_metrics))
    
    # Compute performance drops (if trained and tested on different datasets)
    if train_dataset in all_metrics:
        intra_metrics = all_metrics[train_dataset]
        
        print("\n" + "="*80)
        print("PERFORMANCE DROP ANALYSIS")
        print("="*80)
        
        for dataset_name, metrics in all_metrics.items():
            if dataset_name != train_dataset:
                drops = compute_performance_drop(intra_metrics, metrics)
                print(f"\n{train_dataset} → {dataset_name}:")
                print(f"  Accuracy drop: {drops['accuracy_drop']:.4f} ({drops['accuracy_drop_pct']:.2f}%)")
                print(f"  Precision drop: {drops['precision_drop']:.4f} ({drops['precision_drop_pct']:.2f}%)")
                print(f"  Recall drop: {drops['recall_drop']:.4f} ({drops['recall_drop_pct']:.2f}%)")
                print(f"  F1-Score drop: {drops['f1_drop']:.4f} ({drops['f1_drop_pct']:.2f}%)")
                print(f"  ROC-AUC drop: {drops['roc_auc_drop']:.4f} ({drops['roc_auc_drop_pct']:.2f}%)")
    
    # Save results
    results = {
        'train_dataset': train_dataset,
        'test_datasets': test_datasets,
        'metrics': {name: metrics.to_dict() for name, metrics in all_metrics.items()},
        'checkpoint_path': checkpoint_path
    }
    
    results_path = output_path / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save predictions if requested
    if config['evaluation'].get('save_predictions', True):
        predictions_path = output_path / 'predictions.json'
        with open(predictions_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"Predictions saved to {predictions_path}")
    
    return all_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    cross_dataset_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
