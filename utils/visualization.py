"""Visualization utilities for deepfake detection."""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[2].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = ['Real', 'Fake'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        classes: Class names
        save_path: Path to save the figure
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentage annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = plt.text(
                j + 0.5, i + 0.7,
                f'({cm_normalized[i, j]:.1%})',
                ha='center', va='center',
                fontsize=9, color='red' if cm_normalized[i, j] > 0.5 else 'blue'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'ROC Curve'
):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure
        figsize: Figure size
        title: Plot title
    """
    # If y_prob is 2D, use the positive class probabilities
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()


def plot_cross_dataset_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = 'Cross-Dataset Performance Comparison'
):
    """
    Plot comparison of metrics across different datasets.
    
    Args:
        metrics_dict: Dictionary mapping dataset names to metric dictionaries
        metric_names: List of metric names to plot
        save_path: Path to save the figure
        figsize: Figure size
        title: Plot title
    """
    datasets = list(metrics_dict.keys())
    num_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
    
    for idx, metric_name in enumerate(metric_names):
        values = []
        for dataset in datasets:
            # Handle both PerformanceMetrics objects and dictionaries
            if hasattr(metrics_dict[dataset], metric_name):
                values.append(getattr(metrics_dict[dataset], metric_name))
            else:
                values.append(metrics_dict[dataset].get(metric_name, 0))
        
        axes[idx].bar(datasets, values, color=colors, edgecolor='black', linewidth=1.5)
        axes[idx].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
        axes[idx].set_title(metric_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[idx].set_ylim([0, 1.0])
        axes[idx].grid(True, axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-dataset comparison plot saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
):
    """
    Plot sample predictions with confidence scores.
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        pred_probs: Prediction probabilities
        num_samples: Number of samples to plot
        save_path: Path to save the figure
        figsize: Figure size
    """
    num_samples = min(num_samples, len(images))
    rows = 2
    cols = num_samples // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    class_names = ['Real', 'Fake']
    
    for i in range(num_samples):
        img = images[i]
        
        # Denormalize image if needed (assuming ImageNet normalization)
        if img.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Get prediction info
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        confidence = pred_probs[i, pred_labels[i]] if len(pred_probs.shape) > 1 else pred_probs[i]
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        # Title with true label, prediction, and confidence
        title = f'True: {true_class}\nPred: {pred_class} ({confidence:.2%})'
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to {save_path}")
    
    plt.show()


def plot_baseline_vs_improved(
    baseline_metrics: Dict[str, float],
    improved_metrics: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Baseline vs Improved Model Comparison'
):
    """
    Plot comparison between baseline and improved models.
    
    Args:
        baseline_metrics: Metrics for baseline model
        improved_metrics: Metrics for improved model
        save_path: Path to save the figure
        figsize: Figure size
        title: Plot title
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    baseline_values = [baseline_metrics.get(m, 0) for m in metrics]
    improved_values = [improved_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, improved_values, width, label='Improved', color='lightgreen', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=0)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Baseline vs improved comparison plot saved to {save_path}")
    
    plt.show()
