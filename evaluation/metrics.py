"""Metrics computation for deepfake detection evaluation."""

from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    
    def __str__(self):
        """String representation of metrics."""
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1-Score: {self.f1_score:.4f}\n"
            f"ROC-AUC: {self.roc_auc:.4f}"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'roc_auc': float(self.roc_auc),
            'confusion_matrix': self.confusion_matrix.tolist()
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> PerformanceMetrics:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        
    Returns:
        PerformanceMetrics object containing all metrics
    """
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # ROC-AUC (requires probability scores)
    if y_prob is not None:
        # If y_prob is 2D (probabilities for both classes), use the positive class
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return PerformanceMetrics(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm
    )


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    # If y_prob is 2D, use the positive class probabilities
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with per-class metrics
    """
    classes = ['Real', 'Fake']
    metrics = {}
    
    for i, class_name in enumerate(classes):
        # Binary mask for current class
        mask = (y_true == i)
        
        if mask.sum() == 0:
            continue
        
        # Compute metrics for this class
        class_precision = precision_score(
            y_true == i,
            y_pred == i,
            zero_division=0
        )
        class_recall = recall_score(
            y_true == i,
            y_pred == i,
            zero_division=0
        )
        class_f1 = f1_score(
            y_true == i,
            y_pred == i,
            zero_division=0
        )
        
        metrics[class_name] = {
            'precision': float(class_precision),
            'recall': float(class_recall),
            'f1_score': float(class_f1),
            'support': int(mask.sum())
        }
    
    return metrics


def compute_performance_drop(
    intra_metrics: PerformanceMetrics,
    cross_metrics: PerformanceMetrics
) -> Dict[str, float]:
    """
    Compute performance drop from intra-dataset to cross-dataset.
    
    Args:
        intra_metrics: Metrics on intra-dataset evaluation
        cross_metrics: Metrics on cross-dataset evaluation
        
    Returns:
        Dictionary with performance drops for each metric
    """
    drops = {
        'accuracy_drop': intra_metrics.accuracy - cross_metrics.accuracy,
        'precision_drop': intra_metrics.precision - cross_metrics.precision,
        'recall_drop': intra_metrics.recall - cross_metrics.recall,
        'f1_drop': intra_metrics.f1_score - cross_metrics.f1_score,
        'roc_auc_drop': intra_metrics.roc_auc - cross_metrics.roc_auc,
        
        # Relative drops (percentage)
        'accuracy_drop_pct': ((intra_metrics.accuracy - cross_metrics.accuracy) / intra_metrics.accuracy * 100) if intra_metrics.accuracy > 0 else 0,
        'precision_drop_pct': ((intra_metrics.precision - cross_metrics.precision) / intra_metrics.precision * 100) if intra_metrics.precision > 0 else 0,
        'recall_drop_pct': ((intra_metrics.recall - cross_metrics.recall) / intra_metrics.recall * 100) if intra_metrics.recall > 0 else 0,
        'f1_drop_pct': ((intra_metrics.f1_score - cross_metrics.f1_score) / intra_metrics.f1_score * 100) if intra_metrics.f1_score > 0 else 0,
        'roc_auc_drop_pct': ((intra_metrics.roc_auc - cross_metrics.roc_auc) / intra_metrics.roc_auc * 100) if intra_metrics.roc_auc > 0 else 0,
    }
    
    return drops


def format_metrics_table(metrics_dict: Dict[str, PerformanceMetrics]) -> str:
    """
    Format metrics as a table string.
    
    Args:
        metrics_dict: Dictionary mapping dataset names to metrics
        
    Returns:
        Formatted table string
    """
    header = f"{'Dataset':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}"
    separator = "=" * len(header)
    
    lines = [separator, header, separator]
    
    for dataset_name, metrics in metrics_dict.items():
        line = (
            f"{dataset_name:<20} "
            f"{metrics.accuracy:<10.4f} "
            f"{metrics.precision:<10.4f} "
            f"{metrics.recall:<10.4f} "
            f"{metrics.f1_score:<10.4f} "
            f"{metrics.roc_auc:<10.4f}"
        )
        lines.append(line)
    
    lines.append(separator)
    
    return "\n".join(lines)
