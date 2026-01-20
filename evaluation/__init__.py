"""Evaluation module for deepfake detection."""

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    PerformanceMetrics
)
from .evaluate import evaluate_model, cross_dataset_evaluation

__all__ = [
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_roc_curve',
    'PerformanceMetrics',
    'evaluate_model',
    'cross_dataset_evaluation',
]
