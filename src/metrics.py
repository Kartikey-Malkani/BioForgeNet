"""
Metrics for evaluating image forgery detection
Implements Dice, IoU, F1, Precision, Recall for binary segmentation
"""

import torch
import numpy as np
from typing import Tuple


def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient
    
    Args:
        predictions: Predictions [B, 1, H, W] (logits or probabilities)
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient (scalar)
    """
    # Apply sigmoid if needed (check if values are in [0, 1])
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Binarize predictions
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute Dice
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice.item()


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute Intersection over Union (IoU)
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        IoU score (scalar)
    """
    # Apply sigmoid if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Binarize
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute IoU
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute pixel-wise accuracy
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
    
    Returns:
        Accuracy (scalar)
    """
    # Apply sigmoid if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Binarize
    predictions = (predictions > threshold).float()
    
    # Compute accuracy
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    accuracy = correct / total
    
    return accuracy.item()


def precision_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute precision (positive predictive value)
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Precision (scalar)
    """
    # Apply sigmoid if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Binarize
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute precision
    true_positive = (predictions * targets).sum()
    predicted_positive = predictions.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    
    return precision.item()


def recall_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute recall (sensitivity, true positive rate)
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Recall (scalar)
    """
    # Apply sigmoid if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Binarize
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute recall
    true_positive = (predictions * targets).sum()
    actual_positive = targets.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)
    
    return recall.item()


def f1_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute F1 score (harmonic mean of precision and recall)
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        F1 score (scalar)
    """
    precision = precision_score(predictions, targets, threshold, smooth)
    recall = recall_score(predictions, targets, threshold, smooth)
    
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    return f1


def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Compute all metrics at once
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        threshold: Threshold for binarization
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'dice': dice_coefficient(predictions, targets, threshold),
        'iou': iou_score(predictions, targets, threshold),
        'accuracy': pixel_accuracy(predictions, targets, threshold),
        'precision': precision_score(predictions, targets, threshold),
        'recall': recall_score(predictions, targets, threshold),
        'f1': f1_score(predictions, targets, threshold)
    }
    
    return metrics


def find_best_threshold(predictions: torch.Tensor, targets: torch.Tensor, 
                       thresholds: list = None, metric: str = 'dice') -> Tuple[float, float]:
    """
    Find best threshold by sweeping values
    
    Args:
        predictions: Predictions [B, 1, H, W]
        targets: Ground truth [B, 1, H, W]
        thresholds: List of thresholds to try
        metric: Metric to optimize ('dice', 'iou', 'f1')
    
    Returns:
        (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        if metric == 'dice':
            score = dice_coefficient(predictions, targets, threshold)
        elif metric == 'iou':
            score = iou_score(predictions, targets, threshold)
        elif metric == 'f1':
            score = f1_score(predictions, targets, threshold)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


class MetricsTracker:
    """
    Track metrics across batches/epochs
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.metrics = {
            'dice': [],
            'iou': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predictions [B, 1, H, W]
            targets: Ground truth [B, 1, H, W]
            threshold: Threshold for binarization
        """
        batch_metrics = compute_all_metrics(predictions, targets, threshold)
        
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self) -> dict:
        """Get average of all accumulated metrics"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def get_summary(self) -> str:
        """Get formatted summary string"""
        avg_metrics = self.get_average()
        summary = " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        return summary


def test_metrics():
    """Test metric functions"""
    print("Testing metrics...")
    
    # Create dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 256, 256)  # Logits
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test individual metrics
    dice = dice_coefficient(predictions, targets)
    print(f"Dice: {dice:.4f}")
    
    iou = iou_score(predictions, targets)
    print(f"IoU: {iou:.4f}")
    
    acc = pixel_accuracy(predictions, targets)
    print(f"Accuracy: {acc:.4f}")
    
    precision = precision_score(predictions, targets)
    print(f"Precision: {precision:.4f}")
    
    recall = recall_score(predictions, targets)
    print(f"Recall: {recall:.4f}")
    
    f1 = f1_score(predictions, targets)
    print(f"F1: {f1:.4f}")
    
    # Test all metrics
    all_metrics = compute_all_metrics(predictions, targets)
    print(f"All metrics: {all_metrics}")
    
    # Test threshold tuning
    best_thr, best_score = find_best_threshold(predictions, targets, metric='dice')
    print(f"Best threshold: {best_thr:.2f} (Dice: {best_score:.4f})")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(predictions, targets)
    tracker.update(predictions, targets)
    avg_metrics = tracker.get_average()
    print(f"Tracked metrics: {avg_metrics}")
    print(f"Summary: {tracker.get_summary()}")
    
    print("Metrics tests passed! ✓")


if __name__ == "__main__":
    test_metrics()
