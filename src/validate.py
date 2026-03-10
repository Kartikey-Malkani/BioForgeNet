"""
Validation utilities including threshold tuning
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import get_model
from dataset import ForgeryDataset, get_dataloader
from augmentations import get_validation_augmentation
from metrics import MetricsTracker, find_best_threshold, compute_all_metrics
from utils import get_device, load_checkpoint


def validate_model(
    model,
    val_loader,
    device,
    threshold: float = 0.5
) -> dict:
    """
    Validate model on validation set
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run on
        threshold: Prediction threshold
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics_tracker.update(outputs, masks, threshold=threshold)
    
    avg_metrics = metrics_tracker.get_average()
    
    return avg_metrics


def tune_threshold(
    model,
    val_loader,
    device,
    thresholds: list = None,
    metric: str = 'dice'
) -> tuple:
    """
    Tune threshold on validation set
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run on
        thresholds: List of thresholds to try
        metric: Metric to optimize
    
    Returns:
        (best_threshold, best_score, all_results)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\nTuning threshold on {metric}...")
    print(f"Trying thresholds: {thresholds}")
    
    model.eval()
    
    # Collect all predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(masks.cpu())
    
    # Concatenate all
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Try different thresholds
    results = {}
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        metrics = compute_all_metrics(all_predictions, all_targets, threshold)
        score = metrics[metric]
        results[threshold] = metrics
        
        print(f"  Threshold {threshold:.2f}: {metric}={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"\n✓ Best threshold: {best_threshold:.2f} ({metric}={best_score:.4f})")
    
    return best_threshold, best_score, results


def validate_fold(
    fold: int,
    config,
    tune_thr: bool = True
):
    """
    Validate a single fold model
    
    Args:
        fold: Fold index
        config: Configuration object
        tune_thr: Whether to tune threshold
    
    Returns:
        Validation results
    """
    device = get_device()
    
    # Load fold split
    from dataset import prepare_folds
    folds = prepare_folds(
        image_dir=config.TRAIN_IMAGES_DIR,
        mask_dir=config.TRAIN_MASKS_DIR,
        n_folds=config.N_FOLDS,
        seed=config.SEED,
        stratify=config.STRATIFY
    )
    
    train_ids, val_ids = folds[fold]
    
    # Create validation dataset
    val_transform = get_validation_augmentation(config.IMAGE_SIZE)
    val_dataset = ForgeryDataset(
        image_dir=config.TRAIN_IMAGES_DIR,
        mask_dir=config.TRAIN_MASKS_DIR,
        case_ids=val_ids,
        transform=val_transform,
        mode="val"
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    checkpoint_path = config.CHECKPOINT_DIR / f"best_fold{fold}.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return None
    
    model = get_model(
        architecture=config.ARCHITECTURE,
        encoder_name=config.ENCODER_NAME,
        encoder_weights=None,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES
    )
    
    checkpoint_info = load_checkpoint(checkpoint_path, model)
    model.to(device)
    
    print(f"\nValidating Fold {fold}")
    print(f"Checkpoint epoch: {checkpoint_info['epoch']}")
    
    # Validate with default threshold
    print(f"\nValidation with threshold={config.THRESHOLD}:")
    metrics = validate_model(model, val_loader, device, threshold=config.THRESHOLD)
    
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Tune threshold if requested
    best_threshold = config.THRESHOLD
    tuned_metrics = metrics
    
    if tune_thr:
        best_threshold, best_score, all_results = tune_threshold(
            model, val_loader, device,
            thresholds=config.THRESHOLD_RANGE,
            metric='dice'
        )
        tuned_metrics = all_results[best_threshold]
    
    return {
        'fold': fold,
        'default_metrics': metrics,
        'best_threshold': best_threshold,
        'tuned_metrics': tuned_metrics
    }


def validate_all_folds(config, tune_thr: bool = True):
    """
    Validate all trained folds
    
    Args:
        config: Configuration object
        tune_thr: Whether to tune thresholds
    
    Returns:
        List of validation results
    """
    print("\n" + "="*60)
    print("VALIDATING ALL FOLDS")
    print("="*60)
    
    results = []
    
    for fold in config.TRAIN_FOLDS:
        result = validate_fold(fold, config, tune_thr)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    mean_dice = np.mean([r['tuned_metrics']['dice'] for r in results])
    mean_threshold = np.mean([r['best_threshold'] for r in results])
    
    for result in results:
        print(f"Fold {result['fold']}: "
              f"Dice={result['tuned_metrics']['dice']:.4f}, "
              f"Threshold={result['best_threshold']:.2f}")
    
    print(f"\nMean Dice: {mean_dice:.4f}")
    print(f"Mean Threshold: {mean_threshold:.2f}")
    print("="*60)
    
    return results


def main():
    """Main validation function"""
    from config import Config
    
    config = Config()
    
    # Validate all folds
    results = validate_all_folds(config, tune_thr=True)
    
    return results


if __name__ == "__main__":
    main()
