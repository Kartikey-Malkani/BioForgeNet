"""
Utility functions for reproducibility, visualization, and general helpers
"""

import os
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device():
    """Get the best available device (cuda/cpu)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model):
    """Count trainable parameters in a model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


def save_checkpoint(model, optimizer, scheduler, epoch, fold, metrics, filepath):
    """
    Save model checkpoint with training state
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        fold: Current fold number
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint and optionally restore training state
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
    
    Returns:
        Dictionary with epoch, fold, and metrics
    """
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    return {
        'epoch': checkpoint.get('epoch', 0),
        'fold': checkpoint.get('fold', 0),
        'metrics': checkpoint.get('metrics', {})
    }


def visualize_batch(images, masks, predictions=None, num_samples=4, save_path=None):
    """
    Visualize a batch of images with masks and predictions
    
    Args:
        images: Tensor of shape [B, C, H, W]
        masks: Tensor of shape [B, 1, H, W]
        predictions: Tensor of shape [B, 1, H, W] (optional)
        num_samples: Number of samples to visualize
        save_path: Path to save figure (optional)
    """
    num_samples = min(num_samples, images.shape[0])
    num_cols = 3 if predictions is not None else 2
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 5, num_samples * 5))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Convert image to numpy
        image = images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        mask = masks[i, 0].cpu().numpy()
        
        # Plot image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Plot prediction if available
        if predictions is not None:
            pred = predictions[i, 0].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.close()


def overlay_mask_on_image(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay a binary mask on an image
    
    Args:
        image: RGB image (H, W, 3) or (C, H, W) tensor
        mask: Binary mask (H, W) or (1, H, W) tensor
        alpha: Transparency
        color: RGB color tuple for mask
    
    Returns:
        Image with overlay
    """
    # Convert to numpy if tensor
    if torch.is_tensor(image):
        image = image.cpu().permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    
    if torch.is_tensor(mask):
        mask = mask.cpu().squeeze().numpy()
    
    # Ensure mask is uint8
    mask = (mask * 255).astype(np.uint8)
    
    # Create color overlay
    overlay = image.copy()
    overlay[mask > 0] = color
    
    # Blend
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_training_info(epoch, epochs, batch_idx, num_batches, loss, lr, metrics=None):
    """Print formatted training information"""
    info = f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{num_batches}] "
    info += f"Loss: {loss:.4f} LR: {lr:.6f}"
    
    if metrics:
        for key, value in metrics.items():
            info += f" {key}: {value:.4f}"
    
    print(info)


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CPU mode"
