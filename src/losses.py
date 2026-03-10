"""
Loss functions for image forgery detection
Implements hybrid loss: BCE + Dice + Focal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Handles empty masks safely
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits [B, 1, H, W]
            targets: Binary masks [B, 1, H, W]
        
        Returns:
            Dice loss (scalar)
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # Return Dice loss
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits [B, 1, H, W]
            targets: Binary masks [B, 1, H, W]
        
        Returns:
            Focal loss (scalar)
        """
        # Apply sigmoid
        probs = torch.sigmoid(predictions)
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()


class HybridLoss(nn.Module):
    """
    Hybrid loss combining BCE, Dice, and Focal losses
    """
    
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5,
        pos_weight=None,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # BCE with optional positive class weighting
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Dice loss
        self.dice = DiceLoss()
        
        # Focal loss
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits [B, 1, H, W]
            targets: Binary masks [B, 1, H, W]
        
        Returns:
            Combined loss (scalar)
        """
        # Move pos_weight to same device if needed
        if self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(predictions.device)
        
        # Compute individual losses
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        focal_loss = self.focal(predictions, targets)
        
        # Combine with weights
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )
        
        return total_loss, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'total': total_loss.item()
        }


def get_loss_function(config):
    """
    Create loss function from config
    
    Args:
        config: Configuration object
    
    Returns:
        Loss function
    """
    loss_fn = HybridLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT,
        pos_weight=config.POS_WEIGHT,
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA
    )
    
    return loss_fn


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test individual losses
    dice_loss = DiceLoss()
    dice_val = dice_loss(predictions, targets)
    print(f"Dice loss: {dice_val.item():.4f}")
    
    focal_loss = FocalLoss()
    focal_val = focal_loss(predictions, targets)
    print(f"Focal loss: {focal_val.item():.4f}")
    
    # Test hybrid loss
    hybrid_loss = HybridLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
    total_loss, loss_dict = hybrid_loss(predictions, targets)
    print(f"Hybrid loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test with empty mask
    empty_targets = torch.zeros(batch_size, 1, 256, 256)
    dice_val_empty = dice_loss(predictions, empty_targets)
    print(f"Dice loss (empty mask): {dice_val_empty.item():.4f}")
    
    print("Loss tests passed! ✓")


if __name__ == "__main__":
    test_losses()
