"""
Dataset classes for Image Forgery Detection
Handles train/val/test splits with proper mask merging
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold


class ForgeryDataset(Dataset):
    """
    Dataset for image forgery detection
    
    Handles:
    - Multiple masks per image (merged into single binary mask)
    - Images without masks (authentic images)
    - Proper resizing and normalization
    """
    
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Optional[Path] = None,
        case_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        """
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks (None for test mode)
            case_ids: List of case IDs to include (None = all)
            transform: Albumentations transformation
            mode: "train", "val", or "test"
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.mode = mode
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        
        # Filter by case_ids if provided
        if case_ids is not None:
            self.image_files = [
                f for f in self.image_files 
                if f.stem in case_ids
            ]
        
        # Build mask mapping (for images with masks)
        self.mask_mapping = {}
        if self.mask_dir and self.mask_dir.exists():
            for img_file in self.image_files:
                case_id = img_file.stem
                # Find all masks for this image
                mask_files = list(self.mask_dir.glob(f"{case_id}_*.tif"))
                if mask_files:
                    self.mask_mapping[case_id] = mask_files
        
        print(f"{mode.capitalize()} dataset:")
        print(f"  Total images: {len(self.image_files)}")
        print(f"  Images with forgery: {len(self.mask_mapping)}")
        print(f"  Authentic images: {len(self.image_files) - len(self.mask_mapping)}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with keys:
            - "image": Tensor [C, H, W]
            - "mask": Tensor [1, H, W] (only for train/val)
            - "case_id": str
        """
        img_file = self.image_files[idx]
        case_id = img_file.stem
        
        # Load image
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or create mask
        if self.mode in ["train", "val"]:
            mask = self._load_mask(case_id, image.shape[:2])
        else:
            # Test mode: no mask needed
            mask = None
        
        # Apply transformations
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                # Ensure mask has channel dimension
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask[None]).float()
        
        # Build result
        result = {
            "image": image,
            "case_id": case_id
        }
        
        if mask is not None:
            result["mask"] = mask.float()
        
        return result
    
    def _load_mask(self, case_id: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load and merge all masks for a given case_id
        
        Args:
            case_id: Image case ID
            image_shape: (height, width) of original image
        
        Returns:
            Binary mask of shape (H, W) with values 0 or 1
        """
        if case_id not in self.mask_mapping:
            # No mask = authentic image
            return np.zeros(image_shape, dtype=np.uint8)
        
        mask_files = self.mask_mapping[case_id]
        merged_mask = np.zeros(image_shape, dtype=np.uint8)
        
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize if needed
                if mask.shape != image_shape:
                    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                # Binarize and merge (union of all masks)
                merged_mask = np.maximum(merged_mask, (mask > 127).astype(np.uint8))
        
        return merged_mask
    
    def get_label(self, idx: int) -> int:
        """
        Get binary label for stratification
        
        Returns:
            0 for authentic, 1 for forged
        """
        case_id = self.image_files[idx].stem
        return 1 if case_id in self.mask_mapping else 0


def prepare_folds(
    image_dir: Path,
    mask_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
    stratify: bool = True
) -> List[Tuple[List[str], List[str]]]:
    """
    Prepare stratified K-fold splits
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        n_folds: Number of folds
        seed: Random seed
        stratify: Whether to stratify by authentic/forged
    
    Returns:
        List of (train_ids, val_ids) tuples for each fold
    """
    # Get all image files
    image_files = sorted(list(Path(image_dir).glob("*.tif")))
    case_ids = [f.stem for f in image_files]
    
    # Determine labels for stratification
    if stratify:
        mask_files_all = list(Path(mask_dir).glob("*.tif"))
        forged_cases = set([f.stem.rsplit('_', 1)[0] for f in mask_files_all])
        labels = [1 if cid in forged_cases else 0 for cid in case_ids]
    else:
        labels = [0] * len(case_ids)
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(case_ids, labels)):
        train_ids = [case_ids[i] for i in train_idx]
        val_ids = [case_ids[i] for i in val_idx]
        
        # Count forged images in each split
        train_forged = sum([labels[i] for i in train_idx])
        val_forged = sum([labels[i] for i in val_idx])
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_ids)} images ({train_forged} forged, {len(train_ids)-train_forged} authentic)")
        print(f"  Val: {len(val_ids)} images ({val_forged} forged, {len(val_ids)-val_forged} authentic)")
        
        folds.append((train_ids, val_ids))
    
    return folds


def create_weighted_sampler(dataset: ForgeryDataset, weight_ratio: float = 2.0) -> WeightedRandomSampler:
    """
    Create weighted sampler to oversample images with forgeries
    
    Args:
        dataset: ForgeryDataset instance
        weight_ratio: How much to oversample forged images (e.g., 2.0 = 2x)
    
    Returns:
        WeightedRandomSampler
    """
    # Get labels
    labels = [dataset.get_label(i) for i in range(len(dataset))]
    
    # Compute weights
    weights = []
    for label in labels:
        if label == 1:  # Forged
            weights.append(weight_ratio)
        else:  # Authentic
            weights.append(1.0)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    num_forged = sum(labels)
    num_authentic = len(labels) - num_forged
    print(f"Weighted sampler created:")
    print(f"  Forged images: {num_forged} (weight={weight_ratio})")
    print(f"  Authentic images: {num_authentic} (weight=1.0)")
    
    return sampler


def get_dataloader(
    dataset: ForgeryDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler: Optional[WeightedRandomSampler] = None
) -> DataLoader:
    """
    Create DataLoader with proper settings
    
    Args:
        dataset: ForgeryDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler is provided)
        num_workers: Number of worker processes
        sampler: Optional WeightedRandomSampler
    
    Returns:
        DataLoader
    """
    if sampler is not None:
        shuffle = False  # Cannot use shuffle with sampler
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return loader


# Test dataset functionality
def test_dataset():
    """Test dataset loading and mask merging"""
    from config import Config
    from augmentations import get_training_augmentation
    
    print("Testing dataset...")
    
    # This would need actual data to run
    # Placeholder for testing structure
    print("Dataset module loaded successfully ✓")


if __name__ == "__main__":
    test_dataset()
