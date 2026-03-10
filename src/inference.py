"""
Inference pipeline with Test Time Augmentation (TTA) and ensemble support
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

from config import Config
from models import get_model, load_model_checkpoint
from dataset import ForgeryDataset, get_dataloader
from augmentations import get_validation_augmentation, get_tta_augmentation, tta_inverse_transform
from postprocess import postprocess_mask
from rle import mask_to_rle
from utils import get_device


class InferenceEngine:
    """
    Inference engine with TTA and ensemble support
    """
    
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device else get_device()
        self.models = []
    
    def load_models(self, checkpoint_paths: List[Path]):
        """
        Load models from checkpoints
        
        Args:
            checkpoint_paths: List of checkpoint file paths
        """
        print(f"Loading {len(checkpoint_paths)} models...")
        
        for checkpoint_path in checkpoint_paths:
            model = get_model(
                architecture=self.config.ARCHITECTURE,
                encoder_name=self.config.ENCODER_NAME,
                encoder_weights=None,
                in_channels=self.config.IN_CHANNELS,
                classes=self.config.NUM_CLASSES
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            print(f"  ✓ Loaded: {checkpoint_path.name}")
        
        print(f"Total models loaded: {len(self.models)}")
    
    def predict_single(self, image: torch.Tensor, use_tta: bool = False) -> np.ndarray:
        """
        Predict mask for a single image
        
        Args:
            image: Image tensor [C, H, W] or [1, C, H, W]
            use_tta: Whether to use test time augmentation
        
        Returns:
            Predicted mask [H, W] as probabilities
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        if use_tta and self.config.USE_TTA:
            # Predict with TTA
            prediction = self._predict_with_tta(image)
        else:
            # Simple prediction (ensemble of models)
            prediction = self._predict_ensemble(image)
        
        # Convert to numpy
        prediction = prediction.squeeze().cpu().numpy()
        
        return prediction
    
    def _predict_ensemble(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict using ensemble of models
        
        Args:
            image: Image tensor [1, C, H, W]
        
        Returns:
            Averaged prediction [1, 1, H, W]
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(image)
                # Apply sigmoid to get probabilities
                output = torch.sigmoid(output)
                predictions.append(output)
        
        # Average predictions
        if self.config.ENSEMBLE_METHOD == "mean":
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        elif self.config.ENSEMBLE_METHOD == "max":
            ensemble_pred = torch.stack(predictions).max(dim=0)[0]
        else:
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    def _predict_with_tta(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict with test time augmentation
        
        Args:
            image: Image tensor [1, C, H, W]
        
        Returns:
            Averaged TTA prediction [1, 1, H, W]
        """
        tta_predictions = []
        
        # Original image
        with torch.no_grad():
            pred = self._predict_ensemble(image)
            tta_predictions.append(pred)
        
        # Horizontal flip
        if "horizontal_flip" in self.config.TTA_AUGMENTATIONS:
            flipped = torch.flip(image, dims=[-1])
            with torch.no_grad():
                pred = self._predict_ensemble(flipped)
                pred = torch.flip(pred, dims=[-1])  # Flip back
                tta_predictions.append(pred)
        
        # Vertical flip
        if "vertical_flip" in self.config.TTA_AUGMENTATIONS:
            flipped = torch.flip(image, dims=[-2])
            with torch.no_grad():
                pred = self._predict_ensemble(flipped)
                pred = torch.flip(pred, dims=[-2])  # Flip back
                tta_predictions.append(pred)
        
        # Rotate 90
        if "rotate_90" in self.config.TTA_AUGMENTATIONS:
            rotated = torch.rot90(image, k=1, dims=[-2, -1])
            with torch.no_grad():
                pred = self._predict_ensemble(rotated)
                pred = torch.rot90(pred, k=-1, dims=[-2, -1])  # Rotate back
                tta_predictions.append(pred)
        
        # Average all TTA predictions
        tta_pred = torch.stack(tta_predictions).mean(dim=0)
        
        return tta_pred
    
    def predict_test_set(
        self,
        test_dataset: ForgeryDataset,
        batch_size: int = 4,
        use_tta: bool = False,
        save_predictions: bool = False
    ) -> pd.DataFrame:
        """
        Predict on entire test set and generate submission file
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for inference
            use_tta: Whether to use TTA
            save_predictions: Whether to save prediction masks
        
        Returns:
            DataFrame with case_id and prediction columns
        """
        print("\nStarting inference on test set...")
        
        # Create dataloader
        test_loader = get_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        results = []
        
        # Inference loop
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch['image']
            case_ids = batch['case_id']
            
            # Predict for each image in batch
            for i in range(len(images)):
                image = images[i]
                case_id = case_ids[i]
                
                # Predict
                pred_mask = self.predict_single(image, use_tta=use_tta)
                
                # Postprocess
                processed_mask = postprocess_mask(
                    pred_mask,
                    threshold=self.config.THRESHOLD,
                    min_area=self.config.MIN_AREA,
                    use_morphology=self.config.USE_MORPHOLOGY,
                    morph_kernel_size=self.config.MORPH_KERNEL_SIZE
                )
                
                # Convert to RLE
                rle_string = mask_to_rle(processed_mask, threshold=0.5)
                
                # Save prediction if requested
                if save_predictions:
                    pred_dir = self.config.VISUALIZATION_DIR / "predictions"
                    pred_dir.mkdir(exist_ok=True, parents=True)
                    pred_path = pred_dir / f"{case_id}.png"
                    cv2.imwrite(str(pred_path), (processed_mask * 255).astype(np.uint8))
                
                results.append({
                    'case_id': case_id,
                    'prediction': rle_string
                })
        
        # Create DataFrame
        submission_df = pd.DataFrame(results)
        
        print(f"\nInference completed: {len(results)} predictions")
        print(f"Authentic: {(submission_df['prediction'] == 'authentic').sum()}")
        print(f"Forged: {(submission_df['prediction'] != 'authentic').sum()}")
        
        return submission_df
    
    def create_submission(
        self,
        test_image_dir: Path,
        output_path: Path,
        use_tta: bool = True
    ):
        """
        Create submission file for Kaggle
        
        Args:
            test_image_dir: Directory containing test images
            output_path: Path to save submission.csv
            use_tta: Whether to use TTA
        """
        # Create test dataset
        test_transform = get_validation_augmentation(
            image_size=self.config.IMAGE_SIZE
        )
        
        test_dataset = ForgeryDataset(
            image_dir=test_image_dir,
            mask_dir=None,
            transform=test_transform,
            mode="test"
        )
        
        # Predict
        submission_df = self.predict_test_set(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            use_tta=use_tta,
            save_predictions=True
        )
        
        # Save submission
        submission_df.to_csv(output_path, index=False)
        print(f"\n✓ Submission saved: {output_path}")
        
        # Show sample
        print("\nSample predictions:")
        print(submission_df.head(10))
        
        return submission_df


def generate_submission(
    config,
    fold_indices: Optional[List[int]] = None,
    use_tta: bool = True
):
    """
    Generate submission file using trained models
    
    Args:
        config: Configuration object
        fold_indices: Which folds to use for ensemble (None = all)
        use_tta: Whether to use TTA
    """
    # Setup
    device = get_device()
    
    # Get checkpoint paths
    if fold_indices is None:
        fold_indices = config.ENSEMBLE_FOLDS
    
    checkpoint_paths = [
        config.CHECKPOINT_DIR / f"best_fold{fold}.pth"
        for fold in fold_indices
    ]
    
    # Check if checkpoints exist
    existing_paths = [p for p in checkpoint_paths if p.exists()]
    
    if not existing_paths:
        print("Error: No trained models found!")
        print(f"Expected checkpoints in: {config.CHECKPOINT_DIR}")
        return None
    
    print(f"Using {len(existing_paths)} fold models for ensemble")
    
    # Create inference engine
    engine = InferenceEngine(config, device)
    engine.load_models(existing_paths)
    
    # Generate submission
    submission_path = config.SUBMISSION_DIR / config.SUBMISSION_FILE
    
    submission_df = engine.create_submission(
        test_image_dir=config.TEST_IMAGES_DIR,
        output_path=submission_path,
        use_tta=use_tta
    )
    
    return submission_df


def main():
    """Main inference function"""
    from config import Config
    
    # Load config
    config = Config()
    
    # Update for Kaggle if needed
    # config.update_for_kaggle()
    
    # Generate submission
    submission_df = generate_submission(
        config=config,
        fold_indices=None,  # Use all folds
        use_tta=True
    )
    
    print("\nInference completed!")
    return submission_df


if __name__ == "__main__":
    main()
