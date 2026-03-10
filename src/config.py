"""
Central configuration file for Image Forgery Detection Pipeline
All hyperparameters and paths are defined here for easy experimentation
"""

import os
from pathlib import Path


class Config:
    """Configuration class for all training, validation, and inference settings"""
    
    # ============ PATHS ============
    # These paths should be adjusted for Kaggle environment
    DATA_DIR = Path("../input/recod-ai-luc-scientific-image-forgery-detection")
    TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
    TRAIN_MASKS_DIR = DATA_DIR / "train_masks"
    SUPPLEMENTAL_IMAGES_DIR = DATA_DIR / "supplemental_images"
    SUPPLEMENTAL_MASKS_DIR = DATA_DIR / "supplemental_masks"
    TEST_IMAGES_DIR = DATA_DIR / "test_images"
    
    OUTPUT_DIR = Path("./outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    SUBMISSION_DIR = OUTPUT_DIR / "submissions"
    VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
    
    # ============ DATA ============
    IMAGE_SIZE = 512  # Resize all images to this size
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary segmentation
    
    # ============ MODEL ============
    ENCODER_NAME = "efficientnet-b3"  # Options: efficientnet-b{0-7}, resnet{34,50,101}, etc.
    ENCODER_WEIGHTS = "imagenet"  # Use pretrained weights
    ARCHITECTURE = "Unet"  # Options: Unet, UnetPlusPlus, FPN, DeepLabV3Plus
    
    # Advanced architecture options (for Phase 2)
    USE_ATTENTION = False  # Add attention modules (scSE)
    USE_DEEP_SUPERVISION = False
    
    # ============ TRAINING ============
    BATCH_SIZE = 8  # Adjust based on GPU memory (T4 has 16GB)
    NUM_WORKERS = 2  # DataLoader workers
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Optimizer
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    
    # Scheduler
    SCHEDULER = "CosineAnnealingLR"  # Options: CosineAnnealingLR, ReduceLROnPlateau
    T_MAX = 50  # For CosineAnnealingLR
    
    # Training optimization
    USE_AMP = True  # Mixed precision training
    GRADIENT_CLIP_VAL = 1.0
    ACCUMULATION_STEPS = 1  # Gradient accumulation
    
    # ============ CROSS-VALIDATION ============
    N_FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # Which folds to train
    STRATIFY = True  # Stratify by authentic vs forged
    
    # ============ LOSS FUNCTION ============
    # Hybrid loss weights (BCE + Dice + Focal)
    BCE_WEIGHT = 1.0
    DICE_WEIGHT = 1.0
    FOCAL_WEIGHT = 0.5
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Class imbalance handling
    USE_WEIGHTED_SAMPLER = True  # Oversample images with forgeries
    POS_WEIGHT = 10.0  # Weight for positive class in BCE
    
    # ============ POSTPROCESSING ============
    THRESHOLD = 0.5  # Binary threshold for predictions
    MIN_AREA = 100  # Remove connected components smaller than this (pixels)
    USE_MORPHOLOGY = False  # Apply morphological operations
    MORPH_KERNEL_SIZE = 3
    
    # Threshold tuning
    TUNE_THRESHOLD = True
    THRESHOLD_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # ============ TEST TIME AUGMENTATION ============
    USE_TTA = True
    TTA_AUGMENTATIONS = [
        "original",
        "horizontal_flip",
        "vertical_flip",
        "rotate_90"
    ]
    
    # ============ ENSEMBLE ============
    ENSEMBLE_FOLDS = [0, 1, 2, 3, 4]  # Which fold models to ensemble
    ENSEMBLE_METHOD = "mean"  # Options: mean, max
    
    # ============ REPRODUCIBILITY ============
    SEED = 42
    DETERMINISTIC = True  # Use deterministic algorithms (slower but reproducible)
    
    # ============ LOGGING ============
    VERBOSE = True
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_BEST_ONLY = True
    
    # ============ KAGGLE SPECIFIC ============
    KAGGLE_MODE = False  # Set to True when running on Kaggle
    SUBMISSION_FILE = "submission.csv"
    
    @classmethod
    def update_for_kaggle(cls):
        """Update paths and settings for Kaggle environment"""
        cls.KAGGLE_MODE = True
        cls.NUM_WORKERS = 2  # Kaggle kernel optimization
        cls.DATA_DIR = Path("/kaggle/input/datasets/llkh0a/recod-ailuc-scientific-image-forgery-detection")
        cls.TRAIN_IMAGES_DIR = cls.DATA_DIR / "train_images"
        cls.TRAIN_MASKS_DIR = cls.DATA_DIR / "train_masks"
        cls.SUPPLEMENTAL_IMAGES_DIR = cls.DATA_DIR / "supplemental_images"
        cls.SUPPLEMENTAL_MASKS_DIR = cls.DATA_DIR / "supplemental_masks"
        cls.TEST_IMAGES_DIR = cls.DATA_DIR / "test_images"
        cls.OUTPUT_DIR = Path("/kaggle/working")
        cls.CHECKPOINT_DIR = cls.OUTPUT_DIR / "checkpoints"
        cls.SUBMISSION_DIR = cls.OUTPUT_DIR
        cls.VISUALIZATION_DIR = cls.OUTPUT_DIR / "visualizations"
    
    @classmethod
    def create_directories(cls):
        """Create necessary output directories"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        cls.SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)
        cls.VISUALIZATION_DIR.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 60)
