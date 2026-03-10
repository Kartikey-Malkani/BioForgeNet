"""
Image Forgery Detection Pipeline
A modular PyTorch solution for detecting copy-move forgeries in biomedical images
"""

__version__ = "1.0.0"
__author__ = "Kaggle Grandmaster"

# Import key components for easy access
from .config import Config
from .utils import set_seed, get_device
from .models import get_model
from .losses import get_loss_function
from .metrics import compute_all_metrics
from .dataset import ForgeryDataset, prepare_folds
from .augmentations import get_training_augmentation, get_validation_augmentation
from .train import Trainer
from .inference import InferenceEngine, generate_submission
from .postprocess import postprocess_mask
from .rle import mask_to_rle

__all__ = [
    'Config',
    'set_seed',
    'get_device',
    'get_model',
    'get_loss_function',
    'compute_all_metrics',
    'ForgeryDataset',
    'prepare_folds',
    'get_training_augmentation',
    'get_validation_augmentation',
    'Trainer',
    'InferenceEngine',
    'generate_submission',
    'postprocess_mask',
    'mask_to_rle',
]
