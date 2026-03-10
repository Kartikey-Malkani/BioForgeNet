"""
Augmentation pipeline using Albumentations
Separate pipelines for training, validation, and test
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation(image_size=512, augmentation_level="medium"):
    """
    Get training augmentation pipeline
    
    Args:
        image_size: Target image size
        augmentation_level: "light", "medium", or "heavy"
    
    Returns:
        Albumentations composition
    """
    
    if augmentation_level == "light":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif augmentation_level == "medium":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05, p=1.0),
                A.GridDistortion(distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            ], p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif augmentation_level == "heavy":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT,
                p=0.6
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 70.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=9, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=1.0),
                A.GridDistortion(distort_limit=0.1, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=12,
                max_height=48,
                max_width=48,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.4
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    return transform


def get_validation_augmentation(image_size=512):
    """
    Get validation/test augmentation pipeline (only resize and normalize)
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations composition
    """
    transform = A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    return transform


def get_tta_augmentation(augmentation_type, image_size=512):
    """
    Get specific TTA augmentation
    
    Args:
        augmentation_type: Type of augmentation ("original", "hflip", "vflip", "rotate90")
        image_size: Target image size
    
    Returns:
        Albumentations composition
    """
    
    if augmentation_type == "original":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif augmentation_type == "horizontal_flip":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif augmentation_type == "vertical_flip":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.VerticalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif augmentation_type == "rotate_90":
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Rotate(limit=(90, 90), p=1.0, border_mode=cv2.BORDER_REFLECT),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError(f"Unknown TTA augmentation type: {augmentation_type}")
    
    return transform


def tta_inverse_transform(prediction, augmentation_type):
    """
    Apply inverse transformation to TTA prediction
    
    Args:
        prediction: Prediction tensor of shape [C, H, W] or [B, C, H, W]
        augmentation_type: Type of augmentation applied
    
    Returns:
        Inversely transformed prediction
    """
    import torch
    
    if augmentation_type == "original":
        return prediction
    
    elif augmentation_type == "horizontal_flip":
        return torch.flip(prediction, dims=[-1])  # Flip along width
    
    elif augmentation_type == "vertical_flip":
        return torch.flip(prediction, dims=[-2])  # Flip along height
    
    elif augmentation_type == "rotate_90":
        # Rotate back by -90 degrees (equivalent to 3 * 90 degrees)
        return torch.rot90(prediction, k=3, dims=[-2, -1])
    
    else:
        raise ValueError(f"Unknown TTA augmentation type: {augmentation_type}")
