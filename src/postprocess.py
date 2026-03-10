"""
Postprocessing utilities for predictions
Includes thresholding, morphological operations, and small component removal
"""

import cv2
import numpy as np
from typing import Tuple
from scipy import ndimage


def threshold_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply threshold to convert probabilities to binary mask
    
    Args:
        mask: Probability mask [H, W] with values in [0, 1]
        threshold: Threshold value
    
    Returns:
        Binary mask [H, W] with values 0 or 1
    """
    return (mask > threshold).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove small connected components from binary mask
    
    Args:
        mask: Binary mask [H, W]
        min_area: Minimum component area (in pixels)
    
    Returns:
        Cleaned binary mask
    """
    if mask.sum() == 0:
        return mask
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    # Create output mask
    output_mask = np.zeros_like(mask)
    
    # Keep components larger than min_area (skip background label 0)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output_mask[labels == label] = 1
    
    return output_mask


def morphological_operations(mask: np.ndarray, operation: str = "open", kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological operations to clean mask
    
    Args:
        mask: Binary mask [H, W]
        operation: Type of operation ("open", "close", "dilate", "erode")
        kernel_size: Size of morphological kernel
    
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == "open":
        # Remove small noise
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        # Fill small holes
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == "dilate":
        # Expand regions
        result = cv2.dilate(mask, kernel)
    elif operation == "erode":
        # Shrink regions
        result = cv2.erode(mask, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask
    
    Args:
        mask: Binary mask [H, W]
    
    Returns:
        Mask with holes filled
    """
    # Use scipy ndimage to fill holes
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return filled


def postprocess_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100,
    use_morphology: bool = False,
    morph_kernel_size: int = 3,
    fill_holes_flag: bool = False
) -> np.ndarray:
    """
    Apply full postprocessing pipeline to prediction mask
    
    Args:
        mask: Probability mask [H, W] with values in [0, 1]
        threshold: Threshold for binarization
        min_area: Minimum component area
        use_morphology: Whether to apply morphological operations
        morph_kernel_size: Morphological kernel size
        fill_holes_flag: Whether to fill holes
    
    Returns:
        Postprocessed binary mask
    """
    # Threshold
    binary_mask = threshold_mask(mask, threshold)
    
    # Fill holes if requested
    if fill_holes_flag:
        binary_mask = fill_holes(binary_mask)
    
    # Morphological operations
    if use_morphology:
        # Apply opening to remove noise
        binary_mask = morphological_operations(
            binary_mask, operation="open", kernel_size=morph_kernel_size
        )
        # Apply closing to fill gaps
        binary_mask = morphological_operations(
            binary_mask, operation="close", kernel_size=morph_kernel_size
        )
    
    # Remove small components
    binary_mask = remove_small_components(binary_mask, min_area)
    
    return binary_mask


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component
    
    Args:
        mask: Binary mask [H, W]
    
    Returns:
        Mask with only largest component
    """
    if mask.sum() == 0:
        return mask
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:  # Only background
        return np.zeros_like(mask)
    
    # Find largest component (skip background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create output mask with only largest component
    output_mask = (labels == largest_label).astype(np.uint8)
    
    return output_mask


def adaptive_threshold_postprocess(mask: np.ndarray, percentile: float = 95) -> Tuple[np.ndarray, float]:
    """
    Use adaptive thresholding based on percentile of positive predictions
    
    Args:
        mask: Probability mask [H, W]
        percentile: Percentile for adaptive threshold
    
    Returns:
        (binary_mask, threshold_used)
    """
    # Get non-zero values
    positive_values = mask[mask > 0]
    
    if len(positive_values) == 0:
        return np.zeros_like(mask, dtype=np.uint8), 0.0
    
    # Compute adaptive threshold
    threshold = np.percentile(positive_values, percentile)
    threshold = max(0.3, min(0.9, threshold))  # Clamp to reasonable range
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    return binary_mask, threshold


def smooth_mask_edges(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth mask edges using Gaussian blur
    
    Args:
        mask: Binary mask [H, W]
        sigma: Gaussian blur sigma
    
    Returns:
        Smoothed mask
    """
    if mask.sum() == 0:
        return mask
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), sigma)
    
    # Re-threshold
    smoothed = (blurred > 0.5).astype(np.uint8)
    
    return smoothed


def test_postprocessing():
    """Test postprocessing functions"""
    print("Testing postprocessing...")
    
    # Create dummy mask with noise
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[100:150, 100:150] = 0.8  # Main region
    mask[50:55, 50:55] = 0.9  # Small noise
    mask[200:205, 200:205] = 0.9  # Another small region
    
    print(f"Original mask sum: {mask.sum():.0f}")
    
    # Test thresholding
    binary = threshold_mask(mask, threshold=0.5)
    print(f"After threshold: {binary.sum():.0f} pixels")
    
    # Test small component removal
    cleaned = remove_small_components(binary, min_area=100)
    print(f"After removing small components: {cleaned.sum():.0f} pixels")
    
    # Test morphological operations
    opened = morphological_operations(cleaned, operation="open", kernel_size=3)
    print(f"After morphological open: {opened.sum():.0f} pixels")
    
    # Test full pipeline
    processed = postprocess_mask(
        mask,
        threshold=0.5,
        min_area=100,
        use_morphology=True,
        morph_kernel_size=3
    )
    print(f"After full postprocessing: {processed.sum():.0f} pixels")
    
    # Test largest component
    largest = get_largest_component(binary)
    print(f"Largest component: {largest.sum():.0f} pixels")
    
    print("Postprocessing tests passed! ✓")


if __name__ == "__main__":
    test_postprocessing()
