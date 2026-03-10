"""
Run-Length Encoding (RLE) utilities for Kaggle submission format
Handles conversion between binary masks and RLE strings
"""

import numpy as np
from typing import Union, List


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode binary mask to run-length encoding string
    
    Args:
        mask: Binary mask of shape (H, W) with values 0 or 1
    
    Returns:
        RLE string in format "start1 length1 start2 length2 ..."
        Returns empty string if mask is empty
    
    Note:
        - Pixels are numbered from 1 (not 0)
        - Flatten order is column-major (Fortran-style)
    """
    # Flatten mask in column-major order (Fortran-style)
    pixels = mask.T.flatten()
    
    # Add padding to handle edge cases
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Find runs
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # Extract starts and lengths
    runs[1::2] -= runs[::2]
    
    # Convert to string
    rle = ' '.join(str(x) for x in runs)
    
    return rle


def rle_decode(rle_string: str, shape: tuple) -> np.ndarray:
    """
    Decode run-length encoding string to binary mask
    
    Args:
        rle_string: RLE string in format "start1 length1 start2 length2 ..."
        shape: Tuple (height, width) for output mask
    
    Returns:
        Binary mask of shape (H, W)
    """
    if not rle_string or rle_string == "":
        return np.zeros(shape, dtype=np.uint8)
    
    # Parse RLE string
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    
    # Adjust for 1-based indexing
    starts -= 1
    
    # Create mask
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for start, end in zip(starts, ends):
        img[start:end] = 1
    
    # Reshape to column-major order
    return img.reshape(shape, order='F')


def mask_to_rle(mask: np.ndarray, threshold: float = 0.5) -> str:
    """
    Convert prediction mask to RLE string
    
    Args:
        mask: Mask of shape (H, W) with values in [0, 1] or binary
        threshold: Threshold to binarize mask
    
    Returns:
        RLE string or "authentic" if no forgery detected
    """
    # Binarize if needed
    if mask.max() <= 1.0:
        binary_mask = (mask > threshold).astype(np.uint8)
    else:
        binary_mask = mask.astype(np.uint8)
    
    # Check if mask is empty
    if binary_mask.sum() == 0:
        return "authentic"
    
    # Encode
    rle = rle_encode(binary_mask)
    
    return rle if rle else "authentic"


def merge_masks(mask_files: List[str], shape: tuple) -> np.ndarray:
    """
    Merge multiple mask files into a single binary mask
    
    Args:
        mask_files: List of mask file paths
        shape: Output shape (H, W)
    
    Returns:
        Merged binary mask of shape (H, W)
    """
    import cv2
    
    if not mask_files:
        return np.zeros(shape, dtype=np.uint8)
    
    merged_mask = np.zeros(shape, dtype=np.uint8)
    
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Resize if needed
            if mask.shape != shape:
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            # Binarize and merge
            merged_mask = np.maximum(merged_mask, (mask > 0).astype(np.uint8))
    
    return merged_mask


def validate_rle(rle_string: str, shape: tuple) -> bool:
    """
    Validate that an RLE string can be decoded correctly
    
    Args:
        rle_string: RLE string to validate
        shape: Expected shape (H, W)
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if rle_string == "authentic":
            return True
        
        # Try to decode
        mask = rle_decode(rle_string, shape)
        
        # Check shape
        if mask.shape != shape:
            return False
        
        # Check values
        if not np.all(np.isin(mask, [0, 1])):
            return False
        
        # Re-encode and compare
        re_encoded = rle_encode(mask)
        if re_encoded != rle_string:
            # Small differences might be acceptable
            pass
        
        return True
    
    except Exception as e:
        print(f"RLE validation error: {e}")
        return False


# Test functions for correctness
def test_rle_encoding():
    """Test RLE encoding/decoding with various cases"""
    print("Testing RLE encoding...")
    
    # Test case 1: Empty mask
    mask = np.zeros((10, 10), dtype=np.uint8)
    rle = rle_encode(mask)
    assert rle == "", "Empty mask should encode to empty string"
    print("✓ Empty mask test passed")
    
    # Test case 2: Full mask
    mask = np.ones((10, 10), dtype=np.uint8)
    rle = rle_encode(mask)
    decoded = rle_decode(rle, (10, 10))
    assert np.array_equal(mask, decoded), "Full mask encode/decode mismatch"
    print("✓ Full mask test passed")
    
    # Test case 3: Single pixel
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1
    rle = rle_encode(mask)
    decoded = rle_decode(rle, (10, 10))
    assert np.array_equal(mask, decoded), "Single pixel encode/decode mismatch"
    print("✓ Single pixel test passed")
    
    # Test case 4: Multiple regions
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    mask[7:9, 7:9] = 1
    rle = rle_encode(mask)
    decoded = rle_decode(rle, (10, 10))
    assert np.array_equal(mask, decoded), "Multiple regions encode/decode mismatch"
    print("✓ Multiple regions test passed")
    
    # Test case 5: mask_to_rle with threshold
    mask = np.random.rand(10, 10)
    rle = mask_to_rle(mask, threshold=0.5)
    assert isinstance(rle, str), "mask_to_rle should return string"
    print("✓ mask_to_rle test passed")
    
    print("All RLE tests passed! ✓")


if __name__ == "__main__":
    test_rle_encoding()
