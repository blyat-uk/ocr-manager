"""Image processing utilities for brightness threshold previews."""
import cv2
import numpy as np


def apply_brightness_threshold(img: np.ndarray, threshold: int) -> np.ndarray:
    """Apply VideOCR brightness masking logic.

    This replicates the brightness threshold used by VideOCR to preview
    what characters will be visible at different brightness levels.

    Args:
        img: Input image as numpy array (BGR format from cv2)
        threshold: Brightness threshold (0-255)

    Returns:
        Masked image where only pixels above threshold are visible
    """
    t = max(0, min(255, int(threshold)))
    mask = cv2.inRange(img, (t, t, t), (255, 255, 255))
    return cv2.bitwise_and(img, img, mask=mask)
