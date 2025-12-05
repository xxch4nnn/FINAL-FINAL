"""
cleaner.py
Implements data cleaning, outlier detection, and smoothing for piano motion data.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt

def sanity_check_velocity(points_3d: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Filters out frames where movement exceeds human limits (teleportation).

    Args:
        points_3d: (N, 21, 3) array of 3D coordinates.
        threshold: Max allowed displacement per frame (meters/unit).

    Returns:
        Cleaned points_3d (N, 21, 3) with bad frames replaced by NaN (or interpolated).
    """
    if len(points_3d) < 2:
        return points_3d

    # Calculate displacement
    diff = np.linalg.norm(np.diff(points_3d, axis=0), axis=2) # (N-1, 21)

    # Check max displacement across all joints
    max_diff = np.max(diff, axis=1) # (N-1,)

    # Identify bad frames (indices in diff are i, i+1)
    # If i->i+1 jump is huge, i+1 is likely bad (or i was bad and we returned)
    # Simple strategy: Mask invalid frames
    mask = max_diff > threshold

    # We create a copy
    cleaned = points_3d.copy()

    # Naive repair: if jump is too big, hold previous frame
    # (In a real pipeline, we might drop the frame, but for fixed-size array processing
    #  holding or interpolating is better to keep shapes consistent if needed,
    #  though for training we can just drop rows)
    # Here we will mark as NaN and interpolate later, or just drop.
    # Let's just drop them? But this breaks the sequence for `labeler`.
    # Better: Replace with previous valid frame.

    count = 0
    for i in range(len(mask)):
        if mask[i]:
            cleaned[i+1] = cleaned[i]
            count += 1

    if count > 0:
        print(f"  [Cleaner] Repaired {count} frames with excessive velocity.")

    return cleaned

def smooth_data(points_3d: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Applies Savitzky-Golay filter to smooth jitter.
    """
    if len(points_3d) < window_length:
        return points_3d

    try:
        # Smooth each axis
        smoothed = savgol_filter(points_3d, window_length=window_length, polyorder=polyorder, axis=0)
        return smoothed
    except Exception as e:
        print(f"  [Cleaner] Smoothing failed: {e}")
        return points_3d

def remove_zeros(points_3d: np.ndarray) -> np.ndarray:
    """
    Detects frames where all points are exactly (0,0,0) - MediaPipe failure.
    Replaces them with previous valid frame.
    """
    cleaned = points_3d.copy()
    count = 0

    for i in range(len(cleaned)):
        # Check if wrist (0) is (0,0,0) -> good proxy for full failure
        if np.allclose(cleaned[i, 0], [0, 0, 0]):
            if i > 0:
                cleaned[i] = cleaned[i-1]
            count += 1

    if count > 0:
        print(f"  [Cleaner] Repaired {count} zero-frames.")

    return cleaned

def process_sequence(points_3d: np.ndarray) -> np.ndarray:
    """
    Main entry point for cleaning a sequence of 3D points.
    Order: Remove Zeros -> Sanity Check -> Smooth
    """
    points_3d = remove_zeros(points_3d)
    points_3d = sanity_check_velocity(points_3d, threshold=0.1) # 10cm/frame is huge @ 30fps
    points_3d = smooth_data(points_3d)
    return points_3d
