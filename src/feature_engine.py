"""
feature_engine.py
Unified Feature Extraction Engine for PianoMotion Project.
Ensures congruency between Data Generation (Sync) and Live Runtime.
"""

import numpy as np
import logging
from collections import deque
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple

# --- Constants ---
# Defined here to ensure single source of truth
FEATURE_COLUMNS = [
    # 1. Position (Normalized 2D)
    'finger_pos_x', 'finger_pos_y',
    'wrist_pos_x', 'wrist_pos_y',

    # 2. Velocity
    'finger_vel_x', 'finger_vel_y', 'finger_speed',
    'wrist_vel_x', 'wrist_vel_y', 'wrist_speed',

    # 3. Acceleration
    'finger_acc_x', 'finger_acc_y', 'finger_acc_mag',

    # 4. Relative
    'rel_finger_pos_x', 'rel_finger_pos_y',
    'rel_finger_vel_x', 'rel_finger_vel_y',

    # 5. Distances
    'dist_wrist', 'dist_palm', 'posture_dist',

    # 6. Relative Depth (Z-proxy)
    'rel_depth',

    # 7. Averages
    'avg_speed', 'avg_acc_mag',

    # 8. Lag Features
    'lag_speed_2', 'lag_speed_4', 'lag_speed_6',

    # 9. Rolling Variance
    'rolling_var_speed'
]

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def project_3d_to_2d(points_3d: np.ndarray, cam_w=1920, cam_h=1080) -> np.ndarray:
    """
    Projects 3D points (x, y, z) to Normalized 2D screen coordinates (u_norm, v_norm).
    Uses pinhole camera model with fixed intrinsics.
    Matches SyncPianoMotionDataset logic.

    Args:
        points_3d: (N, 21, 3) array of 3D coordinates.

    Returns:
        (N, 21, 2) array of normalized 2D coordinates [0, 1].
    """
    FX = 1000
    FY = 1000
    CX = 960
    CY = 540

    # Unpack
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    # Avoid division by zero
    z = np.where(z == 0, 1e-6, z)

    # Pinhole Projection
    u = (x / z) * FX + CX
    v = (y / z) * FY + CY

    # Normalize
    u_norm = u / cam_w
    v_norm = v / cam_h

    return np.stack([u_norm, v_norm], axis=-1)

def construct_feature_row(
    tip_p, tip_v, tip_a,
    wrist_p, wrist_v,
    palm_p, dip_p,
    rel_depth,
    avg_speed, avg_acc_mag, rolling_var_speed,
    lags: Dict[int, float]
) -> Dict[str, float]:
    """
    Constructs a single feature dictionary with validated keys.
    Used by both Sync (batch) and Live (stream) extractors.
    """
    row = {}

    # 1. Position (Normalized)
    row['finger_pos_x'] = float(tip_p[0])
    row['finger_pos_y'] = float(tip_p[1])
    row['wrist_pos_x'] = float(wrist_p[0])
    row['wrist_pos_y'] = float(wrist_p[1])

    # 2. Velocity (Normalized/sec)
    row['finger_vel_x'] = float(tip_v[0])
    row['finger_vel_y'] = float(tip_v[1])
    row['finger_speed'] = float(np.linalg.norm(tip_v))

    row['wrist_vel_x'] = float(wrist_v[0])
    row['wrist_vel_y'] = float(wrist_v[1])
    row['wrist_speed'] = float(np.linalg.norm(wrist_v))

    # 3. Acceleration
    row['finger_acc_x'] = float(tip_a[0])
    row['finger_acc_y'] = float(tip_a[1])
    row['finger_acc_mag'] = float(np.linalg.norm(tip_a))

    # 4. Relative (Tip - Wrist)
    rel_pos = tip_p - wrist_p
    rel_vel = tip_v - wrist_v
    row['rel_finger_pos_x'] = float(rel_pos[0])
    row['rel_finger_pos_y'] = float(rel_pos[1])
    row['rel_finger_vel_x'] = float(rel_vel[0])
    row['rel_finger_vel_y'] = float(rel_vel[1])

    # 5. Distances
    row['dist_wrist'] = float(np.linalg.norm(rel_pos))
    row['dist_palm'] = float(np.linalg.norm(tip_p - palm_p))
    row['posture_dist'] = float(np.linalg.norm(tip_p - dip_p)) # Tip to DIP

    # 6. Relative Depth
    row['rel_depth'] = float(rel_depth)

    # 7. Rolling Averages
    row['avg_speed'] = float(avg_speed)
    row['avg_acc_mag'] = float(avg_acc_mag)

    # 8. Lags
    for lag_key, val in lags.items():
        row[f'lag_speed_{lag_key}'] = float(val)

    # 9. Rolling Variance
    row['rolling_var_speed'] = float(rolling_var_speed)

    return row

# --- Live Feature Extractor ---

class LiveFeatureExtractor:
    """
    Real-time Feature Extractor using buffer.
    Designed for use in main_runtime.py.
    """
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.feature_names = FEATURE_COLUMNS
        # Need to handle model loading outside or pass it in?
        # Typically the extractor just extracts. The model usage is separate.
        # But for 'predict' method convenience, we can keep it here or separate.
        # The runtime script had 'predict' inside. We'll keep extraction pure here.

    def update(self, landmarks):
        """
        Push new landmarks to buffer.
        landmarks: MediaPipe NormalizedLandmarkList (or compatible object)
        """
        if not landmarks:
            return

        frame_data = {
            'wrist': np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z]),
            'tip': np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z]),
            'dip': np.array([landmarks[7].x, landmarks[7].y, landmarks[7].z]),
            'palm': np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z]) # Using 9 as proxy
        }
        self.buffer.append(frame_data)

    def extract(self) -> Optional[np.ndarray]:
        """
        Calculates features from buffer.
        Returns: (1, N_FEATURES) array or None if buffering.
        """
        if len(self.buffer) < 5: # Need minimal history for smoothing/vel
            return None

        # Convert buffer to arrays
        # Shape: (T, 3)
        wrist_hist = np.array([d['wrist'] for d in self.buffer])
        tip_hist = np.array([d['tip'] for d in self.buffer])
        dip_hist = np.array([d['dip'] for d in self.buffer])
        palm_hist = np.array([d['palm'] for d in self.buffer])

        # 1. Smooth Positions (SavGol) - Optional but good
        try:
            tip_s = savgol_filter(tip_hist[:, :2], window_length=5, polyorder=2, axis=0)
            wrist_s = savgol_filter(wrist_hist[:, :2], window_length=5, polyorder=2, axis=0)
        except:
            tip_s = tip_hist[:, :2]
            wrist_s = wrist_hist[:, :2]

        # 2. Calculate Derivatives (dt = 1/30)
        dt = 1.0 / 30.0

        # Velocity
        tip_v = np.gradient(tip_s, axis=0) / dt
        wrist_v = np.gradient(wrist_s, axis=0) / dt

        # Acceleration
        tip_a = np.gradient(tip_v, axis=0) / dt

        # Current Frame Indices
        curr = -1 # Last element

        # Extract Values
        t_p = tip_s[curr]
        t_v = tip_v[curr]
        t_a = tip_a[curr]
        w_p = wrist_s[curr]
        w_v = wrist_v[curr]

        # Relative Depth (Tip Z - Wrist Z) - using raw MP Z
        rel_depth = tip_hist[curr, 2] - wrist_hist[curr, 2]

        # Rolling Averages (Last 5)
        start_idx = max(0, len(self.buffer) - 5)
        recent_speeds = np.linalg.norm(tip_v[start_idx:], axis=1)
        recent_accs = np.linalg.norm(tip_a[start_idx:], axis=1)

        avg_speed = np.mean(recent_speeds)
        avg_acc_mag = np.mean(recent_accs)
        rolling_var_speed = np.var(recent_speeds)

        # Lags
        lags = {}
        for lag in [2, 4, 6]:
            idx = len(self.buffer) - 1 - lag
            if idx >= 0:
                lags[lag] = np.linalg.norm(tip_v[idx])
            else:
                lags[lag] = 0.0

        # Construct Row
        feat_dict = construct_feature_row(
            t_p, t_v, t_a,
            w_p, w_v,
            palm_hist[curr, :2], dip_hist[curr, :2],
            rel_depth,
            avg_speed, avg_acc_mag, rolling_var_speed,
            lags
        )

        # Vectorize in order
        vector = []
        for k in self.feature_names:
            vector.append(feat_dict.get(k, 0.0))

        return np.array([vector]) # Shape (1, F)
