"""
feature_gen.py
Unified Feature Extraction Engine for PianoMotion Project.
Ensures congruency between Data Generation (Sync) and Live Runtime.
Refactored to support cleaned data input.
"""

import numpy as np
import logging
from collections import deque
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple

# --- Constants ---
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

# --- Camera Constants ---
CAM_W = 1920
CAM_H = 1080
FX = 1000
FY = 1000
CX = 960
CY = 540
FPS = 30.0
FRAME_DURATION = 1.0 / FPS

def project_3d_to_2d(points_3d: np.ndarray) -> np.ndarray:
    """
    Projects 3D points (x, y, z) to Normalized 2D screen coordinates (u_norm, v_norm).
    Uses pinhole camera model with fixed intrinsics.

    Args:
        points_3d: (N, 21, 3) array of 3D coordinates.

    Returns:
        (N, 21, 2) array of normalized 2D coordinates [0, 1].
    """
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    # Avoid division by zero
    z = np.where(z == 0, 1e-6, z)

    # Pinhole Projection
    u = (x / z) * FX + CX
    v = (y / z) * FY + CY

    # Normalize
    u_norm = u / CAM_W
    v_norm = v / CAM_H

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

def extract_features_from_sequence(points_3d: np.ndarray) -> List[Dict]:
    """
    Extracts features from a full sequence of 3D points.
    Assumes points_3d has been CLEANED already (no smoothing inside here to avoid double smoothing).

    Returns:
        List of feature dictionaries (one per frame/finger combination logic handled upstream,
        BUT wait - original logic handled per-finger.
        Here we assume we process specific finger indices?
        Original extracted for ALL 5 fingers? Yes.

        Refactoring Note:
        The original `Sync` script iterated frames, then fingers.
        This function should probably return a DataFrame structure or list of rows,
        including the finger index meta-data if needed.
        However, usually we train on *a* finger.

        Let's standardize: This function returns features for specific target fingers
        OR we pass the finger index to extract?

        For simplicity/congruency with `SyncPianoMotionDataset.py` original logic:
        It extracted 5 rows per frame (one per finger).
        """

    num_frames = len(points_3d)

    # Project 3D -> 2D
    points_2d = project_3d_to_2d(points_3d)

    # Calculate Derivatives (2D)
    vel_2d = np.gradient(points_2d, axis=0) / FRAME_DURATION
    acc_2d = np.gradient(vel_2d, axis=0) / FRAME_DURATION

    # Clip (Double check, though cleaner should handle outliers, clipping is safe)
    vel_2d = np.clip(vel_2d, -10.0, 10.0)
    acc_2d = np.clip(acc_2d, -50.0, 50.0)

    wrist_pos = points_2d[:, 0, :]
    palm_pos = points_2d[:, 9, :] # Middle MCP
    wrist_vel = vel_2d[:, 0, :]
    wrist_z_3d = points_3d[:, 0, 2]

    fingertip_indices = [4, 8, 12, 16, 20]
    dip_indices = [3, 7, 11, 15, 19]

    all_rows = []

    for f in range(num_frames):
        for i, (tip_idx, dip_idx) in enumerate(zip(fingertip_indices, dip_indices)):

            tip_p = points_2d[f, tip_idx]
            tip_v = vel_2d[f, tip_idx]
            tip_a = acc_2d[f, tip_idx]

            # Relative Depth
            tip_z_3d = points_3d[f, tip_idx, 2]
            rel_depth = float(tip_z_3d - wrist_z_3d[f])

            # Rolling (Window 5)
            s_idx = max(0, f-4)
            recent_vels = vel_2d[s_idx:f+1, tip_idx]
            recent_accs = acc_2d[s_idx:f+1, tip_idx]

            avg_speed = float(np.mean(np.linalg.norm(recent_vels, axis=1)))
            avg_acc_mag = float(np.mean(np.linalg.norm(recent_accs, axis=1)))

            rolling_var_speed = float(np.var(np.linalg.norm(recent_vels, axis=1))) if f > 4 else 0.0

            lags = {}
            for lag in [2, 4, 6]:
                if f >= lag:
                    l_idx = f - lag
                    lags[lag] = float(np.linalg.norm(vel_2d[l_idx, tip_idx]))
                else:
                    lags[lag] = 0.0

            row = construct_feature_row(
                tip_p, tip_v, tip_a,
                wrist_pos[f], wrist_vel[f],
                palm_pos[f], points_2d[f, dip_idx],
                rel_depth,
                avg_speed, avg_acc_mag, rolling_var_speed,
                lags
            )

            # Meta
            row['frame'] = f
            row['finger_idx'] = i # 0-4

            all_rows.append(row)

    return all_rows

# --- Live Feature Extractor ---
class LiveFeatureExtractor:
    """
    Real-time Feature Extractor using buffer.
    """
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.feature_names = FEATURE_COLUMNS

    def update(self, landmarks):
        if not landmarks: return
        frame_data = {
            'wrist': np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z]),
            'tip': np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z]),
            'dip': np.array([landmarks[7].x, landmarks[7].y, landmarks[7].z]),
            'palm': np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        }
        self.buffer.append(frame_data)

    def extract(self) -> Optional[np.ndarray]:
        if len(self.buffer) < 5: return None

        # Convert buffer to arrays
        # Note: In runtime we only track INDEX finger (usually).
        # But 'update' above hardcodes indices 8 (Index Tip).
        # This matches main_runtime logic.

        wrist_hist = np.array([d['wrist'] for d in self.buffer])
        tip_hist = np.array([d['tip'] for d in self.buffer])
        dip_hist = np.array([d['dip'] for d in self.buffer])
        palm_hist = np.array([d['palm'] for d in self.buffer])

        # Simple smooth
        try:
            tip_s = savgol_filter(tip_hist[:, :2], window_length=5, polyorder=2, axis=0)
            wrist_s = savgol_filter(wrist_hist[:, :2], window_length=5, polyorder=2, axis=0)
        except:
            tip_s = tip_hist[:, :2]
            wrist_s = wrist_hist[:, :2]

        dt = 1.0 / 30.0
        tip_v = np.gradient(tip_s, axis=0) / dt
        wrist_v = np.gradient(wrist_s, axis=0) / dt
        tip_a = np.gradient(tip_v, axis=0) / dt

        curr = -1
        t_p = tip_s[curr]
        t_v = tip_v[curr]
        t_a = tip_a[curr]
        w_p = wrist_s[curr]
        w_v = wrist_v[curr]

        rel_depth = tip_hist[curr, 2] - wrist_hist[curr, 2]

        start_idx = max(0, len(self.buffer) - 5)
        recent_speeds = np.linalg.norm(tip_v[start_idx:], axis=1)
        recent_accs = np.linalg.norm(tip_a[start_idx:], axis=1)

        avg_speed = np.mean(recent_speeds)
        avg_acc_mag = np.mean(recent_accs)
        rolling_var_speed = np.var(recent_speeds)

        lags = {}
        for lag in [2, 4, 6]:
            idx = len(self.buffer) - 1 - lag
            if idx >= 0:
                lags[lag] = np.linalg.norm(tip_v[idx])
            else:
                lags[lag] = 0.0

        feat_dict = construct_feature_row(
            t_p, t_v, t_a,
            w_p, w_v,
            palm_hist[curr, :2], dip_hist[curr, :2],
            rel_depth,
            avg_speed, avg_acc_mag, rolling_var_speed,
            lags
        )

        vector = []
        for k in self.feature_names:
            vector.append(feat_dict.get(k, 0.0))

        return np.array([vector])
