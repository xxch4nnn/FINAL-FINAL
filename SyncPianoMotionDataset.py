"""
SyncPianoMotionDataset.py
The Unified Data Engine for PianoMotion10M.
Handles Downloading, Parsing, 3D-to-2D Projection, Feature Extraction, and Batch Generation.
Fixed for robust file matching and multiprocessing.
"""

import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from tqdm import tqdm
import mido
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Generator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
CAM_W = 1920
CAM_H = 1080
FX = 1000
FY = 1000
CX = 960
CY = 540
FPS = 30.0
FRAME_DURATION = 1.0 / FPS

# --- Helper Functions (Global for Pickling) ---

def project_3d_to_2d(points_3d: np.ndarray) -> np.ndarray:
    """
    Projects 3D points (x, y, z) to Normalized 2D screen coordinates (u_norm, v_norm).
    Uses pinhole camera model with fixed intrinsics.

    Args:
        points_3d: (N, 21, 3) array of 3D coordinates.

    Returns:
        (N, 21, 2) array of normalized 2D coordinates [0, 1].
    """
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
    u_norm = u / CAM_W
    v_norm = v / CAM_H

    return np.stack([u_norm, v_norm], axis=-1)

def load_midi_labels(midi_file: Path) -> List[Tuple[int, float, float]]:
    """
    Parses MIDI to get note events.
    Returns list of (note, start_time, end_time).
    """
    notes_log = []
    try:
        mid = mido.MidiFile(str(midi_file))
        tempo = 500000

        # Get tempo
        for msg in mido.merge_tracks(mid.tracks):
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        time_sec = 0.0
        active_notes = {} # note -> start_time

        for msg in mido.merge_tracks(mid.tracks):
            time_sec += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = time_sec
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start = active_notes.pop(msg.note)
                    end = time_sec
                    notes_log.append((msg.note, start, end))

        return notes_log

    except Exception as e:
        logger.error(f"MIDI Error {midi_file}: {e}")
        return []

def extract_features(kinematics_3d: np.ndarray, note_events: List, seq_id: str) -> pd.DataFrame:
    """
    Extracts 2D Features from 3D Kinematics.
    Labels using Group & Rank heuristic.
    IMPROVED: Robust smoothing and outlier handling for velocity/acceleration.
    """
    if kinematics_3d is None or len(kinematics_3d) == 0:
        return pd.DataFrame()

    num_frames = len(kinematics_3d)

    # --- 1. Pre-Projection Features (Depth) ---
    # Calculate Relative Depth (Normalized Z) - before projection logic loss
    # rel_depth = tip.z - wrist.z
    # wrist is index 0
    wrist_z_3d = kinematics_3d[:, 0, 2]

    # --- 2. Smooth 3D poses BEFORE projection (Important!) ---
    try:
        if num_frames >= 5:
            # Apply Savitzky-Golay filter to 3D coordinates first
            kinematics_3d = savgol_filter(kinematics_3d, window_length=5, polyorder=2, axis=0)
            logger.debug(f"Applied smoothing to {num_frames} frames")
    except Exception as e:
        logger.debug(f"Smoothing skipped: {e}")

    # --- 3. Project 3D -> 2D ---
    points_2d = project_3d_to_2d(kinematics_3d) # (Frames, 21, 2)

    # --- 4. Smooth 2D points again ---
    try:
        if num_frames >= 5:
            points_2d = savgol_filter(points_2d, window_length=5, polyorder=2, axis=0)
    except Exception:
        pass # Skip smoothing if too short

    # --- 5. Calculate Derivatives (2D) with outlier clipping ---
    vel_2d = np.gradient(points_2d, axis=0) / FRAME_DURATION
    acc_2d = np.gradient(vel_2d, axis=0) / FRAME_DURATION
    
    # Clip extreme velocities and accelerations (likely noise)
    vel_clip_threshold = 10.0  # reasonable limit for hand movement
    acc_clip_threshold = 50.0  # reasonable limit for acceleration
    
    vel_2d = np.clip(vel_2d, -vel_clip_threshold, vel_clip_threshold)
    acc_2d = np.clip(acc_2d, -acc_clip_threshold, acc_clip_threshold)
    
    logger.debug(f"Velocity clipped to [{-vel_clip_threshold}, {vel_clip_threshold}]")
    logger.debug(f"Acceleration clipped to [{-acc_clip_threshold}, {acc_clip_threshold}]")

    # --- 6. Prepare Landmarks ---
    # Indices: Wrist=0, Index=8, Middle=12
    # Using 9 (Middle MCP) as Palm Center Proxy
    wrist_pos = points_2d[:, 0, :]
    palm_pos = points_2d[:, 9, :]
    wrist_vel = vel_2d[:, 0, :]

    fingertip_indices = [4, 8, 12, 16, 20] # Thumb to Pinky
    dip_indices = [3, 7, 11, 15, 19]

    all_rows = []

    # --- 7. Labeling Logic ---
    labels = np.zeros((num_frames, 5), dtype=int)

    # Group notes by start time (Chord grouping)
    note_events.sort(key=lambda x: x[1])

    # Simple grouping: events within 66ms (2 frames)
    groups = []
    if note_events:
        curr_group = [note_events[0]]
        for i in range(1, len(note_events)):
            if (note_events[i][1] - curr_group[0][1]) < 0.07: # ~2 frames at 30fps
                curr_group.append(note_events[i])
            else:
                groups.append(curr_group)
                curr_group = [note_events[i]]
        groups.append(curr_group)

    # Assign labels
    for group in groups:
        # Time to frame
        start_t = np.mean([n[1] for n in group])
        start_f = int(start_t * FPS)

        if start_f >= num_frames: continue

        # Calculate 3D Z-vel for labeling only
        z_vals = kinematics_3d[:, :, 2]
        z_vel = np.gradient(z_vals, axis=0)

        finger_z_vels = []
        for f_idx in fingertip_indices:
            finger_z_vels.append(z_vel[start_f, f_idx])

        # Rank: Most negative (downward) first
        ranked_indices = np.argsort(finger_z_vels) # Ascending (neg -> pos)

        # Assign
        for i, note_tuple in enumerate(group):
            if i < 5:
                f_real_idx = ranked_indices[i] # 0..4 represents which finger index in [0,1,2,3,4]

                # Label frames
                n_start = int(note_tuple[1] * FPS)
                n_end = int(note_tuple[2] * FPS)
                duration = n_end - n_start

                # 4-State Logic: 0=Hover, 1=Press, 2=Hold, 3=Release
                # 1=Press (3 frames), 2=Hold, 3=Release (3 frames)
                # Ensure bounds
                n_start = max(0, n_start)
                n_end = min(num_frames, n_end)

                if n_start >= n_end: continue

                if duration > 6:
                    labels[n_start:n_start+3, f_real_idx] = 1
                    labels[n_start+3:n_end-3, f_real_idx] = 2
                    labels[n_end-3:n_end, f_real_idx] = 3
                else:
                    mid = n_start + (duration // 2)
                    labels[n_start:mid, f_real_idx] = 1
                    labels[mid:n_end, f_real_idx] = 3

    # --- 8. Feature Construction ---
    for f in range(num_frames):
        for i, (tip_idx, dip_idx) in enumerate(zip(fingertip_indices, dip_indices)):
            row = {}

            # Base vectors
            tip_p = points_2d[f, tip_idx]
            tip_v = vel_2d[f, tip_idx]
            tip_a = acc_2d[f, tip_idx]
            wrist_p = wrist_pos[f]
            wrist_v = wrist_vel[f]
            palm_p = palm_pos[f]
            dip_p = points_2d[f, dip_idx]

            # 1. Position (Normalized)
            row['finger_pos_x'] = float(tip_p[0])
            row['finger_pos_y'] = float(tip_p[1])
            row['wrist_pos_x'] = float(wrist_p[0])
            row['wrist_pos_y'] = float(wrist_p[1])

            # 2. Velocity (Normalized/sec) - now clipped
            row['finger_vel_x'] = float(tip_v[0])
            row['finger_vel_y'] = float(tip_v[1])
            row['finger_speed'] = float(np.linalg.norm(tip_v))

            row['wrist_vel_x'] = float(wrist_v[0])
            row['wrist_vel_y'] = float(wrist_v[1])
            row['wrist_speed'] = float(np.linalg.norm(wrist_v))

            # 3. Acceleration - now clipped
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

            # 6. Relative Depth (Feature Requirement)
            # tip.z - wrist.z (from raw 3D)
            tip_z_3d = kinematics_3d[f, tip_idx, 2]
            row['rel_depth'] = float(tip_z_3d - wrist_z_3d[f])

            # 7. Rolling Averages (Last 5 frames)
            s_idx = max(0, f-4)
            row['avg_speed'] = float(np.mean(np.linalg.norm(vel_2d[s_idx:f+1, tip_idx], axis=1)))
            row['avg_acc_mag'] = float(np.mean(np.linalg.norm(acc_2d[s_idx:f+1, tip_idx], axis=1)))

            # 8. Lags (Speed)
            for lag in [2, 4, 6]:
                if f >= lag:
                    l_idx = f - lag
                    row[f'lag_speed_{lag}'] = float(np.linalg.norm(vel_2d[l_idx, tip_idx]))
                else:
                    row[f'lag_speed_{lag}'] = 0.0

            # 9. Rolling Variance (Stability)
            if f > 4:
                row['rolling_var_speed'] = float(np.var(np.linalg.norm(vel_2d[s_idx:f+1, tip_idx], axis=1)))
            else:
                row['rolling_var_speed'] = 0.0

            # Meta
            row['sequence_id'] = seq_id
            row['ground_truth_label'] = int(labels[f, i])

            all_rows.append(row)

    return pd.DataFrame(all_rows)

def process_sequence_file(seq_meta: Dict) -> pd.DataFrame:
    """
    Worker function to process a single sequence file.
    Global to be picklable.
    """
    try:
        # Load JSON
        with open(seq_meta['pose_path'], 'r') as f:
            raw = json.load(f)

        # Handle structure variants
        data = None
        if isinstance(raw, dict):
            if 'right' in raw:
                data = raw['right']
            elif 'left' in raw:
                data = raw['left']
        elif isinstance(raw, list):
             data = raw

        if data is None:
            return pd.DataFrame()

        # Fix shape and Padding Logic
        frames = []
        padding_warned = False

        for fr in data:
            arr = np.array(fr)
            # Padding Logic: If 62, append 0.0
            if len(fr) == 62:
                if not padding_warned:
                    # logging might be tricky in mp, but let's try
                    # logger.debug(f"Padding frames for {seq_meta['sequence']}")
                    padding_warned = True
                arr = np.append(arr, 0.0)

            if len(arr) == 63:
                frames.append(arr.reshape(21, 3))
            else:
                # Skip malformed frames
                continue

        if not frames:
            return pd.DataFrame()

        points_3d = np.array(frames)

        # Load MIDI
        note_events = load_midi_labels(seq_meta['midi_path'])

        # Extract
        df = extract_features(points_3d, note_events, seq_meta['sequence'])
        return df

    except Exception as e:
        # logger.warning(f"Failed to process {seq_meta['sequence']}: {e}")
        return pd.DataFrame()

# --- Main Class ---

class SyncPianoMotionDataset:
    """
    Unified Data Engine.
    """
    def __init__(self, dataset_root: Path = None):
        if dataset_root is None:
             self.dataset_root = Path(__file__).parent / "PianoMotion10M"
        else:
            self.dataset_root = dataset_root

        self.output_dir = Path(__file__).parent

    def scan_files(self) -> List[Dict]:
        """
        Scans for JSON and MIDI files and maps them.
        Returns list of dicts with 'sequence', 'pose_path', 'midi_path'.
        """
        logger.info(f"Scanning {self.dataset_root}...")

        all_jsons = list(self.dataset_root.rglob("*.json"))
        all_midis = list(self.dataset_root.rglob("*.mid"))

        logger.info(f"Found {len(all_jsons)} JSONs and {len(all_midis)} MIDIs.")

        # Map MIDIs
        # Strategy: key = filename stem (no ext)
        midi_map = { f.stem: f for f in all_midis }

        sequences = []

        for json_path in all_jsons:
            if "annotation" in json_path.name: continue # Skip separate annotation files if any

            stem = json_path.stem
            midi_path = None

            # 1. Exact Match
            if stem in midi_map:
                midi_path = midi_map[stem]
            else:
                # 2. Split by '_seq_'
                parts = stem.split('_seq_')
                if len(parts) > 1:
                    base_stem = parts[0]
                    if base_stem in midi_map:
                        midi_path = midi_map[base_stem]

            if midi_path:
                sequences.append({
                    'sequence': stem,
                    'pose_path': json_path,
                    'midi_path': midi_path
                })

        logger.info(f"Matched {len(sequences)} sequences.")
        return sorted(sequences, key=lambda x: x['sequence'])

    def run(self, max_sequences: int = None, limit: int = None, use_multiprocessing: bool = False):
        """
        Main execution method.
        
        Args:
            max_sequences: Maximum number of sequences to process (None = process all)
            limit: Maximum number of samples to extract (None = no limit)
            use_multiprocessing: If False, runs sequentially for better debugging
        """
        sequences = self.scan_files()
        if not sequences:
            logger.warning("No sequences found. Please check dataset path.")
            return

        # Limit number of sequences
        if max_sequences is not None:
            sequences = sequences[:max_sequences]
            logger.info(f"Processing first {len(sequences)} sequences")

        all_dfs = []
        total_samples = 0

        logger.info(f"Max sequences: {max_sequences if max_sequences else 'All'}")
        logger.info(f"Sample limit: {limit if limit else 'No limit'}")
        logger.info(f"Processing mode: {'Multiprocessing' if use_multiprocessing else 'Sequential (for accuracy)'}")

        if use_multiprocessing:
            # Multiprocessing mode
            max_workers = os.cpu_count() or 1
            logger.info(f"Starting processing with {max_workers} workers...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                from concurrent.futures import as_completed
                future_to_seq = {executor.submit(process_sequence_file, seq): seq for seq in sequences}

                for future in tqdm(as_completed(future_to_seq), total=len(sequences), desc="Processing Sequences"):
                    try:
                        df = future.result()
                        if not df.empty:
                            all_dfs.append(df)
                            total_samples += len(df)

                            if limit is not None and total_samples >= limit:
                                logger.info(f"Sample limit reached: {total_samples} samples.")
                                break
                    except Exception as e:
                        logger.error(f"Worker error: {e}")
        else:
            # Sequential mode - for debugging and accuracy
            logger.info("Starting sequential processing...")
            
            for idx, seq_meta in enumerate(tqdm(sequences, desc="Processing Sequences"), 1):
                try:
                    df = process_sequence_file(seq_meta)
                    if not df.empty:
                        all_dfs.append(df)
                        total_samples += len(df)
                        
                        # Periodic logging
                        if idx % 10 == 0:
                            logger.info(f"Processed {idx}/{len(sequences)} sequences - {total_samples} samples extracted")
                        
                        if limit is not None and total_samples >= limit:
                            logger.info(f"Sample limit reached: {total_samples} samples.")
                            break
                    else:
                        logger.debug(f"No features extracted from {seq_meta['sequence']}")
                        
                except Exception as e:
                    logger.error(f"Error processing {seq_meta['sequence']}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)

            # Enforce limit exactly if needed (optional, but good)
            if limit is not None and len(final_df) > limit:
                final_df = final_df.iloc[:limit]

            out_path = self.output_dir / "features.csv"
            final_df.to_csv(out_path, index=False)
            logger.info(f"\n✅ Saved {len(final_df)} samples to {out_path}")
            logger.info(f"   Feature columns: {len(final_df.columns)}")
            logger.info(f"   Features: {list(final_df.columns[:10])}{'...' if len(final_df.columns) > 10 else ''}")
            logger.info(f"   Label distribution:\n{final_df['ground_truth_label'].value_counts().to_dict()}")
            
            # Data validation
            logger.info(f"\n📊 Data Quality Check:")
            logger.info(f"   Non-null values: {final_df.isnull().sum().sum()} nulls in {len(final_df) * len(final_df.columns)} total values")
            
            # Check for extreme values
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                max_val = final_df[col].abs().max()
                min_val = final_df[col].min()
                if max_val > 1e6:
                    logger.warning(f"   ⚠️  {col}: extreme values detected (max abs: {max_val:.2e})")
        else:
            logger.warning("No features extracted.")

if __name__ == "__main__":
    engine = SyncPianoMotionDataset()
    engine.run(max_sequences=20, use_multiprocessing=False)  # Sequential processing for accuracy
