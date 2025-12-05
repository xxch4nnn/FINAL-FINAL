"""
generate_data.py
Orchestrates the Data Pipeline: Load -> Clean -> Features -> Label -> Save.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.processing.loader import load_json_sequence, load_midi_labels, compute_binary_labels
from src.processing.cleaner import process_sequence as clean_sequence
from src.processing.feature_gen import extract_features_from_sequence
from src.processing.labeler import StateLabeler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "PianoMotion10M"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

def process_single_file(meta: dict) -> pd.DataFrame:
    try:
        # 1. Load Raw
        points_3d = load_json_sequence(meta['pose_path'])
        if points_3d is None: return pd.DataFrame()

        # 2. Clean (Outlier removal & Smoothing)
        cleaned_points = clean_sequence(points_3d)

        # 3. Load MIDI & Compute Binary Labels
        note_events = load_midi_labels(meta['midi_path'])
        binary_labels = compute_binary_labels(cleaned_points, note_events)

        # 4. Extract Features
        rows = extract_features_from_sequence(cleaned_points)
        df = pd.DataFrame(rows)

        if df.empty: return df

        # Add Binary Label
        labels_flat = []
        for _, row in df.iterrows():
            f = int(row['frame'])
            i = int(row['finger_idx'])
            if f < len(binary_labels):
                labels_flat.append(binary_labels[f, i])
            else:
                labels_flat.append(0)

        df['is_pressed'] = labels_flat

        # 5. Apply State Machine (4-State Labeling)
        final_dfs = []
        for f_idx in range(5):
            finger_df = df[df['finger_idx'] == f_idx].sort_values('frame')
            labeled_finger_df = StateLabeler.apply(finger_df, binary_col='is_pressed')
            final_dfs.append(labeled_finger_df)

        final_df = pd.concat(final_dfs).sort_values(['frame', 'finger_idx'])
        final_df['sequence_id'] = meta['sequence']

        return final_df

    except Exception as e:
        logger.error(f"Error processing {meta['sequence']}: {e}")
        return pd.DataFrame()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scan files
    all_jsons = list(RAW_DATA_DIR.rglob("*.json"))
    all_midis = list(RAW_DATA_DIR.rglob("*.mid"))
    midi_map = { f.stem: f for f in all_midis }

    sequences = []
    for json_path in all_jsons:
        if "annotation" in json_path.name: continue
        stem = json_path.stem
        midi_path = midi_map.get(stem)
        if not midi_path:
             parts = stem.split('_seq_')
             if len(parts) > 1 and parts[0] in midi_map:
                 midi_path = midi_map[parts[0]]

        if midi_path:
            sequences.append({'sequence': stem, 'pose_path': json_path, 'midi_path': midi_path})

    sequences = sorted(sequences, key=lambda x: x['sequence'])
    logger.info(f"Found {len(sequences)} valid sequences.")

    if not sequences:
        logger.warning("No data found. Exiting.")
        return

    # Run Pipeline
    all_dfs = []
    processed_count = 0

    # Use max_workers=None (defaults to cpu_count)
    with ProcessPoolExecutor() as executor:
        future_to_seq = {executor.submit(process_single_file, seq): seq for seq in sequences}

        for future in tqdm(as_completed(future_to_seq), total=len(sequences)):
            try:
                res = future.result()
                if not res.empty:
                    all_dfs.append(res)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Pipeline error: {e}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_path = OUTPUT_DIR / "features.csv"
        full_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(full_df)} samples to {out_path}")
        logger.info(f"Label Dist:\n{full_df['state_label'].value_counts()}")
    else:
        logger.warning("No features extracted.")

if __name__ == "__main__":
    main()
