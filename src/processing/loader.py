"""
loader.py
Handles file scanning, loading JSON/MIDI, and basic alignment.
Extracts the logic previously in SyncPianoMotionDataset.py.
"""

import os
import json
import mido
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def load_midi_labels(midi_file: Path) -> List[Tuple[int, float, float]]:
    """
    Parses MIDI to get note events.
    Returns list of (note, start_time, end_time).
    """
    notes_log = []
    try:
        mid = mido.MidiFile(str(midi_file))
        tempo = 500000

        for msg in mido.merge_tracks(mid.tracks):
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        time_sec = 0.0
        active_notes = {}

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

def load_json_sequence(json_path: Path) -> np.ndarray:
    """
    Loads raw JSON pose data.
    Returns: (Frames, 21, 3) array or None.
    """
    try:
        with open(json_path, 'r') as f:
            raw = json.load(f)

        data = None
        if isinstance(raw, dict):
            if 'right' in raw:
                data = raw['right']
            elif 'left' in raw:
                data = raw['left']
        elif isinstance(raw, list):
             data = raw

        if data is None:
            return None

        frames = []
        for fr in data:
            arr = np.array(fr)
            if len(fr) == 62:
                arr = np.append(arr, 0.0)

            if len(arr) == 63:
                frames.append(arr.reshape(21, 3))

        if not frames:
            return None

        return np.array(frames)

    except Exception as e:
        logger.warning(f"JSON Load Error {json_path}: {e}")
        return None

def compute_binary_labels(points_3d: np.ndarray, note_events: List, fps=30.0) -> np.ndarray:
    """
    Generates a binary 'is_pressed' label for each frame/finger.

    This replaces the complex 'Group & Rank' heuristic for the initial binary mask.
    The output will be used by StateLabeler to generate 4-state logic.

    Assumption: We still need to map notes to fingers.
    The original logic:
    - Group notes by time (chords).
    - Sort chord notes (pitch).
    - Rank fingers by Z-velocity (lowest/most-downward = pressed).
    - Map lowest finger to lowest note?

    Actually, the original logic mapped note events to fingers using Z-velocity ranking.
    We must preserve this logic to know WHICH finger pressed.

    Returns:
        (NumFrames, 5) array of binary labels (0 or 1).
    """
    num_frames = len(points_3d)
    binary_labels = np.zeros((num_frames, 5), dtype=int)

    # 1. Group notes (Chords)
    note_events.sort(key=lambda x: x[1])
    groups = []
    if note_events:
        curr_group = [note_events[0]]
        for i in range(1, len(note_events)):
            if (note_events[i][1] - curr_group[0][1]) < 0.07:
                curr_group.append(note_events[i])
            else:
                groups.append(curr_group)
                curr_group = [note_events[i]]
        groups.append(curr_group)

    fingertip_indices = [4, 8, 12, 16, 20]

    # 2. Assign
    for group in groups:
        start_t = np.mean([n[1] for n in group])
        start_f = int(start_t * fps)

        if start_f >= num_frames: continue

        # Calc Z-velocity at start_f for ranking
        z_vals = points_3d[:, :, 2]
        # We need a small window to get velocity. simple diff
        # Handle edges
        s_idx = max(0, start_f - 1)
        e_idx = min(num_frames, start_f + 1)

        if e_idx <= s_idx: continue

        # Calculate velocity approximation
        vels = []
        for f_idx in fingertip_indices:
            v = z_vals[e_idx-1, f_idx] - z_vals[s_idx, f_idx]
            vels.append(v)

        # Rank: Most negative (downward) -> Pressed
        ranked_indices = np.argsort(vels) # 0..4

        for i, note_tuple in enumerate(group):
            if i < 5:
                f_real_idx = ranked_indices[i] # This is the finger index (0=Thumb...4=Pinky) relative to our 5 fingers

                n_start = int(note_tuple[1] * fps)
                n_end = int(note_tuple[2] * fps)

                n_start = max(0, n_start)
                n_end = min(num_frames, n_end)

                if n_start < n_end:
                    binary_labels[n_start:n_end, f_real_idx] = 1

    return binary_labels
