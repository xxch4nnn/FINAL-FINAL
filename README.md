# PianoMotion10M: The Unified AI Piano System

**Project Vision:**
PianoMotion10M is an end-to-end Machine Learning pipeline that translates raw 3D hand motion into musical execution. By analyzing the "Digital Twin" of a pianist's hand, we can predict subtle articulations (Hover, Press, Hold, Release) with high precision, enabling virtual piano performance using only a webcam.

This repository contains the complete lifecycle:
1.  **Data Generation:** Procedural 3D Projection & Feature Extraction.
2.  **Machine Learning:** Training robust classifiers (Random Forest / SVM) on 10M data points.
3.  **Real-Time Runtime:** A low-latency inference engine that plays music in real-time.

---

## 📦 Setup Guide

### 1. Prerequisites
*   Python 3.10+
*   Webcam (720p+ recommended)
*   ArUco Marker Printout (DICT_4X4_50, ID=0, 50mm)

### 2. Installation
Create a clean environment and install dependencies:

```bash
# Create Conda Environment (Optional but recommended)
conda create -n pianomotion python=3.10
conda activate pianomotion

# Install Dependencies
pip install -r requirements.txt
```

---

## 🚀 The "One-Day Plan"

Follow these steps to go from zero to playing music.

### Step 1: Data Generation & Processing
We generate a congruous dataset by processing raw 3D poses (or downloading them).

```bash
# Run the Data Engine (Processes raw JSONs -> features.csv)
# Ensure your raw data is in src/PianoMotion10M (or edit path)
python src/SyncPianoMotionDataset.py
```
*   **Output:** `data/features.csv`
*   *Note: If you don't have the 10M dataset, the script handles a small sample if provided.*

### Step 2: Model Training
Train the classifier on the generated features.

```bash
# Train Models (SVM & Random Forest)
python src/ML_Pipeline_Prep.py
```
*   **Output:** `models/rf_model.pkl`, `models/scaler.pkl`, `models/selected_features.pkl`
*   **Verify:** Check `results/model_comparison.png` for performance metrics.

### Step 3: The Performance (Runtime)
Run the real-time system. Place your ArUco marker on a table to project the virtual keys.

```bash
# Start the Piano
python src/main_runtime.py
```
*   **Controls:** 'q' to quit.
*   **Interaction:** Align camera so the ArUco marker is visible. Virtual keys will appear. Use your fingers to "press" the air keys!

---

## 🛠 Troubleshooting

**Q: The audio is lagging.**
**A:** Ensure you are running on a native OS (Windows/Mac/Linux) and not a VM/Container for audio. The `SoundEngine` tries to init low-latency buffers. If simple `pygame` latency is too high, consider ASIO drivers (outside python scope) or reducing `BUFFER_SIZE` in `src/main_runtime.py`.

**Q: ArUco marker isn't detected.**
**A:** Ensure adequate lighting. The marker must be flat and clearly visible. The system expects `DICT_4X4_50`. If using a different dictionary, update `CONFIG` in `src/main_runtime.py`.

**Q: "Model not found" warning.**
**A:** You must run **Step 2** (`ML_Pipeline_Prep.py`) successfully first. The runtime falls back to a heuristic (simple depth check) if no model is found, which is less accurate.

**Q: "Congruency Check Failed" during development.**
**A:** Run `python tests/test_congruency.py` to debug feature mismatch issues.

---

## 📂 File Structure

*   `src/`: All source code.
    *   `feature_engine.py`: **Core Logic.** Single source of truth for features.
    *   `SyncPianoMotionDataset.py`: Data Pipeline.
    *   `ML_Pipeline_Prep.py`: Training Pipeline.
    *   `main_runtime.py`: Live Application.
*   `data/`: Stores `features.csv` and raw datasets.
*   `models/`: Stores trained `.pkl` artifacts.
*   `tests/`: System verification scripts.
