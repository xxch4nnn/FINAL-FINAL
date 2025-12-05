# FINAL-FINAL Piano Motion Project

## 🏗️ Structure
This project uses a modular architecture:

*   `src/processing`: Data cleaning, smoothing, feature extraction, and labeling logic.
*   `src/training`: GPU-accelerated model definitions (PyTorch & XGBoost).
*   `src/runtime`: Real-time application logic.
*   `scripts`: Entry points for data generation and training.
*   `data`: Organized storage for raw and processed datasets.

## 🚀 Usage

### 1. Data Generation
Process raw JSON/MIDI data into a clean, labeled dataset.

```bash
python scripts/generate_data.py
```
*   Reads from `data/raw/PianoMotion10M`
*   Cleans outliers & smooths trajectories.
*   Generates features.
*   Applies 4-state logic (Hover, Press, Hold, Release).
*   Saves to `data/processed/features.csv`.

### 2. Training
Train GPU-accelerated models.

```bash
python scripts/train_model.py
```
*   Trains PyTorch Linear Model.
*   Trains XGBoost (GPU Hist) Model.
*   Saves artifacts to `models/`.

### 3. Runtime
Run the piano application.

```bash
python src/runtime/piano.py
```
*   Uses trained models for real-time inference.

## 🔧 Requirements
*   Python 3.10+
*   NVIDIA GPU (Recommended) for training.
*   See `requirements.txt`.
