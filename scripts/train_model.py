"""
train_model.py
Orchestrates training using GPU-accelerated models.
"""

import sys
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Add root
sys.path.append(str(Path(__file__).parent.parent))

from src.training.gpu_models import PianoXGBoost, PyTorchTrainer
# We also need to know feature columns to filter?
# Or we assume features.csv is clean.
# features.csv has 'state_label', 'is_pressed', 'frame', 'finger_idx', 'sequence_id' which are NOT features.
from src.processing.feature_gen import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        logger.error(f"Data not found at {DATA_PATH}. Run scripts/generate_data.py first.")
        return

    logger.info("Loading Data...")
    df = pd.read_csv(DATA_PATH)

    # Prepare X, y
    # Filter columns
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values
    y = df['state_label'].values

    # Handle NaN
    if np.isnan(X).any():
        logger.warning("NaNs found in features. Imputing with 0.")
        X = np.nan_to_num(X)

    # Scale
    logger.info("Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Train PyTorch
    logger.info("--- PyTorch Training ---")
    pt_model = PyTorchTrainer(input_dim=X_train.shape[1], use_gpu=True)
    pt_model.fit(X_train, y_train, epochs=15, batch_size=128)

    pt_pred = pt_model.predict(X_test)
    acc_pt = accuracy_score(y_test, pt_pred)
    logger.info(f"PyTorch Accuracy: {acc_pt:.4f}")
    pt_model.save(MODELS_DIR / "pytorch_model.pth")

    # 2. Train XGBoost
    logger.info("--- XGBoost Training ---")
    xgb_model = PianoXGBoost(use_gpu=True, n_estimators=200)
    # Note: XGBoost custom class needs internal fix for predict_proba if used later,
    # but for now we fit/predict
    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, xgb_pred)
    logger.info(f"XGBoost Accuracy: {acc_xgb:.4f}")
    xgb_model.save(MODELS_DIR / "xgboost_model.json") # XGB uses JSON/UBJSON

    # Compare
    logger.info("\nComparison:")
    logger.info(f"PyTorch: {acc_pt:.4f}")
    logger.info(f"XGBoost: {acc_xgb:.4f}")

    # Save Best?
    # Usually XGBoost wins on tabular data.

    logger.info("Done.")

if __name__ == "__main__":
    main()
