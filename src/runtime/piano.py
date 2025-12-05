"""
piano.py
Refactored Main Runtime.
Integrates LiveFeatureExtractor, SoundEngine, and Inference.
"""

import cv2
import numpy as np
import mediapipe as mp
import pygame
import joblib
import time
import threading
import sys
import logging
import torch
import xgboost as xgb
from pathlib import Path

# Add root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processing.feature_gen import LiveFeatureExtractor, FEATURE_COLUMNS
from src.training.gpu_models import PianoPyTorch # For loading class structure

# --- CONFIG ---
CONFIG = {
    'CAM_ID': 0,
    'WIDTH': 1280,
    'HEIGHT': 720,
    'MODELS_DIR': Path(__file__).parent.parent.parent / "models",
    'USE_MODEL': 'pytorch', # 'xgboost' or 'pytorch'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AUDIO ---
class SoundEngine:
    def __init__(self):
        self.sounds = {}
        self.active = False
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.init()
            pygame.mixer.set_num_channels(16)
            self.active = True
            self._load_sounds()
        except:
            pass

    def _load_sounds(self):
        notes = {0: 261.63, 1: 293.66, 2: 329.63, 3: 349.23, 4: 392.00, 5: 440.00, 6: 493.88}
        for i, freq in notes.items():
            self.sounds[i] = self._generate_sine(freq)

    def _generate_sine(self, freq):
        if not self.active: return None
        rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(rate*duration), False)
        wave = np.sin(2*np.pi*freq*t) * 0.3
        audio = (wave * 32767).astype(np.int16)
        return pygame.mixer.Sound(np.column_stack((audio, audio)))

    def play(self, idx):
        if self.active and idx in self.sounds:
            self.sounds[idx].play()

# --- PREDICTOR ---
class RuntimePredictor:
    def __init__(self):
        self.extractor = LiveFeatureExtractor()
        self.model = None
        self.scaler = None
        self.load_models()

    def load_models(self):
        try:
            self.scaler = joblib.load(CONFIG['MODELS_DIR'] / "scaler.pkl")

            if CONFIG['USE_MODEL'] == 'pytorch':
                # Load PyTorch
                # Need input dim.
                # FEATURE_COLUMNS length
                input_dim = len(FEATURE_COLUMNS)
                self.model = PianoPyTorch(input_dim)
                self.model.load_state_dict(torch.load(CONFIG['MODELS_DIR'] / "pytorch_model.pth", map_location='cpu'))
                self.model.eval()
                self.mode = 'pt'
            else:
                # Load XGBoost
                self.model = xgb.Booster()
                self.model.load_model(CONFIG['MODELS_DIR'] / "xgboost_model.json")
                self.mode = 'xgb'

            logger.info(f"Loaded {self.mode} model.")
        except Exception as e:
            logger.warning(f"Model load failed: {e}. Using dummy.")
            self.model = None

    def predict(self, vec):
        if self.model is None or vec is None: return 0

        try:
            # Scale
            # vec is (1, F)
            vec_s = self.scaler.transform(vec)

            if self.mode == 'pt':
                with torch.no_grad():
                    t = torch.tensor(vec_s, dtype=torch.float32)
                    out = self.model(t)
                    pred = torch.argmax(out, dim=1).item()
                    return pred
            else:
                dtest = xgb.DMatrix(vec_s)
                # XGBoost predict returns float array
                # If trained with multi:softmax, returns class index
                pred = self.model.predict(dtest)
                return int(pred[0])
        except Exception as e:
            # logger.error(e)
            return 0

# --- MAIN ---
def main():
    cap = cv2.VideoCapture(CONFIG['CAM_ID'])
    cap.set(3, CONFIG['WIDTH'])
    cap.set(4, CONFIG['HEIGHT'])

    audio = SoundEngine()
    predictor = RuntimePredictor()

    hands = mp.solutions.hands.Hands(max_num_hands=1)

    logger.info("Running Piano...")

    last_state = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            mp.solutions.drawing_utils.draw_landmarks(frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

            predictor.extractor.update(lm)
            vec = predictor.extractor.extract()
            state = predictor.predict(vec)

            # Simple Trigger
            # State 1 = Press
            if state == 1 and last_state != 1:
                # Dummy key mapping (middle C)
                audio.play(0)
                cv2.circle(frame, (100, 100), 20, (0, 255, 0), -1)

            last_state = state

            cv2.putText(frame, f"State: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Piano", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
