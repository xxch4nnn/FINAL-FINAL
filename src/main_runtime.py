"""
main_runtime.py
The Final Integration Script for the PianoMotion Project.
Merges Vision (Phase 1), ML Model (Phase 2), and Audio Engine.

Features:
- Threaded Video Capture (High Performance)
- Live Feature Extraction (replicating SyncPianoMotionDataset.py via feature_engine)
- Random Forest Inference (or Heuristic Fallback)
- Low-Latency Audio (Pygame + Synth)
- Virtual Piano Projection (ArUco)
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
from collections import deque
from pathlib import Path
from scipy.signal import savgol_filter

# Import shared feature engine
try:
    from feature_engine import LiveFeatureExtractor, FEATURE_COLUMNS
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from feature_engine import LiveFeatureExtractor, FEATURE_COLUMNS

# --- CONFIGURATION ---
CONFIG = {
    'CAM_ID': 0,
    'WIDTH': 1280,
    'HEIGHT': 720,
    'FPS_TARGET': 30,
    'MARKER_SIZE': 0.05, # Meters (5cm)
    'KEY_WIDTH': 0.024,  # Meters (approx standard key width)
    'NUM_KEYS': 7,
    'BUFFER_SIZE': 10,
    'HISTORY_LAG': 6,    # For lag features
    'PROB_THRESHOLD': 0.5,
    'DEBOUNCE_FRAMES': 3,
    'MODELS_DIR': Path(__file__).parent.parent / "models",
    'SCALER_NAME': "scaler.pkl",
    'MODEL_NAME': "rf_model.pkl",
    'FEATURES_NAME': "selected_features.pkl"
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PianoMotion] - %(message)s')
logger = logging.getLogger(__name__)

# --- AUDIO ENGINE ---
class SoundEngine:
    """
    Low-latency audio engine using Pygame.
    Generates synthetic tones if files are missing.
    Handles headless environments gracefully.
    """
    def __init__(self):
        self.sounds = {}
        self.active = False
        try:
            # Try to init audio
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.init()
            pygame.mixer.set_num_channels(16)
            self.active = True
            self._load_sounds()
        except Exception as e:
            logger.warning(f"Audio Engine Disabled (Headless/Error): {e}")
            self.active = False

    def _load_sounds(self):
        # C Major Scale frequencies
        notes = {
            0: ('C', 261.63),
            1: ('D', 293.66),
            2: ('E', 329.63),
            3: ('F', 349.23),
            4: ('G', 392.00),
            5: ('A', 440.00),
            6: ('B', 493.88)
        }
        
        # Try loading wavs, fallback to synth
        for idx, (name, freq) in notes.items():
            filename = f"{name}.wav"
            if Path(filename).exists():
                try:
                    self.sounds[idx] = pygame.mixer.Sound(filename)
                except:
                    self.sounds[idx] = self._generate_sine_wave(freq)
            else:
                self.sounds[idx] = self._generate_sine_wave(freq)
        
        logger.info("Audio Engine Ready (Synthetic/Hybrid Mode)")

    def _generate_sine_wave(self, frequency, duration=0.5):
        """Generates a sine wave Sound object on the fly."""
        if not self.active: return None
        try:
            sample_rate = 44100
            n_samples = int(sample_rate * duration)
            
            # Generate wave
            t = np.linspace(0, duration, n_samples, False)
            wave = np.sin(2 * np.pi * frequency * t) * 0.3 # 0.3 Volume
            
            # Convert to 16-bit signed integer
            audio_data = (wave * 32767).astype(np.int16)
            
            # Stereo duplication
            stereo_data = np.column_stack((audio_data, audio_data))
            
            return pygame.mixer.Sound(buffer=stereo_data)
        except:
            return None

    def play(self, key_index):
        if self.active and key_index in self.sounds:
            try:
                self.sounds[key_index].play()
            except:
                pass

# --- VISION ENGINE (THREADED) ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['WIDTH'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['HEIGHT'])
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            try:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.1)
            except:
                break

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.t.join()
        if self.cap.isOpened():
            self.cap.release()

# --- PREDICTION WRAPPER ---
class PianoMotionPredictor:
    """
    Wraps the LiveFeatureExtractor and the Model.
    """
    def __init__(self):
        self.extractor = LiveFeatureExtractor(buffer_size=CONFIG['BUFFER_SIZE'])
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.has_model = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            m_path = CONFIG['MODELS_DIR'] / CONFIG['MODEL_NAME']
            s_path = CONFIG['MODELS_DIR'] / CONFIG['SCALER_NAME']
            f_path = CONFIG['MODELS_DIR'] / CONFIG['FEATURES_NAME']

            if m_path.exists() and s_path.exists() and f_path.exists():
                self.model = joblib.load(m_path)
                self.scaler = joblib.load(s_path)
                self.selected_features = joblib.load(f_path)
                self.has_model = True
                logger.info("ML Models Loaded Successfully.")
            else:
                logger.warning(f"ML Models NOT found at {CONFIG['MODELS_DIR']}. Using Heuristic Fallback.")
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            self.has_model = False

    def update(self, landmarks):
        self.extractor.update(landmarks)

    def predict(self):
        # Extract 1 vector
        vec = self.extractor.extract() # (1, F)
        
        if vec is None:
            return 0 # Buffering

        if self.has_model:
            try:
                # Filter to selected features if needed
                # The vec currently contains ALL features in FEATURE_COLUMNS order.
                # If model was trained on a subset (RFE), we must filter.

                # Create DataFrame for name-based indexing
                import pandas as pd
                df = pd.DataFrame(vec, columns=FEATURE_COLUMNS)

                # Select only trained features
                if self.selected_features is not None:
                    # Check overlap
                    valid_feats = [f for f in self.selected_features if f in df.columns]
                    X_input = df[valid_feats]
                else:
                    X_input = df

                # Scale
                X_scaled = self.scaler.transform(X_input)

                # Predict
                prob = self.model.predict_proba(X_scaled)[0]
                pred = np.argmax(prob)
                
                if prob[1] > CONFIG['PROB_THRESHOLD']:
                    return 1
                elif pred == 1 and prob[1] < CONFIG['PROB_THRESHOLD']:
                    return 0
                
                return pred
            except Exception as e:
                # logger.error(f"Prediction Error: {e}")
                pass
        
        # Fallback (simple)
        return 0


# --- MAIN RUNTIME ---

def get_hand_in_aruco_space(hand_norm, rvec, tvec, cam_mat, dist_coeffs):
    """
    Projects the Hand (Normalized) onto the ArUco Plane (Z=0).
    Returns: x_aruco (meters)
    """
    if rvec is None or tvec is None:
        return None

    # 1. Undistort/Unproject 2D Point to Camera Ray
    # Normalized (0-1) to Pixel
    u = hand_norm[0] * CONFIG['WIDTH']
    v = hand_norm[1] * CONFIG['HEIGHT']
    
    # Simple Pinhole Inverse
    fx = cam_mat[0, 0]
    fy = cam_mat[1, 1]
    cx = cam_mat[0, 2]
    cy = cam_mat[1, 2]
    
    x_ray = (u - cx) / fx
    y_ray = (v - cy) / fy
    z_ray = 1.0
    ray_cam = np.array([x_ray, y_ray, z_ray]) # Direction vector
    
    # 2. Plane Intersection
    R, _ = cv2.Rodrigues(rvec)
    r1 = R[:, 0]
    r2 = R[:, 1]
    
    A = np.column_stack((r1, r2, -ray_cam))
    b = -tvec.flatten()
    
    try:
        sol = np.linalg.solve(A, b)
        x_a = sol[0]
        return x_a
    except:
        return None

def main():
    # Init Modules
    cam = ThreadedCamera(CONFIG['CAM_ID'])
    audio = SoundEngine()
    predictor = PianoMotionPredictor()
    
    # CV2 / MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # State
    running = True
    last_state = 0 # Hover
    
    # ArUco Marker Points (for SolvePnP)
    s = CONFIG['MARKER_SIZE']
    obj_points = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]
    ], dtype=np.float32)

    logger.info("Starting Main Loop...")
    
    try:
        while running:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Copy for visualization
            vis_frame = frame.copy()
            
            # 1. Camera Intrinsics (Est)
            h, w = frame.shape[:2]
            f = w # Focal length approx
            K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
            D = np.zeros((4,1))
            
            # 2. ArUco Detection
            corners, ids, _ = aruco_detector.detectMarkers(frame)
            rvec, tvec = None, None
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
                # Find Marker 0
                for i, mid in enumerate(ids.flatten()):
                    if mid == 0: # Assuming Marker 0 is the anchor
                        success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], K, D)
                        if success:
                            cv2.drawFrameAxes(vis_frame, K, D, rvec, tvec, 0.05)
                            
                            # Draw Virtual Keys
                            # Start from right edge of marker
                            start_x = s + 0.01 # 1cm gap
                            for k in range(CONFIG['NUM_KEYS']):
                                k_x = start_x + (k * CONFIG['KEY_WIDTH'])
                                
                                # Project Key bounds
                                pts_3d = np.array([
                                    [k_x, -0.05, 0], [k_x + CONFIG['KEY_WIDTH'], -0.05, 0],
                                    [k_x + CONFIG['KEY_WIDTH'], 0.15, 0], [k_x, 0.15, 0]
                                ], dtype=np.float32)
                                
                                pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
                                pts_2d = np.int32(pts_2d).reshape(-1, 2)
                                cv2.polylines(vis_frame, [pts_2d], True, (255, 255, 0), 1)

            # 3. Hand Tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                # Get First Hand
                lm = results.multi_hand_landmarks[0].landmark
                
                # Draw
                mp.solutions.drawing_utils.draw_landmarks(vis_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                
                # Update Features
                predictor.update(lm)
                
                # Predict
                state = predictor.predict() # 0, 1, 2, 3
                
                # 4. Trigger Logic (Debounced in mind, or simple state change)
                # PRESS (1) or HOLD (2)? Usually sound on 0->1 or 0->2
                # If last was not press, and now is press
                
                # Get Spatial Position (Key Index)
                key_idx = -1
                if rvec is not None:
                    # Index Tip
                    tip_norm = np.array([lm[8].x, lm[8].y])
                    x_aruco = get_hand_in_aruco_space(tip_norm, rvec, tvec, K, D)
                    
                    if x_aruco is not None:
                         # Map to key
                         # x starts at Marker_Size + Gap
                         offset = x_aruco - (CONFIG['MARKER_SIZE'] + 0.01)
                         if offset > 0:
                             k_i = int(offset / CONFIG['KEY_WIDTH'])
                             if 0 <= k_i < CONFIG['NUM_KEYS']:
                                 key_idx = k_i
                                 
                                 # Visualize Touch
                                 cv2.putText(vis_frame, f"Key: {k_i}", (int(lm[8].x*w), int(lm[8].y*h)-20), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                
                # Trigger Sound
                if state == 1 and last_state != 1:
                    if key_idx != -1:
                        audio.play(key_idx)
                        cv2.circle(vis_frame, (int(lm[8].x*w), int(lm[8].y*h)), 15, (0, 255, 0), -1)
                
                # Visual Feedback of State
                state_str = ["HOVER", "PRESS", "HOLD", "RELEASE"][state]
                color = (0, 0, 255) if state == 0 else (0, 255, 0)
                cv2.putText(vis_frame, f"State: {state_str}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                last_state = state
            
            cv2.imshow("PianoMotion Final Runtime", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        pygame.quit()
        logger.info("Runtime Terminated.")

if __name__ == "__main__":
    main()
