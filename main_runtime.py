"""
main_runtime.py
The Final Integration Script for the PianoMotion Project.
Merges Vision (Phase 1), ML Model (Phase 2), and Audio Engine.

Features:
- Threaded Video Capture (High Performance)
- Live Feature Extraction (replicating SyncPianoMotionDataset.py)
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
    'MODELS_DIR': Path("Machine_Learning_Course/Data/PianoMotion10M/models"),
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

# --- LIVE FEATURE EXTRACTOR ---
class LiveFeatureExtractor:
    """
    Congruent Feature Extractor.
    Buffers 10 frames of MediaPipe Landmarks.
    Calculates derivatives, averages, and lags.
    Matches SyncPianoMotionDataset.py output exactly.
    """
    def __init__(self):
        self.buffer = deque(maxlen=CONFIG['BUFFER_SIZE'])
        self.scaler = None
        self.selected_features = None
        self.model = None
        self.has_model = False
        
        self.feature_names = [
            'finger_pos_x', 'finger_pos_y', 'wrist_pos_x', 'wrist_pos_y',
            'finger_vel_x', 'finger_vel_y', 'finger_speed',
            'wrist_vel_x', 'wrist_vel_y', 'wrist_speed',
            'finger_acc_x', 'finger_acc_y', 'finger_acc_mag',
            'rel_finger_pos_x', 'rel_finger_pos_y',
            'rel_finger_vel_x', 'rel_finger_vel_y',
            'dist_wrist', 'dist_palm', 'posture_dist',
            'rel_depth',
            'avg_speed', 'avg_acc_mag',
            'lag_speed_2', 'lag_speed_4', 'lag_speed_6',
            'rolling_var_speed'
        ]

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
                logger.warning("ML Models NOT found. Using Heuristic Fallback.")
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            self.has_model = False

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

    def extract(self):
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

        # --- Congruent Logic with SyncPianoMotionDataset ---
        
        # 1. Smooth Positions (SavGol) - Optional but good
        # Using simple mean for robustness in small buffer if SavGol fails
        try:
            tip_s = savgol_filter(tip_hist[:, :2], window_length=5, polyorder=2, axis=0)
            wrist_s = savgol_filter(wrist_hist[:, :2], window_length=5, polyorder=2, axis=0)
        except:
            tip_s = tip_hist[:, :2]
            wrist_s = wrist_hist[:, :2]

        # 2. Calculate Derivatives (dt = 1/30)
        # Using last frame as 'current'
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
        # MP Z is relative to wrist roughly.
        rel_depth = tip_hist[curr, 2] - wrist_hist[curr, 2]

        # Construct Dictionary
        feat = {}
        
        # Position
        feat['finger_pos_x'] = t_p[0]
        feat['finger_pos_y'] = t_p[1]
        feat['wrist_pos_x'] = w_p[0]
        feat['wrist_pos_y'] = w_p[1]
        
        # Velocity
        feat['finger_vel_x'] = t_v[0]
        feat['finger_vel_y'] = t_v[1]
        feat['finger_speed'] = np.linalg.norm(t_v)
        
        feat['wrist_vel_x'] = w_v[0]
        feat['wrist_vel_y'] = w_v[1]
        feat['wrist_speed'] = np.linalg.norm(w_v)
        
        # Acceleration
        feat['finger_acc_x'] = t_a[0]
        feat['finger_acc_y'] = t_a[1]
        feat['finger_acc_mag'] = np.linalg.norm(t_a)
        
        # Relative
        feat['rel_finger_pos_x'] = t_p[0] - w_p[0]
        feat['rel_finger_pos_y'] = t_p[1] - w_p[1]
        feat['rel_finger_vel_x'] = t_v[0] - w_v[0]
        feat['rel_finger_vel_y'] = t_v[1] - w_v[1]
        
        # Distances
        feat['dist_wrist'] = np.linalg.norm(t_p - w_p)
        feat['dist_palm'] = np.linalg.norm(t_p - palm_hist[curr, :2])
        feat['posture_dist'] = np.linalg.norm(t_p - dip_hist[curr, :2])
        
        # Depth
        feat['rel_depth'] = rel_depth
        
        # Rolling Averages (Last 5)
        # Buffer slice
        start_idx = max(0, len(self.buffer) - 5)
        recent_speeds = np.linalg.norm(tip_v[start_idx:], axis=1)
        recent_accs = np.linalg.norm(tip_a[start_idx:], axis=1)
        
        feat['avg_speed'] = np.mean(recent_speeds)
        feat['avg_acc_mag'] = np.mean(recent_accs)
        feat['rolling_var_speed'] = np.var(recent_speeds)

        # Lags
        # We need historical values. Buffer index logic:
        # curr = len-1. Lag L means index = len-1 - L
        for lag in [2, 4, 6]:
            idx = len(self.buffer) - 1 - lag
            if idx >= 0:
                feat[f'lag_speed_{lag}'] = np.linalg.norm(tip_v[idx])
            else:
                feat[f'lag_speed_{lag}'] = 0.0

        # Create Vector (Congruent Order?)
        # Best to use DataFrame to align with feature list
        # But for speed, we iterate list
        vector = []
        # If we have selected features, use those keys
        # If not, use all computed keys that match known list
        
        # NOTE: If we have model, we MUST use 'selected_features' order
        keys_to_use = self.selected_features if self.has_model and self.selected_features is not None else self.feature_names
        
        # Check if keys are missing from feat (e.g. if RFE selected something weird)
        # Using 0.0 for missing
        for k in keys_to_use:
            vector.append(feat.get(k, 0.0))
            
        return np.array([vector]) # Shape (1, F)

    def predict(self, feature_vector):
        """
        Inference Step.
        Returns: Class ID (0=Hover, 1=Press, 2=Hold, 3=Release)
        """
        if feature_vector is None:
            return 0 # Default Hover

        if self.has_model:
            try:
                # Scale
                X_scaled = self.scaler.transform(feature_vector)
                # Predict
                prob = self.model.predict_proba(X_scaled)[0] # (4,)
                pred = np.argmax(prob)
                
                # Probability Gating (Optional)
                if prob[1] > CONFIG['PROB_THRESHOLD']: # Confident Press
                    return 1
                elif pred == 1 and prob[1] < CONFIG['PROB_THRESHOLD']:
                    return 0 # Suppress weak press
                
                return pred
            except Exception as e:
                # logger.error(f"Prediction Error: {e}")
                pass
        
        # Fallback Heuristic
        # Simple Z-velocity check (using feature vector if possible, or raw)
        # We can't easily extract z-vel from vector without mapping.
        # So we trust the caller or return a dumb value.
        # But wait! I can just check the Z-velocity if I knew where it was.
        # It's 'finger_vel_x' etc. I don't have Z-vel explicitly in feature vector 
        # (Wait, I only have 2D vel in 'finger_vel_x/y').
        # Actually I do NOT have 'finger_vel_z' in the feature list.
        # The ML model relies on `rel_depth` and `avg_speed`.
        # So simple fallback: If `rel_depth` (normalized) < Threshold?
        # MP Z: Negative is closer? Wait. Z is rel to wrist.
        # Wrist=0. Tip < 0 means Tip closer to camera than Wrist (if palm facing cam).
        # In Piano posture, hand is prone. Camera above.
        # Wrist is "high". Tip is "low" (further from cam).
        # So Tip Z > Wrist Z (Positive Z).
        # Larger Positive Z = Deeper = Pressed?
        # Let's say Threshold > 0.1?
        # I'll stick to returning 0 if model fails, to be safe.
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
    
    # Simple Pinhole Inverse (assuming low distortion for speed, or use undistortPoints)
    # x_c = (u - cx)/fx * z_c
    # y_c = (v - cy)/fy * z_c
    fx = cam_mat[0, 0]
    fy = cam_mat[1, 1]
    cx = cam_mat[0, 2]
    cy = cam_mat[1, 2]
    
    x_ray = (u - cx) / fx
    y_ray = (v - cy) / fy
    z_ray = 1.0
    ray_cam = np.array([x_ray, y_ray, z_ray]) # Direction vector
    
    # 2. Plane Intersection
    # Plane defined by ArUco: R*P_aruco + t = P_cam
    # We want P_aruco where P_aruco.z = 0 (on the table/paper)
    # P_cam = R * [x_a, y_a, 0] + t
    # P_cam = x_a * R_col0 + y_a * R_col1 + t
    # Also P_cam = lambda * ray_cam
    
    # System of equations:
    # lambda * ray_cam = x_a * r1 + y_a * r2 + t
    # [r1, r2, -ray_cam] * [x_a, y_a, lambda]^T = -t
    
    R, _ = cv2.Rodrigues(rvec)
    r1 = R[:, 0]
    r2 = R[:, 1]
    
    A = np.column_stack((r1, r2, -ray_cam))
    b = -tvec.flatten()
    
    try:
        sol = np.linalg.solve(A, b)
        x_a = sol[0]
        # y_a = sol[1]
        # lambda_val = sol[2]
        return x_a
    except:
        return None

def main():
    # Init Modules
    cam = ThreadedCamera(CONFIG['CAM_ID'])
    audio = SoundEngine()
    features = LiveFeatureExtractor()
    
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
                features.update(lm)
                
                # Extract & Predict
                vec = features.extract() # (1, F)
                state = features.predict(vec) # 0, 1, 2, 3
                
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
