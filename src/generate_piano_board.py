import cv2
import numpy as np
from cv2 import aruco

# --- SHARED CONFIGURATION (THE SOURCE OF TRUTH) ---
# Units = Pixels. We draw in pixels, and the Vision Engine treats 1 Pixel = 1 Unit.
CONFIG = {
    'MARKER_SIZE': 200,      # Large 200px marker
    'SAFETY_GAP': 50,        # White space around marker (Critical for detection)
    'BORDER_WIDTH': 20,      # Blue decorative border
    'KEY_WIDTH': 100,        # Width of one white key
    'KEY_HEIGHT': 400,       # Vertical length of keys
    'NUM_KEYS': 7,           # A-G
    'MARGIN': 100            # Paper margins
}

def generate_board():
    # 1. Calculate Canvas Size
    piano_width = CONFIG['KEY_WIDTH'] * CONFIG['NUM_KEYS']
    
    # Structure: [Marker0] [Gap] [Border] [Piano] [Border] [Gap] [Marker1]
    internal_width = (CONFIG['MARKER_SIZE'] * 2) + \
                     (CONFIG['SAFETY_GAP'] * 2) + \
                     (CONFIG['BORDER_WIDTH'] * 2) + \
                     piano_width
                     
    img_w = internal_width + (CONFIG['MARGIN'] * 2)
    img_h = max(CONFIG['MARKER_SIZE'], CONFIG['KEY_HEIGHT']) + (CONFIG['MARGIN'] * 2) + 100
    
    # White Background
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # 2. Setup ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # 3. Draw Elements (Left to Right)
    cursor_x = CONFIG['MARGIN']
    cursor_y = CONFIG['MARGIN']
    
    # --- LEFT MARKER (ID 0) ---
    # This Top-Left corner (cursor_x, cursor_y) is the WORLD ORIGIN (0,0) in the Vision Engine
    marker0 = aruco.generateImageMarker(aruco_dict, 0, CONFIG['MARKER_SIZE'])
    marker0 = cv2.cvtColor(marker0, cv2.COLOR_GRAY2BGR)
    img[cursor_y:cursor_y+CONFIG['MARKER_SIZE'], cursor_x:cursor_x+CONFIG['MARKER_SIZE']] = marker0
    cursor_x += CONFIG['MARKER_SIZE']
    
    # --- SAFETY GAP (White) ---
    cursor_x += CONFIG['SAFETY_GAP']
    
    # --- BLUE BORDER (Left) ---
    cv2.rectangle(img, (cursor_x, cursor_y), 
                  (cursor_x + CONFIG['BORDER_WIDTH'], cursor_y + CONFIG['KEY_HEIGHT']), 
                  (255, 0, 0), -1)
    cursor_x += CONFIG['BORDER_WIDTH']
    
    # --- PIANO KEYS ---
    for i in range(CONFIG['NUM_KEYS']):
        # Draw Key
        top_left = (cursor_x, cursor_y)
        bottom_right = (cursor_x + CONFIG['KEY_WIDTH'], cursor_y + CONFIG['KEY_HEIGHT'])
        cv2.rectangle(img, top_left, bottom_right, (0,0,0), 3)
        
        # Label
        label = chr(65 + i) # A, B, C...
        cv2.putText(img, label, (cursor_x + 30, cursor_y + 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        
        cursor_x += CONFIG['KEY_WIDTH']
        
    # --- BLUE BORDER (Right) ---
    cv2.rectangle(img, (cursor_x, cursor_y), 
                  (cursor_x + CONFIG['BORDER_WIDTH'], cursor_y + CONFIG['KEY_HEIGHT']), 
                  (255, 0, 0), -1)
    cursor_x += CONFIG['BORDER_WIDTH']
    
    # --- SAFETY GAP (Right) ---
    cursor_x += CONFIG['SAFETY_GAP']
    
    # --- RIGHT MARKER (ID 1) ---
    marker1 = aruco.generateImageMarker(aruco_dict, 1, CONFIG['MARKER_SIZE'])
    marker1 = cv2.cvtColor(marker1, cv2.COLOR_GRAY2BGR)
    img[cursor_y:cursor_y+CONFIG['MARKER_SIZE'], cursor_x:cursor_x+CONFIG['MARKER_SIZE']] = marker1
    
    # 4. Output
    cv2.imwrite("piano_board.png", img)
    print(f"✅ Generated 'piano_board.png' ({img_w}x{img_h})")
    print("   Print this image at 100% scale.")

if __name__ == "__main__":
    generate_board()