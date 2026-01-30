"""
Data Collection Script
======================
Captures screenshots from CS 1.6 for training YOLO.

Usage:
1. Start CS 1.6 in windowed mode
2. Run this script
3. Press 'S' to save a screenshot when you see enemies
4. Press 'Q' to quit

Tips for good training data:
- Capture enemies at different distances (close, medium, far)
- Capture different player models (T and CT)
- Capture on different maps
- Capture enemies in different poses (standing, crouching, running)
- Capture some images WITHOUT enemies (negative samples)
- Aim for 200-300 images total
"""

import cv2
import mss
import numpy as np
import time
import os
from datetime import datetime


# =============================================================================
# CONFIGURATION - ADJUST FOR YOUR CS 1.6 WINDOW
# =============================================================================

# Set this to match your CS 1.6 window position and size
# Run CS 1.6 in windowed mode at 800x600 for best results
GAME_REGION = {
    "left": 0,      # X position of game window (adjust this!)
    "top": 0,       # Y position of game window (adjust this!)
    "width": 800,   # CS 1.6 width
    "height": 600   # CS 1.6 height
}

# Where to save screenshots
SAVE_DIR = "data/images/raw"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Screenshots will be saved to: {SAVE_DIR}")


def generate_filename():
    """Generate a unique filename based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"cs16_{timestamp}.png"


def save_screenshot(frame, filename):
    """Save a screenshot to disk."""
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def draw_overlay(frame, count, fps):
    """Draw helpful information on the preview."""
    h, w = frame.shape[:2]

    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Text info
    cv2.putText(frame, f"Screenshots: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "S=Save | Q=Quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Crosshair
    center_x, center_y = w // 2, h // 2
    cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 255), 1)
    cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 255), 1)

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CS 1.6 DATA COLLECTION")
    print("=" * 60)
    print()
    print("INSTRUCTIONS:")
    print("1. Start CS 1.6 in WINDOWED mode (800x600)")
    print("2. Position the game window at the top-left of your screen")
    print("   (or adjust GAME_REGION in this script)")
    print("3. Join a game with bots")
    print("4. Press 'S' when you see enemies to save screenshots")
    print("5. Press 'Q' to quit")
    print()
    print("TIPS FOR GOOD DATA:")
    print("- Get enemies at different distances")
    print("- Get different maps (dust2, inferno, etc.)")
    print("- Get enemies standing, crouching, running")
    print("- Get some images WITHOUT enemies too (10-20%)")
    print("- Aim for 200-300 total images")
    print()
    print("=" * 60)

    ensure_directories()

    screenshot_count = 0
    fps_data = {'counter': 0, 'start_time': time.time(), 'current_fps': 0}

    with mss.mss() as sct:
        print(f"\nCapturing region: {GAME_REGION}")
        print("Starting capture... (Press 'S' to save, 'Q' to quit)\n")

        while True:
            # Capture frame
            screenshot = sct.grab(GAME_REGION)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Keep a clean copy for saving (no overlay)
            clean_frame = frame.copy()

            # Calculate FPS
            fps_data['counter'] += 1
            elapsed = time.time() - fps_data['start_time']
            if elapsed >= 1.0:
                fps_data['current_fps'] = fps_data['counter'] / elapsed
                fps_data['counter'] = 0
                fps_data['start_time'] = time.time()

            # Draw overlay on display copy
            display_frame = draw_overlay(frame, screenshot_count, fps_data['current_fps'])

            # Show preview
            cv2.imshow("Data Collection - S=Save, Q=Quit", display_frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                filename = generate_filename()
                filepath = save_screenshot(clean_frame, filename)
                screenshot_count += 1
                print(f"[{screenshot_count}] Saved: {filename}")

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(f"Collection complete! Saved {screenshot_count} screenshots.")
    print(f"Location: {os.path.abspath(SAVE_DIR)}")
    print()
    print("NEXT STEPS:")
    print("1. Open the images in a labeling tool (LabelImg or Roboflow)")
    print("2. Draw bounding boxes around enemies")
    print("3. Export in YOLO format")
    print("=" * 60)


if __name__ == "__main__":
    main()
