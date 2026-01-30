"""
Screen Capture Module
=====================
This module captures frames from your screen in real-time.

Key Concepts:
- mss grabs the screen as a numpy array
- OpenCV (cv2) displays the image and handles the window
- We measure FPS to ensure real-time performance
- ROI (Region of Interest) capture improves performance
"""

import cv2
import mss
import numpy as np
import time


# =============================================================================
# CONFIGURATION - Adjust these for your setup
# =============================================================================

# Option 1: Full screen capture (slower)
# Set to None to capture full screen
# CAPTURE_REGION = None

# Option 2: Specific region capture (faster)
# Format: {"left": x, "top": y, "width": w, "height": h}
# CS 1.6 default resolution is 800x600
# Adjust left/top to where your game window will be
CAPTURE_REGION = {
    "left": 100,    # X position of top-left corner
    "top": 100,     # Y position of top-left corner
    "width": 800,   # Width to capture
    "height": 600   # Height to capture
}


def get_capture_region(sct):
    """
    Returns the region to capture.
    If CAPTURE_REGION is None, returns the primary monitor.
    """
    if CAPTURE_REGION is None:
        return sct.monitors[1]  # Primary monitor
    return CAPTURE_REGION


def calculate_fps(fps_data):
    """
    Calculate and update FPS.

    Args:
        fps_data: dict with 'counter', 'start_time', 'current_fps'

    Returns:
        Updated fps_data dict
    """
    fps_data['counter'] += 1
    elapsed = time.time() - fps_data['start_time']

    if elapsed >= 1.0:
        fps_data['current_fps'] = fps_data['counter'] / elapsed
        fps_data['counter'] = 0
        fps_data['start_time'] = time.time()

    return fps_data


def draw_info(frame, fps, region):
    """
    Draw information overlay on the frame.
    """
    # FPS counter
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # Capture region info
    cv2.putText(
        frame, f"Region: {region['width']}x{region['height']}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # Instructions
    cv2.putText(
        frame, "Press 'q' to quit | 'f' for fullscreen toggle", (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    return frame


def main():
    """
    Main capture loop.
    """
    print("=" * 50)
    print("SCREEN CAPTURE TEST")
    print("=" * 50)

    with mss.mss() as sct:
        region = get_capture_region(sct)

        print(f"Capturing region: {region}")
        print(f"Resolution: {region['width']}x{region['height']}")
        print("-" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'f' - Toggle fullscreen capture")
        print("-" * 50)

        # FPS tracking
        fps_data = {
            'counter': 0,
            'start_time': time.time(),
            'current_fps': 0
        }

        # State
        fullscreen_mode = (CAPTURE_REGION is None)

        while True:
            # Determine capture region
            if fullscreen_mode:
                current_region = sct.monitors[1]
            else:
                current_region = CAPTURE_REGION

            # Capture
            screenshot = sct.grab(current_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Update FPS
            fps_data = calculate_fps(fps_data)

            # Draw overlay
            frame = draw_info(frame, fps_data['current_fps'], current_region)

            # Draw crosshair in center (useful for aiming reference)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
            cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)

            # Display
            cv2.imshow("Screen Capture", frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('f'):
                fullscreen_mode = not fullscreen_mode
                mode_str = "FULLSCREEN" if fullscreen_mode else "ROI (800x600)"
                print(f"Switched to: {mode_str}")

        cv2.destroyAllWindows()
        print("Capture stopped.")


if __name__ == "__main__":
    main()
