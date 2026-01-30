"""
CS2 Aimbot - Full Implementation (Optimized)
=============================================
Combines screen capture, YOLO detection, and mouse control.

CONTROLS:
- Hold RIGHT MOUSE BUTTON (ADS) to activate aimbot
- Release to deactivate
- Press 'END' key to quit

IMPORTANT: For educational/portfolio use only.
Only use on offline servers with bots.
"""

import cv2
import mss
import numpy as np
import time
import win32api
import win32con
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model path
MODEL_PATH = "models/best.pt"

# Screen capture region (2K monitor)
CAPTURE_REGION = {
    "left": 0,
    "top": 0,
    "width": 2560,
    "height": 1440
}

# Performance: Resize frame before inference (smaller = faster)
# Set to 1.0 for no resize, 0.5 for half size, etc.
INFERENCE_SCALE = 1.0  # Capture at full resolution

# YOLO inference size (smaller = faster, 640/480/416/320)
INFERENCE_SIZE = 416  # Sweet spot for speed vs accuracy

# Detection settings
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Aimbot settings
AIM_SPEED = 0.8          # 0.0-1.0, higher = faster (increased for responsiveness)
HEADSHOT_OFFSET = 0.15   # Aim at top 15% of box (head area)
AIM_FOV = 500            # Only aim at targets within this pixel radius from crosshair

# Auto-fire settings
AUTO_FIRE = True         # Set to False to disable auto-clicking
FIRE_DELAY = 0.05        # Minimum seconds between shots
LOCK_THRESHOLD = 30      # Fire when within this many pixels of target

# Activation key
ACTIVATION_KEY = win32con.VK_RBUTTON  # Right mouse button


# =============================================================================
# MOUSE CONTROL
# =============================================================================

def get_screen_center():
    """Get the center of the capture region (crosshair position)."""
    return (
        CAPTURE_REGION["left"] + CAPTURE_REGION["width"] // 2,
        CAPTURE_REGION["top"] + CAPTURE_REGION["height"] // 2
    )


def move_mouse_relative(dx, dy):
    """Move mouse by relative amount using raw input."""
    dx = int(dx)
    dy = int(dy)
    if dx != 0 or dy != 0:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)


def click_mouse():
    """Perform a mouse click."""
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def is_key_pressed(key):
    """Check if a key is currently pressed."""
    return win32api.GetAsyncKeyState(key) & 0x8000 != 0


# =============================================================================
# TARGET SELECTION
# =============================================================================

def get_target_point(box):
    """
    Get the aim point for a detection box.
    YOLO returns coordinates in original image space, so no scaling needed.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Center X
    target_x = (x1 + x2) // 2

    # Head Y (offset from top)
    box_height = y2 - y1
    target_y = y1 + int(box_height * HEADSHOT_OFFSET)

    return target_x, target_y


def distance_to_crosshair(target_x, target_y):
    """Calculate distance from target to crosshair (screen center)."""
    center_x, center_y = get_screen_center()
    return np.sqrt((target_x - center_x)**2 + (target_y - center_y)**2)


def select_best_target(results):
    """
    Select the best target from detections.
    Returns the target closest to the crosshair within FOV.
    """
    best_target = None
    best_distance = float('inf')

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            target_x, target_y = get_target_point(box)
            distance = distance_to_crosshair(target_x, target_y)

            # Only consider targets within FOV
            if distance > AIM_FOV:
                continue

            # Select closest target
            if distance < best_distance:
                best_distance = distance
                best_target = (target_x, target_y, conf, distance)

    return best_target


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("CS2 AIMBOT (OPTIMIZED)")
    print("=" * 60)
    print()
    print("CONTROLS:")
    print("  Hold RIGHT MOUSE BUTTON - Activate aimbot")
    print("  Press END key          - Quit")
    print()
    print("SETTINGS:")
    print(f"  Capture: {CAPTURE_REGION['width']}x{CAPTURE_REGION['height']}")
    print(f"  Inference size: {INFERENCE_SIZE}px")
    print(f"  Aim Speed: {AIM_SPEED}")
    print(f"  Aim FOV: {AIM_FOV}px")
    print(f"  Auto-fire: {'ON' if AUTO_FIRE else 'OFF'}")
    print()
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = YOLO(MODEL_PATH)
    model.to('cuda')  # Ensure GPU
    print("Model loaded on GPU!")

    # Warm up
    print("Warming up...")
    dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
    for _ in range(5):
        model(dummy, verbose=False, half=True, imgsz=INFERENCE_SIZE)
    print("Ready!\n")

    # Tracking
    last_fire_time = 0
    frame_times = []

    with mss.mss() as sct:
        print("Aimbot running. Hold RIGHT MOUSE to aim. Press END to quit.")
        print("-" * 60)

        while True:
            loop_start = time.perf_counter()

            # Check for quit key (END)
            if is_key_pressed(win32con.VK_END):
                print("\nEND key pressed. Quitting...")
                break

            # Capture frame
            screenshot = sct.grab(CAPTURE_REGION)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run detection (use half precision and smaller imgsz for speed)
            # YOLO handles resizing internally - more efficient
            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
                half=True,
                imgsz=INFERENCE_SIZE  # Smaller = faster
            )

            # Check if activation key is held
            aimbot_active = is_key_pressed(ACTIVATION_KEY)

            if aimbot_active:
                # Select best target
                target = select_best_target(results)

                if target:
                    target_x, target_y, conf, distance = target
                    center_x, center_y = get_screen_center()

                    # Calculate movement needed
                    dx = (target_x - center_x) * AIM_SPEED
                    dy = (target_y - center_y) * AIM_SPEED

                    # Move mouse
                    move_mouse_relative(dx, dy)

                    # Auto-fire if close enough to target
                    if AUTO_FIRE and distance < LOCK_THRESHOLD:
                        current_time = time.time()
                        if current_time - last_fire_time >= FIRE_DELAY:
                            click_mouse()
                            last_fire_time = current_time

            # FPS tracking
            frame_time = time.perf_counter() - loop_start
            frame_times.append(frame_time)

            if len(frame_times) >= 30:
                avg_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                detections = sum(len(r.boxes) for r in results)
                status = "ACTIVE" if aimbot_active else "READY"
                print(f"[{status}] FPS: {fps:.0f} | Frame time: {avg_time*1000:.1f}ms | Detections: {detections}")
                frame_times = []

    print("\nAimbot stopped.")


if __name__ == "__main__":
    main()
