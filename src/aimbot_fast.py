"""
CS2 Aimbot - ULTRA OPTIMIZED + STABLE TRACKING
===============================================
Smooth, stable aim with fire persistence.

CONTROLS:
- Hold RIGHT MOUSE BUTTON - Activate aimbot
- Press END key - Quit
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

MODEL_PATH = "models/best.engine"

# Screen setup
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
CAPTURE_SIZE = 800

CAPTURE_REGION = {
    "left": (SCREEN_WIDTH - CAPTURE_SIZE) // 2,
    "top": (SCREEN_HEIGHT - CAPTURE_SIZE) // 2,
    "width": CAPTURE_SIZE,
    "height": CAPTURE_SIZE
}

# Detection
CONFIDENCE_THRESHOLD = 0.35

# =============================================================================
# AIM SETTINGS - TUNED FOR STABLE TRACKING
# =============================================================================

AIM_SMOOTHING = 0.6       # How fast to move (0.1=slow, 1.0=instant)
AIM_DEADZONE = 3          # Pixels - don't move if closer than this
TARGET_SMOOTHING = 0.3    # Smooth target position to reduce jitter

HEADSHOT_OFFSET = 0.18    # Aim at 18% from top of box (head area)
IGNORE_BOTTOM_PERCENT = 0.25  # Ignore bottom 25% (your own feet)

# =============================================================================
# FIRING SETTINGS - MORE FORGIVING
# =============================================================================

AUTO_FIRE = True
LOCK_THRESHOLD = 40       # Fire when within 40px (was 15 - more forgiving now)
FIRE_DELAY = 0.06         # 60ms between shots

# Fire persistence - keep firing briefly even if target jitters
FIRE_PERSIST_TIME = 0.15  # Keep firing for 150ms after losing lock

# Target memory - remember target position if detection drops briefly
TARGET_MEMORY_TIME = 0.1  # Remember target for 100ms if lost

# Activation
ACTIVATION_KEY = win32con.VK_RBUTTON

# =============================================================================
# TRACKING STATE
# =============================================================================

SCREEN_CENTER_X = SCREEN_WIDTH // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2

# Smoothed target
smooth_x = SCREEN_CENTER_X
smooth_y = SCREEN_CENTER_Y

# Tracking state
last_target_time = 0      # When we last saw a target
last_fire_time = 0        # When we last fired
last_lock_time = 0        # When we were last locked on
is_tracking = False       # Are we currently tracking a target


# =============================================================================
# FUNCTIONS
# =============================================================================

def move_mouse(dx, dy):
    dx = int(round(dx))
    dy = int(round(dy))
    if dx or dy:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)


def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def is_pressed(key):
    return win32api.GetAsyncKeyState(key) & 0x8000


def get_best_target(results):
    """Find closest valid target."""
    best_dist_sq = float('inf')
    best_x = None
    best_y = None

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Head position
            target_x = (x1 + x2) / 2
            target_y = y1 + (y2 - y1) * HEADSHOT_OFFSET

            # Skip bottom of screen (your feet)
            if target_y > CAPTURE_SIZE * (1 - IGNORE_BOTTOM_PERCENT):
                continue

            # Convert to screen coords
            screen_x = CAPTURE_REGION["left"] + target_x
            screen_y = CAPTURE_REGION["top"] + target_y

            # Distance to crosshair
            dx = screen_x - SCREEN_CENTER_X
            dy = screen_y - SCREEN_CENTER_Y
            dist_sq = dx * dx + dy * dy

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_x = screen_x
                best_y = screen_y

    if best_x is not None:
        return best_x, best_y
    return None


def update_aim(target):
    """
    Update aim with smoothing and memory.
    Returns (move_x, move_y, distance, should_fire)
    """
    global smooth_x, smooth_y, is_tracking, last_target_time, last_lock_time

    now = time.perf_counter()

    if target:
        target_x, target_y = target
        last_target_time = now

        if not is_tracking:
            # First frame - snap closer to target
            smooth_x = SCREEN_CENTER_X + (target_x - SCREEN_CENTER_X) * 0.5
            smooth_y = SCREEN_CENTER_Y + (target_y - SCREEN_CENTER_Y) * 0.5
            is_tracking = True
        else:
            # Smooth the target position
            smooth_x += (target_x - smooth_x) * (1 - TARGET_SMOOTHING)
            smooth_y += (target_y - smooth_y) * (1 - TARGET_SMOOTHING)

    elif is_tracking and (now - last_target_time) < TARGET_MEMORY_TIME:
        # No target this frame, but remember last position briefly
        pass  # Keep using smooth_x, smooth_y from last frame

    else:
        # Lost target
        is_tracking = False
        return 0, 0, 999, False

    # Calculate movement
    dx = smooth_x - SCREEN_CENTER_X
    dy = smooth_y - SCREEN_CENTER_Y
    distance = (dx * dx + dy * dy) ** 0.5

    # Check if we should fire
    should_fire = False
    if distance < LOCK_THRESHOLD:
        last_lock_time = now
        should_fire = True
    elif (now - last_lock_time) < FIRE_PERSIST_TIME:
        # We were locked recently - keep firing (persistence)
        should_fire = True

    # Deadzone
    if distance < AIM_DEADZONE:
        return 0, 0, distance, should_fire

    # Apply smoothing to movement
    move_x = dx * AIM_SMOOTHING
    move_y = dy * AIM_SMOOTHING

    return move_x, move_y, distance, should_fire


def reset_tracking():
    """Reset all tracking state."""
    global is_tracking, last_target_time, last_lock_time
    is_tracking = False
    last_target_time = 0
    last_lock_time = 0


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    global last_fire_time

    print("=" * 60)
    print("CS2 AIMBOT - STABLE TRACKING")
    print("=" * 60)
    print(f"Lock threshold: {LOCK_THRESHOLD}px")
    print(f"Fire persistence: {FIRE_PERSIST_TIME*1000:.0f}ms")
    print(f"Target memory: {TARGET_MEMORY_TIME*1000:.0f}ms")
    print()
    print("Hold RIGHT MOUSE to aim | END to quit")
    print("=" * 60)

    # Load model
    print("\nLoading TensorRT engine...")
    model = YOLO(MODEL_PATH, task='detect')

    # Warm up
    print("Warming up...")
    dummy = np.zeros((CAPTURE_SIZE, CAPTURE_SIZE, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False, imgsz=416)
    print("Ready!\n")

    frame_count = 0
    start_time = time.perf_counter()

    with mss.mss() as sct:
        print("Running...")

        while True:
            if is_pressed(win32con.VK_END):
                break

            # Capture
            frame = np.array(sct.grab(CAPTURE_REGION))[:, :, :3]

            # Detect
            results = model(frame, verbose=False, imgsz=416)

            # Aim
            if is_pressed(ACTIVATION_KEY):
                target = get_best_target(results)
                move_x, move_y, dist, should_fire = update_aim(target)

                # Move
                if move_x or move_y:
                    move_mouse(move_x, move_y)

                # Fire
                if AUTO_FIRE and should_fire:
                    now = time.perf_counter()
                    if now - last_fire_time >= FIRE_DELAY:
                        click()
                        last_fire_time = now
            else:
                reset_tracking()

            # FPS
            frame_count += 1
            if frame_count >= 100:
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.0f} | Frame: {elapsed/frame_count*1000:.1f}ms")
                frame_count = 0
                start_time = time.perf_counter()

    print("\nStopped.")


if __name__ == "__main__":
    main()
