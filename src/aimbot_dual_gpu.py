"""
CS2 Aimbot - DUAL GPU MODE
==========================
Dedicated inference GPU for maximum performance.
Run game on GPU 0, inference on GPU 1.

SETUP:
1. Export TensorRT engine on GPU 1 (see docs/DUAL_GPU_SETUP.md)
2. Update MODEL_PATH and INFERENCE_GPU below

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
import torch
from ultralytics import YOLO
from threading import Thread, Lock


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "models/best_gpu1.engine"  # TensorRT engine exported on GPU 1
INFERENCE_GPU = 1  # Dedicated GPU for inference (0 = first GPU, 1 = second GPU)

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
CAPTURE_SIZE = 640

CAPTURE_REGION = {
    "left": (SCREEN_WIDTH - CAPTURE_SIZE) // 2,
    "top": (SCREEN_HEIGHT - CAPTURE_SIZE) // 2,
    "width": CAPTURE_SIZE,
    "height": CAPTURE_SIZE
}

# Detection
CONFIDENCE_THRESHOLD = 0.40
INFERENCE_SIZE = 416  # Must match TensorRT engine export size

# =============================================================================
# AIM SETTINGS
# =============================================================================

AIM_SMOOTHING = 0.65
AIM_DEADZONE = 3
TARGET_SMOOTHING = 0.25

HEADSHOT_OFFSET = 0.18
IGNORE_BOTTOM_PERCENT = 0.25

# Firing
AUTO_FIRE = True
LOCK_THRESHOLD = 45
FIRE_DELAY = 0.05
FIRE_PERSIST_TIME = 0.15
TARGET_MEMORY_TIME = 0.1

ACTIVATION_KEY = win32con.VK_RBUTTON

# =============================================================================
# THREADED CAPTURE
# =============================================================================

class FrameGrabber:
    """Captures frames in a separate thread."""

    def __init__(self, region):
        self.region = region
        self.frame = None
        self.lock = Lock()
        self.running = True

    def start(self):
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        with mss.mss() as sct:
            while self.running:
                screenshot = sct.grab(self.region)
                frame = np.array(screenshot)[:, :, :3]

                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# =============================================================================
# TRACKING STATE
# =============================================================================

SCREEN_CENTER_X = SCREEN_WIDTH // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2

smooth_x = SCREEN_CENTER_X
smooth_y = SCREEN_CENTER_Y
last_target_time = 0
last_fire_time = 0
last_lock_time = 0
is_tracking = False


# =============================================================================
# FUNCTIONS
# =============================================================================

def move_mouse(dx, dy):
    dx, dy = int(round(dx)), int(round(dy))
    if dx or dy:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)


def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def is_pressed(key):
    return win32api.GetAsyncKeyState(key) & 0x8000


def get_best_target(results):
    best_dist_sq = float('inf')
    best_x = best_y = None

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            target_x = (x1 + x2) / 2
            target_y = y1 + (y2 - y1) * HEADSHOT_OFFSET

            if target_y > CAPTURE_SIZE * (1 - IGNORE_BOTTOM_PERCENT):
                continue

            screen_x = CAPTURE_REGION["left"] + target_x
            screen_y = CAPTURE_REGION["top"] + target_y

            dx = screen_x - SCREEN_CENTER_X
            dy = screen_y - SCREEN_CENTER_Y
            dist_sq = dx * dx + dy * dy

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_x, best_y = screen_x, screen_y

    return (best_x, best_y) if best_x else None


def update_aim(target):
    global smooth_x, smooth_y, is_tracking, last_target_time, last_lock_time

    now = time.perf_counter()

    if target:
        target_x, target_y = target
        last_target_time = now

        if not is_tracking:
            smooth_x = SCREEN_CENTER_X + (target_x - SCREEN_CENTER_X) * 0.6
            smooth_y = SCREEN_CENTER_Y + (target_y - SCREEN_CENTER_Y) * 0.6
            is_tracking = True
        else:
            smooth_x += (target_x - smooth_x) * (1 - TARGET_SMOOTHING)
            smooth_y += (target_y - smooth_y) * (1 - TARGET_SMOOTHING)

    elif is_tracking and (now - last_target_time) < TARGET_MEMORY_TIME:
        pass
    else:
        is_tracking = False
        return 0, 0, 999, False

    dx = smooth_x - SCREEN_CENTER_X
    dy = smooth_y - SCREEN_CENTER_Y
    distance = (dx * dx + dy * dy) ** 0.5

    should_fire = False
    if distance < LOCK_THRESHOLD:
        last_lock_time = now
        should_fire = True
    elif (now - last_lock_time) < FIRE_PERSIST_TIME:
        should_fire = True

    if distance < AIM_DEADZONE:
        return 0, 0, distance, should_fire

    return dx * AIM_SMOOTHING, dy * AIM_SMOOTHING, distance, should_fire


def reset_tracking():
    global is_tracking, last_target_time, last_lock_time
    is_tracking = False
    last_target_time = last_lock_time = 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    global last_fire_time

    print("=" * 60)
    print("CS2 AIMBOT - DUAL GPU MODE")
    print("=" * 60)

    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    print(f"\nDetected GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_count < 2:
        print("\nWARNING: Only 1 GPU detected!")
        print("This script is designed for dual GPU setup.")
        print("Falling back to GPU 0...")
        inference_gpu = 0
    else:
        inference_gpu = INFERENCE_GPU
        print(f"\nUsing GPU {inference_gpu} for inference")

    print()
    print(f"Capture: {CAPTURE_SIZE}x{CAPTURE_SIZE} (threaded)")
    print(f"Inference: {INFERENCE_SIZE}px on cuda:{inference_gpu}")
    print()
    print("Hold RIGHT MOUSE to aim | END to quit")
    print("=" * 60)

    # Load model on dedicated GPU
    print(f"\nLoading TensorRT engine on GPU {inference_gpu}...")

    # Clear any cached memory
    torch.cuda.empty_cache()

    model = YOLO(MODEL_PATH, task='detect')
    model.to(f'cuda:{inference_gpu}')

    # Verify model is on correct GPU
    print(f"Model loaded on: cuda:{inference_gpu}")

    # Warm up on the inference GPU
    print("Warming up...")
    dummy = np.zeros((CAPTURE_SIZE, CAPTURE_SIZE, 3), dtype=np.uint8)
    for _ in range(20):
        model(dummy, verbose=False, imgsz=INFERENCE_SIZE, device=inference_gpu)
    print("Starting capture thread...")

    # Start threaded capture
    grabber = FrameGrabber(CAPTURE_REGION)
    grabber.start()

    # Wait for first frame
    time.sleep(0.1)
    print("Ready!\n")

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            if is_pressed(win32con.VK_END):
                break

            # Get latest frame from capture thread
            frame = grabber.get_frame()
            if frame is None:
                continue

            # Run inference on dedicated GPU
            results = model(frame, verbose=False, imgsz=INFERENCE_SIZE, device=inference_gpu)

            # Process aim
            if is_pressed(ACTIVATION_KEY):
                target = get_best_target(results)
                move_x, move_y, dist, should_fire = update_aim(target)

                if move_x or move_y:
                    move_mouse(move_x, move_y)

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
                ms = elapsed / frame_count * 1000
                print(f"FPS: {fps:.0f} | Frame: {ms:.1f}ms | GPU: {inference_gpu}")
                frame_count = 0
                start_time = time.perf_counter()

    finally:
        grabber.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
