"""
Live Detection Script
=====================
Runs YOLO inference on screen capture in real-time.

This combines:
1. Screen capture (from Phase 2)
2. YOLO model (from Phase 4)
3. Visualization (bounding boxes + confidence)
"""

import cv2
import mss
import numpy as np
import time
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to trained model
MODEL_PATH = "models/best.pt"

# Capture region - adjust to match your CS2 window
# Set to None for fullscreen capture
CAPTURE_REGION = {
    "left": 0,
    "top": 0,
    "width": 2560,   # 2K monitor
    "height": 1440
}

# Detection settings
CONFIDENCE_THRESHOLD = 0.35  # Adjusted threshold
IOU_THRESHOLD = 0.45        # Non-max suppression threshold


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame: numpy array (BGR image)
        results: YOLO results object

    Returns:
        frame with detections drawn
    """
    # Get frame dimensions
    h, w = frame.shape[:2]

    # Process each detection
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Get coordinates (xyxy format: x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Skip low confidence detections
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Colors based on confidence
            if conf > 0.8:
                color = (0, 255, 0)    # Green - high confidence
            elif conf > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - lower confidence

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"Enemy {conf:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw center point (for aiming)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Draw head position (top center of box - for headshots)
            head_x = center_x
            head_y = y1 + int((y2 - y1) * 0.15)  # 15% from top
            cv2.circle(frame, (head_x, head_y), 5, (255, 0, 255), -1)

    return frame


def draw_info(frame, fps, inference_time, num_detections):
    """Draw performance information overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Info text
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {num_detections}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Q=Quit | C=Change region", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Crosshair in center
    center_x, center_y = w // 2, h // 2
    cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 255), 1)
    cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 255), 1)

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LIVE DETECTION")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")
    print()

    # Warm up the model (first inference is slow)
    print("Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    print("Ready!")
    print()

    print("Controls:")
    print("  Q - Quit")
    print("  C - Print capture region config")
    print()
    print("=" * 60)

    # FPS tracking
    fps_data = {'counter': 0, 'start_time': time.time(), 'current_fps': 0}

    with mss.mss() as sct:
        region = CAPTURE_REGION if CAPTURE_REGION else sct.monitors[1]
        print(f"Capturing: {region['width']}x{region['height']}")

        while True:
            # Capture frame
            screenshot = sct.grab(region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run inference
            inference_start = time.time()
            results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            inference_time = (time.time() - inference_start) * 1000  # ms

            # Count detections
            num_detections = sum(len(r.boxes) for r in results)

            # Print detections to console (for single monitor setup)
            if num_detections > 0:
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        print(f"DETECTED: conf={conf:.0%} center=({center_x}, {center_y})")

            # Draw detections
            frame = draw_detections(frame, results)

            # Calculate FPS
            fps_data['counter'] += 1
            elapsed = time.time() - fps_data['start_time']
            if elapsed >= 1.0:
                fps_data['current_fps'] = fps_data['counter'] / elapsed
                fps_data['counter'] = 0
                fps_data['start_time'] = time.time()

            # Draw info overlay
            frame = draw_info(frame, fps_data['current_fps'], inference_time, num_detections)

            # Resize for display if needed
            display_scale = 0.75
            display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

            # Show frame (disable for single-monitor fullscreen testing)
            # cv2.imshow("Live Detection - Q to quit", display_frame)

            # Handle input - press Ctrl+C in terminal to stop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print(f"\nCurrent capture region:")
                print(f"CAPTURE_REGION = {{")
                print(f'    "left": {region["left"]},')
                print(f'    "top": {region["top"]},')
                print(f'    "width": {region["width"]},')
                print(f'    "height": {region["height"]}')
                print(f"}}")

    cv2.destroyAllWindows()
    print("\nDetection stopped.")


if __name__ == "__main__":
    main()
