"""
YOLO Training Script
====================
Trains a YOLOv8 model on the CS:GO dataset.

Key Concepts:
- We start with a PRE-TRAINED model (transfer learning)
- The model already knows basic shapes/patterns from millions of images
- We fine-tune it to recognize CS:GO players specifically
- This is MUCH faster than training from scratch

Training Parameters Explained:
- epochs: How many times to go through the entire dataset
- imgsz: Input image size (larger = more accurate but slower)
- batch: Images processed together (higher = faster but needs more VRAM)
- device: 'cuda' for GPU, 'cpu' for CPU (GPU is 10-50x faster)
"""

from ultralytics import YOLO
import torch
import os


def check_gpu():
    """Check if CUDA GPU is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name}")
        print(f"VRAM: {vram:.1f} GB")
        return True
    else:
        print("No GPU detected. Training will use CPU (slower).")
        return False


def main():
    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)
    print()

    # Check hardware
    has_gpu = check_gpu()
    device = 'cuda' if has_gpu else 'cpu'
    print(f"Using device: {device}")
    print()

    # Training configuration
    # Optimized for RTX 5080 16GB VRAM
    config = {
        'data': 'data/csgo-dataset/data.yaml',  # Dataset config
        'epochs': 50,                            # Training iterations
        'imgsz': 640,                            # Image size
        'batch': 32,                             # Large batch for 16GB VRAM
        'device': device,
        'patience': 10,                          # Early stopping patience
        'save': True,                            # Save checkpoints
        'project': 'runs/detect',                # Output directory
        'name': 'csgo-player-detection',         # Run name
        'exist_ok': True,                        # Overwrite if exists
        'pretrained': True,                      # Use pretrained weights
        'verbose': True,                         # Show progress
        'workers': 8,                            # Data loading threads
        'amp': True,                             # Mixed precision (faster)
    }

    print("Training Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)
    print()

    # Load a pretrained YOLOv8 model
    # Options: yolov8n (nano/fast), yolov8s (small), yolov8m (medium), yolov8l (large)
    # We use 'nano' for speed - good enough for this use case
    print("Loading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')

    print()
    print("Starting training...")
    print("=" * 60)
    print()

    # Train the model
    results = model.train(**config)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print("Model saved to: runs/detect/csgo-player-detection/weights/best.pt")
    print()
    print("Next steps:")
    print("1. Copy best.pt to models/ folder")
    print("2. Run detect_live.py to test on CS2")
    print()

    # Copy best model to models folder
    best_model_path = 'runs/detect/csgo-player-detection/weights/best.pt'
    if os.path.exists(best_model_path):
        import shutil
        os.makedirs('models', exist_ok=True)
        shutil.copy(best_model_path, 'models/best.pt')
        print("Copied best.pt to models/best.pt")


if __name__ == "__main__":
    main()
