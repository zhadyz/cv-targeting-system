# Dual GPU Setup Guide

## Overview

This guide explains how to configure the aimbot to use a dedicated GPU for inference, separate from the GPU rendering the game. This eliminates GPU contention and maximizes FPS.

## Hardware Configuration

| GPU | Role | Task |
|-----|------|------|
| GPU 0 (RTX 5080) | Display/Gaming | Renders CS2 |
| GPU 1 (RTX 6000 Pro) | Inference | Runs YOLO detection |

## Step 1: Verify Both GPUs are Detected

```bash
# Activate the TensorRT environment
cd C:\Users\eclip\Desktop\new\cs16-aimbot
venv-tensorrt\Scripts\activate

# Check available GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

Expected output:
```
GPUs: 2
  0: NVIDIA GeForce RTX 5080
  1: NVIDIA RTX 6000 Ada Generation
```

## Step 2: Export TensorRT Engine on Second GPU

TensorRT engines are GPU-specific. You must export on the GPU you'll use for inference.

```bash
# Set environment to use second GPU
set CUDA_VISIBLE_DEVICES=1

# Export the engine (will be optimized for RTX 6000 Pro)
yolo export model=models/best.pt format=engine imgsz=416 half=True device=0

# Rename to indicate which GPU it's for
move models\best.engine models\best_gpu1.engine
```

Note: After `CUDA_VISIBLE_DEVICES=1`, the second GPU becomes `device=0` from PyTorch's perspective.

## Step 3: Update the Aimbot Code

Modify `src/aimbot_ultra.py`:

```python
# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "models/best_gpu1.engine"  # Use GPU1-specific engine

# Add this constant
INFERENCE_GPU = 1  # Use second GPU for inference (0 = first, 1 = second)
```

Then update the model loading in `main()`:

```python
def main():
    global last_fire_time

    print("=" * 60)
    print("CS2 AIMBOT - DUAL GPU MODE")
    print("=" * 60)
    print(f"Game GPU: cuda:0")
    print(f"Inference GPU: cuda:{INFERENCE_GPU}")
    print(f"Capture: {CAPTURE_SIZE}x{CAPTURE_SIZE} (threaded)")
    print(f"Inference: {INFERENCE_SIZE}px")
    print()
    print("Hold RIGHT MOUSE to aim | END to quit")
    print("=" * 60)

    # Load model on second GPU
    print(f"\nLoading TensorRT engine on GPU {INFERENCE_GPU}...")
    model = YOLO(MODEL_PATH, task='detect')
    model.to(f'cuda:{INFERENCE_GPU}')

    # ... rest of the code remains the same
```

## Step 4: Verify It's Using the Correct GPU

Run the aimbot and check GPU usage:

```bash
# In a separate terminal, monitor GPU usage
nvidia-smi -l 1
```

You should see:
- GPU 0 (5080): High usage from CS2
- GPU 1 (6000 Pro): Moderate usage from Python/YOLO

## Expected Performance Gains

| Configuration | FPS (Combat) | FPS (Idle) |
|--------------|--------------|------------|
| Single GPU (shared) | 60-100 | 200-240 |
| Dual GPU (dedicated) | 150-200+ | 200-250+ |

## Troubleshooting

### "CUDA out of memory" on GPU 1
The RTX 6000 Pro has 48GB VRAM - this shouldn't happen. If it does:
```python
# Clear cache before loading
import torch
torch.cuda.empty_cache()
```

### Model not using correct GPU
Verify with:
```python
print(f"Model device: {next(model.model.parameters()).device}")
```

### TensorRT engine fails to load
TensorRT engines are not portable between GPU architectures. You must re-export on the target GPU.

## Alternative: Environment Variable Method

Instead of code changes, you can force GPU selection via environment:

```bash
# Only expose the second GPU to Python
set CUDA_VISIBLE_DEVICES=1

# Now cuda:0 in Python = physical GPU 1
python src/aimbot_ultra.py
```

This makes the RTX 6000 Pro appear as the only GPU to Python, while Windows still uses the 5080 for display.

## Full Modified aimbot_ultra.py

See `src/aimbot_dual_gpu.py` for a complete implementation with dual GPU support.
