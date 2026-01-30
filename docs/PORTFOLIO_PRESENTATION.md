# Real-Time Computer Vision Targeting System
## Portfolio Project - Object Detection & Tracking in Simulated Environments

---

## Executive Summary

This project demonstrates a **real-time computer vision system** capable of detecting, tracking, and targeting human figures in a dynamic simulated environment. Built using state-of-the-art deep learning techniques, the system achieves **sub-15ms latency** with **98.9% detection accuracy**, suitable for applications in:

- Training simulation systems
- Automated surveillance and threat detection
- Drone/UAV target acquisition
- Robotics and autonomous systems
- Defense simulation and wargaming

The simulation environment (Counter-Strike 2) was chosen for its realistic human models, dynamic scenarios, and accessibility for rapid prototyping.

---

## Technical Specifications

| Specification | Value |
|--------------|-------|
| Detection Model | YOLOv8 (Custom Trained) |
| Detection Accuracy (mAP50) | 98.9% |
| End-to-End Latency | 10-15ms |
| Inference Speed | 60-100 FPS (active) / 200+ FPS (idle) |
| Model Size | 3.2M parameters |
| Optimization | TensorRT + Multi-threaded Pipeline |
| Hardware | NVIDIA RTX GPU |

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TARGETING SYSTEM PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────┐ │
│  │   VIDEO    │──▶│  DETECTION │──▶│   TARGET   │──▶│ OUTPUT │ │
│  │   INPUT    │   │   ENGINE   │   │  TRACKING  │   │ SYSTEM │ │
│  └────────────┘   └────────────┘   └────────────┘   └────────┘ │
│                                                                  │
│  • Screen Capture   • YOLOv8 CNN     • Kalman Filter   • API    │
│  • Camera Feed      • TensorRT       • Smoothing       • HID    │
│  • Video Stream     • GPU Accel.     • Prediction      • Serial │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

**1. Video Input Module**
- High-speed frame capture (640×640 region of interest)
- Multi-threaded architecture for zero-latency buffering
- Supports multiple input sources (screen, camera, video stream)

**2. Detection Engine**
- YOLOv8 neural network (custom trained on human figures)
- TensorRT optimization for NVIDIA GPU acceleration
- FP16 inference for maximum throughput
- Single-pass detection (no region proposals)

**3. Target Tracking Module**
- Exponential moving average for position smoothing
- Target memory system (maintains lock during brief occlusions)
- Priority-based target selection (closest to reticle)
- Configurable engagement zones

**4. Output Interface**
- Real-time coordinate output
- HID device control capability
- Configurable for various output systems

---

## Deep Learning Model

### Architecture: YOLOv8

**YOLO (You Only Look Once)** is a state-of-the-art real-time object detection system. Unlike two-stage detectors that first propose regions then classify, YOLO performs detection in a single forward pass.

```
Input Image (640×640×3)
         │
         ▼
┌─────────────────────┐
│   CSPDarknet53      │  ◄── Backbone (Feature Extraction)
│   Backbone          │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   PANet Neck        │  ◄── Multi-scale Feature Aggregation
│   (FPN + PAN)       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Decoupled Head    │  ◄── Classification + Regression
│   (Anchor-free)     │
└─────────────────────┘
         │
         ▼
    Detections
    (x, y, w, h, confidence, class)
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Training Dataset | 2,153 annotated images |
| Epochs | 100 |
| Batch Size | 16 |
| Input Resolution | 640×640 |
| Optimizer | SGD (momentum=0.937) |
| Learning Rate | 0.01 → 0.0001 (cosine decay) |
| Data Augmentation | Mosaic, MixUp, HSV, Flip, Scale |

### Performance Metrics

| Metric | Score |
|--------|-------|
| Precision | 98.9% |
| Recall | 96.8% |
| mAP@50 | 98.9% |
| mAP@50-95 | 68.7% |
| F1 Score | 97.8% |

---

## Optimization Techniques

### 1. TensorRT Compilation

NVIDIA TensorRT optimizes deep learning models for production deployment:

- **Layer Fusion:** Combines Conv + BatchNorm + Activation into single kernel
- **Precision Calibration:** FP32 → FP16 with minimal accuracy loss
- **Kernel Auto-Tuning:** Selects optimal CUDA kernels for target GPU
- **Memory Optimization:** Reduces memory footprint and bandwidth

**Result:** 2-4× inference speedup

### 2. Multi-Threaded Pipeline

```python
# Traditional Sequential Pipeline
capture → inference → process → output  # 18ms total

# Optimized Parallel Pipeline
Thread 1: capture → capture → capture → ...
Thread 2:          inference → inference → ...
                              ↓
                         process → output
# Effective: 10-12ms per frame
```

### 3. Region of Interest (ROI) Processing

Instead of processing full 1440p frames, we capture a centered 640×640 region:
- **4× fewer pixels** to process
- **Maintains accuracy** (targets near crosshair are most relevant)
- **Reduces memory bandwidth** requirements

---

## Real-World Applications

### 1. Military Training Simulations
The system can provide automated threat detection in virtual training environments, offering:
- Instant feedback on target acquisition
- Performance metrics for trainees
- Scenario-based threat recognition

### 2. Surveillance Systems
Adaptable for real-world camera feeds:
- Perimeter security
- Crowd monitoring
- Intruder detection

### 3. Autonomous Systems
Core technology applicable to:
- Drone target acquisition
- Robotic vision systems
- Autonomous vehicle pedestrian detection

### 4. Sports & Analytics
Human tracking capabilities extend to:
- Player tracking in broadcast
- Performance analysis
- Automated camera systems

---

## Technical Challenges & Solutions

### Challenge 1: Latency Requirements
**Problem:** System must respond within human reaction time (~150ms)
**Solution:** TensorRT optimization + threaded capture achieves <15ms latency

### Challenge 2: Detection Stability
**Problem:** Frame-to-frame detection jitter causes erratic tracking
**Solution:** Exponential smoothing filter with target memory system

### Challenge 3: Self-Detection
**Problem:** System detecting its own character model as a target
**Solution:** Configurable exclusion zone (bottom 25% of frame)

### Challenge 4: GPU Resource Contention
**Problem:** Simulation and detection compete for GPU resources
**Solution:** Dual-GPU architecture (documented for future implementation)

---

## Performance Benchmarks

### Inference Speed by Optimization Level

| Configuration | FPS | Latency |
|--------------|-----|---------|
| PyTorch (baseline) | 20 | 50ms |
| + Reduced resolution | 35 | 28ms |
| + FP16 precision | 45 | 22ms |
| + TensorRT | 55 | 18ms |
| + Threaded capture | 80 | 12ms |

### Hardware Utilization

| Component | Usage |
|-----------|-------|
| GPU Compute | 40-60% |
| GPU Memory | 2.1 GB |
| CPU | 15-25% |
| RAM | 800 MB |

---

## Future Development

### Planned Enhancements

1. **Multi-Target Tracking (SORT/DeepSORT)**
   - Maintain persistent IDs across frames
   - Handle occlusions and re-identification

2. **Pose Estimation Integration**
   - Identify specific body parts (head, torso, limbs)
   - Enable precision targeting zones

3. **Thermal/IR Compatibility**
   - Train on thermal imagery datasets
   - Enable night-vision applications

4. **Edge Deployment**
   - Optimize for NVIDIA Jetson platforms
   - Enable embedded/portable systems

5. **Dual-GPU Architecture**
   - Dedicated inference GPU
   - Target: 200+ FPS sustained

---

## Repository Structure

```
project/
├── src/
│   ├── capture_module.py      # Video input system
│   ├── detection_engine.py    # YOLO inference
│   ├── tracking_system.py     # Target tracking
│   └── targeting_system.py    # Main application
├── models/
│   ├── best.pt               # PyTorch weights
│   └── best.engine           # TensorRT engine
├── data/
│   └── training/             # Training dataset
├── docs/
│   ├── STUDY_GUIDE.md        # Technical documentation
│   └── DUAL_GPU_SETUP.md     # Multi-GPU configuration
└── requirements.txt          # Dependencies
```

---

## Conclusion

This project demonstrates proficiency in:

- **Deep Learning:** Custom YOLO model training and optimization
- **Computer Vision:** Real-time object detection and tracking
- **Systems Engineering:** Low-latency pipeline design
- **GPU Programming:** TensorRT optimization, CUDA utilization
- **Software Architecture:** Multi-threaded, production-ready code

The techniques developed here are directly applicable to defense, security, autonomous systems, and other domains requiring real-time visual intelligence.

---

## Contact

*[Your Name]*
*[Your Email]*
*[LinkedIn/GitHub]*

---

*This project was developed for educational and portfolio purposes using a commercial game as a simulation environment. The underlying computer vision techniques are applicable to legitimate real-world systems.*
