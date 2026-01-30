# Real-Time Computer Vision Targeting System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/TensorRT-Optimized-76B900.svg" alt="TensorRT">
  <img src="https://img.shields.io/badge/CUDA-12.x-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Accuracy-98.9%25-brightgreen.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/Latency-<15ms-blue.svg" alt="Latency">
</p>

<p align="center">
  <strong>A production-grade computer vision pipeline for real-time human detection and tracking in simulated environments.</strong>
</p>

<p align="center">
  Built for defense simulations, training systems, autonomous platforms, and surveillance applications.
</p>

---

## Overview

This project implements an end-to-end computer vision system capable of detecting, tracking, and acquiring human targets in real-time. Leveraging state-of-the-art deep learning with aggressive hardware optimization, the system achieves **sub-15 millisecond latency** at **98.9% detection accuracy**—meeting the stringent requirements of time-critical applications.

The simulation environment (Counter-Strike 2) serves as a controlled testbed, providing realistic human models, dynamic scenarios, and immediate performance feedback. The underlying technology is directly transferable to real-world applications including military training simulations, perimeter security, drone systems, and autonomous robotics.

---

## Key Metrics

<table>
  <tr>
    <td align="center"><strong>98.9%</strong><br>Detection Accuracy<br>(mAP@50)</td>
    <td align="center"><strong>10-15ms</strong><br>End-to-End<br>Latency</td>
    <td align="center"><strong>100+ FPS</strong><br>Sustained<br>Throughput</td>
    <td align="center"><strong>5×</strong><br>Optimization<br>Speedup</td>
  </tr>
</table>

---

## System Architecture


```mermaid
graph TD
    %% Node Definitions
    A([ <b>CAPTURE MODULE</b><br><small>Threaded I/O • Zero-copy • ROI</small>])
    B([ <b>DETECTION ENGINE</b><br><small>YOLOv8 • TensorRT • FP16</small>])
    C([ <b>TRACKING SYSTEM</b><br><small>EMA Smoothing • Memory • Priority</small>])
    D([ <b>OUTPUT</b><br><small>HID • Serial • API</small>])

    %% Connections
    A ==> B ==> C ==> D

    %% Styling
    classDef capture fill:#1a1a1a,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef detect fill:#1a1a1a,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef track fill:#1a1a1a,stroke:#fdcb6e,stroke-width:2px,color:#fff;
    classDef out fill:#1a1a1a,stroke:#00b894,stroke-width:2px,color:#fff;

    %% Apply Styles
    class A capture;
    class B detect;
    class C track;
    class D out;
  ```

### Pipeline Components

| Component | Technology | Function |
|-----------|------------|----------|
| **Capture Module** | MSS + Threading | Asynchronous frame acquisition with double buffering |
| **Detection Engine** | YOLOv8 + TensorRT | Real-time object detection with hardware acceleration |
| **Tracking System** | Kalman + EMA | Smooth target pursuit with occlusion handling |
| **Output Interface** | Win32 API | Low-latency control signal generation |

---

## Technical Implementation

### Detection Model

The system employs **YOLOv8n** (nano variant), a single-stage anchor-free detector optimized for speed without sacrificing accuracy.

| Specification | Value |
|---------------|-------|
| Architecture | CSPDarknet + PANet + Decoupled Head |
| Parameters | 3.2M |
| Training Data | 2,153 annotated images |
| Input Resolution | 640×640 |
| Inference Resolution | 416×416 |
| Precision | FP16 (Half) |

### Training Results

| Metric | Score |
|--------|-------|
| Precision | 98.9% |
| Recall | 96.8% |
| mAP@50 | 98.9% |
| mAP@50-95 | 68.7% |
| F1 Score | 97.8% |

### Optimization Pipeline

The system underwent systematic optimization, achieving a **5× performance improvement**:

| Stage | Technique | FPS | Improvement |
|-------|-----------|-----|-------------|
| Baseline | PyTorch FP32 | 20 | — |
| Stage 1 | Reduced inference resolution | 35 | +75% |
| Stage 2 | FP16 half precision | 45 | +29% |
| Stage 3 | TensorRT compilation | 55 | +22% |
| Stage 4 | Multi-threaded capture | 100+ | +82% |

---

## Installation

### Prerequisites

- **Operating System:** Windows 10/11
- **GPU:** NVIDIA RTX 20-series or newer (Tensor Cores required)
- **CUDA:** 12.x with cuDNN 8.x
- **Python:** 3.11+

### Setup

```bash
# Clone the repository
git clone https://github.com/zhadyz/cv-targeting-system.git
cd cv-targeting-system

# Create isolated environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Export TensorRT engine (first run only)
yolo export model=models/best.pt format=engine imgsz=416 half=True
```

---

## Usage

### Training (Optional)

Train on custom dataset:

```bash
python src/train.py
```

### Detection Visualization

Run live detection with bounding box overlay:

```bash
python src/detect_live.py
```

### Full Targeting System

Deploy the complete pipeline:

```bash
python src/targeting_system.py
```

**Controls:**
- `Right Mouse Button` — Activate targeting
- `END` — Terminate system

---

## Project Structure

```
cv-targeting-system/
│
├── src/
│   ├── train.py                 # Model training pipeline
│   ├── detect_live.py           # Real-time detection visualization
│   ├── targeting_system.py      # Production targeting system
│   ├── targeting_fast.py        # TensorRT-optimized variant
│   ├── targeting_dual_gpu.py    # Multi-GPU configuration
│   ├── screen_capture.py        # Capture module benchmarks
│   ├── collect_data.py          # Dataset collection utility
│   └── download_dataset.py      # Training data acquisition
│
├── models/
│   └── best.pt                  # Trained YOLOv8 weights
│
├── docs/
│   ├── PORTFOLIO_PRESENTATION.md    # Executive summary
│   └── DUAL_GPU_SETUP.md            # Multi-GPU configuration guide
│
└── requirements.txt             # Python dependencies
```

---

## Applications

The core technology developed in this project is directly applicable to:

| Domain | Use Case |
|--------|----------|
| **Defense & Simulation** | Training systems, wargaming, threat detection |
| **Surveillance** | Perimeter security, crowd monitoring, intruder detection |
| **Autonomous Systems** | UAV targeting, robotic vision, autonomous vehicles |
| **Sports Analytics** | Player tracking, broadcast automation, performance analysis |

---

## Future Roadmap

- [ ] **Multi-Object Tracking (MOT):** Integrate DeepSORT for persistent target identification
- [ ] **Pose Estimation:** Add keypoint detection for anatomical targeting zones
- [ ] **Thermal Imaging:** Extend to IR/thermal sensor inputs
- [ ] **Edge Deployment:** Optimize for NVIDIA Jetson embedded platforms
- [ ] **Dual-GPU Architecture:** Dedicated inference GPU for 200+ FPS sustained

---

## Documentation

| Document | Description |
|----------|-------------|
| [Portfolio Presentation](docs/PORTFOLIO_PRESENTATION.md) | Executive overview for stakeholders |
| [Dual GPU Setup](docs/DUAL_GPU_SETUP.md) | Multi-GPU configuration instructions |

---

## License

This project is developed for **educational and portfolio demonstration purposes**. The computer vision techniques demonstrated herein are intended for legitimate applications in defense, security, and autonomous systems.

---

## Author

**Abdul Bari**

*Co-developed with Mendicant_Bias*

---

<p align="center">
  <em>Demonstrating real-time computer vision, deep learning optimization, and low-latency systems engineering.</em>
</p>
