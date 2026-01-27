# Poultry-Pi-System

A production-ready computer vision system for automated monitoring and behavior analysis of poultry in cage-free environments. Built on **YOLOv12** with **ByteTrack** tracking, supporting dual-camera setups with frame synchronization.

The system detects individual hens, feeders, and waterers in real-time, classifying hen behaviors (feeding, drinking, idle) while maintaining spatial state in world coordinates.

**Research Paper**: [MDPI Agriculture 15(18):1963](https://www.mdpi.com/2077-0472/15/18/1963)

## Sample Inference

![Inference Preview](assets/inference_preview.gif)

The model successfully detects and tracks:
- **Hens** with real-time behavior classification (Feeding, Drinking, Idle)
- **Feeders** and **Waterers** as resource zones
- Individual hen tracking with cumulative time spent feeding and drinking

## Project Structure

```
poultry-pi-system/
├── src/
│   ├── core/           # Runtime engine (inference, tracking, behavior)
│   └── tools/          # Dataset tools & pose labeler
├── config/             # YAML configuration files
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for evaluation
├── scripts/            # CLI entry points
├── models/             # Trained model weights
└── ultralytics/        # Custom YOLOv12 fork
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU inference, optional)
- Conda (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ogbidaniel/poultry-vision.git
   cd poultry-vision
   ```

2. **Create conda environment:**
   ```bash
   conda create --name poultry-vision python=3.12
   conda activate poultry-vision
   ```

3. **Install dependencies:**
   ```bash
   # Install required packages
   pip install -r requirements.txt

   # Install custom ultralytics fork (REQUIRED)
   pip install -e .
   ```

---

## Quick Start

### Run Inference

```bash
# USB camera (device 0)
python scripts/run_inference.py --source 0

# Video file
python scripts/run_inference.py --source samplevideos/video.mp4

# RTSP stream
python scripts/run_inference.py --source "rtsp://ip:port/stream"

# Headless mode with video output
python scripts/run_inference.py --source video.mp4 --no-display --save-video output.mp4
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--source`, `-s` | `0` | Video source (camera index, URL, or file) |
| `--model`, `-m` | `models/poultry-yolov12n-v1.pt` | Model path |
| `--config`, `-c` | `config/system_config.yaml` | Config file |
| `--calibration` | `pen_config.npy` | Pen calibration |
| `--no-display` | False | Run without display |
| `--save-video` | None | Output video path |

### Controls

- Press **'q'** to quit
- Press **'s'** to save screenshot

---

## Calibration

Before running inference, calibrate the pen boundaries:

```bash
python -m src.tools.calibrate --source samplevideos/video.mp4 --output pen_config.npy
```

Click the 4 corners of the pen in order: top-left, top-right, bottom-right, bottom-left.

---

## Pose Labeler Tool

Create pose annotations for training data:

```bash
python -m src.tools.labeler.main
```

Features:
- 10-keypoint pose schema for poultry
- Bounding box + keypoint annotation
- YOLO format export

---

## Configuration

### System Config (`config/system_config.yaml`)

```yaml
pen:
  width_cm: 120
  height_cm: 200
  calibration_file: pen_config.npy
  interaction_radius_cm: 15
```

### Camera Config (`config/cameras.yaml`)

```yaml
cameras:
  top:
    type: usb
    source: "0"
    fps: 30
  side:
    type: rtsp
    source: "rtsp://ip:port/stream"
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src/core --cov-report=html
```

---

## Notebooks

Jupyter notebooks for model evaluation and analysis:

```bash
cd notebooks
jupyter notebook
```

| Notebook | Description |
|----------|-------------|
| `01_model_evaluation.ipynb` | mAP, precision, recall, speed benchmarks |
| `02_dataset_analysis.ipynb` | Dataset statistics, class distribution |
| `03_system_metrics.ipynb` | Latency, sync rate, memory usage |
| `04_pose_estimation.ipynb` | PCK metrics, skeleton visualization |

---

## Model Information

| Model | File | Size | Classes |
|-------|------|------|---------|
| Detection | `poultry-yolov12n-v1.pt` | 5.2 MB | feeder, hen, waterer |
| Pose | `yolo11-pose.pt` | - | 10 keypoints |
| Segmentation | `poultry-seg-large.pt` | - | instance masks |

### Training Results

| Overall Results | Confusion Matrix |
| :---: | :---: |
| ![Training Results](assets/training_results/results.png) | ![Confusion Matrix](assets/training_results/confusion_matrix.png) |

| PR Curve | Validation Prediction |
| :---: | :---: |
| ![PR Curve](assets/training_results/PR_curve.png) | ![Validation Prediction](assets/training_results/val_batch0_pred.jpg) |

---

## Python API

```python
from src.core import (
    CameraView,
    PenStateMachine,
    HenBehaviorMonitor,
    get_homography_matrix
)

# Load model with coordinate transformation
camera = CameraView(
    name="top",
    model_path="models/poultry-yolov12n-v1.pt",
    pen_polygon=pen_corners,
    homography_matrix=H
)

# Process frame
annotated_frame, detections = camera.process_frame(frame)

# Update state
pen_state = PenStateMachine(width_cm=120, height_cm=200)
pen_state.update_from_detections(detections, timestamp_ms)

# Generate minimap
minimap = pen_state.generate_minimap(render_scale=3)
```

---

## Target Hardware

- **Desktop**: GPU recommended for real-time inference
- **Raspberry Pi 5**: With Hailo-8L AI Hat (8 TOPS NPU)

---

## Technical Notes

- Uses **custom ultralytics package** for YOLOv12 attention-centric detection
- **Scaled dot product attention** by default (flash-attention optional)
- Dual-camera frame synchronization with 50ms tolerance
- Homography-based coordinate transformation for world-space tracking

---

## License

MIT License - See LICENSE file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{poultry2025,
  title={Automated Poultry Behavior Assessment Using Computer Vision},
  journal={MDPI Agriculture},
  volume={15},
  number={18},
  pages={1963},
  year={2025}
}
```
