# Yolo26-Finetuning-Road-Crack-Detection

Fine-tuned YOLO26 nano model for road crack/defect detection using a synthetic drone image dataset.

## Overview

| Component | Details |
|-----------|---------|
| **Base Model** | YOLO26n (2.5M params, 5.9 GFLOPs) |
| **Dataset** | 240 synthetic drone images (2560×1920), 7 defect classes |
| **Splits** | 200 train / 18 val / 22 test |
| **GPU** | AMD Radeon RX 9060 XT (16GB) via ROCm 7.1 |
| **Training Time** | ~18 minutes (75 epochs, early stopped) |
| **Model Size** | 5.3 MB |

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | D00 | Crack type D00 |
| 1 | D10 | Crack type D10 |
| 2 | D20 | Crack type D20 |
| 3 | D40 | Crack type D40 |
| 4 | repair | Repair patch |
| 5 | edge_break | Edge break |
| 6 | manhole | Manhole cover |

## Test Set Results

| Metric | Value |
|--------|-------|
| Precision | 0.709 |
| Recall | 0.381 |
| mAP50 | 0.369 |
| mAP50-95 | 0.281 |

### Per-Class AP50

| Class | AP50 | Instances |
|-------|------|-----------|
| D00 | 0.096 | 4 |
| D10 | 0.532 | 6 |
| D20 | 0.000 | 2 |
| D40 | 0.709 | 14 |
| repair | 0.252 | 5 |
| edge_break | 0.000 | 3 |
| manhole | 0.995 | 3 |

> **Note:** Low per-class metrics for D00, D20, and edge_break are expected due to very few test instances (2-4 each). The model performs well on D40 (14 instances) and manhole (3 instances).

## Project Structure

```
├── yolo26n.pt                          # Base pretrained model (COCO)
├── models/
│   └── yolo26n_road_crack_best.pt      # Fine-tuned model (best validation)
├── scripts/
│   ├── prepare_data.py                 # Organize dataset into train/val/test
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Test set evaluation
│   └── predict.py                      # Inference utility
├── runs/                               # Training outputs (plots, weights)
├── results/                            # Evaluation outputs
├── Synthetic_dataset/
│   └── road-defect-dataset-sample/
│       ├── dataset/                    # Prepared dataset (generated)
│       ├── images/                     # Source images (240 PNGs)
│       ├── labels/                     # Source labels (YOLO format)
│       ├── labels_seg/                 # Segmentation labels
│       ├── overlays/                   # Visualization overlays
│       ├── images.csv                  # Image metadata + splits
│       └── objects.csv                 # Object-level metadata
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- AMD GPU with ROCm support (tested on RX 9060 XT with ROCm 7.1)
- ~2GB disk space for dependencies

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with ROCm 7.1
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/rocm7.1

# Install remaining dependencies
pip install ultralytics opencv-python-headless pandas matplotlib
```

### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True / AMD Radeon RX 9060 XT
```

## Usage

### 1. Prepare Data (if starting fresh)

```bash
python scripts/prepare_data.py
```

This organizes images/labels into `dataset/images/{train,val,test}/` and `dataset/labels/{train,val,test}/` based on the `split` column in `images.csv`.

### 2. Train

```bash
python scripts/train.py
```

Key hyperparameters (configured in `scripts/train.py`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `imgsz` | 1024 | Preserves crack detail from 2560px source |
| `batch` | 16 | Fits in 16GB VRAM |
| `epochs` | 100 | With early stopping |
| `patience` | 20 | Stop if no improvement for 20 epochs |
| `optimizer` | auto | AdamW selected automatically |
| `close_mosaic` | 10 | Disable mosaic augmentation last 10 epochs |

The best model is automatically saved to `models/yolo26n_road_crack_best.pt`.

### 3. Evaluate

```bash
python scripts/evaluate.py
```

Runs inference on the held-out test set and saves metrics to `results/test_metrics.json`.

### 4. Predict

```bash
# Single image
python scripts/predict.py path/to/image.jpg

# Folder of images with custom confidence
python scripts/predict.py path/to/folder/ --conf 0.3 --save

# Video
python scripts/predict.py path/to/video.mp4 --save
```

### 5. Use Model in Code

```python
from ultralytics import YOLO

# Load fine-tuned model
model = YOLO("models/yolo26n_road_crack_best.pt")

# Predict on an image
results = model.predict("test_image.jpg", conf=0.25, imgsz=1024)

# Access results
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"{model.names[cls]}: {conf:.2f} at {xyxy}")
```

## Training Configuration

Full training args are saved in `runs/road_crack_v1/args.yaml`.

### Reproducing Training

```bash
# From a fresh clone:
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
```

## Improvement Ideas

If higher accuracy is needed:

1. **More data** — The 200-image training set is small. Collecting real road images would significantly improve generalization.
2. **Larger model** — Try `yolo26s.pt` (more capacity, still fast on 16GB VRAM).
3. **Higher resolution** — Increase `imgsz` to 1280 or 1536 for finer crack details.
4. **Class balancing** — D20 and edge_break have very few instances; consider oversampling or class weights.
5. **Test-time augmentation** — Enable TTA for inference: `model.predict(..., augment=True)`.
6. **Longer training** — Increase `patience` to 40 or disable early stopping if metrics are still improving.

## License

See [LICENSE](LICENSE).
