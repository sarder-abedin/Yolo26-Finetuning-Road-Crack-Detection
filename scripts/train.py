#!/usr/bin/env python3
"""
Fine-tune YOLO26 nano for road crack detection.

Loads the pretrained yolo26n.pt (COCO weights) and fine-tunes on the
synthetic road crack dataset using the AMD GPU via ROCm.

Usage:
    python scripts/train.py
"""

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
DATA_YAML = (
    PROJECT_ROOT
    / "Synthetic_dataset"
    / "road-defect-dataset-sample"
    / "dataset"
    / "data.yaml"
)
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"
RUN_NAME = "road_crack_v1"


def main():
    # Verify prerequisites
    if not MODEL_PATH.exists():
        print(f"ERROR: Base model not found at {MODEL_PATH}")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        print("Run 'python scripts/prepare_data.py' first.")
        sys.exit(1)

    # Verify GPU
    import torch

    if not torch.cuda.is_available():
        print("WARNING: GPU not available. Training will be very slow on CPU.")
        print("Consider checking ROCm installation.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")

    print(f"\nBase model: {MODEL_PATH}")
    print(f"Data config: {DATA_YAML}")
    print(f"Output: {RUNS_DIR / RUN_NAME}")
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    # Load pretrained model
    model = YOLO(str(MODEL_PATH))

    # Train
    results = model.train(
        data=str(DATA_YAML),
        # Training duration
        epochs=100,
        patience=20,  # early stopping
        # Image & batch
        imgsz=1024,
        batch=16,
        # Device
        device=0,
        # Optimizer & learning rate
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        # Augmentation (conservative for small dataset)
        close_mosaic=10,  # disable mosaic in last 10 epochs
        # Output
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        # Logging
        verbose=True,
    )

    # Save best model to models/ directory
    best_weight = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if best_weight.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        final_path = MODELS_DIR / "yolo26n_road_crack_best.pt"
        shutil.copy2(best_weight, final_path)
        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"Best model saved to: {final_path}")
        print(f"{'=' * 60}")
    else:
        print(f"\nWARNING: Best weight not found at {best_weight}")
        print(f"Check {RUNS_DIR / RUN_NAME} for outputs.")

    return results


if __name__ == "__main__":
    main()
