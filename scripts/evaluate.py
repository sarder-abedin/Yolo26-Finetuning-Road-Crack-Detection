#!/usr/bin/env python3
"""
Evaluate the fine-tuned YOLO26 model on the test set.

Runs inference on the held-out test images and reports per-class metrics.

Usage:
    python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_road_crack_best.pt"
FALLBACK_MODEL = PROJECT_ROOT / "runs" / "road_crack_v1" / "weights" / "best.pt"
DATA_YAML = (
    PROJECT_ROOT
    / "Synthetic_dataset"
    / "road-defect-dataset-sample"
    / "dataset"
    / "data.yaml"
)
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    # Find model
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    if not model_path.exists():
        print(f"ERROR: Model not found. Looked in:")
        print(f"  {MODEL_PATH}")
        print(f"  {FALLBACK_MODEL}")
        print("Run training first: python scripts/train.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    print(f"Data config: {DATA_YAML}\n")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=1024,
        batch=16,
        device=0,
        project=str(RESULTS_DIR),
        name="test_eval",
        exist_ok=True,
        verbose=True,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")

    # Per-class metrics
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        names = model.names
        print(f"\n  Per-class AP50:")
        for i, name in names.items():
            ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            print(f"    {name:12s}: {ap50:.4f}")

    # Save metrics to JSON
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_file = RESULTS_DIR / "test_metrics.json"
    metrics_data = {
        "model": str(model_path),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    metrics_file.write_text(json.dumps(metrics_data, indent=2))
    print(f"\nMetrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
