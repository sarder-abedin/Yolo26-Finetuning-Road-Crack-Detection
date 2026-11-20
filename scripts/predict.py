#!/usr/bin/env python3
"""
Run inference with the fine-tuned YOLO26 road crack detection model.

Usage:
    python scripts/predict.py <image_or_folder> [--conf 0.25] [--source-type image]

Examples:
    python scripts/predict.py path/to/image.jpg
    python scripts/predict.py path/to/folder/ --conf 0.3
    python scripts/predict.py path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_road_crack_best.pt"
FALLBACK_MODEL = PROJECT_ROOT / "runs" / "road_crack_v1" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Road crack detection inference")
    parser.add_argument("source", help="Image, folder, or video to predict on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--save", action="store_true", help="Save annotated results")
    parser.add_argument("--save-txt", action="store_true", help="Save results as txt")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to model (overrides default)"
    )
    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run training first: python scripts/train.py")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"Image size: {args.imgsz}")
    print()

    # Load model and predict
    model = YOLO(str(model_path))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=0,
        save=args.save,
        save_txt=args.save_txt,
        verbose=True,
    )

    print(f"\nProcessed {len(results)} image(s).")
    if args.save:
        print(f"Results saved to: {PROJECT_ROOT / 'runs' / 'predict'}")


if __name__ == "__main__":
    main()
