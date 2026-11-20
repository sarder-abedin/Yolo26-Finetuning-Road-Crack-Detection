#!/usr/bin/env python3
"""
Prepare the road crack detection dataset for YOLO training.

Reorganizes images and labels from the flat dataset into train/val/test splits
based on the 'split' column in images.csv.

Usage:
    python scripts/prepare_data.py
"""

import csv
import shutil
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Synthetic_dataset" / "road-defect-dataset-sample"
IMAGES_CSV = DATASET_DIR / "images.csv"
SRC_IMAGES = DATASET_DIR / "images"
SRC_LABELS = DATASET_DIR / "labels"
DEST_DIR = DATASET_DIR / "dataset"


def main():
    print(f"Dataset source: {DATASET_DIR}")
    print(f"Output directory: {DEST_DIR}")

    # Create output directories
    for split in ["train", "val", "test"]:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Read CSV and organize files
    splits = {"train": 0, "val": 0, "test": 0}

    with open(IMAGES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            # Image filename (e.g., "s0104_f_0015.png")
            img_name = Path(row["sample_image"]).name
            # Label filename (same base name, .txt extension)
            lbl_name = Path(img_name).stem + ".txt"

            # Copy image
            src_img = SRC_IMAGES / img_name
            dst_img = DEST_DIR / "images" / split / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                print(f"  WARNING: Image not found: {src_img}")
                continue

            # Copy label
            src_lbl = SRC_LABELS / lbl_name
            dst_lbl = DEST_DIR / "labels" / split / lbl_name
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"  WARNING: Label not found: {src_lbl}")

            splits[split] += 1

    print(f"\nFiles organized:")
    for split, count in splits.items():
        print(f"  {split}: {count} images")
    print(f"  Total: {sum(splits.values())}")

    # Write data.yaml
    data_yaml = DEST_DIR / "data.yaml"
    yaml_content = f"""\
# Road Crack Detection Dataset
# 240 synthetic drone images, 7 defect classes
# Splits: 200 train / 18 val / 22 test

path: {DEST_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: D00
  1: D10
  2: D20
  3: D40
  4: repair
  5: edge_break
  6: manhole
"""
    data_yaml.write_text(yaml_content)
    print(f"\nWrote: {data_yaml}")
    print("\nDone! Dataset ready for training.")


if __name__ == "__main__":
    main()
