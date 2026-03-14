# copy_dataset.py
# Copies 200 images per class from dataset/train into dataset_small/train
# Then splits everything into train / val / test
# Run: python copy_dataset.py

import os
import shutil
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SOURCE_TRAIN = Path("dataset/train")      # original full dataset
OUTPUT_DIR   = Path("dataset_split")     # final output with train/val/test
IMAGES_PER_CLASS = 200
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
# ────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

class_dirs = sorted([d for d in SOURCE_TRAIN.iterdir() if d.is_dir()])
print(f"Found {len(class_dirs)} classes in '{SOURCE_TRAIN}'")
print("Copying and splitting...\n")

for i, class_dir in enumerate(class_dirs, 1):
    # Get up to 200 images
    images = [f for f in class_dir.iterdir()
              if f.is_file() and f.suffix in EXTENSIONS]
    random.shuffle(images)
    images = images[:IMAGES_PER_CLASS]

    n       = len(images)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))

    splits = {
        "train": images[:n_train],
        "val":   images[n_train: n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, files in splits.items():
        dest = OUTPUT_DIR / split / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)

    print(f"[{i:>2}/{len(class_dirs)}] {class_dir.name:<55} "
          f"train={len(splits['train']):>3}  "
          f"val={len(splits['val']):>3}  "
          f"test={len(splits['test']):>3}")

print(f"\n✅ Done! Dataset saved to '{OUTPUT_DIR}/'")
print("\nSummary:")
for split in ["train", "val", "test"]:
    folders = list((OUTPUT_DIR / split).iterdir())
    total   = sum(len(list(f.iterdir())) for f in folders)
    print(f"  {split:>5}: {len(folders)} classes, {total} images")

print(f"\nNow run:")
print(f"  python train.py --data_dir {OUTPUT_DIR} --num_workers 0 --batch_size 16 --epochs_p1 5 --epochs_p2 10")