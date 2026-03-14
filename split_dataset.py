"""
split_dataset.py
----------------
Splits a flat dataset (one folder per class) into train / val / test splits.

Expected input layout
---------------------
raw_dataset/
    Tomato_Healthy/
        img001.jpg
        img002.jpg
        ...
    Tomato_Early_Blight/
        ...

Output layout
-------------
dataset/
    train/
        Tomato_Healthy/
        Tomato_Early_Blight/
    val/
        Tomato_Healthy/
        Tomato_Early_Blight/
    test/
        Tomato_Healthy/
        Tomato_Early_Blight/

Usage
-----
    python split_dataset.py

    # Custom paths and ratios
    python split_dataset.py --src raw_dataset --dst dataset --train 0.7 --val 0.15 --test 0.15

    # Dry run (see what would happen without copying files)
    python split_dataset.py --dry_run
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Split a flat image dataset into train/val/test")
    p.add_argument("--src",   default="raw_dataset", help="Source folder (one sub-dir per class)")
    p.add_argument("--dst",   default="dataset",     help="Destination folder for the split dataset")
    p.add_argument("--train", type=float, default=0.70, help="Train split ratio (default 0.70)")
    p.add_argument("--val",   type=float, default=0.15, help="Val   split ratio (default 0.15)")
    p.add_argument("--test",  type=float, default=0.15, help="Test  split ratio (default 0.15)")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed for reproducibility")
    p.add_argument("--copy",  action="store_true", default=True,
                   help="Copy files (default). Use --no-copy to symlink instead (Linux/Mac only)")
    p.add_argument("--no_copy", dest="copy", action="store_false")
    p.add_argument("--dry_run", action="store_true", help="Print plan without copying any files")
    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def gather_images(class_dir: Path):
    """Return sorted list of image paths inside a class directory."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def split_list(items, train_r, val_r):
    """Split a list into (train, val, test) portions."""
    n       = len(items)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def transfer(src: Path, dst: Path, use_copy: bool, dry_run: bool):
    """Copy or symlink src → dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate ratios
    total = round(args.train + args.val + args.test, 6)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0 — got {total:.4f}")

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: '{src}'")

    # Collect class directories
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No sub-directories found in '{src}'")

    random.seed(args.seed)

    split_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    total_counts = {"train": 0, "val": 0, "test": 0}

    print(f"\n{'─'*60}")
    print(f"  Source      : {src}")
    print(f"  Destination : {dst}")
    print(f"  Ratios      : train={args.train}  val={args.val}  test={args.test}")
    print(f"  Seed        : {args.seed}")
    print(f"  Mode        : {'DRY RUN' if args.dry_run else 'copy' if args.copy else 'symlink'}")
    print(f"{'─'*60}\n")

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images   = gather_images(cls_dir)

        if not images:
            print(f"  [SKIP] {cls_name} — no images found")
            continue

        random.shuffle(images)
        train_imgs, val_imgs, test_imgs = split_list(images, args.train, args.val)

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for img in split_imgs:
                dst_path = dst / split_name / cls_name / img.name
                transfer(img, dst_path, args.copy, args.dry_run)
            split_counts[cls_name][split_name] = len(split_imgs)
            total_counts[split_name] += len(split_imgs)

        print(
            f"  {cls_name:<40}  "
            f"train={split_counts[cls_name]['train']:>5}  "
            f"val={split_counts[cls_name]['val']:>4}  "
            f"test={split_counts[cls_name]['test']:>4}  "
            f"total={len(images):>5}"
        )

    print(f"\n{'─'*60}")
    print(
        f"  TOTAL{' ':35}"
        f"train={total_counts['train']:>5}  "
        f"val={total_counts['val']:>4}  "
        f"test={total_counts['test']:>4}  "
        f"total={sum(total_counts.values()):>5}"
    )
    print(f"{'─'*60}")

    if args.dry_run:
        print("\n  [DRY RUN] No files were copied. Remove --dry_run to execute.")
    else:
        action = "copied" if args.copy else "symlinked"
        print(f"\n  ✓ All files {action} into '{dst}/'")
        print( "  Ready to train → python train.py --data_dir dataset")


if __name__ == "__main__":
    main()