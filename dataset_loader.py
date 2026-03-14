# dataset_loader.py
# Handles image loading, augmentation, and DataLoader creation

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Image size expected by EfficientNet-B0 / ResNet50 ──────────────────────
IMG_SIZE = 224

# ── Mean & std from ImageNet (used because backbone was pretrained on it) ───
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train"):
    """
    Return torchvision transform pipelines.

    Training gets heavy augmentation; val/test get only resize + normalize.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """
    Build DataLoaders for train / val / test splits.

    Args:
        data_dir    : Root folder that contains train/, val/, test/ sub-folders.
        batch_size  : Images per mini-batch.
        num_workers : Parallel data-loading workers (set 0 on Windows).

    Returns:
        dataloaders : dict with keys 'train', 'val', 'test'
        class_names : list of class name strings (from train split, sorted)
        num_classes : int
    """
    data_dir = Path(data_dir)

    # ── Validate that all three split folders exist ─────────────────────────
    for split in ["train", "val", "test"]:
        assert (data_dir / split).exists(), (
            f"Expected folder '{data_dir / split}' not found.\n"
            "Ensure your dataset follows the structure:\n"
            "  dataset/train/<class>/\n"
            "  dataset/val/<class>/\n"
            "  dataset/test/<class>/"
        )

    # ── Build ImageFolder datasets ──────────────────────────────────────────
    image_datasets = {
        split: datasets.ImageFolder(
            root=str(data_dir / split),
            transform=get_transforms(split),
        )
        for split in ["train", "val", "test"]
    }

    # ── Class names come from the train split (the full set) ───────────────
    class_names = image_datasets["train"].classes
    num_classes  = len(class_names)

    # ── Validate val: must have exactly the same classes as train ──────────
    if image_datasets["val"].classes != class_names:
        val_extra   = set(image_datasets["val"].classes) - set(class_names)
        val_missing = set(class_names) - set(image_datasets["val"].classes)
        msg = "Class mismatch between 'train' and 'val'!\n"
        if val_extra:
            msg += f"  Extra in val (not in train)      : {val_extra}\n"
        if val_missing:
            msg += f"  Missing in val (present in train): {val_missing}\n"
        raise AssertionError(msg)

    # ── Validate test: allowed to be a SUBSET of train classes ────────────
    test_classes  = image_datasets["test"].classes
    extra_in_test = set(test_classes) - set(class_names)
    if extra_in_test:
        raise ValueError(
            f"'test' contains classes not present in 'train': {extra_in_test}\n"
            "Rename those folders to match the train class names exactly."
        )

    if len(test_classes) < num_classes:
        print(
            f"  Note: 'test' has {len(test_classes)} of {num_classes} classes — "
            f"evaluation will cover only those {len(test_classes)} classes."
        )

    # ── Build DataLoaders ──────────────────────────────────────────────────
    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"Total classes ({num_classes}) detected from train split:")
    for name in class_names:
        print(f"    {name}")
    print()
    for split in ["train", "val", "test"]:
        n = len(image_datasets[split])
        c = len(image_datasets[split].classes)
        print(f"  {split:>5} : {n:>6} images  |  {c} classes")
    print()

    return dataloaders, class_names, num_classes