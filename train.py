# train.py
# Full training pipeline: phase 1 (frozen backbone) → phase 2 (fine-tune all)

import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_loader import get_dataloaders
from model import build_model, unfreeze_model, count_parameters
from utils import (
    EarlyStopping,
    run_inference,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)


# ── CLI arguments ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--data_dir",    type=str,   default="dataset",
                        help="Root folder with train/val/test sub-folders")
    parser.add_argument("--backbone",    type=str,   default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--epochs_p1",   type=int,   default=10,
                        help="Epochs with frozen backbone (phase 1)")
    parser.add_argument("--epochs_p2",   type=int,   default=20,
                        help="Epochs with full fine-tuning (phase 2)")
    parser.add_argument("--lr_p1",       type=float, default=1e-3,
                        help="Learning rate for phase 1")
    parser.add_argument("--lr_p2",       type=float, default=1e-4,
                        help="Learning rate for phase 2 fine-tune")
    parser.add_argument("--patience",    type=int,   default=7,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="outputs")
    return parser.parse_args()


# ── One epoch of training ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ── One epoch of validation ──────────────────────────────────────────────────

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

    return running_loss / total, correct / total


# ── Training loop for one phase ──────────────────────────────────────────────

def run_phase(model, dataloaders, criterion, optimizer, scheduler,
              device, num_epochs, patience, checkpoint_path, history):
    early_stopper = EarlyStopping(patience=patience)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    ✓ Checkpoint saved (val_loss={val_loss:.4f})")

        if early_stopper(val_loss):
            break

    return history


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\n── Loading data ─────────────────────────────────────")
    dataloaders, class_names, num_classes = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers)

    # Save class names for later use by predict.py
    class_info = {"class_names": class_names, "num_classes": num_classes}
    with open(os.path.join(args.output_dir, "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)
    print(f"class_info.json saved to {args.output_dir}/")

    # Model
    print("\n── Building model ───────────────────────────────────")
    model = build_model(num_classes, backbone=args.backbone, freeze_base=True)
    model = model.to(device)
    count_parameters(model)

    criterion      = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(args.output_dir, "plant_disease_model.pth")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Phase 1: Train only the new classifier head ──────────────────────────
    print(f"\n── Phase 1: Train classifier head ({args.epochs_p1} epochs) ──")
    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_p1)
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.epochs_p1, eta_min=1e-6)

    history = run_phase(
        model, dataloaders, criterion, optimizer_p1, scheduler_p1,
        device, args.epochs_p1, args.patience, checkpoint_path, history)

    # ── Phase 2: Fine-tune the whole network ─────────────────────────────────
    print(f"\n── Phase 2: Full fine-tuning ({args.epochs_p2} epochs) ────────")
    unfreeze_model(model)
    count_parameters(model)

    optimizer_p2 = optim.AdamW(model.parameters(), lr=args.lr_p2, weight_decay=1e-4)
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=args.epochs_p2, eta_min=1e-7)

    history = run_phase(
        model, dataloaders, criterion, optimizer_p2, scheduler_p2,
        device, args.epochs_p2, args.patience, checkpoint_path, history)

    # ── Load best weights & evaluate on test set ──────────────────────────────
    print("\n── Evaluating on test set ───────────────────────────")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = run_inference(model, dataloaders["test"], device)
    metrics = compute_metrics(all_labels, all_preds, class_names)

    # Save metrics
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"test_metrics.json saved to {args.output_dir}/")

    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png"))

    # Training curves
    plot_training_curves(
        history,
        save_path=os.path.join(args.output_dir, "training_curves.png"))

    print(f"\n✅ All done! Best model → {checkpoint_path}")


if __name__ == "__main__":
    main()