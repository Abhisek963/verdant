# utils.py
# Evaluation metrics, confusion matrix, and training-curve visualisation

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ── Metric computation ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, class_names):
    """
    Compute accuracy, per-class precision/recall/F1, and macro averages.

    Args:
        y_true       : list/array of ground-truth integer labels
        y_pred       : list/array of predicted integer labels
        class_names  : list of class name strings

    Returns:
        metrics dict
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names)))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro Precision  : {macro_p:.4f}")
    print(f"Macro Recall     : {macro_r:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")

    return {
        "accuracy":  acc,
        "precision": macro_p,
        "recall":    macro_r,
        "f1":        macro_f1,
        "per_class": {
            class_names[i]: {
                "precision": precision[i],
                "recall":    recall[i],
                "f1":        f1[i],
                "support":   int(support[i]),
            }
            for i in range(len(class_names))
        },
    }


# ── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ── Training-curve plots ────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path="training_curves.png"):
    """
    Plot loss and accuracy curves for train and val splits.

    Args:
        history   : dict with keys 'train_loss', 'val_loss',
                                   'train_acc',  'val_acc'
        save_path : where to write the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=4, label="Val Acc")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ── Inference helper (used by both train.py evaluation & predict.py) ────────

def run_inference(model, dataloader, device):
    """
    Run the model over a DataLoader and collect predictions + ground truths.

    Returns:
        all_preds  : list of int
        all_labels : list of int
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_preds, all_labels


# ── Early stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience  : epochs to wait after last improvement
        min_delta : minimum change to qualify as improvement
        verbose   : print messages
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.counter    = 0
        self.best_loss  = np.inf
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("  Early stopping triggered.")
        return self.should_stop