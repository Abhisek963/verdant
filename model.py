# model.py
# Defines the transfer learning model for plant disease classification

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, backbone: str = "efficientnet_b0", freeze_base: bool = True):
    """
    Build a transfer learning model.

    Args:
        num_classes  : Number of disease categories (auto-detected from dataset).
        backbone     : 'efficientnet_b0' or 'resnet50'.
        freeze_base  : If True, freeze all backbone weights so only the
                       new classifier trains in the first phase.

    Returns:
        model (nn.Module): Ready-to-train PyTorch model.
    """

    if backbone == "efficientnet_b0":
        # Load ImageNet-pretrained EfficientNet-B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False  # Freeze everything

        # Replace the final classification head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose 'efficientnet_b0' or 'resnet50'.")

    return model


def unfreeze_model(model):
    """Unfreeze all model parameters for fine-tuning (phase 2 of training)."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")