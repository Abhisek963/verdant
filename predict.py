# predict.py
# Load a saved model and predict the disease class of a single leaf image.
#
# Usage:
#   python predict.py --image path/to/leaf.jpg
#   python predict.py --image path/to/leaf.jpg --top_k 3

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image

from dataset_loader import get_transforms
from model import build_model


def load_model(checkpoint_path: str, class_info_path: str, backbone: str, device):
    """Load class names and rebuild the model from a saved checkpoint."""

    # Class information saved during training
    with open(class_info_path, "r") as f:
        class_info = json.load(f)
    class_names = class_info["class_names"]
    num_classes  = class_info["num_classes"]

    # Rebuild model architecture and load weights
    model = build_model(num_classes, backbone=backbone, freeze_base=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"Model loaded from  : {checkpoint_path}")
    print(f"Classes ({num_classes}): {class_names}")

    return model, class_names


def predict_image(model, image_path: str, class_names: list, device, top_k: int = 1):
    """
    Predict the disease class of a single image.

    Args:
        model       : loaded PyTorch model
        image_path  : path to the leaf image file
        class_names : list of class strings
        device      : torch.device
        top_k       : number of top predictions to return

    Returns:
        List of (class_name, probability) tuples, sorted by probability descending.
    """
    transform = get_transforms("val")   # No augmentation for inference

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)   # shape: [1, 3, H, W]

    with torch.no_grad():
        logits = model(tensor)                         # shape: [1, num_classes]
        probs  = F.softmax(logits, dim=1)[0]           # shape: [num_classes]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))
    results = [
        (class_names[idx.item()], top_probs[i].item())
        for i, idx in enumerate(top_indices)
    ]
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Plant disease inference")
    parser.add_argument("--image",      type=str,   required=True,
                        help="Path to the input leaf image")
    parser.add_argument("--checkpoint", type=str,   default="outputs/plant_disease_model.pth")
    parser.add_argument("--class_info", type=str,   default="outputs/class_info.json")
    parser.add_argument("--backbone",   type=str,   default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--top_k",      type=int,   default=3,
                        help="Show top-k predictions")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Validate paths
    for path, name in [(args.image, "Image"), (args.checkpoint, "Checkpoint"),
                       (args.class_info, "class_info.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    model, class_names = load_model(
        args.checkpoint, args.class_info, args.backbone, device)

    print(f"\nAnalysing: {args.image}")
    results = predict_image(model, args.image, class_names, device, top_k=args.top_k)

    print("\n── Predictions ──────────────────────────────────────")
    for rank, (cls, prob) in enumerate(results, start=1):
        bar = "█" * int(prob * 40)
        print(f"  #{rank} {cls:<35} {prob*100:5.1f}%  {bar}")

    top_class, top_prob = results[0]
    print(f"\n✅ Predicted disease : {top_class}")
    print(f"   Confidence        : {top_prob*100:.2f}%")


if __name__ == "__main__":
    main()