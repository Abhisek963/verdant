# 🌿 Plant Disease Detection — Deep Learning with PyTorch

A complete transfer-learning pipeline that classifies plant leaf images into
healthy / diseased categories using **EfficientNet-B0** or **ResNet-50**
pretrained on ImageNet.

---

## Project Structure

```
plant_disease_project/
├── dataset_loader.py   # Dataset, transforms, DataLoader factory
├── model.py            # Model builder (EfficientNet / ResNet)
├── train.py            # Full training + evaluation script
├── predict.py          # Single-image or batch inference
├── utils.py            # EarlyStopping, metrics, plots, checkpoints
└── README.md
```

---

## 1 · Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# or CPU-only:
pip install torch torchvision

pip install scikit-learn matplotlib seaborn pillow
```

Python ≥ 3.9 recommended.

---

## 2 · Prepare Your Dataset

Organise images in the following layout (standard ImageFolder format):

```
dataset/
├── train/
│   ├── Tomato_Early_Blight/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Tomato_Healthy/
│   └── ...
├── val/
│   ├── Tomato_Early_Blight/
│   └── ...
└── test/
    ├── Tomato_Early_Blight/
    └── ...
```

> **Tip:** A good starting dataset is the
> [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
> (~87 K images, 38 classes).

---

## 3 · Train the Model

### Minimal command
```bash
python train.py --data_dir dataset
```

### Full options
```bash
python train.py \
  --data_dir      dataset          \   # path to dataset root
  --arch          efficientnet_b0  \   # or resnet50
  --epochs        40               \   # max epochs
  --batch_size    32               \
  --lr            1e-3             \   # head learning rate
  --finetune_lr   1e-4             \   # backbone LR after warm-up
  --warmup_epochs 5                \   # epochs to train head only
  --patience      10               \   # early-stopping patience
  --num_workers   4                \
  --dropout       0.3              \
  --out_dir       outputs          \   # where to save artefacts
  --amp                                # mixed precision (faster on GPU)
```

### Training strategy (two-phase)
| Phase | Epochs | What trains |
|-------|--------|-------------|
| Warm-up | 1 – `warmup_epochs` | Classification head only |
| Fine-tune | `warmup_epochs+1` → end | Entire network (lower LR) |

### Outputs produced
| File | Description |
|------|-------------|
| `plant_disease_model.pth` | Best checkpoint (lowest val loss) |
| `training_curves.png` | Loss & accuracy curves |
| `confusion_matrix.png` | Normalised confusion matrix |
| `training_history.json` | Raw epoch metrics |

---

## 4 · Run Inference

### Single image
```bash
python predict.py \
  --image  path/to/leaf.jpg \
  --model  plant_disease_model.pth \
  --top_k  5
```

### Batch (directory of images)
```bash
python predict.py \
  --image_dir  my_leaf_images/ \
  --model      plant_disease_model.pth \
  --output     results.json
```

### Example output
```
  Image     : leaf.jpg
  Prediction: Tomato_Early_Blight
  Confidence: 94.73%

  Top-3 predictions:
    1. Tomato_Early_Blight            94.73%  ████████████████████████████
    2. Tomato_Late_Blight              3.81%  █
    3. Tomato_Healthy                  1.46%
```

---

## 5 · Quick Sanity Checks

```bash
# Verify data loader
python dataset_loader.py dataset

# Verify model builds correctly
python model.py

# Verify utils (no dataset needed)
python -c "from utils import get_device; get_device()"
```

---

## 6 · Tips for Better Results

| Tip | Detail |
|-----|--------|
| **Class imbalance** | Use `WeightedRandomSampler` or class weights in CrossEntropyLoss |
| **Small dataset** | Reduce `finetune_lr` to `5e-5`; increase augmentation |
| **Large dataset** | Increase `batch_size`, enable `--amp`, set `warmup_epochs 0` |
| **Overfitting** | Increase `--dropout`, add more augmentation, reduce `finetune_lr` |
| **Resume training** | Load checkpoint manually and adjust epoch counter in train.py |

---

## 7 · Architecture Details

```
Input (224×224 RGB)
      │
EfficientNet-B0 backbone (pretrained, ImageNet)
      │
  Global Avg Pool
      │
  Dropout (p=0.3)
      │
  Linear(1280 → num_classes)
      │
Softmax → predicted class
```

For ResNet-50, the final `fc` layer is replaced identically.

---

## 8 · Metrics Reported

- **Accuracy** — overall correct / total
- **Precision** — macro-averaged across classes
- **Recall** — macro-averaged across classes
- **F1 Score** — macro-averaged harmonic mean of precision & recall
- **Per-class classification report** (printed to console)
- **Confusion matrix** (saved as PNG)