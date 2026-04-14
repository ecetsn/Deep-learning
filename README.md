# CS515 HW3 — Data Augmentation and Adversarial Samples

## Overview

This repository extends previous homework with five experiments on the robustness of deep neural networks:

| Task | Description |
|---|---|
| 1 | Baseline robustness on CIFAR-10-C (15 corruptions × 5 severities) |
| 2 | AugMix fine-tuning with JSD consistency loss |
| 3 | PGD-20 adversarial evaluation (L∞ + L2) + Grad-CAM + t-SNE |
| 4 | Knowledge distillation using the robust AugMix teacher |
| 5 | Adversarial transferability (teacher → student) |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CIFAR-10-C

```bash
mkdir -p data/CIFAR-10-C
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar -P data/
tar -xf data/CIFAR-10-C.tar -C data/
# The directory data/CIFAR-10-C/ should now contain 15 .npy files + labels.npy
```

### 3. Directory structure

```
Deep-learning-mobilenet/
├── main.py            # Entry point — all subcommands
├── parameters.py      # Typed dataclass configs
├── train.py           # Training routines (clean, AugMix, KD)
├── test.py            # Evaluation routines
├── augmix.py          # AugMix augmentation + JSD loss
├── attacks.py         # PGD L∞ and L2 (from scratch)
├── gradcam.py         # Grad-CAM implementation (~40 lines)
├── data_utils.py      # CIFAR-10 / CIFAR-10-C loaders
├── models/
│   ├── ResNet.py      # Teacher (ResNet-18 equivalent)
│   └── student.py     # Student CNN (~2.2M params)
├── checkpoints/       # Saved model weights
├── results/           # Figures and report_log.md
└── data/
    ├── cifar-10-batches-py/
    └── CIFAR-10-C/
```

---

## Exact Commands to Reproduce All Experiments

### Task 1 — Baseline robustness (clean teacher + CIFAR-10-C)

```bash
# 1a. Train the clean teacher (30 epochs, SGD with momentum)
python main.py train-clean --epochs 30 --lr 0.01

# 1b. Evaluate on clean CIFAR-10 test set
python main.py eval-clean --ckpt checkpoints/teacher_clean.pth

# 1c. Evaluate on CIFAR-10-C (downloads to data/CIFAR-10-C/)
python main.py eval-cifar10c --ckpt checkpoints/teacher_clean.pth
```

### Task 2 — AugMix fine-tuning

```bash
# 2a. Train with AugMix (JSD consistency loss enabled)
python main.py train-augmix --epochs 30 --lr 0.01 --jsd

# 2b. Evaluate AugMix model on clean test set
python main.py eval-clean --ckpt checkpoints/teacher_augmix.pth

# 2c. Evaluate on CIFAR-10-C → comparison with Task 1
python main.py eval-cifar10c --ckpt checkpoints/teacher_augmix.pth
```

### Task 3 — PGD-20 + Grad-CAM + t-SNE

```bash
# 3a. Clean teacher — L∞ attack (ε=4/255) + Grad-CAM + t-SNE
python main.py eval-pgd \
    --ckpt checkpoints/teacher_clean.pth \
    --norm linf --eps 0.0157 \
    --gradcam --tsne

# 3b. Clean teacher — L2 attack (ε=0.25)
python main.py eval-pgd \
    --ckpt checkpoints/teacher_clean.pth \
    --norm l2 --eps 0.25

# 3c. AugMix teacher — L∞ attack + Grad-CAM + t-SNE
python main.py eval-pgd \
    --ckpt checkpoints/teacher_augmix.pth \
    --norm linf --eps 0.0157 \
    --gradcam --tsne

# 3d. AugMix teacher — L2 attack
python main.py eval-pgd \
    --ckpt checkpoints/teacher_augmix.pth \
    --norm l2 --eps 0.25
```

### Task 4 — Robust teacher → student distillation

```bash
python main.py distill \
    --teacher checkpoints/teacher_augmix.pth \
    --alpha 0.9 --temperature 4 --epochs 30 --lr 0.01
```

### Task 5 — Adversarial transferability

```bash
python main.py transfer-attack \
    --teacher checkpoints/teacher_augmix.pth \
    --student checkpoints/student_augmix.pth \
    --eps 0.0157
```

### Run all tasks in order

```bash
python main.py train-clean --epochs 30 --lr 0.01
python main.py eval-clean --ckpt checkpoints/teacher_clean.pth
python main.py eval-cifar10c --ckpt checkpoints/teacher_clean.pth
python main.py train-augmix --epochs 30 --lr 0.01 --jsd
python main.py eval-clean --ckpt checkpoints/teacher_augmix.pth
python main.py eval-cifar10c --ckpt checkpoints/teacher_augmix.pth
python main.py eval-pgd --ckpt checkpoints/teacher_clean.pth --norm linf --eps 0.0157 --gradcam --tsne
python main.py eval-pgd --ckpt checkpoints/teacher_clean.pth --norm l2 --eps 0.25
python main.py eval-pgd --ckpt checkpoints/teacher_augmix.pth --norm linf --eps 0.0157 --gradcam --tsne
python main.py eval-pgd --ckpt checkpoints/teacher_augmix.pth --norm l2 --eps 0.25
python main.py distill --teacher checkpoints/teacher_augmix.pth --alpha 0.9 --temperature 4 --epochs 30 --lr 0.01
python main.py transfer-attack --teacher checkpoints/teacher_augmix.pth --student checkpoints/student_augmix.pth --eps 0.0157
```