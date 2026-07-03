# MCSAM: Edge-Guided Multi-Scale Channel-Spatial Attention Module

Official PyTorch implementation of the **Multi-Scale Channel-Spatial Attention Module (MCSAM)** integrated with the **ConvNeXt-Tiny** backbone, as proposed in the paper *"MCSAM: An Edge-Guided Multi-Scale Channel-Spatial Attention Module for Automated Waste Classification"*.

This repository provides a reproducible, production-ready framework for integrating MCSAM into modern CNN architectures. It is specifically tailored for tasks requiring robust boundary detection and multi-scale feature extraction (e.g., automated waste classification)
---

## 🌟 Architecture Overview

MCSAM is a drop-in replacement for standard attention mechanisms like CBAM. It consists of two sequential, novel components:

1. **Multi-Scale Channel Attention (MSCA)**: Replaces single-scale channel pooling with three parallel convolutional branches (1×1, 3×3, 5×5) to capture point-wise, local, and broader spatial channel relationships.
2. **Edge-Guided Spatial Attention (EGSA)**: Enhances standard spatial attention by explicitly injecting structural boundary cues using fixed, non-learnable Sobel filters. This allows the network to focus on object boundaries and suppress cluttered backgrounds.

---

## 📦 Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- numpy

Install dependencies via:
```bash
pip install torch torchvision numpy

1. Initialize the Model
import torch
from mcsam_convnext import get_model, count_parameters

# Initialize model for 6-class waste classification (e.g., TrashNet)
# pretrained=True loads ImageNet weights for the ConvNeXt-Tiny backbone
model = get_model(num_classes=6, pretrained=True, device='cuda')

# Check parameter counts
count_parameters(model)

2. Data Preprocessing
from mcsam_convnext import get_transforms

train_transform, val_transform = get_transforms()

 3. Training Utilities
from mcsam_convnext import get_optimizer, get_scheduler

# Optimizer applies 1/10th learning rate to the backbone, full LR to MCSAM & Head
optimizer = get_optimizer(model, lr=1e-4, weight_decay=1e-4)
scheduler = get_scheduler(optimizer, epochs=50)

# Note: Pass these to your standard PyTorch training loop. 
# The full training loop is also available via: from mcsam_convnext import train
