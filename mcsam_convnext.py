"""
MCSAM: Multi-Scale Channel-Spatial Attention Module
Backbone: ConvNeXt-Tiny
Task: Waste Classification

Architecture:
  - Multi-Scale Channel Attention: parallel 1x1, 3x3, 5x5 convolutions → channel weights
  - Edge-Guided Spatial Attention: Sobel filter branch → spatial map focused on boundaries
  - Backbone: ConvNeXt-Tiny (pretrained ImageNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ─────────────────────────────────────────────
# 1.  Multi-Scale Channel Attention (MSCA)
# ─────────────────────────────────────────────
class MultiScaleChannelAttention(nn.Module):
    """
    Replaces CBAM's single-scale channel attention.
    Uses three parallel convolution branches (1x1, 3x3, 5x5) on
    globally pooled features, then fuses them for richer channel weights.

    Steps:
      1. Global Average Pool  → (B, C, 1, 1)
      2. Global Max Pool      → (B, C, 1, 1)
      3. Three parallel conv branches on each pooled tensor
      4. Fuse all branches → sigmoid → channel weight
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(MultiScaleChannelAttention, self).__init__()

        reduced = max(in_channels // reduction_ratio, 8)

        # Branch 1 — 1x1 convolution (point-wise)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        )

        # Branch 2 — 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        )

        # Branch 3 — 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        )

        # Learnable fusion weights for avg and max pool streams
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # Global pooling → (B, C, 1, 1)
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)

        # Multi-scale features from each pool
        avg_feat = self.branch1(avg) + self.branch2(avg) + self.branch3(avg)
        max_feat = self.branch1(mx)  + self.branch2(mx)  + self.branch3(mx)

        # Fuse avg + max streams
        combined = torch.cat([avg_feat, max_feat], dim=1)   # (B, 2C, 1, 1)
        channel_weight = self.sigmoid(self.fusion(combined)) # (B, C, 1, 1)

        return x * channel_weight


# ─────────────────────────────────────────────
# 2.  Edge-Guided Spatial Attention (EGSA)
# ─────────────────────────────────────────────
class EdgeGuidedSpatialAttention(nn.Module):
    """
    Replaces CBAM's plain spatial attention.
    Adds a Sobel edge-detection branch so the spatial map is
    explicitly guided by object boundaries — critical for deformed
    or crumpled waste objects.

    Steps:
      1. Standard spatial stream: channel-wise avg + max → conv → sigmoid
      2. Edge stream: Sobel filter on grayscale of x → learned refinement
      3. Final spatial map = sigmoid( standard + edge )
    """

    def __init__(self, in_channels):
        super(EdgeGuidedSpatialAttention, self).__init__()

        # Standard spatial branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )

        # Edge refinement branch (operates on single-channel edge map)
        self.edge_refine = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1)
        )

        # Fixed Sobel kernels — not learned, represent true edge detection
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Register as buffers so they move with .to(device) but are not trained
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.sigmoid = nn.Sigmoid()

    def _compute_edges(self, x):
        """Convert feature map to grayscale, apply Sobel, return edge magnitude."""
        # Grayscale approximation: mean across channels → (B, 1, H, W)
        gray = x.mean(dim=1, keepdim=True)

        # Normalize to [0, 1] per sample for stable Sobel response
        b_min = gray.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        b_max = gray.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        gray  = (gray - b_min) / (b_max - b_min + 1e-8)

        # Sobel gradients
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)

        # Edge magnitude
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return edge   # (B, 1, H, W)

    def forward(self, x):
        # ── Standard spatial stream ──────────────────────────
        avg_map = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_map, _ = x.max(dim=1, keepdim=True)        # (B, 1, H, W)
        spatial_in = torch.cat([avg_map, max_map], 1)  # (B, 2, H, W)
        spatial_out = self.spatial_conv(spatial_in)    # (B, 1, H, W)

        # ── Edge stream ──────────────────────────────────────
        edge_map    = self._compute_edges(x)           # (B, 1, H, W)
        edge_out    = self.edge_refine(edge_map)       # (B, 1, H, W)

        # ── Combine and apply ────────────────────────────────
        spatial_weight = self.sigmoid(spatial_out + edge_out)  # (B, 1, H, W)
        return x * spatial_weight


# ─────────────────────────────────────────────
# 3.  MCSAM — full attention module
# ─────────────────────────────────────────────
class MCSAM(nn.Module):
    """
    Multi-Scale Channel-Spatial Attention Module.

    Pipeline:
        x → MultiScaleChannelAttention → EdgeGuidedSpatialAttention → output

    Drop-in replacement for CBAM.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(MCSAM, self).__init__()
        self.channel_att = MultiScaleChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = EdgeGuidedSpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ─────────────────────────────────────────────
# 4.  ConvNeXt-Tiny + MCSAM  (full model)
# ─────────────────────────────────────────────
class ConvNeXtTinyMCSAM(nn.Module):
    """
    ConvNeXt-Tiny backbone with MCSAM attention injected after each
    of the four stages.

    Args:
        num_classes  : number of waste categories  (default 7 for TrashNet)
        pretrained   : load ImageNet weights for ConvNeXt-Tiny
        reduction    : channel reduction ratio inside MSCA
    
    Stage output channels for ConvNeXt-Tiny:
        stage 0 → 96
        stage 1 → 192
        stage 2 → 384
        stage 3 → 768
    """

    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, num_classes=7, pretrained=True, reduction=16):
        super(ConvNeXtTinyMCSAM, self).__init__()

        # ── Load pretrained ConvNeXt-Tiny ────────────────────
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # ConvNeXt feature extractor has 4 sequential stages inside `features`
        # features[0] = stem, features[1..7] = stages & downsampling layers
        # We expose each stage separately so we can insert MCSAM in between.
        self.stem    = backbone.features[0]   # stem conv

        # Stage 1  (output 96 ch)
        self.stage1  = backbone.features[1]
        self.mcsam1  = MCSAM(self.STAGE_CHANNELS[0], reduction)

        # Downsample 1→2
        self.down1   = backbone.features[2]

        # Stage 2  (output 192 ch)
        self.stage2  = backbone.features[3]
        self.mcsam2  = MCSAM(self.STAGE_CHANNELS[1], reduction)

        # Downsample 2→3
        self.down2   = backbone.features[4]

        # Stage 3  (output 384 ch)
        self.stage3  = backbone.features[5]
        self.mcsam3  = MCSAM(self.STAGE_CHANNELS[2], reduction)

        # Downsample 3→4
        self.down3   = backbone.features[6]

        # Stage 4  (output 768 ch)
        self.stage4  = backbone.features[7]
        self.mcsam4  = MCSAM(self.STAGE_CHANNELS[3], reduction)

        # ── Classifier head ──────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(p=0.4),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.mcsam1(x)

        x = self.down1(x)
        x = self.stage2(x)
        x = self.mcsam2(x)

        x = self.down2(x)
        x = self.stage3(x)
        x = self.mcsam3(x)

        x = self.down3(x)
        x = self.stage4(x)
        x = self.mcsam4(x)

        x = self.head(x)
        return x


# ─────────────────────────────────────────────
# 5.  Training utilities
# ─────────────────────────────────────────────
def get_model(num_classes=7, pretrained=True, device='cuda'):
    model = ConvNeXtTinyMCSAM(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """
    Differential learning rates:
      - backbone stages get a lower lr (fine-tuning)
      - MCSAM modules and head get the full lr
    """
    backbone_params = []
    mcsam_params    = []
    head_params     = []

    for name, param in model.named_parameters():
        if 'mcsam' in name:
            mcsam_params.append(param)
        elif 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},   # backbone: lr/10
        {'params': mcsam_params,    'lr': lr},          # MCSAM: full lr
        {'params': head_params,     'lr': lr},          # head: full lr
    ], weight_decay=weight_decay)

    return optimizer


def get_scheduler(optimizer, epochs=50):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


# ─────────────────────────────────────────────
# 6.  Training loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs      = model(imgs)
        loss         = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, val_loader, epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mcsam_model.pth')
            print(f"  ✅ Best model saved (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")


# ─────────────────────────────────────────────
# 7.  Grad-CAM utility
# ─────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM visualization for ConvNeXtTinyMCSAM.
    Target layer: stage4 (last convolutional stage, 768 ch).

    Usage:
        gradcam = GradCAM(model)
        heatmap = gradcam(img_tensor, class_idx=None)  # None → predicted class
    """

    def __init__(self, model):
        self.model    = model
        self.gradients = None
        self.activations = None
        # Hook onto stage4 (last stage before MCSAM4 + head)
        model.stage4.register_forward_hook(self._save_activation)
        model.stage4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()


# ─────────────────────────────────────────────
# 8.  Data transforms (recommended)
# ─────────────────────────────────────────────
def get_transforms():
    """
    Standard ImageNet-style transforms for training and validation.
    Paste into your dataset/dataloader setup.
    """
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────────
# 9.  Quick sanity check
# ─────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Build model
    model = get_model(num_classes=7, pretrained=False, device=device)

    # Parameter count
    count_parameters(model)

    # Forward pass test
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # should be (4, 7)

    # Grad-CAM test
    gradcam   = GradCAM(model)
    single    = torch.randn(1, 3, 224, 224).to(device)
    heatmap   = gradcam(single)
    print(f"Grad-CAM map : {heatmap.shape}")  # (224, 224)

    print("\n✅ MCSAM + ConvNeXt-Tiny model ready for training.")
