"""
GateNet training script with Dice + BCE loss (matching MonoRace paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.gatenet import GateNet


class GateSegDataset(Dataset):
    """Dataset of gate images and binary segmentation masks."""
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.png')
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image = cv2.imread(os.path.join(self.image_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        # Runtime augmentations
        if self.augment:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # To tensors: image [C, H, W], mask [1, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def dice_loss(pred, target, smooth=1.0):
    """Differentiable Dice loss."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def dice_score(pred, target, threshold=0.5):
    """Dice score metric (non-differentiable)."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target.sum() + 1e-8)


def train(data_dir="data", epochs=100, batch_size=16, lr=1e-3, base_filters=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Datasets
    train_ds = GateSegDataset(
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "train", "masks"),
        augment=True,
    )
    val_ds = GateSegDataset(
        os.path.join(data_dir, "val", "images"),
        os.path.join(data_dir, "val", "masks"),
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples", flush=True)

    # Model
    model = GateNet(n_channels=3, n_classes=1, base_filters=base_filters).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"GateNet parameters: {param_count:,} (base_filters={base_filters})")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            pred = model(images)
            loss = dice_loss(pred, masks) + 2.0 * F.binary_cross_entropy(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                pred = model(images)
                loss = dice_loss(pred, masks) + 2.0 * F.binary_cross_entropy(pred, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_score(pred, masks).item() * images.size(0)

        val_loss /= len(val_ds)
        val_dice /= len(val_ds)

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "checkpoints/gatenet_best.pth")

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"LR: {lr_now:.6f}"
                  f"{' *best*' if val_dice >= best_val_dice else ''}",
                  flush=True)

    # Save final model
    torch.save(model.state_dict(), "checkpoints/gatenet_final.pth")
    print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
    print(f"Models saved to checkpoints/gatenet_best.pth and gatenet_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GateNet")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-filters", type=int, default=16)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_filters=args.base_filters,
    )
