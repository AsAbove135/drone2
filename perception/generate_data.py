"""
Procedural synthetic data generator for GateNet training.
Generates 384x384 images of racing gates on random backgrounds
with corresponding binary segmentation masks.
"""
import cv2
import numpy as np
import os
import argparse


def random_background(size=384, rng=None):
    """Generate a random background simulating indoor environments."""
    rng = rng or np.random.default_rng()
    bg = np.zeros((size, size, 3), dtype=np.uint8)

    choice = rng.integers(0, 4)

    if choice == 0:
        # Solid color with noise
        color = rng.integers(20, 200, size=3).tolist()
        bg[:] = color
        noise = rng.integers(-15, 15, bg.shape, dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif choice == 1:
        # Vertical gradient (floor/wall)
        wall_color = rng.integers(50, 180, size=3)
        floor_color = rng.integers(30, 120, size=3)
        split = rng.integers(size // 3, 2 * size // 3)
        for i in range(size):
            if i < split:
                t = i / max(split, 1)
                bg[i] = (wall_color * (1 - t * 0.3)).astype(np.uint8)
            else:
                t = (i - split) / max(size - split, 1)
                bg[i] = (floor_color * (1 - t * 0.2)).astype(np.uint8)

    elif choice == 2:
        # Random colored rectangles (clutter)
        base = rng.integers(30, 100, size=3)
        bg[:] = base.tolist()
        for _ in range(rng.integers(3, 10)):
            x1, y1 = rng.integers(0, size, size=2)
            w, h = rng.integers(20, size // 2, size=2)
            color = rng.integers(20, 220, size=3).tolist()
            cv2.rectangle(bg, (x1, y1), (x1 + w, y1 + h), color, -1)

    else:
        # Diagonal gradient
        c1 = rng.integers(30, 180, size=3).astype(np.float64)
        c2 = rng.integers(30, 180, size=3).astype(np.float64)
        for i in range(size):
            t = i / size
            bg[i] = ((1 - t) * c1 + t * c2).astype(np.uint8)

    return bg


def render_gate_on_image(size=384, rng=None):
    """
    Render a racing gate with random perspective on a random background.
    Returns (image, mask) both as (size, size) arrays.
    """
    rng = rng or np.random.default_rng()

    # Background
    image = random_background(size, rng)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Gate parameters
    gate_side = rng.integers(60, 280)  # pixel size of gate
    border = rng.integers(8, max(9, gate_side // 6))  # frame thickness

    # Gate color (orange, white, blue, red, green — typical racing gates)
    gate_colors = [
        (0, 140, 255),    # orange (BGR)
        (230, 230, 230),  # white
        (255, 100, 30),   # blue
        (50, 50, 220),    # red
        (50, 200, 50),    # green
        (0, 200, 255),    # yellow
    ]
    gate_color = gate_colors[rng.integers(0, len(gate_colors))]

    # Source corners of the gate frame (outer rectangle)
    outer = np.array([
        [0, 0],
        [gate_side, 0],
        [gate_side, gate_side],
        [0, gate_side],
    ], dtype=np.float32)

    # Inner opening (not part of the frame)
    inner = np.array([
        [border, border],
        [gate_side - border, border],
        [gate_side - border, gate_side - border],
        [border, gate_side - border],
    ], dtype=np.float32)

    # Apply random perspective transform
    # Random destination corners (simulating 3D viewpoint variation)
    cx = rng.integers(gate_side // 2 + 20, size - gate_side // 2 - 20)
    cy = rng.integers(gate_side // 2 + 20, size - gate_side // 2 - 20)

    # Perspective jitter per corner
    jitter = gate_side * 0.25
    dst = np.array([
        [cx - gate_side // 2 + rng.uniform(-jitter, jitter),
         cy - gate_side // 2 + rng.uniform(-jitter, jitter)],
        [cx + gate_side // 2 + rng.uniform(-jitter, jitter),
         cy - gate_side // 2 + rng.uniform(-jitter, jitter)],
        [cx + gate_side // 2 + rng.uniform(-jitter, jitter),
         cy + gate_side // 2 + rng.uniform(-jitter, jitter)],
        [cx - gate_side // 2 + rng.uniform(-jitter, jitter),
         cy + gate_side // 2 + rng.uniform(-jitter, jitter)],
    ], dtype=np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(outer, dst)

    # Transform inner corners too
    inner_h = np.hstack([inner, np.ones((4, 1), dtype=np.float32)])
    inner_dst = (M @ inner_h.T).T
    inner_dst = inner_dst[:, :2] / inner_dst[:, 2:3]
    inner_dst = inner_dst.astype(np.int32)

    # Draw the gate frame (outer - inner) on image
    dst_int = dst.astype(np.int32)

    # Draw filled outer polygon
    cv2.fillPoly(image, [dst_int], gate_color)
    cv2.fillPoly(mask, [dst_int], 255)

    # Cut out the inner opening (restore background there)
    bg_patch = random_background(size, rng)  # different background visible through gate
    roi_mask_inner = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(roi_mask_inner, [inner_dst], 255)

    # Restore background in the inner opening
    image[roi_mask_inner > 0] = bg_patch[roi_mask_inner > 0]
    mask[roi_mask_inner > 0] = 0

    # Add some edge detail / anti-aliasing on the gate frame
    cv2.polylines(image, [dst_int], True, tuple(max(0, c - 40) for c in gate_color), 2)

    return image, mask


def apply_augmentations(image, rng=None):
    """Apply paper-specified augmentations to the image (not the mask)."""
    rng = rng or np.random.default_rng()

    # HSV color jitter
    if rng.random() > 0.3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + rng.integers(-10, 10), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.integers(-50, 50), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.integers(-50, 50), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gaussian noise
    if rng.random() > 0.3:
        sigma = rng.uniform(5, 25)
        noise = rng.normal(0, sigma, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Motion blur
    if rng.random() > 0.5:
        ksize = int(rng.integers(3, 8)) * 2 + 1  # odd kernel
        angle = float(rng.uniform(0, 180))
        M_blur = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1.0)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0
        kernel = cv2.warpAffine(kernel, M_blur, (ksize, ksize))
        kernel /= kernel.sum() + 1e-8
        image = cv2.filter2D(image, -1, kernel)

    # Brightness variation
    if rng.random() > 0.3:
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Gaussian blur (slight)
    if rng.random() > 0.5:
        k = rng.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    return image


def generate_dataset(output_dir, num_samples, seed=42):
    """Generate a dataset of synthetic gate images and masks."""
    rng = np.random.default_rng(seed)

    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Also generate some negative samples (no gate)
    num_negatives = num_samples // 10

    for i in range(num_samples + num_negatives):
        if i < num_samples:
            image, mask = render_gate_on_image(384, rng)
            image = apply_augmentations(image, rng)
        else:
            # Negative sample: just background, no gate
            image = random_background(384, rng)
            image = apply_augmentations(image, rng)
            mask = np.zeros((384, 384), dtype=np.uint8)

        cv2.imwrite(os.path.join(img_dir, f"{i:05d}.png"), image)
        cv2.imwrite(os.path.join(mask_dir, f"{i:05d}.png"), mask)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples + num_negatives} samples")

    print(f"Dataset saved to {output_dir} ({num_samples} positive + {num_negatives} negative)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GateNet training data")
    parser.add_argument("--train-samples", type=int, default=3500)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    print("Generating training set...")
    generate_dataset(
        os.path.join(args.output, "train"),
        args.train_samples,
        seed=42,
    )

    print("\nGenerating validation set...")
    generate_dataset(
        os.path.join(args.output, "val"),
        args.val_samples,
        seed=123,
    )

    print("\nDone! Preview some samples:")
    print(f"  {args.output}/train/images/00000.png")
    print(f"  {args.output}/train/masks/00000.png")
