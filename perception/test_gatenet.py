"""
GateNet visualization script.
Generates test images, runs segmentation, extracts corners,
and saves a visual grid showing the results.
"""
import torch
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.gatenet import GateNet
from perception.quadgate import QuAdGate
from perception.generate_data import render_gate_on_image


def visualize_predictions(model_path="checkpoints/gatenet_best.pth",
                          num_samples=8, base_filters=16, output_dir="test_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GateNet(n_channels=3, n_classes=1, base_filters=base_filters).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    quadgate = QuAdGate()
    rng = np.random.default_rng(999)

    os.makedirs(output_dir, exist_ok=True)

    # Generate test samples
    rows = []
    for i in range(num_samples):
        image_bgr, gt_mask = render_gate_on_image(384, rng)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run model
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            pred_mask = model(img_tensor.to(device)).squeeze().cpu().numpy()

        # Extract corners from prediction
        corners = quadgate.extract_corners(pred_mask)

        # Build visualization row: [original, GT mask, pred mask, corners overlay]
        # Original
        panel_orig = image_bgr.copy()

        # GT mask (colorized)
        panel_gt = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        # Predicted mask (colorized with heatmap)
        pred_vis = (pred_mask * 255).clip(0, 255).astype(np.uint8)
        panel_pred = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)

        # Corners overlay on original
        panel_corners = image_bgr.copy()
        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(panel_corners, [pts], True, (0, 255, 0), 2)
            for j, (cx, cy) in enumerate(corners):
                cv2.circle(panel_corners, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(panel_corners, str(j), (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            status = "4 corners detected"
        else:
            status = f"{len(corners)} corners (need 4)"
        cv2.putText(panel_corners, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Concatenate panels horizontally
        row = np.hstack([panel_orig, panel_gt, panel_pred, panel_corners])
        rows.append(row)

    # Add column headers
    header = np.zeros((40, 384 * 4, 3), dtype=np.uint8)
    labels = ["Input Image", "Ground Truth", "Prediction", "Detected Corners"]
    for j, label in enumerate(labels):
        cv2.putText(header, label, (j * 384 + 80, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stack all rows vertically
    grid = np.vstack([header] + rows)

    output_path = os.path.join(output_dir, "gatenet_predictions.png")
    cv2.imwrite(output_path, grid)
    print(f"\nVisualization saved to: {output_path}")
    print(f"Grid size: {grid.shape[1]}x{grid.shape[0]} pixels ({num_samples} samples)")

    # Also save individual predictions for closer inspection
    for i in range(min(3, num_samples)):
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}.png"), rows[i])

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test GateNet predictions")
    parser.add_argument("--model", type=str, default="checkpoints/gatenet_best.pth")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--output", type=str, default="test_results")
    args = parser.parse_args()

    visualize_predictions(
        model_path=args.model,
        num_samples=args.samples,
        base_filters=args.base_filters,
        output_dir=args.output,
    )
