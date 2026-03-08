"""
Save the latest training run to permanent storage.
Usage: python control/save_run.py "short description of this run"
"""
import shutil
import sys
import os
from datetime import datetime

TRAIN_BASE = "D:/drone2_training"
LATEST_DIR = os.path.join(TRAIN_BASE, "latest")
SAVED_DIR = os.path.join(TRAIN_BASE, "saved")


def save_run(description):
    if not os.path.exists(LATEST_DIR):
        print("Nothing to save — no latest/ folder found.")
        return

    files = os.listdir(LATEST_DIR)
    if not files:
        print("Nothing to save — latest/ folder is empty.")
        return

    # Build destination name: 2026-03-07_15-30_description
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_desc = description.replace(" ", "_").replace("/", "-")[:50]
    dest_name = f"{timestamp}_{safe_desc}"
    dest_path = os.path.join(SAVED_DIR, dest_name)

    os.makedirs(SAVED_DIR, exist_ok=True)
    shutil.copytree(LATEST_DIR, dest_path)

    total_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(dest_path) for f in fns
    ) / (1024 * 1024)

    print(f"Saved to: {dest_path}")
    print(f"Size: {total_mb:.1f} MB")
    print(f"Files: {len(os.listdir(dest_path))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python control/save_run.py "description of this run"')
        sys.exit(1)
    save_run(" ".join(sys.argv[1:]))
