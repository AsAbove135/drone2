"""
Package drone2 project files into a zip for Google Colab upload.
Run from project root: python colab/package_for_colab.py
Creates: colab/drone2_colab.zip
"""
import zipfile
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "colab", "drone2_colab.zip")

# Files needed for GPU RPPO training
FILES = [
    "config.py",
    "control/__init__.py",
    "control/gpu_env.py",
    "control/dynamics.py",
    "control/plot_training.py",
    "control_rppo/__init__.py",
    "control_rppo/train_gpu_rppo.py",
    "control_rppo/eval_gpu_rppo.py",
    "control_mamba/__init__.py",
    "control_mamba/train_gpu_mamba.py",
    "control_mamba/eval_gpu_mamba.py",
]

def main():
    # Check all files exist
    missing = [f for f in FILES if not os.path.exists(os.path.join(PROJECT_ROOT, f))]
    if missing:
        print(f"Missing files: {missing}")
        return

    with zipfile.ZipFile(OUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in FILES:
            full = os.path.join(PROJECT_ROOT, f)
            zf.write(full, f"drone2/{f}")
            print(f"  Added: {f}")

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nCreated: {OUT_PATH} ({size_kb:.1f} KB)")
    print("Upload this zip to your Google Drive, then run the Colab notebook.")

if __name__ == "__main__":
    main()
