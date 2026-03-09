"""
Generate training charts from CSV log.
Usage: python control/plot_training.py [--csv PATH] [--out-dir PATH] [--run-name NAME]

Reads training_stats.csv and produces charts in results_tracking/<run_name>/:
  1. reward.png — mean episode reward over training
  2. avg_gates.png — average gates passed per episode
  3. gate_distribution.png — % of episodes passing >= N gates
  4. curriculum.png — gate size and spawn distance over training
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_csv(csv_path):
    """Load training_stats.csv into a dict of arrays."""
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        rows = []
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(',')])
    if not rows:
        print("No data rows found in CSV.")
        sys.exit(1)
    data = {}
    for i, col in enumerate(header):
        data[col] = np.array([r[i] for r in rows])
    return data, header


def plot_reward(data, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = data['steps'] / 1e6
    ax.plot(steps, data['reward_mean'], 'b-', linewidth=1.5)
    ax.set_xlabel('Training Steps (M)')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Episode Reward Over Training')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'reward.png'), dpi=150)
    plt.close()


def plot_avg_gates(data, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = data['steps'] / 1e6
    ax.plot(steps, data['avg_gates'], 'g-', linewidth=1.5)
    ax.set_xlabel('Training Steps (M)')
    ax.set_ylabel('Avg Gates Passed / Episode')
    ax.set_title('Average Gates Passed Per Episode')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'avg_gates.png'), dpi=150)
    plt.close()


def plot_gate_distribution(data, header, out_dir):
    """Plot % of episodes passing >= N gates for each N."""
    pct_cols = [c for c in header if c.startswith('pct_ge_')]
    if not pct_cols:
        return
    num_thresholds = len(pct_cols)
    steps = data['steps'] / 1e6

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.viridis
    for i, col in enumerate(pct_cols):
        gate_num = int(col.split('_')[-1])
        color = cmap(i / max(num_thresholds - 1, 1))
        ax.plot(steps, data[col], color=color, linewidth=1.2, label=f'>= {gate_num} gate{"s" if gate_num > 1 else ""}')

    ax.set_xlabel('Training Steps (M)')
    ax.set_ylabel('% of Episodes')
    ax.set_title('Gate Passage Distribution (% of episodes passing >= N gates)')
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gate_distribution.png'), dpi=150)
    plt.close()


def plot_curriculum(data, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    steps = data['steps'] / 1e6

    ax1.plot(steps, data['gate_size'], 'r-', linewidth=1.5)
    ax1.set_ylabel('Gate Size (m)')
    ax1.set_title('Curriculum Progression')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, data['spawn_dist_max'], 'm-', linewidth=1.5)
    ax2.set_xlabel('Training Steps (M)')
    ax2.set_ylabel('Max Spawn Distance (m)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curriculum.png'), dpi=150)
    plt.close()


def plot_ppo_stats(data, out_dir):
    """Plot entropy, explained variance, approx KL, and clip fraction."""
    has_stats = all(k in data for k in ['entropy', 'explained_var', 'approx_kl', 'clip_fraction'])
    if not has_stats:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    steps = data['steps'] / 1e6

    # Entropy
    ax = axes[0, 0]
    ax.plot(steps, data['entropy'], 'purple', linewidth=1.5)
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy (exploration diversity)')
    ax.grid(True, alpha=0.3)

    # Explained Variance
    ax = axes[0, 1]
    ax.plot(steps, data['explained_var'], 'teal', linewidth=1.5)
    ax.set_ylabel('Explained Variance')
    ax.set_title('Value Function Explained Variance (critic accuracy)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Approx KL
    ax = axes[1, 0]
    ax.plot(steps, data['approx_kl'], 'orange', linewidth=1.5)
    ax.set_xlabel('Training Steps (M)')
    ax.set_ylabel('Approx KL')
    ax.set_title('Policy Update Size (KL divergence)')
    ax.grid(True, alpha=0.3)

    # Clip Fraction
    ax = axes[1, 1]
    ax.plot(steps, data['clip_fraction'], 'red', linewidth=1.5)
    ax.set_xlabel('Training Steps (M)')
    ax.set_ylabel('Clip Fraction')
    ax.set_title('PPO Clip Fraction (gradient clipping rate)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ppo_stats.png'), dpi=150)
    plt.close()
    return True


def generate_charts(csv_path, out_dir, run_name=None):
    """Generate all charts from a training CSV."""
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_run"

    chart_dir = os.path.join(out_dir, run_name)
    os.makedirs(chart_dir, exist_ok=True)

    data, header = load_csv(csv_path)
    print(f"Loaded {len(data['steps'])} data points from {csv_path}")
    print(f"Steps range: {data['steps'][0]:,.0f} — {data['steps'][-1]:,.0f}")

    plot_reward(data, chart_dir)
    print(f"  Saved reward.png")
    plot_avg_gates(data, chart_dir)
    print(f"  Saved avg_gates.png")
    plot_gate_distribution(data, header, chart_dir)
    print(f"  Saved gate_distribution.png")
    plot_curriculum(data, chart_dir)
    print(f"  Saved curriculum.png")
    if plot_ppo_stats(data, chart_dir):
        print(f"  Saved ppo_stats.png")

    print(f"\nAll charts saved to: {os.path.abspath(chart_dir)}")
    return chart_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="D:/drone2_training/latest/training_stats.csv",
                        help="Path to training_stats.csv")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: results_tracking/)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run's chart folder")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_tracking')

    generate_charts(args.csv, args.out_dir, args.run_name)
