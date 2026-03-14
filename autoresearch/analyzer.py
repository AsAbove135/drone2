"""
Post-training analysis: reads CSV logs, computes metrics, generates plots,
and produces a structured summary for the experiment report.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(csv_path):
    """Load training_stats.csv into a dict of arrays."""
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        rows = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append([float(x) for x in line.split(',')])
                except ValueError:
                    continue
    if not rows:
        return None, header
    data = {}
    for i, col in enumerate(header):
        data[col] = np.array([r[i] for r in rows])
    return data, header


def compute_metrics(data):
    """Extract key metrics from training data."""
    if data is None:
        return {"error": "no data"}

    metrics = {}
    steps = data.get('steps', np.array([]))

    # Reward
    if 'reward' in data:
        r = data['reward']
        metrics['reward_final'] = float(r[-1]) if len(r) > 0 else 0
        metrics['reward_peak'] = float(np.max(r)) if len(r) > 0 else 0
        metrics['reward_peak_step'] = int(steps[np.argmax(r)]) if len(r) > 0 else 0
        # Smoothed final (last 10%)
        n10 = max(1, len(r) // 10)
        metrics['reward_final_smooth'] = float(np.mean(r[-n10:]))

    # Gate passage
    if 'pass_rate' in data:
        pr = data['pass_rate']
        metrics['pass_rate_final'] = float(pr[-1]) if len(pr) > 0 else 0
        metrics['pass_rate_peak'] = float(np.max(pr)) if len(pr) > 0 else 0
        n10 = max(1, len(pr) // 10)
        metrics['pass_rate_final_smooth'] = float(np.mean(pr[-n10:]))

    if 'gates_passed' in data and 'episodes' in data:
        gp = data['gates_passed']
        ep = data['episodes']
        # Compute avg gates per episode over last 10% of training
        n10 = max(1, len(gp) // 10)
        total_gates = np.sum(gp[-n10:])
        total_eps = np.sum(ep[-n10:])
        metrics['avg_gates'] = float(total_gates / max(total_eps, 1))

    # Training stability
    if 'entropy' in data:
        ent = data['entropy']
        metrics['entropy_final'] = float(ent[-1]) if len(ent) > 0 else 0
        metrics['entropy_trend'] = "rising" if len(ent) > 10 and ent[-1] > ent[len(ent)//2] else "falling"

    if 'clip_frac' in data:
        cf = data['clip_frac']
        metrics['clip_frac_final'] = float(cf[-1]) if len(cf) > 0 else 0

    # FPS
    if 'fps' in data:
        metrics['fps_avg'] = float(np.mean(data['fps']))

    # Total steps
    metrics['total_steps'] = int(steps[-1]) if len(steps) > 0 else 0

    return metrics


def generate_plots(data, out_dir):
    """Generate analysis plots."""
    os.makedirs(out_dir, exist_ok=True)
    if data is None:
        return []

    plots = []
    steps = data.get('steps', np.array([]))
    step_m = steps / 1e6  # In millions

    # 1. Reward curve
    if 'reward' in data:
        fig, ax = plt.subplots(figsize=(10, 4))
        r = data['reward']
        ax.plot(step_m, r, alpha=0.3, color='blue', linewidth=0.5)
        # Smoothed
        window = max(1, len(r) // 50)
        if window > 1:
            smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
            ax.plot(step_m[:len(smoothed)], smoothed, color='blue', linewidth=1.5, label='smoothed')
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, 'reward.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # 2. Gate passage rate
    if 'pass_rate' in data:
        fig, ax = plt.subplots(figsize=(10, 4))
        pr = data['pass_rate']
        ax.plot(step_m, pr, alpha=0.3, color='green', linewidth=0.5)
        window = max(1, len(pr) // 50)
        if window > 1:
            smoothed = np.convolve(pr, np.ones(window)/window, mode='valid')
            ax.plot(step_m[:len(smoothed)], smoothed, color='green', linewidth=1.5, label='smoothed')
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Pass Rate (gates/episode)')
        ax.set_title('Gate Passage Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, 'pass_rate.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # 3. Loss & entropy panel
    loss_keys = ['pg_loss', 'v_loss', 'entropy', 'clip_frac']
    available = [k for k in loss_keys if k in data]
    if available:
        fig, axes = plt.subplots(len(available), 1, figsize=(10, 3 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]
        for ax, key in zip(axes, available):
            vals = data[key]
            ax.plot(step_m, vals, linewidth=0.8)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Steps (M)')
        fig.suptitle('Training Diagnostics')
        fig.tight_layout()
        path = os.path.join(out_dir, 'diagnostics.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    return plots


def generate_report(config, metrics, eval_results, plots, out_path):
    """Generate a markdown experiment report."""
    lines = []
    lines.append(f"# Experiment: {config.name}")
    lines.append(f"**ID**: {config.experiment_id}")
    lines.append(f"**Date**: {config.started_at or 'unknown'}")
    lines.append(f"**Status**: {config.status}")
    lines.append(f"**Model**: {config.model_type}")
    lines.append(f"**Total Steps**: {config.total_timesteps:,}")
    if config.parent:
        lines.append(f"**Parent Experiment**: {config.parent}")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append(config.hypothesis)
    lines.append("")

    lines.append("## Rationale")
    lines.append(config.rationale)
    lines.append("")

    # Cost breakdown
    lines.append("## Cost")
    lines.append("| | Estimated | Actual |")
    lines.append("|--|-----------|--------|")
    lines.append(f"| Hours | {config.estimated_hours:.2f}h | {config.actual_hours:.2f}h |")
    lines.append(f"| Cost | ${config.estimated_cost:.2f} | ${config.actual_cost:.2f} |")
    if config.gpu_type or config.gpu_cost_per_hour > 0:
        lines.append(f"\n**GPU**: {config.gpu_type} @ ${config.gpu_cost_per_hour:.2f}/hr")
    if config.cost_efficiency > 0:
        lines.append(f"\n**Cost efficiency**: {config.cost_efficiency:.1f} avg_gates per dollar")
    accuracy_pct = 0
    if config.estimated_cost > 0:
        accuracy_pct = (1 - abs(config.actual_cost - config.estimated_cost) / config.estimated_cost) * 100
        lines.append(f"\n**Estimate accuracy**: {accuracy_pct:.0f}%")
    lines.append("")

    # Key config differences from defaults
    lines.append("## Key Configuration")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    key_params = [
        ("model_type", config.model_type),
        ("total_timesteps", f"{config.total_timesteps:,}"),
        ("num_envs", config.num_envs),
        ("n_steps", config.n_steps),
        ("lr", config.lr),
        ("ent_coef", config.ent_coef),
        ("cosine_lr", config.cosine_lr),
    ]
    if config.model_type == "fsppo":
        key_params.append(("frame_stack", config.frame_stack))
    elif config.model_type == "rppo":
        key_params.extend([
            ("lstm_hidden", config.lstm_hidden),
            ("seq_len", config.seq_len),
        ])
    for name, val in key_params:
        lines.append(f"| {name} | {val} |")
    # Reward lambdas
    for attr in ["lambda_prog", "lambda_gate", "lambda_gate_inc", "lambda_rate"]:
        val = getattr(config, attr)
        if val is not None:
            lines.append(f"| {attr} | {val} |")
    lines.append("")

    # Metrics
    lines.append("## Training Metrics")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # Evaluation results
    if eval_results:
        lines.append("## Evaluation Results")
        for track_name, result in eval_results.items():
            lines.append(f"### Track: {track_name}")
            if isinstance(result, dict):
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for k, v in result.items():
                    if isinstance(v, float):
                        lines.append(f"| {k} | {v:.4f} |")
                    else:
                        lines.append(f"| {k} | {v} |")
            else:
                lines.append(str(result))
            lines.append("")

    # Plots
    if plots:
        lines.append("## Charts")
        for p in plots:
            rel = os.path.basename(p)
            lines.append(f"![{rel}]({rel})")
        lines.append("")

    # Conclusion placeholder
    lines.append("## Conclusion")
    lines.append("_To be filled by researcher agent or user._")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return out_path
