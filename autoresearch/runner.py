"""
Experiment runner: orchestrates train → evaluate → analyze → report.
Each experiment gets its own directory under D:/drone2_training/autoresearch/.
"""
import os
import sys
import json
import subprocess
import shutil
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autoresearch.experiment import ExperimentConfig

# GPU cost from environment (set by cloud worker, 0 for local runs)
GPU_COST_PER_HOUR = float(os.environ.get("GPU_COST_PER_HOUR", "0.0"))
GPU_TYPE = os.environ.get("GPU_TYPE", "unknown")

# FPS benchmarks for cost estimation (steps/sec on single GPU)
_FPS_BENCHMARKS = {
    "fsppo": {"RTX_4090": 65000, "RTX_3090": 45000, "A100": 80000, "A10": 35000, "unknown": 50000},
    "rppo":  {"RTX_4090": 27000, "RTX_3090": 18000, "A100": 35000, "A10": 15000, "unknown": 20000},
    "mamba": {"RTX_4090": 25000, "RTX_3090": 17000, "A100": 32000, "A10": 12000, "unknown": 18000},
    "ppo":   {"RTX_4090": 56000, "RTX_3090": 38000, "A100": 70000, "A10": 30000, "unknown": 40000},
}


def estimate_cost(model_type: str, total_timesteps: int,
                  gpu_type: str, cost_per_hour: float) -> dict:
    """Estimate training cost before running."""
    fps = _FPS_BENCHMARKS.get(model_type, {}).get(gpu_type, 20000)
    hours = total_timesteps / fps / 3600
    cost = hours * cost_per_hour
    return {
        "estimated_hours": round(hours, 3),
        "estimated_cost": round(cost, 4),
        "estimated_fps": fps,
    }

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use D: drive if available (GPU training machine), else fall back to local
_DEFAULT_BASE = "D:/drone2_training/autoresearch"
if not os.path.exists("D:/"):
    _DEFAULT_BASE = os.path.join(PROJECT_ROOT, "training_output", "autoresearch")
AUTORESEARCH_BASE = os.environ.get("AUTORESEARCH_DIR", _DEFAULT_BASE)


def get_git_hash():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def run_experiment(config: ExperimentConfig, dry_run=False):
    """Run a single experiment end-to-end."""

    # Generate ID and set metadata
    if not config.experiment_id:
        config.generate_id()
    config.git_hash = get_git_hash()
    config.started_at = datetime.now().isoformat()
    config.status = "running"

    exp_dir = os.path.join(AUTORESEARCH_BASE, config.experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(exp_dir, "config.json")
    config.save(config_path)

    # Save hypothesis
    hyp_path = os.path.join(exp_dir, "hypothesis.md")
    with open(hyp_path, 'w') as f:
        f.write(f"# {config.name}\n\n")
        f.write(f"**Hypothesis**: {config.hypothesis}\n\n")
        f.write(f"**Rationale**: {config.rationale}\n\n")
        if config.parent:
            f.write(f"**Parent**: {config.parent}\n\n")
        f.write(f"**Success metric**: {config.success_metric}\n")
        if config.success_threshold is not None:
            f.write(f"**Threshold**: {config.success_threshold}\n")

    # ── Cost estimation ──
    config.gpu_cost_per_hour = config.gpu_cost_per_hour or GPU_COST_PER_HOUR
    config.gpu_type = config.gpu_type or GPU_TYPE
    est = estimate_cost(config.model_type, config.total_timesteps,
                        config.gpu_type, config.gpu_cost_per_hour)
    config.estimated_hours = est["estimated_hours"]
    config.estimated_cost = est["estimated_cost"]

    print(f"{'='*70}")
    print(f"EXPERIMENT: {config.experiment_id}")
    print(f"Model: {config.model_type} | Steps: {config.total_timesteps:,}")
    print(f"GPU: {config.gpu_type} @ ${config.gpu_cost_per_hour:.2f}/hr")
    print(f"Estimated: {config.estimated_hours:.1f}h / ${config.estimated_cost:.2f}")
    print(f"Hypothesis: {config.hypothesis}")
    print(f"Output: {exp_dir}")
    print(f"{'='*70}")

    config.save(config_path)  # Save with estimates before training starts

    # Build training command
    train_args = config.to_train_args()
    train_args.extend(["--save-dir", exp_dir])

    cmd = [sys.executable, "-m", config.train_script] + train_args

    if dry_run:
        print(f"\n[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        config.status = "dry_run"
        config.save(config_path)
        return config

    # Run training
    print(f"\nStarting training...")
    print(f"Command: {' '.join(cmd[:6])}...")
    train_start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output to console AND log file
        log_path = os.path.join(exp_dir, "training.log")
        with open(log_path, 'w') as log_file:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()

        process.wait()

        if process.returncode != 0:
            config.status = "failed"
            config.completed_at = datetime.now().isoformat()
            config.save(config_path)
            print(f"\nTraining FAILED with return code {process.returncode}")
            return config

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        config.status = "interrupted"
        config.completed_at = datetime.now().isoformat()
        config.save(config_path)
        # Still analyze what we have
    except Exception as e:
        config.status = "failed"
        config.completed_at = datetime.now().isoformat()
        config.save(config_path)
        print(f"\nTraining error: {e}")
        return config

    config.completed_at = datetime.now().isoformat()

    # ── Compute actual cost ──
    config.actual_hours = round((time.time() - train_start_time) / 3600, 4)
    config.actual_cost = round(config.actual_hours * config.gpu_cost_per_hour, 4)
    print(f"\n  Wall time: {config.actual_hours:.2f}h")
    print(f"  Actual cost: ${config.actual_cost:.2f}"
          f" (estimated: ${config.estimated_cost:.2f},"
          f" {'over' if config.actual_cost > config.estimated_cost else 'under'}"
          f" by ${abs(config.actual_cost - config.estimated_cost):.2f})")

    # ── Analyze ──
    print(f"\nAnalyzing results...")
    csv_path = os.path.join(exp_dir, "training_stats.csv")
    metrics = {}
    plots = []

    if os.path.exists(csv_path):
        from autoresearch.analyzer import load_csv, compute_metrics, generate_plots
        data, header = load_csv(csv_path)
        metrics = compute_metrics(data)
        analysis_dir = os.path.join(exp_dir, "analysis")
        plots = generate_plots(data, analysis_dir)

        # Save metrics JSON
        metrics_path = os.path.join(analysis_dir, "metrics.json")
        os.makedirs(analysis_dir, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  Reward (final smooth): {metrics.get('reward_final_smooth', 'N/A'):.4f}")
        print(f"  Avg gates: {metrics.get('avg_gates', 'N/A')}")
        print(f"  Pass rate (final): {metrics.get('pass_rate_final_smooth', 'N/A')}")
    else:
        print(f"  No training_stats.csv found")

    # ── Evaluate ──
    print(f"\nRunning evaluation...")
    eval_results = {}
    final_model = find_final_model(exp_dir, config.model_type)

    if final_model:
        for track in config.eval_tracks:
            try:
                result = run_evaluation(
                    final_model, config.model_type, track,
                    config.eval_episodes, config.frame_stack
                )
                eval_results[track] = result
                print(f"  {track}: {result.get('avg_gates', 'N/A')} avg gates, "
                      f"{result.get('completion_rate', 'N/A')} completion")
            except Exception as e:
                eval_results[track] = {"error": str(e)}
                print(f"  {track}: evaluation failed - {e}")
    else:
        print(f"  No final model found for evaluation")

    # Save eval results
    if eval_results:
        eval_path = os.path.join(exp_dir, "eval_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

    # ── Cost efficiency ──
    avg_gates = metrics.get("avg_gates", 0)
    if config.actual_cost > 0 and avg_gates > 0:
        config.cost_efficiency = round(avg_gates / config.actual_cost, 2)

    # ── Generate Report ──
    if config.status != "failed":
        config.status = "completed"
    from autoresearch.analyzer import generate_report
    report_path = os.path.join(exp_dir, "report.md")
    generate_report(config, metrics, eval_results, plots, report_path)
    config.save(config_path)

    # ── Update journal ──
    update_journal(config, metrics, eval_results)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {config.experiment_id}")
    print(f"Status: {config.status}")
    print(f"Report: {report_path}")
    print(f"{'='*70}")

    return config


def find_final_model(exp_dir, model_type):
    """Find the final model checkpoint in experiment directory."""
    # Look for *_final.pt
    for f in os.listdir(exp_dir):
        if f.endswith('_final.pt'):
            return os.path.join(exp_dir, f)
    # Fall back to latest checkpoint
    checkpoints = sorted([
        f for f in os.listdir(exp_dir)
        if f.endswith('.pt') and 'final' not in f
    ])
    if checkpoints:
        return os.path.join(exp_dir, checkpoints[-1])
    return None


def run_evaluation(checkpoint, model_type, track, episodes, frame_stack=None):
    """Run evaluation and return results dict."""
    cmd = [
        sys.executable, "-m", "evaluation.evaluate",
        "--checkpoint", checkpoint,
        "--model-type", model_type,
        "--tracks", track,
        "--episodes", str(episodes),
        "--device", "cuda",
    ]
    if frame_stack and model_type == "fsppo":
        cmd.extend(["--frame-stack", str(frame_stack)])

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=600
    )

    # Parse output for key metrics
    output = result.stdout + result.stderr
    metrics = parse_eval_output(output, track)
    return metrics


def parse_eval_output(output, track):
    """Parse evaluation script output for metrics."""
    metrics = {"raw_output": output[-500:] if len(output) > 500 else output}

    for line in output.split('\n'):
        line = line.strip()
        # Look for common patterns in eval output
        if 'avg gates' in line.lower() or 'average gates' in line.lower():
            try:
                val = float(line.split(':')[-1].strip().split()[0])
                metrics['avg_gates'] = val
            except (ValueError, IndexError):
                pass
        if 'completion' in line.lower():
            try:
                val = line.split(':')[-1].strip()
                metrics['completion_rate'] = val
            except (ValueError, IndexError):
                pass
        if 'lap time' in line.lower() or 'avg time' in line.lower():
            try:
                val = float(line.split(':')[-1].strip().rstrip('s').strip())
                metrics['avg_time'] = val
            except (ValueError, IndexError):
                pass

    return metrics


def update_journal(config, metrics, eval_results):
    """Append experiment summary to the master journal."""
    journal_path = os.path.join(AUTORESEARCH_BASE, "JOURNAL.md")

    # Create journal if it doesn't exist
    if not os.path.exists(journal_path):
        with open(journal_path, 'w') as f:
            f.write("# Autoresearch Experiment Journal\n\n")
            f.write("Automated experiment log. Each entry links to its full report.\n\n")
            f.write("---\n\n")

    with open(journal_path, 'a') as f:
        f.write(f"## {config.experiment_id}\n\n")
        f.write(f"- **Model**: {config.model_type} | **Steps**: {config.total_timesteps:,}\n")
        f.write(f"- **Hypothesis**: {config.hypothesis}\n")
        f.write(f"- **Status**: {config.status}\n")
        f.write(f"- **Git**: {config.git_hash}\n")

        # Cost tracking
        f.write(f"- **GPU**: {config.gpu_type} @ ${config.gpu_cost_per_hour:.2f}/hr\n")
        f.write(f"- **Cost**: ${config.actual_cost:.2f} actual"
                f" / ${config.estimated_cost:.2f} estimated"
                f" ({config.actual_hours:.2f}h actual"
                f" / {config.estimated_hours:.1f}h estimated)\n")
        if config.cost_efficiency > 0:
            f.write(f"- **Cost efficiency**: {config.cost_efficiency:.1f} avg_gates/$\n")

        if metrics:
            f.write(f"- **Reward (smooth)**: {metrics.get('reward_final_smooth', 'N/A')}\n")
            f.write(f"- **Avg gates**: {metrics.get('avg_gates', 'N/A')}\n")
            f.write(f"- **Pass rate**: {metrics.get('pass_rate_final_smooth', 'N/A')}\n")

        if eval_results:
            for track, result in eval_results.items():
                if isinstance(result, dict) and 'avg_gates' in result:
                    f.write(f"- **Eval {track}**: {result['avg_gates']} avg gates\n")

        f.write(f"- **Report**: [{config.experiment_id}/report.md]({config.experiment_id}/report.md)\n")
        f.write(f"\n---\n\n")


def run_queue(queue_dir=None, dry_run=False):
    """Run all pending experiments from the queue directory."""
    if queue_dir is None:
        queue_dir = os.path.join(PROJECT_ROOT, "autoresearch", "experiments")

    if not os.path.exists(queue_dir):
        print(f"No experiment queue at {queue_dir}")
        return []

    configs = sorted([
        f for f in os.listdir(queue_dir)
        if f.endswith('.json') and f != 'TEMPLATE.json'
    ])

    if not configs:
        print("No experiments in queue.")
        return []

    print(f"Found {len(configs)} experiments in queue:")
    for c in configs:
        print(f"  - {c}")
    print()

    results = []
    for config_file in configs:
        path = os.path.join(queue_dir, config_file)
        config = ExperimentConfig.load(path)

        if config.status == "completed":
            print(f"Skipping {config_file} (already completed)")
            continue

        result = run_experiment(config, dry_run=dry_run)
        results.append(result)

        # Move completed experiment file to done/ subfolder
        if result.status in ("completed", "failed", "interrupted"):
            done_dir = os.path.join(queue_dir, "done")
            os.makedirs(done_dir, exist_ok=True)
            shutil.move(path, os.path.join(done_dir, config_file))

    return results
