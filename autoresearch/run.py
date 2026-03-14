"""
CLI entry point for autoresearch.

Usage:
    # Run a single experiment from YAML
    python -m autoresearch.run experiment.yaml

    # Run all queued experiments
    python -m autoresearch.run --queue

    # Dry run (show what would execute)
    python -m autoresearch.run experiment.yaml --dry-run

    # Run the full queue in dry-run mode
    python -m autoresearch.run --queue --dry-run

    # Analyze a completed experiment
    python -m autoresearch.run --analyze D:/drone2_training/autoresearch/<exp_id>

    # Compare experiments
    python -m autoresearch.run --compare <exp_id_1> <exp_id_2> ...

    # Show experiment history
    python -m autoresearch.run --history
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autoresearch.experiment import ExperimentConfig
from autoresearch.runner import run_experiment, run_queue, AUTORESEARCH_BASE


def cmd_run_single(args):
    """Run a single experiment from YAML config."""
    config = ExperimentConfig.load(args.config)
    run_experiment(config, dry_run=args.dry_run)


def cmd_run_queue(args):
    """Run all pending experiments from the queue."""
    run_queue(dry_run=args.dry_run)


def cmd_analyze(args):
    """Re-analyze a completed experiment."""
    exp_dir = args.path
    if not os.path.isabs(exp_dir):
        exp_dir = os.path.join(AUTORESEARCH_BASE, exp_dir)

    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"No config.json found in {exp_dir}")
        return

    config = ExperimentConfig.load(config_path)
    csv_path = os.path.join(exp_dir, "training_stats.csv")

    if not os.path.exists(csv_path):
        print(f"No training_stats.csv found in {exp_dir}")
        return

    from autoresearch.analyzer import load_csv, compute_metrics, generate_plots, generate_report
    data, header = load_csv(csv_path)
    metrics = compute_metrics(data)
    analysis_dir = os.path.join(exp_dir, "analysis")
    plots = generate_plots(data, analysis_dir)

    with open(os.path.join(analysis_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Load eval results if they exist
    eval_path = os.path.join(exp_dir, "eval_results.json")
    eval_results = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)

    report_path = os.path.join(exp_dir, "report.md")
    generate_report(config, metrics, eval_results, plots, report_path)

    print(f"Analysis complete. Report: {report_path}")
    print(f"\nKey metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def cmd_compare(args):
    """Compare multiple experiments side by side."""
    exp_ids = args.experiments

    print(f"\n{'Metric':<30}", end="")
    for eid in exp_ids:
        short = eid[:25]
        print(f" {short:>25}", end="")
    print()
    print("-" * (30 + 26 * len(exp_ids)))

    all_metrics = {}
    for eid in exp_ids:
        exp_dir = os.path.join(AUTORESEARCH_BASE, eid)
        metrics_path = os.path.join(exp_dir, "analysis", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                all_metrics[eid] = json.load(f)
        else:
            # Try to compute on the fly
            csv_path = os.path.join(exp_dir, "training_stats.csv")
            if os.path.exists(csv_path):
                data, _ = load_csv(csv_path)
                all_metrics[eid] = compute_metrics(data)
            else:
                all_metrics[eid] = {"error": "no data"}

    # Collect all metric keys
    all_keys = set()
    for m in all_metrics.values():
        all_keys.update(m.keys())

    for key in sorted(all_keys):
        if key in ('error',):
            continue
        print(f"{key:<30}", end="")
        for eid in exp_ids:
            val = all_metrics.get(eid, {}).get(key, "N/A")
            if isinstance(val, float):
                print(f" {val:>25.4f}", end="")
            else:
                print(f" {str(val):>25}", end="")
        print()


def cmd_history(args):
    """Show experiment history."""
    if not os.path.exists(AUTORESEARCH_BASE):
        print("No experiments found.")
        return

    experiments = []
    for entry in sorted(os.listdir(AUTORESEARCH_BASE)):
        config_path = os.path.join(AUTORESEARCH_BASE, entry, "config.json")
        if os.path.exists(config_path):
            try:
                config = ExperimentConfig.load(config_path)
                experiments.append(config)
            except Exception:
                pass

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'ID':<40} {'Model':<8} {'Steps':>10} {'Status':<12}")
    print("-" * 75)
    for exp in experiments:
        steps_m = f"{exp.total_timesteps/1e6:.0f}M"
        print(f"{(exp.experiment_id or 'unknown'):<40} {exp.model_type:<8} {steps_m:>10} {exp.status:<12}")

    print(f"\nTotal: {len(experiments)} experiments")
    print(f"Location: {AUTORESEARCH_BASE}")


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch — automated experiment runner for MonoRace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Run single experiment
    run_parser = subparsers.add_parser("run", help="Run a single experiment from YAML")
    run_parser.add_argument("config", help="Path to experiment YAML config")
    run_parser.add_argument("--dry-run", action="store_true", help="Show what would execute")

    # Run queue
    queue_parser = subparsers.add_parser("queue", help="Run all queued experiments")
    queue_parser.add_argument("--dry-run", action="store_true", help="Show what would execute")

    # Analyze
    analyze_parser = subparsers.add_parser("analyze", help="Re-analyze a completed experiment")
    analyze_parser.add_argument("path", help="Experiment directory or ID")

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiments", nargs="+", help="Experiment IDs to compare")

    # History
    subparsers.add_parser("history", help="Show experiment history")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run_single(args)
    elif args.command == "queue":
        cmd_run_queue(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "history":
        cmd_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
