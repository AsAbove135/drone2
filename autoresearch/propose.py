"""
Experiment proposal helper.
Reads past experiment results and prints a summary for the researcher agent
to use when proposing the next experiment.

Usage:
    python -m autoresearch.propose
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autoresearch.experiment import ExperimentConfig
from autoresearch.runner import AUTORESEARCH_BASE


def gather_history():
    """Collect all completed experiment summaries."""
    if not os.path.exists(AUTORESEARCH_BASE):
        return []

    experiments = []
    for entry in sorted(os.listdir(AUTORESEARCH_BASE)):
        exp_dir = os.path.join(AUTORESEARCH_BASE, entry)
        config_path = os.path.join(exp_dir, "config.json")
        metrics_path = os.path.join(exp_dir, "analysis", "metrics.json")

        if not os.path.exists(config_path):
            continue

        try:
            config = ExperimentConfig.load(config_path)
        except Exception:
            continue

        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

        eval_results = {}
        eval_path = os.path.join(exp_dir, "eval_results.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                eval_results = json.load(f)

        experiments.append({
            "id": config.experiment_id,
            "name": config.name,
            "model_type": config.model_type,
            "total_timesteps": config.total_timesteps,
            "hypothesis": config.hypothesis,
            "status": config.status,
            "metrics": metrics,
            "eval": eval_results,
        })

    return experiments


def print_summary():
    """Print experiment history as a research summary."""
    experiments = gather_history()

    if not experiments:
        print("No completed experiments yet.")
        print("\nTo get started, create experiment YAML files in autoresearch/experiments/")
        print("Use TEMPLATE.yaml as a starting point.")
        return

    print("=" * 70)
    print("AUTORESEARCH EXPERIMENT HISTORY")
    print("=" * 70)

    for exp in experiments:
        print(f"\n## {exp['name']} ({exp['id']})")
        print(f"   Model: {exp['model_type']} | Steps: {exp['total_timesteps']:,}")
        print(f"   Status: {exp['status']}")
        print(f"   Hypothesis: {exp['hypothesis'][:100]}...")

        if exp['metrics']:
            m = exp['metrics']
            print(f"   Results:")
            print(f"     Reward (smooth): {m.get('reward_final_smooth', 'N/A')}")
            print(f"     Avg gates: {m.get('avg_gates', 'N/A')}")
            print(f"     Pass rate: {m.get('pass_rate_final_smooth', 'N/A')}")
            print(f"     Entropy: {m.get('entropy_final', 'N/A')} ({m.get('entropy_trend', '?')})")

        if exp['eval']:
            for track, result in exp['eval'].items():
                if isinstance(result, dict):
                    print(f"   Eval ({track}): {result}")

    print(f"\n{'='*70}")
    print(f"Total: {len(experiments)} experiments")
    print(f"\nBased on these results, consider what to test next.")
    print(f"Key questions:")
    print(f"  - Which hyperparameter had the biggest effect?")
    print(f"  - Are there bottleneck gate segments?")
    print(f"  - Is the model capacity sufficient?")
    print(f"  - Would longer training help or has it plateaued?")


if __name__ == "__main__":
    print_summary()
