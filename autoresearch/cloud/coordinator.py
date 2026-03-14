"""
Coordinator: manages the vast.ai fleet from your local machine.
Searches for cheap GPUs, launches workers, monitors costs, enforces caps.

Usage:
    # Launch workers to process the queue
    python -m autoresearch.cloud.coordinator launch

    # Check status of running workers
    python -m autoresearch.cloud.coordinator status

    # Stop all workers
    python -m autoresearch.cloud.coordinator stop

    # Estimate costs for queued experiments
    python -m autoresearch.cloud.coordinator estimate

    # Show spending summary
    python -m autoresearch.cloud.coordinator costs
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from autoresearch.cloud.costs import CostTracker, Budget
from autoresearch.cloud import vastai
from autoresearch.experiment import ExperimentConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default Docker image — PyTorch + CUDA, you build on top
DOCKER_IMAGE = os.environ.get(
    "AUTORESEARCH_DOCKER_IMAGE",
    "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
)

# Budget config file
BUDGET_PATH = os.path.join(PROJECT_ROOT, "autoresearch", "budget.json")


def load_budget() -> Budget:
    """Load budget from config file or use defaults."""
    if os.path.exists(BUDGET_PATH):
        with open(BUDGET_PATH) as f:
            data = json.load(f)
        return Budget(**data)
    return Budget()


def save_budget(budget: Budget):
    """Save budget config."""
    from dataclasses import asdict
    with open(BUDGET_PATH, 'w') as f:
        json.dump(asdict(budget), f, indent=2)


def get_repo_url() -> str:
    """Get the git remote URL."""
    import subprocess
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    url = result.stdout.strip()
    # Convert SSH to HTTPS if needed for vast.ai workers
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")
    return url


def build_onstart_cmd(repo_url: str, budget: Budget, gpu_cost: float) -> str:
    """Build the startup command for a vast.ai instance."""
    github_token = os.environ.get("GITHUB_TOKEN", "")
    token_env = f"GITHUB_TOKEN={github_token} " if github_token else ""

    return (
        f"cd /workspace && "
        f"apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && "
        f"git clone {repo_url} drone2 && "
        f"cd drone2 && "
        f"pip install -r requirements.txt -q && "
        f"pip install mamba-ssm causal-conv1d -q --no-build-isolation 2>/dev/null; "
        f"REPO_URL={repo_url} "
        f"GPU_COST_PER_HOUR={gpu_cost} "
        f"MAX_COST_PER_WORKER={budget.per_experiment_cap * 3} "
        f"USE_FEATURE_BRANCHES=true "
        f"{token_env}"
        f"python -m autoresearch.cloud.worker"
    )


def cmd_launch(args):
    """Launch worker instances on vast.ai."""
    budget = load_budget()
    tracker = CostTracker(budget)

    # How many workers to launch?
    n_workers = args.workers
    running = tracker.get_running_count()
    available_slots = budget.max_concurrent_instances - running

    if available_slots <= 0:
        print(f"Already at max concurrent instances ({running}/{budget.max_concurrent_instances})")
        return

    n_workers = min(n_workers, available_slots)
    print(f"Launching {n_workers} worker(s) (currently {running} running)")

    # Check budget
    can_launch, reason = tracker.can_launch(budget.max_gpu_price)
    if not can_launch:
        print(f"Cannot launch: {reason}")
        return

    # Count queued experiments
    queue_dir = os.path.join(PROJECT_ROOT, "autoresearch", "queue")
    queued = count_queued_experiments(queue_dir)  # Walks entire tier hierarchy
    if queued == 0:
        print("No experiments in queue. Add experiments to autoresearch/queue/ first.")
        return
    print(f"Found {queued} experiments in queue")

    # Search for cheap GPUs
    gpu_name = args.gpu or budget.preferred_gpu
    max_price = args.max_price or budget.max_gpu_price
    print(f"\nSearching for {gpu_name} instances under ${max_price}/hr...")

    offers = vastai.search_offers(
        gpu_name=gpu_name,
        max_price=max_price,
        num_results=n_workers * 2  # Get extras in case some fail
    )

    if not offers:
        print(f"No offers found for {gpu_name} under ${max_price}/hr")
        # Try fallback GPUs
        for fallback in ["RTX_3090", "A10", "RTX_4080"]:
            if fallback != gpu_name:
                offers = vastai.search_offers(gpu_name=fallback, max_price=max_price, num_results=3)
                if offers:
                    gpu_name = fallback
                    print(f"Found {fallback} offers instead")
                    break
        if not offers:
            print("No GPU offers found at all. Try increasing --max-price.")
            return

    repo_url = get_repo_url()

    launched = 0
    for offer in offers[:n_workers]:
        offer_id = offer.get("id")
        dph = offer.get("dph_total", offer.get("dph", 0))
        gpu = offer.get("gpu_name", gpu_name)

        # Final budget check
        can_launch, reason = tracker.can_launch(dph)
        if not can_launch:
            print(f"Skipping offer {offer_id}: {reason}")
            continue

        print(f"\n  Launching on {gpu} @ ${dph:.3f}/hr (offer {offer_id})...")

        onstart = build_onstart_cmd(repo_url, budget, dph)
        result = vastai.create_instance(
            offer_id=offer_id,
            docker_image=DOCKER_IMAGE,
            disk_gb=40,
            onstart_cmd=onstart,
            label=f"autoresearch_{datetime.now().strftime('%H%M')}",
        )

        if result:
            instance_id = result.get("new_contract") if isinstance(result, dict) else str(result)
            print(f"  Launched instance {instance_id}")
            tracker.start_experiment(
                experiment_id=f"fleet_{instance_id}",
                instance_id=str(instance_id),
                gpu_type=gpu,
                cost_per_hour=dph,
            )
            launched += 1
        else:
            print(f"  Failed to launch on offer {offer_id}")

    print(f"\nLaunched {launched} worker(s)")
    tracker.print_summary()


def cmd_status(args):
    """Show status of running workers and experiments."""
    budget = load_budget()
    tracker = CostTracker(budget)

    # Get vast.ai instances
    instances = vastai.get_instances()

    print(f"\n{'='*60}")
    print("FLEET STATUS")
    print(f"{'='*60}")

    if instances:
        print(f"\n  {'ID':<12} {'GPU':<15} {'Status':<12} {'$/hr':>8} {'Cost':>8}")
        print(f"  {'-'*55}")
        for inst in instances:
            print(f"  {inst.get('id', '?'):<12} "
                  f"{inst.get('gpu_name', '?'):<15} "
                  f"{inst.get('actual_status', inst.get('status', '?')):<12} "
                  f"${inst.get('dph_total', 0):>7.3f} "
                  f"${inst.get('total_cost', 0):>7.2f}")
    else:
        print("\n  No active instances")

    tracker.print_summary()


def cmd_stop(args):
    """Stop all running workers."""
    instances = vastai.get_instances()

    if not instances:
        print("No running instances")
        return

    budget = load_budget()
    tracker = CostTracker(budget)

    print(f"Stopping {len(instances)} instance(s)...")
    for inst in instances:
        inst_id = inst.get("id")
        if inst_id:
            print(f"  Destroying instance {inst_id}...")
            vastai.destroy_instance(inst_id)
            tracker.end_experiment(f"fleet_{inst_id}", status="killed_by_user")

    print("All instances stopped.")
    tracker.print_summary()


def cmd_estimate(args):
    """Estimate costs for all queued experiments."""
    budget = load_budget()
    queue_dir = os.path.join(PROJECT_ROOT, "autoresearch", "queue")

    if not os.path.exists(queue_dir):
        print("No queue directory found")
        return

    total_cost = 0.0
    total_hours = 0.0

    print(f"\n{'Experiment':<35} {'Model':<8} {'Steps':>10} {'Est Hours':>10} {'Est Cost':>10}")
    print("-" * 78)

    for root, dirs, files in sorted(os.walk(queue_dir)):
        for f in sorted(files):
            if not f.endswith('.json') or f == 'TEMPLATE.json':
                continue

            path = os.path.join(root, f)
            try:
                config = ExperimentConfig.load(path)
            except Exception:
                continue

            if config.status != "pending":
                continue

            est = vastai.estimate_experiment_cost(
                config.model_type,
                config.total_timesteps,
                gpu_name=budget.preferred_gpu,
                dph=budget.max_gpu_price,
            )

            tier = os.path.relpath(root, queue_dir)
            name = f"{tier}/{f}" if tier != "." else f
            steps_m = f"{config.total_timesteps/1e6:.0f}M"

            print(f"  {name:<33} {config.model_type:<8} {steps_m:>10} "
                  f"{est['estimated_hours']:>9.1f}h ${est['estimated_cost']:>9.2f}")

            total_cost += est['estimated_cost']
            total_hours += est['estimated_hours']

    print("-" * 78)
    print(f"  {'TOTAL':<33} {'':8} {'':>10} {total_hours:>9.1f}h ${total_cost:>9.2f}")

    if budget.max_concurrent_instances > 1:
        parallel_hours = total_hours / budget.max_concurrent_instances
        print(f"\n  With {budget.max_concurrent_instances} parallel workers: ~{parallel_hours:.1f}h wall time")

    print(f"\n  Budget remaining: ${budget.total_budget - CostTracker(budget).get_total_spend():.2f}")


def cmd_costs(args):
    """Show spending summary."""
    budget = load_budget()
    tracker = CostTracker(budget)
    tracker.print_summary()


def count_queued_experiments(queue_dir: str) -> int:
    """Count pending experiments in queue."""
    count = 0
    if not os.path.exists(queue_dir):
        return 0
    for root, dirs, files in os.walk(queue_dir):
        for f in files:
            if not f.endswith('.json') or f == 'TEMPLATE.json':
                continue
            path = os.path.join(root, f)
            lock_path = path + ".lock"
            if os.path.exists(lock_path):
                continue
            try:
                with open(path) as fp:
                    config = json.load(fp)
                if config.get("status", "pending") == "pending":
                    count += 1
            except Exception:
                pass
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch cloud coordinator — manage vast.ai GPU fleet"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Launch
    launch_p = subparsers.add_parser("launch", help="Launch GPU workers")
    launch_p.add_argument("--workers", type=int, default=1, help="Number of workers to launch")
    launch_p.add_argument("--gpu", type=str, default=None, help="GPU type (e.g. RTX_4090)")
    launch_p.add_argument("--max-price", type=float, default=None, help="Max $/hr per GPU")

    # Status
    subparsers.add_parser("status", help="Show fleet status")

    # Stop
    subparsers.add_parser("stop", help="Stop all workers")

    # Estimate
    subparsers.add_parser("estimate", help="Estimate costs for queued experiments")

    # Costs
    subparsers.add_parser("costs", help="Show spending summary")

    args = parser.parse_args()

    if args.command == "launch":
        cmd_launch(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "costs":
        cmd_costs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
