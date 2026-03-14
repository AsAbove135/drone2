"""
Worker script — runs on vast.ai GPU instances.
Pulls experiment configs from the git repo, runs them, pushes results back.

Lifecycle:
  1. Clone repo (or pull latest)
  2. Check autoresearch/queue/ for unclaimed experiments
  3. Claim one by writing a .lock file and pushing
  4. Run training + evaluation + analysis
  5. Push results (config, metrics, report, checkpoints) to results branch
  6. Repeat until queue is empty or budget exceeded
  7. Self-terminate
"""
import os
import sys
import json
import subprocess
import time
import socket
from datetime import datetime
from pathlib import Path

# Worker identity
WORKER_ID = os.environ.get("WORKER_ID", f"worker_{socket.gethostname()}")
REPO_URL = os.environ.get("REPO_URL", "")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")
RESULTS_BRANCH = os.environ.get("RESULTS_BRANCH", "autoresearch-results")
WORK_DIR = os.environ.get("WORK_DIR", "/workspace/drone2")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/workspace/results")
MAX_COST = float(os.environ.get("MAX_COST_PER_WORKER", "10.0"))
GPU_COST_PER_HOUR = float(os.environ.get("GPU_COST_PER_HOUR", "0.35"))
QUEUE_DIR = os.environ.get("QUEUE_DIR", "autoresearch/queue")
# If True, workers create feature branches + PRs for any code changes
USE_FEATURE_BRANCHES = os.environ.get("USE_FEATURE_BRANCHES", "true").lower() == "true"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def run(cmd, cwd=None, timeout=None, check=True):
    """Run a shell command."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd or WORK_DIR, timeout=timeout
    )
    if check and result.returncode != 0:
        print(f"  CMD FAILED: {cmd}")
        print(f"  STDERR: {result.stderr[:500]}")
    return result


def setup_repo():
    """Clone or update the repo."""
    if os.path.exists(os.path.join(WORK_DIR, ".git")):
        print(f"Pulling latest from {REPO_BRANCH}...")
        run(f"git fetch origin && git checkout {REPO_BRANCH} && git pull origin {REPO_BRANCH}")
    else:
        print(f"Cloning repo...")
        os.makedirs(os.path.dirname(WORK_DIR), exist_ok=True)
        run(f"git clone {REPO_URL} {WORK_DIR}", cwd="/workspace")

    # Set up results branch
    result = run(f"git branch -a", check=False)
    if RESULTS_BRANCH not in result.stdout:
        run(f"git checkout -b {RESULTS_BRANCH}")
        run(f"git push -u origin {RESULTS_BRANCH}", check=False)
    run(f"git checkout {REPO_BRANCH}")


def setup_results_repo():
    """Set up a separate checkout for pushing results (avoids conflicts)."""
    if not os.path.exists(RESULTS_DIR):
        run(f"git clone {REPO_URL} {RESULTS_DIR} -b {RESULTS_BRANCH}", cwd="/workspace", check=False)
        if not os.path.exists(RESULTS_DIR):
            # Branch doesn't exist yet, create from main
            run(f"git clone {REPO_URL} {RESULTS_DIR}", cwd="/workspace")
            run(f"git checkout -b {RESULTS_BRANCH}", cwd=RESULTS_DIR)


def create_experiment_branch(exp_name: str) -> str:
    """Create a feature branch for this experiment's code changes.
    Results and code changes go here, NOT main."""
    branch_name = f"autoresearch/{exp_name}"
    # Create branch from main
    run(f"git checkout {REPO_BRANCH}")
    run(f"git pull origin {REPO_BRANCH}", check=False)
    run(f"git checkout -b {branch_name}", check=False)
    return branch_name


def commit_and_push_branch(branch_name: str, exp_name: str, message: str):
    """Commit any changes on the experiment branch and push."""
    result = run("git status --porcelain", check=False)
    if not result.stdout.strip():
        return  # Nothing to commit

    run("git add -A")
    run(f'git commit -m "autoresearch: {message}"', check=False)
    run(f"git push -u origin {branch_name}", check=False)


def create_pull_request(branch_name: str, exp_name: str, results: dict):
    """Create a GitHub PR from the experiment branch to main.
    Requires GITHUB_TOKEN or gh CLI auth."""

    cost_line = (f"${results.get('compute_cost', 0):.2f} actual "
                 f"/ ${results.get('estimated_cost', 0):.2f} estimated")
    time_line = (f"{results.get('elapsed_hours', 0):.1f}h actual "
                 f"/ {results.get('estimated_hours', 0):.1f}h estimated")

    title = f"[autoresearch] {exp_name}"
    body = (
        f"## Experiment: {exp_name}\n\n"
        f"**Worker**: {WORKER_ID}\n"
        f"**Cost**: {cost_line}\n"
        f"**Time**: {time_line}\n"
        f"**Exit code**: {results.get('exit_code', '?')}\n\n"
        f"### Results\n"
        f"See experiment report and analysis in this branch.\n\n"
        f"### Changes\n"
        f"Review code changes (if any) before merging.\n"
        f"Training results and reports are included.\n\n"
        f"---\n"
        f"*Automated by autoresearch worker {WORKER_ID}*"
    )

    # Try gh CLI first (works if authenticated)
    pr_result = run(
        f'gh pr create --base {REPO_BRANCH} --head {branch_name} '
        f'--title "{title}" --body "{body}"',
        check=False,
    )

    if pr_result.returncode == 0:
        print(f"  PR created: {pr_result.stdout.strip()}")
        return pr_result.stdout.strip()

    # Fallback: use GitHub API directly
    if GITHUB_TOKEN:
        import urllib.request
        import re

        # Extract owner/repo from URL
        match = re.search(r'github\.com[:/](.+?)(?:\.git)?$', REPO_URL)
        if match:
            repo_path = match.group(1)
            api_url = f"https://api.github.com/repos/{repo_path}/pulls"

            data = json.dumps({
                "title": title,
                "body": body,
                "head": branch_name,
                "base": REPO_BRANCH,
            }).encode()

            req = urllib.request.Request(
                api_url, data=data, method="POST",
                headers={
                    "Authorization": f"token {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github.v3+json",
                    "Content-Type": "application/json",
                }
            )
            try:
                resp = urllib.request.urlopen(req)
                pr_data = json.loads(resp.read())
                pr_url = pr_data.get("html_url", "")
                print(f"  PR created: {pr_url}")
                return pr_url
            except Exception as e:
                print(f"  PR creation via API failed: {e}")

    print(f"  Could not create PR (no gh CLI or GITHUB_TOKEN). Branch pushed: {branch_name}")
    return None


def find_unclaimed_experiment() -> str | None:
    """Find the first unclaimed experiment in the queue."""
    queue_path = os.path.join(WORK_DIR, QUEUE_DIR)
    if not os.path.exists(queue_path):
        return None

    # Walk the queue hierarchy
    for root, dirs, files in sorted(os.walk(queue_path)):
        for f in sorted(files):
            if not f.endswith('.json') or f == 'TEMPLATE.json':
                continue

            config_path = os.path.join(root, f)
            lock_path = config_path + ".lock"

            # Skip if locked
            if os.path.exists(lock_path):
                continue

            # Check if status is pending
            try:
                with open(config_path) as fp:
                    config = json.load(fp)
                if config.get("status", "pending") == "pending":
                    return config_path
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def claim_experiment(config_path: str) -> bool:
    """Claim an experiment by creating a lock file and pushing."""
    lock_path = config_path + ".lock"
    lock_data = {
        "worker_id": WORKER_ID,
        "claimed_at": datetime.now().isoformat(),
        "gpu_cost_per_hour": GPU_COST_PER_HOUR,
    }

    # Write lock
    with open(lock_path, 'w') as f:
        json.dump(lock_data, f, indent=2)

    # Try to push the lock (first one wins)
    rel_lock = os.path.relpath(lock_path, WORK_DIR)
    run(f'git add "{rel_lock}"')
    result = run(
        f'git commit -m "claim: {os.path.basename(config_path)} by {WORKER_ID}"',
        check=False
    )
    result = run(f"git pull --rebase origin {REPO_BRANCH} && git push origin {REPO_BRANCH}", check=False)

    if result.returncode != 0:
        # Someone else claimed it first
        os.remove(lock_path)
        run(f"git checkout -- .", check=False)
        return False

    return True


def run_experiment(config_path: str) -> dict:
    """Run a single experiment and return results."""
    with open(config_path) as f:
        config = json.load(f)

    exp_name = config.get("name", "unknown")
    model_type = config.get("model_type", "fsppo")
    timesteps = config.get("total_timesteps", 100_000_000)

    print(f"\n{'='*60}")
    print(f"RUNNING: {exp_name}")
    print(f"Model: {model_type} | Steps: {timesteps:,}")
    print(f"Worker: {WORKER_ID} | GPU cost: ${GPU_COST_PER_HOUR}/hr")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Compute cost estimate before running
    from autoresearch.cloud.vastai import estimate_experiment_cost
    est = estimate_experiment_cost(model_type, timesteps, dph=GPU_COST_PER_HOUR)

    print(f"  Estimated: {est['estimated_hours']:.1f}h / ${est['estimated_cost']:.2f}")

    # Use autoresearch runner — pass GPU cost via env so runner tracks it
    save_dir = f"/workspace/experiment_output/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Detect GPU type
    gpu_type = os.environ.get("GPU_TYPE", "")
    if not gpu_type:
        try:
            import subprocess as sp
            nv = sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=10)
            gpu_type = nv.stdout.strip().split('\n')[0].replace(" ", "_") if nv.returncode == 0 else "unknown"
        except Exception:
            gpu_type = "unknown"

    env_cmd = (
        f"GPU_COST_PER_HOUR={GPU_COST_PER_HOUR} "
        f"GPU_TYPE={gpu_type} "
    )

    result = run(
        f"{env_cmd} {sys.executable} -m autoresearch run \"{config_path}\"",
        timeout=72 * 3600,  # 72 hour max per experiment
        check=False,
    )

    elapsed_hours = (time.time() - start_time) / 3600
    compute_cost = elapsed_hours * GPU_COST_PER_HOUR

    # Collect results
    results = {
        "experiment": exp_name,
        "worker_id": WORKER_ID,
        "gpu_type": gpu_type,
        "elapsed_hours": round(elapsed_hours, 4),
        "compute_cost": round(compute_cost, 4),
        "gpu_cost_per_hour": GPU_COST_PER_HOUR,
        "estimated_hours": est["estimated_hours"],
        "estimated_cost": est["estimated_cost"],
        "cost_accuracy_pct": round(
            (1 - abs(compute_cost - est["estimated_cost"]) / max(est["estimated_cost"], 0.001)) * 100, 1
        ),
        "exit_code": result.returncode,
        "completed_at": datetime.now().isoformat(),
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
    }

    # Save worker results
    results_path = os.path.join(save_dir, "worker_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def push_results(exp_name: str, results: dict):
    """Push experiment results to the results branch."""
    # Copy results to results repo
    src = f"/workspace/experiment_output/{exp_name}"
    if not os.path.exists(src):
        print(f"No results to push for {exp_name}")
        return

    dst = os.path.join(RESULTS_DIR, "autoresearch", "completed", exp_name)
    run(f'cp -r "{src}" "{dst}"', cwd="/workspace", check=False)

    # Git add + commit + push on results branch
    run("git add -A", cwd=RESULTS_DIR)
    cost = results.get("compute_cost", 0)
    hours = results.get("elapsed_hours", 0)
    run(
        f'git commit -m "results: {exp_name} (${cost:.2f}, {hours:.1f}h) by {WORKER_ID}"',
        cwd=RESULTS_DIR, check=False
    )
    run(f"git push origin {RESULTS_BRANCH}", cwd=RESULTS_DIR, check=False)
    print(f"Results pushed to {RESULTS_BRANCH}")


def get_total_cost() -> float:
    """Calculate total cost spent by this worker session."""
    total = 0.0
    output_dir = "/workspace/experiment_output"
    if not os.path.exists(output_dir):
        return 0.0
    for exp_dir in os.listdir(output_dir):
        results_path = os.path.join(output_dir, exp_dir, "worker_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                r = json.load(f)
            total += r.get("compute_cost", 0)
    return total


def main():
    """Main worker loop."""
    print(f"{'='*60}")
    print(f"AUTORESEARCH WORKER: {WORKER_ID}")
    print(f"GPU cost: ${GPU_COST_PER_HOUR}/hr")
    print(f"Max budget: ${MAX_COST}")
    print(f"Repo: {REPO_URL}")
    print(f"{'='*60}\n")

    if not REPO_URL:
        print("ERROR: REPO_URL environment variable not set")
        sys.exit(1)

    # Setup
    setup_repo()
    setup_results_repo()

    # Install dependencies
    print("Installing dependencies...")
    run(f"{sys.executable} -m pip install -r requirements.txt -q", check=False)

    experiments_run = 0

    while True:
        # Check budget
        total_cost = get_total_cost()
        if total_cost >= MAX_COST:
            print(f"\nBudget exhausted: ${total_cost:.2f} >= ${MAX_COST:.2f}")
            break

        remaining = MAX_COST - total_cost
        print(f"\nBudget: ${total_cost:.2f} spent, ${remaining:.2f} remaining")

        # Pull latest queue
        run(f"git pull origin {REPO_BRANCH}", check=False)

        # Find and claim an experiment
        config_path = find_unclaimed_experiment()
        if not config_path:
            print("No unclaimed experiments in queue. Worker done.")
            break

        print(f"Found: {os.path.basename(config_path)}")

        if not claim_experiment(config_path):
            print("  Failed to claim (another worker got it). Retrying...")
            continue

        print(f"  Claimed! Starting experiment...")

        # Create feature branch for this experiment
        exp_name = json.load(open(config_path)).get("name", "unknown")
        branch_name = None
        if USE_FEATURE_BRANCHES:
            branch_name = create_experiment_branch(exp_name)
            print(f"  Branch: {branch_name}")

        # Run it
        results = run_experiment(config_path)
        experiments_run += 1

        print(f"\nExperiment complete:")
        print(f"  Cost: ${results['compute_cost']:.2f} actual / ${results['estimated_cost']:.2f} estimated")
        print(f"  Time: {results['elapsed_hours']:.1f}h actual / {results['estimated_hours']:.1f}h estimated")
        print(f"  Estimate accuracy: {results['cost_accuracy_pct']:.0f}%")
        print(f"  GPU: {results['gpu_type']} @ ${GPU_COST_PER_HOUR}/hr")
        print(f"  Exit: {results['exit_code']}")

        # Commit results to feature branch and create PR
        if branch_name:
            commit_and_push_branch(
                branch_name, exp_name,
                f"{exp_name} results (${results['compute_cost']:.2f}, {results['elapsed_hours']:.1f}h)"
            )
            create_pull_request(branch_name, exp_name, results)
            # Switch back to main for next experiment
            run(f"git checkout {REPO_BRANCH}", check=False)
        else:
            # Legacy: push to results branch
            push_results(results["experiment"], results)

    # Final summary
    total_cost = get_total_cost()
    print(f"\n{'='*60}")
    print(f"WORKER SESSION COMPLETE")
    print(f"  Experiments run: {experiments_run}")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Worker: {WORKER_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
