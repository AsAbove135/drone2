"""
Cost tracking and budget enforcement for cloud experiments.
Maintains a ledger of all spending and enforces caps.
"""
import os
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List

COST_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cost_ledger.json"
)


@dataclass
class CostEntry:
    experiment_id: str
    instance_id: str
    gpu_type: str
    cost_per_hour: float
    started_at: str
    ended_at: Optional[str] = None
    hours_used: float = 0.0
    compute_cost: float = 0.0
    status: str = "running"  # running | completed | failed | killed_by_cap


@dataclass
class Budget:
    """Spending caps. Set any to null/0 to disable that cap."""
    hourly_max: float = 2.0          # Max $/hr across all instances
    daily_cap: float = 20.0          # Max $/day total spend
    weekly_cap: float = 100.0        # Max $/week total spend
    monthly_cap: float = 300.0       # Max $/month total spend
    per_experiment_cap: float = 10.0 # Max $ per single experiment
    total_budget: float = 500.0      # Lifetime cap — hard stop
    max_concurrent_instances: int = 3 # Max simultaneous GPU instances
    preferred_gpu: str = "RTX_4090"  # GPU type to search for
    max_gpu_price: float = 0.50      # Max $/hr per GPU to accept


class CostTracker:
    """Tracks spending and enforces budget caps."""

    def __init__(self, budget: Optional[Budget] = None, ledger_path: str = COST_LEDGER_PATH):
        self.ledger_path = ledger_path
        self.budget = budget or Budget()
        self.entries: List[CostEntry] = []
        self._load()

    def _load(self):
        """Load ledger from disk."""
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path) as f:
                data = json.load(f)
            self.entries = [CostEntry(**e) for e in data.get("entries", [])]
            if "budget" in data:
                self.budget = Budget(**data["budget"])

    def _save(self):
        """Persist ledger to disk."""
        data = {
            "budget": asdict(self.budget),
            "entries": [asdict(e) for e in self.entries],
            "last_updated": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self.ledger_path) or '.', exist_ok=True)
        with open(self.ledger_path, 'w') as f:
            json.dump(data, f, indent=2)

    def start_experiment(self, experiment_id: str, instance_id: str,
                         gpu_type: str, cost_per_hour: float) -> CostEntry:
        """Record the start of a billable experiment."""
        entry = CostEntry(
            experiment_id=experiment_id,
            instance_id=instance_id,
            gpu_type=gpu_type,
            cost_per_hour=cost_per_hour,
            started_at=datetime.now().isoformat(),
        )
        self.entries.append(entry)
        self._save()
        return entry

    def end_experiment(self, experiment_id: str, status: str = "completed"):
        """Record the end of a billable experiment."""
        for entry in reversed(self.entries):
            if entry.experiment_id == experiment_id and entry.status == "running":
                entry.ended_at = datetime.now().isoformat()
                entry.status = status
                start = datetime.fromisoformat(entry.started_at)
                end = datetime.fromisoformat(entry.ended_at)
                entry.hours_used = (end - start).total_seconds() / 3600
                entry.compute_cost = entry.hours_used * entry.cost_per_hour
                self._save()
                return entry
        return None

    def update_running_costs(self):
        """Update cost estimates for currently running experiments."""
        now = datetime.now()
        for entry in self.entries:
            if entry.status == "running":
                start = datetime.fromisoformat(entry.started_at)
                entry.hours_used = (now - start).total_seconds() / 3600
                entry.compute_cost = entry.hours_used * entry.cost_per_hour
        self._save()

    # ── Budget checks ──

    def get_current_hourly_rate(self) -> float:
        """Sum of $/hr for all running instances."""
        return sum(e.cost_per_hour for e in self.entries if e.status == "running")

    def get_running_count(self) -> int:
        """Number of currently running instances."""
        return sum(1 for e in self.entries if e.status == "running")

    def get_spend_in_period(self, hours: float) -> float:
        """Total spend in the last N hours."""
        self.update_running_costs()
        cutoff = datetime.now() - timedelta(hours=hours)
        total = 0.0
        for entry in self.entries:
            start = datetime.fromisoformat(entry.started_at)
            if start >= cutoff or entry.status == "running":
                total += entry.compute_cost
        return total

    def get_total_spend(self) -> float:
        """Lifetime total spend."""
        self.update_running_costs()
        return sum(e.compute_cost for e in self.entries)

    def get_experiment_cost(self, experiment_id: str) -> float:
        """Cost of a specific experiment."""
        return sum(
            e.compute_cost for e in self.entries
            if e.experiment_id == experiment_id
        )

    def check_budget(self) -> dict:
        """Check all budget caps. Returns dict of {cap_name: (ok, current, limit)}."""
        self.update_running_costs()
        checks = {}

        # Hourly rate
        rate = self.get_current_hourly_rate()
        checks["hourly_rate"] = (
            rate <= self.budget.hourly_max,
            rate, self.budget.hourly_max
        )

        # Daily
        daily = self.get_spend_in_period(24)
        checks["daily"] = (
            daily <= self.budget.daily_cap,
            daily, self.budget.daily_cap
        )

        # Weekly
        weekly = self.get_spend_in_period(24 * 7)
        checks["weekly"] = (
            weekly <= self.budget.weekly_cap,
            weekly, self.budget.weekly_cap
        )

        # Monthly
        monthly = self.get_spend_in_period(24 * 30)
        checks["monthly"] = (
            monthly <= self.budget.monthly_cap,
            monthly, self.budget.monthly_cap
        )

        # Total
        total = self.get_total_spend()
        checks["total"] = (
            total <= self.budget.total_budget,
            total, self.budget.total_budget
        )

        # Concurrent instances
        running = self.get_running_count()
        checks["concurrent"] = (
            running <= self.budget.max_concurrent_instances,
            running, self.budget.max_concurrent_instances
        )

        return checks

    def can_launch(self, cost_per_hour: float) -> tuple:
        """Check if we can launch a new instance at the given rate.
        Returns (allowed: bool, reason: str)."""
        checks = self.check_budget()

        for cap_name, (ok, current, limit) in checks.items():
            if not ok:
                return False, f"Budget cap '{cap_name}' exceeded: ${current:.2f} / ${limit:.2f}"

        # Check if adding this instance would exceed hourly cap
        new_rate = self.get_current_hourly_rate() + cost_per_hour
        if new_rate > self.budget.hourly_max:
            return False, f"Would exceed hourly rate cap: ${new_rate:.2f}/hr > ${self.budget.hourly_max:.2f}/hr"

        # Check GPU price
        if cost_per_hour > self.budget.max_gpu_price:
            return False, f"GPU price ${cost_per_hour:.2f}/hr exceeds max ${self.budget.max_gpu_price:.2f}/hr"

        return True, "OK"

    def print_summary(self):
        """Print a spending summary."""
        self.update_running_costs()
        checks = self.check_budget()

        print("\n" + "=" * 60)
        print("COST TRACKER SUMMARY")
        print("=" * 60)

        print(f"\n  Running instances: {self.get_running_count()}")
        print(f"  Current burn rate: ${self.get_current_hourly_rate():.2f}/hr")

        print(f"\n  {'Cap':<20} {'Spent':>10} {'Limit':>10} {'Status':>10}")
        print(f"  {'-'*50}")
        for name, (ok, current, limit) in checks.items():
            status = "OK" if ok else "EXCEEDED"
            if name == "concurrent":
                print(f"  {name:<20} {int(current):>10} {int(limit):>10} {status:>10}")
            else:
                print(f"  {name:<20} ${current:>9.2f} ${limit:>9.2f} {status:>10}")

        total = self.get_total_spend()
        remaining = self.budget.total_budget - total
        print(f"\n  Total spent:     ${total:.2f}")
        print(f"  Budget remaining: ${remaining:.2f}")

        # Recent experiments
        recent = [e for e in self.entries[-10:]]
        if recent:
            print(f"\n  Recent experiments:")
            for e in recent:
                print(f"    {e.experiment_id[:40]:<40} "
                      f"${e.compute_cost:>6.2f} "
                      f"({e.hours_used:.1f}h on {e.gpu_type}) "
                      f"[{e.status}]")

        print("=" * 60)
