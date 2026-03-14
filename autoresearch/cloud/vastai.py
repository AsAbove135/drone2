"""
Vast.ai fleet management: search, launch, monitor, and destroy GPU instances.
Wraps the vastai CLI for programmatic control.
"""
import os
import sys
import json
import subprocess
import time
from typing import Optional, List


def _run_vastai(*args, parse_json=True):
    """Run a vastai CLI command and return output."""
    cmd = ["vastai"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {result.stderr}")
    if parse_json and result.stdout.strip():
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()
    return result.stdout.strip()


def search_offers(gpu_name: str = "RTX_4090", max_price: float = 0.50,
                  min_ram: int = 16, min_disk: int = 30,
                  num_results: int = 5) -> list:
    """Search for available GPU instances on vast.ai."""
    query = (
        f"gpu_name={gpu_name} "
        f"dph<={max_price} "
        f"gpu_ram>={min_ram} "
        f"disk_space>={min_disk} "
        f"inet_up>=100 "
        f"reliability>0.95 "
        f"num_gpus=1"
    )
    try:
        result = _run_vastai(
            "search", "offers", query,
            "--order", "dph",
            "--limit", str(num_results),
            "--raw",
        )
        if isinstance(result, list):
            return result
        return []
    except Exception as e:
        print(f"Error searching offers: {e}")
        return []


def create_instance(offer_id: int, docker_image: str, disk_gb: int = 30,
                    onstart_cmd: str = "", env_vars: dict = None,
                    label: str = "") -> Optional[dict]:
    """Create a vast.ai instance from an offer."""
    args = [
        "create", "instance", str(offer_id),
        "--image", docker_image,
        "--disk", str(disk_gb),
        "--raw",
    ]
    if onstart_cmd:
        args.extend(["--onstart-cmd", onstart_cmd])
    if label:
        args.extend(["--label", label])
    if env_vars:
        env_str = " ".join(f"-e {k}={v}" for k, v in env_vars.items())
        args.extend(["--env", env_str])

    try:
        result = _run_vastai(*args)
        return result
    except Exception as e:
        print(f"Error creating instance: {e}")
        return None


def get_instances() -> list:
    """Get all current instances."""
    try:
        result = _run_vastai("show", "instances", "--raw")
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []


def get_instance(instance_id: int) -> Optional[dict]:
    """Get details for a specific instance."""
    try:
        result = _run_vastai("show", "instance", str(instance_id), "--raw")
        return result
    except Exception:
        return None


def destroy_instance(instance_id: int) -> bool:
    """Destroy (stop and delete) an instance."""
    try:
        _run_vastai("destroy", "instance", str(instance_id), parse_json=False)
        return True
    except Exception as e:
        print(f"Error destroying instance {instance_id}: {e}")
        return False


def scp_to_instance(instance_id: int, local_path: str, remote_path: str) -> bool:
    """Copy files to an instance via SCP."""
    try:
        _run_vastai("scp", f"{local_path}", f"{instance_id}:{remote_path}", parse_json=False)
        return True
    except Exception as e:
        print(f"SCP to {instance_id} failed: {e}")
        return False


def scp_from_instance(instance_id: int, remote_path: str, local_path: str) -> bool:
    """Copy files from an instance via SCP."""
    try:
        _run_vastai("scp", f"{instance_id}:{remote_path}", f"{local_path}", parse_json=False)
        return True
    except Exception as e:
        print(f"SCP from {instance_id} failed: {e}")
        return False


def ssh_command(instance_id: int, command: str, timeout: int = 30) -> str:
    """Run a command on an instance via SSH."""
    try:
        result = subprocess.run(
            ["vastai", "ssh", str(instance_id), command],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"SSH error: {e}"


def get_cheapest_offer(gpu_name: str = "RTX_4090",
                       max_price: float = 0.50) -> Optional[dict]:
    """Find the single cheapest available offer."""
    offers = search_offers(gpu_name, max_price, num_results=1)
    if offers:
        return offers[0]
    return None


def estimate_cost(dph: float, total_timesteps: int, fps_estimate: int) -> dict:
    """Estimate the cost of running an experiment.

    Args:
        dph: Dollars per hour for the instance
        total_timesteps: Training steps
        fps_estimate: Expected training FPS on this GPU
    """
    hours = total_timesteps / fps_estimate / 3600
    cost = hours * dph
    return {
        "estimated_hours": round(hours, 2),
        "estimated_cost": round(cost, 2),
        "cost_per_hour": dph,
        "fps_estimate": fps_estimate,
    }


# FPS benchmarks from experiment history (steps/sec on single GPU)
# mamba: updated for mamba-ssm CUDA kernels (was 4K with pure PyTorch)
GPU_FPS_BENCHMARKS = {
    "fsppo": {"RTX_4090": 65000, "RTX_3090": 45000, "A100": 80000, "A10": 35000},
    "rppo":  {"RTX_4090": 27000, "RTX_3090": 18000, "A100": 35000, "A10": 15000},
    "mamba": {"RTX_4090": 25000, "RTX_3090": 17000, "A100": 32000, "A10": 12000},
    "ppo":   {"RTX_4090": 56000, "RTX_3090": 38000, "A100": 70000, "A10": 30000},
}


def estimate_experiment_cost(model_type: str, total_timesteps: int,
                              gpu_name: str = "RTX_4090",
                              dph: float = 0.35) -> dict:
    """Estimate cost for a specific experiment config."""
    fps = GPU_FPS_BENCHMARKS.get(model_type, {}).get(gpu_name, 20000)
    return estimate_cost(dph, total_timesteps, fps)
