"""
Unified evaluation system for drone racing models.
Supports RPPO, Mamba, and Frame-Stacked PPO (FSPPO) models across any track.
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from config import OBS_DIM, NUM_LAPS, MAX_EPISODE_STEPS, TRACKS


def load_model(model_type, checkpoint_path, device, frame_stack=None):
    """Load model and obs normalizer from checkpoint.
    Returns (policy, obs_norm, ckpt_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == 'rppo':
        from control_rppo.train_gpu_rppo import RecurrentActorCritic, RunningNorm
        policy = RecurrentActorCritic(
            obs_dim=OBS_DIM, act_dim=4, lstm_hidden=128, n_lstm_layers=1
        ).to(device)
    elif model_type == 'mamba':
        from control_mamba.train_gpu_mamba import MambaActorCritic, RunningNorm
        policy = MambaActorCritic(obs_dim=OBS_DIM, act_dim=4).to(device)
    elif model_type == 'fsppo':
        from control_fsppo.train_gpu_fsppo import FrameStackedActorCritic, RunningNorm
        fs = frame_stack or ckpt.get('frame_stack', 16)
        policy = FrameStackedActorCritic(
            obs_dim=OBS_DIM, act_dim=4, frame_stack=fs
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    # Import RunningNorm from whichever module we used
    if model_type == 'rppo':
        from control_rppo.train_gpu_rppo import RunningNorm
    elif model_type == 'mamba':
        from control_mamba.train_gpu_mamba import RunningNorm
    else:
        from control_fsppo.train_gpu_fsppo import RunningNorm

    obs_norm = RunningNorm(OBS_DIM, device)
    obs_norm.mean = ckpt['obs_norm_mean'].to(device)
    obs_norm.var = ckpt['obs_norm_var'].to(device)

    return policy, obs_norm, ckpt


def evaluate_track(
    policy, obs_norm, model_type, track_name, gate_positions,
    num_episodes=10, max_steps=MAX_EPISODE_STEPS, device="cuda",
    frame_stack=None,
):
    """Run evaluation episodes on a single track.
    Returns (trajectories, gates_passed_np, steps_per_episode).
    """
    device = torch.device(device)

    env = BatchedDroneEnv(
        num_envs=num_episodes, device=device,
        fixed_start=True,
        domain_randomize=False,
        single_gate=False,
        random_segments=False,
        gate_size_override=1.5,  # training gate size
        gate_positions=gate_positions,
    )

    obs = env.reset_all()
    obs_n = obs_norm.normalize(obs)

    trajectories = [[] for _ in range(num_episodes)]
    max_gate_idx = torch.zeros(num_episodes, device=device, dtype=torch.long)
    episode_done = torch.zeros(num_episodes, device=device, dtype=torch.bool)
    ep_steps = torch.zeros(num_episodes, device=device, dtype=torch.long)

    # Record initial positions
    positions = env.states[:, 0:3].cpu().numpy()
    for i in range(num_episodes):
        trajectories[i].append(positions[i].copy())

    # Initialize model state
    if model_type == 'rppo':
        actor_hc, critic_hc = policy.initial_state(num_episodes, device)
    elif model_type == 'mamba':
        actor_states, critic_states = policy.initial_state(num_episodes, device)
    elif model_type == 'fsppo':
        from control_fsppo.train_gpu_fsppo import FrameStackBuffer
        fs = frame_stack or 16
        frame_buf = FrameStackBuffer(num_episodes, fs, OBS_DIM, device)
        frame_buf.push(obs_n)

    with torch.no_grad():
        for step in range(max_steps):
            still_running = ~episode_done
            max_gate_idx = torch.where(
                still_running, torch.max(max_gate_idx, env.gate_idx), max_gate_idx
            )
            ep_steps[still_running] += 1

            # Forward pass (deterministic)
            if model_type == 'rppo':
                mean, std, value, actor_hc, critic_hc = policy.forward_single(
                    obs_n, actor_hc, critic_hc
                )
                action = torch.sigmoid(mean)
            elif model_type == 'mamba':
                mean, std, value, actor_states, critic_states = policy.forward_single(
                    obs_n, actor_states, critic_states
                )
                action = torch.sigmoid(mean)
            elif model_type == 'fsppo':
                stacked = frame_buf.get_stacked()
                mean, std, value = policy(stacked)
                action = torch.sigmoid(mean)

            next_obs, rewards, dones = env.step(action)
            next_obs_n = obs_norm.normalize(next_obs)

            # For episodes that just completed (dones=True while still_running),
            # gate_idx was reset inside step(). Infer completed episodes passed all gates.
            newly_done = still_running & dones
            total_gates = len(gate_positions) * NUM_LAPS
            # If episode finished and reward is positive (not a crash), it completed all gates
            completed_mask = newly_done & (rewards > 0)
            max_gate_idx = torch.where(
                completed_mask, torch.tensor(total_gates, device=device), max_gate_idx
            )
            # Also capture gate_idx for non-completed episodes (crashed mid-run)
            max_gate_idx = torch.where(
                still_running & ~completed_mask,
                torch.max(max_gate_idx, env.gate_idx), max_gate_idx
            )

            # Record trajectories
            positions = env.states[:, 0:3].cpu().numpy()
            for i in range(num_episodes):
                if not episode_done[i]:
                    trajectories[i].append(positions[i].copy())

            # Mark newly done
            episode_done = episode_done | dones

            # Reset model state for done episodes
            if dones.any():
                if model_type == 'rppo':
                    done_mask = dones.float().unsqueeze(0).unsqueeze(-1)
                    actor_hc = (actor_hc[0] * (1.0 - done_mask),
                                actor_hc[1] * (1.0 - done_mask))
                    critic_hc = (critic_hc[0] * (1.0 - done_mask),
                                 critic_hc[1] * (1.0 - done_mask))
                elif model_type == 'mamba':
                    done_mask = dones.unsqueeze(-1).unsqueeze(-1)
                    for layer in range(len(actor_states)):
                        ssm, conv = actor_states[layer]
                        actor_states[layer] = (ssm * (1.0 - done_mask),
                                               conv * (1.0 - done_mask))
                        ssm, conv = critic_states[layer]
                        critic_states[layer] = (ssm * (1.0 - done_mask),
                                                conv * (1.0 - done_mask))
                elif model_type == 'fsppo':
                    frame_buf.reset(dones)

            obs_n = next_obs_n
            if model_type == 'fsppo':
                frame_buf.push(obs_n)

            if episode_done.all():
                break

    gates_passed_np = max_gate_idx.cpu().numpy()
    steps_np = ep_steps.cpu().numpy()
    return trajectories, gates_passed_np, steps_np


def plot_trajectories(trajectories, gates_passed_np, gate_positions, track_name,
                      model_type, save_path=None, gate_size=1.5):
    """Plot top-down and side view trajectories for a track."""
    num_gates = len(gate_positions)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    cmap = plt.cm.tab10

    # Draw gates at actual size
    half = gate_size / 2.0
    for i in range(num_gates):
        cx, cy, cz, yaw = gate_positions[i]
        perp_x = -np.sin(yaw) * half
        perp_y = np.cos(yaw) * half
        ax1.plot([cx - perp_x, cx + perp_x], [cy - perp_y, cy + perp_y],
                 color='gray', linewidth=4, zorder=5, solid_capstyle='round')
        ax1.annotate(f'G{i+1}', (cx, cy), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
        ax2.plot([cx, cx], [cz - half, cz + half],
                 color='gray', linewidth=4, zorder=5, solid_capstyle='round')
        ax2.annotate(f'G{i+1}', (cx, cz - half), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        if len(traj) < 2:
            continue
        color = cmap(i % 10 / 10)
        gp = gates_passed_np[i]
        sim_time = len(traj) * 0.01
        label = f'Ep {i+1}: {gp} gates, {sim_time:.1f}s'
        ax1.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1.5, label=label)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=6, zorder=4)
        ax1.plot(traj[-1, 0], traj[-1, 1], 'x', color=color, markersize=8, zorder=4)
        ax2.plot(traj[:, 0], traj[:, 2], color=color, alpha=0.7, linewidth=1.5)

    total_target = num_gates * NUM_LAPS
    avg_gp = gates_passed_np.mean()
    ax1.set_title(
        f'{model_type.upper()} Eval: {track_name} '
        f'({num_gates} gates x {NUM_LAPS} laps = {total_target}) — '
        f'avg {avg_gp:.1f} gates', fontsize=13
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='upper left')

    ax2.set_title('Side View (X-Z, NED: negative Z = up)', fontsize=13)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved trajectory plot to {os.path.abspath(save_path)}")
    plt.close()


def run_evaluation(checkpoint_path, model_type, track_names, experiment_name,
                   num_episodes=10, frame_stack=None, device="cuda"):
    """Run full evaluation across specified tracks."""
    device_obj = torch.device(device)

    print(f"Loading {model_type.upper()} model from {checkpoint_path}")
    policy, obs_norm, ckpt = load_model(model_type, checkpoint_path, device_obj,
                                         frame_stack=frame_stack)
    print(f"  Global step: {ckpt.get('global_step', 'unknown')}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing results to merge (don't overwrite other tracks)
    results_path = os.path.join(results_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {
            'experiment': experiment_name,
            'checkpoint': checkpoint_path,
            'model_type': model_type,
            'global_step': ckpt.get('global_step', None),
            'tracks': {},
        }
    if frame_stack is not None:
        results['frame_stack'] = frame_stack

    for track_name in track_names:
        if track_name not in TRACKS:
            print(f"WARNING: Track '{track_name}' not found in config.TRACKS. Skipping.")
            continue

        gate_positions = TRACKS[track_name]
        num_gates = len(gate_positions)

        print(f"\nEvaluating on '{track_name}' track ({num_gates} gates)...")
        trajectories, gates_passed_np, steps_np = evaluate_track(
            policy, obs_norm, model_type, track_name, gate_positions,
            num_episodes=num_episodes, device=device,
            frame_stack=frame_stack,
        )

        # Print per-episode results
        total_target = num_gates * NUM_LAPS
        print(f"{'='*60}")
        print(f"Results: {track_name} ({num_episodes} episodes)")
        print(f"{'='*60}")
        DT_SIM = 0.01  # 100Hz control rate
        for i in range(num_episodes):
            gp = int(gates_passed_np[i])
            steps = int(steps_np[i])
            sim_time = steps * DT_SIM
            completed = gp >= total_target
            time_str = f"{sim_time:.1f}s" if completed else f"{sim_time:.1f}s (crashed)"
            print(f"  Episode {i+1:2d}: {gp:2d}/{total_target} gates, {steps} steps, {time_str}")

        avg_gates = float(gates_passed_np.mean())
        max_gates = int(gates_passed_np.max())
        pct_ge_1 = float((gates_passed_np >= 1).mean() * 100)
        pct_ge_5 = float((gates_passed_np >= 5).mean() * 100)
        pct_ge_10 = float((gates_passed_np >= 10).mean() * 100)

        print(f"\n  Average: {avg_gates:.1f} gates")
        print(f"  Max: {max_gates} gates")
        print(f"  >= 1 gate: {pct_ge_1:.0f}%")
        print(f"  >= 5 gates: {pct_ge_5:.0f}%")
        print(f"  >= 10 gates: {pct_ge_10:.0f}%")

        # Save track results
        episodes_data = []
        completion_times = []
        for i in range(num_episodes):
            gp = int(gates_passed_np[i])
            steps = int(steps_np[i])
            sim_time = round(steps * DT_SIM, 2)
            ep = {
                'gates_passed': gp,
                'steps': steps,
                'sim_time_s': sim_time,
            }
            if gp >= total_target:
                ep['completed'] = True
                completion_times.append(sim_time)
            else:
                ep['completed'] = False
            episodes_data.append(ep)

        track_results = {
            'episodes': episodes_data,
            'avg_gates': avg_gates,
            'max_gates': max_gates,
            'pct_ge_1': pct_ge_1,
            'pct_ge_5': pct_ge_5,
            'pct_ge_10': pct_ge_10,
            'completions': len(completion_times),
        }
        if completion_times:
            track_results['avg_completion_time_s'] = round(sum(completion_times) / len(completion_times), 2)
            track_results['best_completion_time_s'] = min(completion_times)
            print(f"\n  Completions: {len(completion_times)}/{num_episodes}")
            print(f"  Avg completion time: {track_results['avg_completion_time_s']:.2f}s")
            print(f"  Best completion time: {track_results['best_completion_time_s']:.2f}s")

        results['tracks'][track_name] = track_results

        # Plot trajectories
        plot_path = os.path.join(results_dir, f'{track_name}_trajectories.png')
        plot_trajectories(
            trajectories, gates_passed_np, gate_positions, track_name,
            model_type, save_path=plot_path,
        )

    # Save results JSON
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {os.path.abspath(results_path)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified evaluation for drone racing models"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=['rppo', 'mamba', 'fsppo'],
                        help="Model architecture type")
    parser.add_argument("--tracks", type=str, nargs='+',
                        default=list(TRACKS.keys()),
                        help="Track names to evaluate on (default: all)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per track (default: 10)")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Frame stack size (FSPPO only, default: from checkpoint or 16)")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name for results directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    # Default experiment name from checkpoint filename
    if args.name is None:
        args.name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    run_evaluation(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        track_names=args.tracks,
        experiment_name=args.name,
        num_episodes=args.episodes,
        frame_stack=args.frame_stack,
        device=args.device,
    )
