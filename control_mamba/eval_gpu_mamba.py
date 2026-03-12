"""
Evaluate a trained GPU Mamba-PPO policy on the kidney track.
Runs full episodes, reports gates passed per episode, plots trajectories.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control_mamba.train_gpu_mamba import MambaActorCritic, RunningNorm
from control.gpu_env import BatchedDroneEnv
from config import OBS_DIM, GATE_POSITIONS, NUM_LAPS, ACTIVE_TRACK, MAX_EPISODE_STEPS


def evaluate(
    checkpoint_path,
    num_episodes=16,
    max_steps=MAX_EPISODE_STEPS,
    device="cuda",
):
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Global step: {ckpt.get('global_step', 'unknown')}")

    policy = MambaActorCritic(obs_dim=OBS_DIM, act_dim=4).to(device)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    obs_norm = RunningNorm(OBS_DIM, device)
    obs_norm.mean = ckpt['obs_norm_mean'].to(device)
    obs_norm.var = ckpt['obs_norm_var'].to(device)

    env = BatchedDroneEnv(
        num_envs=num_episodes, device=device,
        fixed_start=True,
        domain_randomize=False,
        single_gate=False,
        random_segments=False,
    )

    obs = env.reset_all()
    obs = obs_norm.normalize(obs)
    actor_states, critic_states = policy.initial_state(num_episodes, device)

    trajectories = [[] for _ in range(num_episodes)]
    max_gate_idx = torch.zeros(num_episodes, device=device, dtype=torch.long)
    episode_done = torch.zeros(num_episodes, device=device, dtype=torch.bool)

    positions = env.states[:, 0:3].cpu().numpy()
    for i in range(num_episodes):
        trajectories[i].append(positions[i].copy())

    with torch.no_grad():
        for step in range(max_steps):
            still_running = ~episode_done
            max_gate_idx = torch.where(still_running, torch.max(max_gate_idx, env.gate_idx), max_gate_idx)

            mean, std, value, actor_states, critic_states = \
                policy.forward_single(obs, actor_states, critic_states)
            action = torch.sigmoid(mean)

            next_obs, rewards, dones = env.step(action)
            next_obs = obs_norm.normalize(next_obs)

            max_gate_idx = torch.where(still_running, torch.max(max_gate_idx, env.gate_idx), max_gate_idx)

            positions = env.states[:, 0:3].cpu().numpy()
            for i in range(num_episodes):
                if not episode_done[i]:
                    trajectories[i].append(positions[i].copy())

            episode_done = episode_done | dones

            if dones.any():
                done_mask = dones.unsqueeze(-1).unsqueeze(-1)
                for layer in range(len(actor_states)):
                    ssm, conv = actor_states[layer]
                    actor_states[layer] = (ssm * (1.0 - done_mask), conv * (1.0 - done_mask))
                    ssm, conv = critic_states[layer]
                    critic_states[layer] = (ssm * (1.0 - done_mask), conv * (1.0 - done_mask))

            obs = next_obs

            if episode_done.all():
                break

    gates_passed_np = max_gate_idx.cpu().numpy()
    num_gates = len(GATE_POSITIONS)
    total_target = num_gates * NUM_LAPS

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({num_episodes} episodes, {ACTIVE_TRACK} track)")
    print(f"{'='*60}")
    for i in range(num_episodes):
        steps = len(trajectories[i])
        gp = gates_passed_np[i]
        print(f"  Episode {i+1:2d}: {gp:2d}/{total_target} gates, {steps} steps")

    avg_gates = gates_passed_np.mean()
    max_gates = gates_passed_np.max()
    pct_any = (gates_passed_np > 0).mean() * 100
    print(f"\n  Average: {avg_gates:.1f} gates")
    print(f"  Max: {max_gates} gates")
    print(f"  Episodes with >=1 gate: {pct_any:.0f}%")

    return trajectories, gates_passed_np


def plot_trajectories(trajectories, gates_passed_np, save_path=None):
    track_gates = GATE_POSITIONS
    num_gates = len(track_gates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    cmap = plt.cm.tab10

    half = 0.5
    for i in range(num_gates):
        cx, cy, cz, yaw = track_gates[i]
        perp_x = -np.sin(yaw) * half
        perp_y = np.cos(yaw) * half
        ax1.plot([cx - perp_x, cx + perp_x], [cy - perp_y, cy + perp_y],
                 color='gray', linewidth=4, zorder=5, solid_capstyle='round')
        ax1.annotate(f'G{i+1}', (cx, cy), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
        vis_half = 0.3
        ax2.plot([cx, cx], [cz - vis_half, cz + vis_half],
                 color='gray', linewidth=4, zorder=5, solid_capstyle='round')
        ax2.annotate(f'G{i+1}', (cx, cz - vis_half), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        if len(traj) < 2:
            continue
        color = cmap(i % 10 / 10)
        gp = gates_passed_np[i]
        label = f'Ep {i+1}: {gp} gates, {len(traj)} steps'
        ax1.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1.5, label=label)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=6, zorder=4)
        ax1.plot(traj[-1, 0], traj[-1, 1], 'x', color=color, markersize=8, zorder=4)
        ax2.plot(traj[:, 0], traj[:, 2], color=color, alpha=0.7, linewidth=1.5)

    total_target = num_gates * NUM_LAPS
    avg_gp = gates_passed_np.mean()
    ax1.set_title(f'Mamba-PPO Eval: {ACTIVE_TRACK} ({num_gates} gates x {NUM_LAPS} laps = {total_target}) — '
                  f'avg {avg_gp:.1f} gates', fontsize=13)
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
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                 'results_tracking', 'mamba_eval.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved trajectory plot to {os.path.abspath(save_path)}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GPU Mamba-PPO policy")
    parser.add_argument("--checkpoint", type=str,
                        default="D:/drone2_training/mamba_latest/mamba_m23_randseg_final.pt")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    trajs, gp = evaluate(args.checkpoint, num_episodes=args.episodes, device=args.device)
    plot_trajectories(trajs, gp)
