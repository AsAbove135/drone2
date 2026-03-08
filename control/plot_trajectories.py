"""
Plot drone trajectories from a trained policy.
Runs N episodes with random segments and plots top-down + side views.
Color-coded by target gate segment.
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from control.train_gpu import ActorCritic, RunningNorm
from config import OBS_DIM, GATE_POSITIONS, GATE_SIZE, NUM_GATES


def run_episodes(checkpoint_path, num_episodes=32, device="cuda"):
    """Run policy and collect trajectories."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    policy = ActorCritic(obs_dim=OBS_DIM, act_dim=4).to(device)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    obs_norm = RunningNorm(OBS_DIM, device)
    obs_norm.mean = ckpt['obs_norm_mean']
    obs_norm.var = ckpt['obs_norm_var']

    env = BatchedDroneEnv(num_envs=num_episodes, device=device,
                          random_segments=True, domain_randomize=True)

    obs = env.reset_all()
    obs = obs_norm.normalize(obs)

    # Store trajectories: list of lists of (x, y, z)
    trajectories = [[] for _ in range(num_episodes)]
    target_gates = env.gate_idx.cpu().numpy().copy()
    passed = [False] * num_episodes
    active = [True] * num_episodes

    # Record initial positions
    for i in range(num_episodes):
        p = env.states[i, 0:3].cpu().numpy()
        trajectories[i].append(p.copy())

    max_steps = 4000
    for step in range(max_steps):
        with torch.no_grad():
            mean, std, value = policy(obs)
            action = torch.sigmoid(mean)  # deterministic (mean) policy

        obs, rewards, dones = env.step(action)
        obs = obs_norm.normalize(obs)

        for i in range(num_episodes):
            if active[i]:
                if dones[i]:
                    # Don't record post-reset state — env already spawned new episode
                    if rewards[i] > 5.0:  # gate reward is 10
                        passed[i] = True
                    active[i] = False
                else:
                    p = env.states[i, 0:3].cpu().numpy()
                    trajectories[i].append(p.copy())

        if not any(active):
            break

    return trajectories, target_gates, passed


def plot_trajectories(trajectories, target_gates, passed):
    """Plot top-down and side views, color-coded by target gate."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # Color map: one distinct color per gate
    cmap = plt.cm.tab20
    gate_colors = [cmap(i / NUM_GATES) for i in range(NUM_GATES)]

    half = GATE_SIZE / 2.0

    # Draw gates
    for i in range(NUM_GATES):
        cx, cy, cz, yaw = GATE_POSITIONS[i]
        gc = gate_colors[i]

        # Top-down: gate as a short line perpendicular to yaw
        perp_x = -np.sin(yaw) * max(half, 0.5)  # min visual size
        perp_y = np.cos(yaw) * max(half, 0.5)
        ax1.plot([cx - perp_x, cx + perp_x], [cy - perp_y, cy + perp_y],
                 color=gc, linewidth=4, zorder=5, solid_capstyle='round')
        ax1.annotate(f'G{i}', (cx, cy), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, fontweight='bold',
                     color=gc)

        # Side view: gate as vertical line
        vis_half = max(half, 0.3)
        ax2.plot([cx, cx], [cz - vis_half, cz + vis_half],
                 color=gc, linewidth=4, zorder=5, solid_capstyle='round')
        ax2.annotate(f'G{i}', (cx, cz - vis_half), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, fontweight='bold',
                     color=gc)

    # Plot trajectories colored by target gate
    for i, traj in enumerate(trajectories):
        if len(traj) < 2:
            continue
        traj = np.array(traj)
        gi = target_gates[i]
        color = gate_colors[gi]
        alpha = 0.9 if passed[i] else 0.5
        lw = 1.5 if passed[i] else 0.8
        style = '-' if passed[i] else '--'

        # Top-down (X vs Y)
        ax1.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha,
                 linewidth=lw, linestyle=style)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=4,
                 alpha=alpha, zorder=4)
        ax1.plot(traj[-1, 0], traj[-1, 1], 'x' if not passed[i] else '*',
                 color=color, markersize=5 if not passed[i] else 8,
                 alpha=alpha, zorder=4)

        # Side view (X vs Z)
        ax2.plot(traj[:, 0], traj[:, 2], color=color, alpha=alpha,
                 linewidth=lw, linestyle=style)
        ax2.plot(traj[0, 0], traj[0, 2], 'o', color=color, markersize=4,
                 alpha=alpha, zorder=4)
        ax2.plot(traj[-1, 0], traj[-1, 2], 'x' if not passed[i] else '*',
                 color=color, markersize=5 if not passed[i] else 8,
                 alpha=alpha, zorder=4)

    # Legend: one entry per gate that has trajectories
    from matplotlib.lines import Line2D
    legend_elements = []
    for gi in sorted(set(target_gates)):
        n_total = sum(1 for t in target_gates if t == gi)
        n_pass = sum(1 for j, t in enumerate(target_gates) if t == gi and passed[j])
        legend_elements.append(
            Line2D([0], [0], color=gate_colors[gi], linewidth=2,
                   label=f'G{gi}: {n_pass}/{n_total} passed'))

    n_passed = sum(passed)
    ax1.set_title(f'Top-Down View (X-Y) — {n_passed}/{len(trajectories)} passed gates',
                  fontsize=13)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2)

    ax2.set_title('Side View (X-Z, NED: negative Z = up)', fontsize=13)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trajectories.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {os.path.abspath(save_path)}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="D:/drone2_training/gcnet_m23_randseg_final.pt")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    trajs, targets, passed = run_episodes(args.checkpoint, args.episodes, args.device)
    plot_trajectories(trajs, targets, passed)
