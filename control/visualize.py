"""
3D trajectory visualizer for trained GCNet policy.
Loads a checkpoint, rolls out episodes, and renders 3D matplotlib plot.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.gpu_env import BatchedDroneEnv
from control.train_gpu import ActorCritic, RunningNorm
from config import (
    OBS_DIM, GATE_POSITIONS, GATE_SIZE, NUM_GATES,
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z,
)


def load_policy(checkpoint_path, device):
    """Load trained policy and observation normalizer."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = ActorCritic(obs_dim=OBS_DIM, act_dim=4).to(device)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    obs_norm = RunningNorm(OBS_DIM, device)
    obs_norm.mean = ckpt['obs_norm_mean']
    obs_norm.var = ckpt['obs_norm_var']

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Global step: {ckpt.get('global_step', 'unknown')}")
    return policy, obs_norm


def rollout(env, policy, obs_norm, n_episodes=4, max_steps=8000):
    """Roll out episodes, recording trajectories. Tracks gate passage BEFORE auto-reset."""
    trajectories = []

    for ep in range(n_episodes):
        env.reset_all()
        obs = env.get_obs()

        traj = {'pos': [], 'rewards': []}
        gates_passed = 0
        prev_gate_idx = env.gate_idx[0].item()

        for step in range(max_steps):
            obs_n = obs_norm.normalize(obs)
            with torch.no_grad():
                mean, std, value = policy(obs_n)
                action = torch.sigmoid(mean)

            # Record position BEFORE step (this is the actual drone position)
            traj['pos'].append(env.states[0, 0:3].cpu().numpy().copy())

            next_obs, reward, done = env.step(action)
            traj['rewards'].append(reward[0].item())

            # Track gate idx changes ACROSS the step
            # After step (but before auto-reset in single_gate mode),
            # gate_idx may have incremented. If done, env is already reset.
            # Use reward to detect gate passage: gate bonus adds LAMBDA_GATE=10
            if reward[0].item() > 5:  # gate bonus signal
                gates_passed += 1

            obs = next_obs
            if done[0]:
                # Record final position from prev_states (before reset)
                traj['pos'].append(env.prev_states[0, 0:3].cpu().numpy().copy())
                break

        traj['pos'] = np.array(traj['pos'])
        traj['total_reward'] = sum(traj['rewards'])
        traj['steps'] = len(traj['rewards'])
        traj['gates_passed'] = gates_passed
        trajectories.append(traj)

        print(f"Episode {ep+1}: {traj['steps']} steps, "
              f"reward={traj['total_reward']:.2f}, "
              f"gates={traj['gates_passed']}")

    return trajectories


def draw_gate(ax, pos, yaw, size, color='red', alpha=0.3):
    """Draw a square gate in 3D. Uses ENU display (X-east, Y-north, Z-up)."""
    half = size / 2
    # Gate corners in local frame (y-z plane of gate)
    # In NED: Y is right of gate normal, Z is down
    corners_local = np.array([
        [0, -half, -half],  # left-top (NED: -y=left, -z=up)
        [0,  half, -half],  # right-top
        [0,  half,  half],  # right-bottom
        [0, -half,  half],  # left-bottom
    ])
    # Rotate by yaw around Z-axis (NED)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners_ned = (R @ corners_local.T).T + pos

    # NED to display: negate Z (up = positive in display)
    corners_disp = corners_ned.copy()
    corners_disp[:, 2] = -corners_disp[:, 2]

    verts = [list(zip(corners_disp[:, 0], corners_disp[:, 1], corners_disp[:, 2]))]
    gate_poly = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                                  edgecolor=color, linewidth=2)
    ax.add_collection3d(gate_poly)


def plot_trajectories(trajectories, save_path=None, gate_size=GATE_SIZE, title=None):
    """Plot 3D trajectories with gates. NED→display: Z negated."""
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw gates
    for i, g in enumerate(GATE_POSITIONS):
        pos = np.array([g[0], g[1], g[2]])
        color = 'red' if i % 2 == 0 else 'orange'
        draw_gate(ax, pos, g[3], size=gate_size, color=color, alpha=0.4)
        # Label at gate center (Z negated for display)
        ax.text(pos[0], pos[1], -pos[2] + 0.3, f'G{i+1}',
                fontsize=8, ha='center', fontweight='bold', color='darkred')

    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(trajectories), 10)))
    for i, traj in enumerate(trajectories):
        pos = traj['pos']
        # NED to display: negate Z
        x, y, z_disp = pos[:, 0], pos[:, 1], -pos[:, 2]

        ax.plot(x, y, z_disp,
                color=colors[i % 10], linewidth=1.8, alpha=0.85,
                label=f"Ep{i+1}: R={traj['total_reward']:.1f}, "
                      f"gates={traj['gates_passed']}, "
                      f"{traj['steps']}steps")
        # Start marker (circle)
        ax.scatter(x[0], y[0], z_disp[0], color=colors[i % 10],
                   marker='o', s=60, zorder=5)
        # End marker (X)
        ax.scatter(x[-1], y[-1], z_disp[-1], color=colors[i % 10],
                   marker='x', s=60, zorder=5, linewidths=2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(title or 'MonoRace M23 — Drone Trajectories')
    ax.legend(loc='upper left', fontsize=7, ncol=2)

    # Axis limits
    ax.set_xlim(BOUNDS_X[0] - 2, BOUNDS_X[1] + 2)
    ax.set_ylim(BOUNDS_Y[0] - 2, BOUNDS_Y[1] + 2)
    ax.set_zlim(0, -BOUNDS_Z[0] + 1)

    # Better viewing angle
    ax.view_init(elev=25, azim=-60)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_from_npz(npz_path, save_path=None, gate_size=GATE_SIZE):
    """Plot trajectories from saved .npz file (from train_sequential full course test)."""
    data = np.load(npz_path, allow_pickle=True)
    gates = data['gates']
    rewards = data['rewards']
    steps = data['steps']

    trajectories = []
    for i in range(len(gates)):
        pos = data[f'pos_{i}']
        trajectories.append({
            'pos': pos,
            'total_reward': float(rewards[i]),
            'gates_passed': int(gates[i]),
            'steps': int(steps[i]),
        })

    plot_trajectories(trajectories, save_path=save_path, gate_size=gate_size,
                      title='MonoRace M23 — Full Course Test (Sequential Training)')


def main():
    parser = argparse.ArgumentParser(description="Visualize trained GCNet policy")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--npz", type=str, default=None,
                        help="Plot from saved .npz results file")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--single-gate", action="store_true")
    parser.add_argument("--gate-size", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    gate_sz = args.gate_size or GATE_SIZE

    if args.npz:
        plot_from_npz(args.npz, save_path=args.save, gate_size=gate_sz)
        return

    if args.checkpoint is None:
        print("Error: --checkpoint or --npz required")
        return

    device = torch.device(args.device)
    policy, obs_norm = load_policy(args.checkpoint, device)

    env = BatchedDroneEnv(num_envs=1, device=args.device,
                          fixed_start=args.fixed_start,
                          domain_randomize=not args.fixed_start,
                          single_gate=args.single_gate,
                          gate_size_override=args.gate_size)
    env.set_difficulty(args.difficulty)

    print(f"\nRolling out {args.episodes} episodes...")
    trajectories = rollout(env, policy, obs_norm, n_episodes=args.episodes)
    plot_trajectories(trajectories, save_path=args.save, gate_size=gate_sz)


if __name__ == "__main__":
    main()
