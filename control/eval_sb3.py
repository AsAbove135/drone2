"""
Evaluate a trained SB3 policy on any track and visualize trajectories.
Usage: python control/eval_sb3.py --checkpoint D:/drone2_training/latest/checkpoints/gcnet_m23_2000000_steps.zip --track easy
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from control.ppo_train import MonoRaceSimEnv
import config


class EvalEnv(MonoRaceSimEnv):
    """MonoRaceSimEnv that always spawns behind gate 0 for evaluation."""
    def __init__(self, track_gates=None):
        super().__init__(near_gate_spawn=True)
        if track_gates is not None:
            self.gates = list(track_gates)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Override: place behind gate 0
        gx, gy, gz, g_yaw = self.gates[0]
        spawn_dist = 3.0
        px = gx - spawn_dist * np.cos(g_yaw)
        py = gy - spawn_dist * np.sin(g_yaw)
        pz = gz

        state_np = self.state.cpu().numpy().squeeze(0)
        state_np[0], state_np[1], state_np[2] = px, py, pz
        state_np[3:6] = 0.0  # zero velocity
        # Facing gate (yaw-only quaternion)
        state_np[6] = np.cos(g_yaw / 2)   # qw
        state_np[7] = 0.0                  # qx
        state_np[8] = 0.0                  # qy
        state_np[9] = np.sin(g_yaw / 2)   # qz
        state_np[10:13] = 0.0  # zero body rates

        self.state = torch.tensor([state_np], dtype=torch.float32)
        self.prev_state_np = state_np.copy()
        self.current_gate_idx = 0
        self.step_count = 0
        self.steps_since_gate = 0
        self._crashed = False

        obs = self._get_obs()
        return obs, info


def evaluate(checkpoint_path, track_name=None, vecnorm_path=None, num_episodes=8, max_steps=4000):
    """Run policy from behind gate 0 on the specified track, collect trajectories."""
    # Resolve track gates
    if track_name and track_name in config.TRACKS:
        track_gates = config.TRACKS[track_name]
    else:
        track_gates = config.GATE_POSITIONS
        track_name = config.ACTIVE_TRACK

    num_gates = len(track_gates)

    # Set gate size to training value for fair eval
    MonoRaceSimEnv.gate_size_current = 1.5

    env = DummyVecEnv([lambda: EvalEnv(track_gates=track_gates)])

    # Try to load VecNormalize stats
    vn_loaded = False
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        vn_loaded = True
        print(f"Loaded VecNormalize from {vecnorm_path}")
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(ckpt_dir)
        import re, glob
        step_match = re.search(r'(\d+)_steps', os.path.basename(checkpoint_path))
        search_dirs = [ckpt_dir, parent_dir]
        for search_dir in search_dirs:
            if step_match:
                vn = os.path.join(search_dir, f"vecnormalize_{step_match.group(1)}_steps.pkl")
                if os.path.exists(vn):
                    env = VecNormalize.load(vn, env)
                    vn_loaded = True
                    print(f"Loaded VecNormalize from {vn}")
                    break
            vn_files = sorted(glob.glob(os.path.join(search_dir, "vecnormalize*.pkl")))
            if vn_files:
                env = VecNormalize.load(vn_files[-1], env)
                vn_loaded = True
                print(f"Loaded VecNormalize from {vn_files[-1]}")
                break

    if not vn_loaded:
        print("WARNING: No VecNormalize found! Policy will get unnormalized obs and likely fail.")

    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False

    model = PPO.load(checkpoint_path, env=env, device="cpu")

    trajectories = []
    gates_passed_list = []

    for ep in range(num_episodes):
        obs = env.reset()
        inner_env = env.envs[0] if hasattr(env, 'envs') else env.venv.envs[0]

        traj = [inner_env.state.cpu().numpy().squeeze(0)[:3].copy()]
        max_gate_idx = 0
        crashed = False

        for step in range(max_steps):
            cur_gate = inner_env.current_gate_idx
            max_gate_idx = max(max_gate_idx, cur_gate)
            was_crashed = inner_env._crashed

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if not done[0]:
                p = inner_env.state.cpu().numpy().squeeze(0)[:3]
                traj.append(p.copy())
                max_gate_idx = max(max_gate_idx, inner_env.current_gate_idx)
            else:
                crashed = was_crashed or inner_env._crashed
                break

        trajectories.append(np.array(traj))
        gates_passed_list.append(max_gate_idx)
        print(f"  Episode {ep+1}/{num_episodes}: {len(traj)} steps, "
              f"{max_gate_idx}/{num_gates} gates"
              f"{' CRASHED' if crashed else ''}")

    return trajectories, gates_passed_list, track_gates, track_name


def plot_eval(trajectories, gates_passed_list, track_gates, track_name):
    """Plot top-down and side views of evaluation trajectories."""
    num_gates = len(track_gates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    cmap = plt.cm.tab10
    eval_gate_size = MonoRaceSimEnv.gate_size_current
    half = max(eval_gate_size / 2.0, 0.5)

    for i in range(num_gates):
        cx, cy, cz, yaw = track_gates[i]
        perp_x = -np.sin(yaw) * half
        perp_y = np.cos(yaw) * half
        ax1.plot([cx - perp_x, cx + perp_x], [cy - perp_y, cy + perp_y],
                 color='gray', linewidth=4, zorder=5, solid_capstyle='round')
        ax1.annotate(f'G{i+1}', (cx, cy), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

        vis_half = max(eval_gate_size / 2.0, 0.3)
        ax2.plot([cx, cx], [cz - vis_half, cz + vis_half],
                 color='gray', linewidth=4, zorder=5, solid_capstyle='round')
        ax2.annotate(f'G{i+1}', (cx, cz - vis_half), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    for i, traj in enumerate(trajectories):
        if len(traj) < 2:
            continue
        color = cmap(i / max(len(trajectories), 1))
        gp = gates_passed_list[i]
        label = f'Ep {i+1}: {gp}/{num_gates} gates, {len(traj)} steps'

        ax1.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.8, linewidth=1.5, label=label)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=6, zorder=4)
        ax1.plot(traj[-1, 0], traj[-1, 1], 'x', color=color, markersize=8, zorder=4)

        ax2.plot(traj[:, 0], traj[:, 2], color=color, alpha=0.8, linewidth=1.5)
        ax2.plot(traj[0, 0], traj[0, 2], 'o', color=color, markersize=6, zorder=4)
        ax2.plot(traj[-1, 0], traj[-1, 2], 'x', color=color, markersize=8, zorder=4)

    total_gates = sum(gates_passed_list)
    max_gates = max(gates_passed_list) if gates_passed_list else 0
    ax1.set_title(f'Track: {track_name} ({num_gates} gates) — {len(trajectories)} episodes, '
                  f'avg {total_gates/max(len(trajectories),1):.1f} gates, max {max_gates}',
                  fontsize=13)
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
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f'eval_{track_name}.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {os.path.abspath(save_path)}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--track", type=str, default=None,
                        choices=list(config.TRACKS.keys()),
                        help="Track to evaluate on (default: active track)")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--all-tracks", action="store_true",
                        help="Evaluate on all available tracks")
    args = parser.parse_args()

    if args.all_tracks:
        for name in config.TRACKS:
            print(f"\n{'='*60}")
            print(f"Evaluating on track: {name} ({len(config.TRACKS[name])} gates)")
            print(f"{'='*60}")
            trajs, gp, tg, tn = evaluate(args.checkpoint, name, args.vecnorm, args.episodes, args.max_steps)
            plot_eval(trajs, gp, tg, tn)
    else:
        track = args.track or config.ACTIVE_TRACK
        print(f"Evaluating {args.checkpoint} on track '{track}' ({len(config.TRACKS[track])} gates)...")
        trajs, gp, tg, tn = evaluate(args.checkpoint, track, args.vecnorm, args.episodes, args.max_steps)
        plot_eval(trajs, gp, tg, tn)
