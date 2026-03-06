import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from control.dynamics import QuadcopterDynamics
from control.utils import quat_to_euler, euler_to_quat
from config import (
    MASS, J, K_OMEGA, K_DRAG, ARM_LENGTH, MOTOR_TAU, MOTOR_K,
    MOTOR_OMEGA_MIN, MOTOR_OMEGA_MAX, DT_SIM, OBS_DIM,
    LAMBDA_PROG, LAMBDA_GATE, LAMBDA_RATE, LAMBDA_OFFSET,
    LAMBDA_PERC, LAMBDA_CRASH, V_MAX,
    GATE_POSITIONS, GATE_SIZE, NUM_LAPS, NUM_GATES,
    DR_RANGE, MAX_EPISODE_STEPS,
    INIT_X_RANGE, INIT_Y_RANGE, INIT_Z_RANGE, INIT_V_RANGE,
    INIT_RP_RANGE, INIT_YAW_RANGE, INIT_OMEGA_RANGE,
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z,
    NUM_ENVS, TOTAL_TIMESTEPS, CHECKPOINT_FREQ,
)


class GCNet(nn.Module):
    """
    Guidance & Control Network (M23 architecture).
    3 FC layers, 64 neurons, ReLU activation, Sigmoid output.
    """
    def __init__(self, state_dim=OBS_DIM, action_dim=4):
        super(GCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class MonoRaceSimEnv(gym.Env):
    """
    Gymnasium environment for M23 PPO training.
    20D gate-relative observation, 7-term reward function, 11-gate track.
    """
    def __init__(self):
        super(MonoRaceSimEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.dynamics = QuadcopterDynamics(dt=DT_SIM)
        self.gates = GATE_POSITIONS
        self.state = None
        self.prev_state_np = None
        self.step_count = 0
        self.current_gate_idx = 0
        self.max_steps = MAX_EPISODE_STEPS

    # ── Gate frame transforms ──────────────────────────────────────

    def _gate_rotation_inv(self, gate_yaw):
        """Inverse rotation matrix for a gate's yaw (rotation about Z)."""
        c, s = np.cos(-gate_yaw), np.sin(-gate_yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _world_to_gate_frame(self, pos_w, vel_w, euler_w, gate_idx):
        """Transform world-frame position/velocity/euler into gate frame."""
        gx, gy, gz, g_yaw = self.gates[gate_idx % NUM_GATES]
        gate_pos = np.array([gx, gy, gz])
        R_inv = self._gate_rotation_inv(g_yaw)

        p_g = R_inv @ (pos_w - gate_pos)
        v_g = R_inv @ vel_w
        euler_g = euler_w.copy()
        euler_g[2] -= g_yaw  # relative yaw
        return p_g, v_g, euler_g

    def _next_gate_in_current_frame(self, gate_idx):
        """Get next gate position and yaw relative to current gate frame."""
        gi = gate_idx % NUM_GATES
        gi_next = (gate_idx + 1) % NUM_GATES

        gx, gy, gz, g_yaw = self.gates[gi]
        gx_n, gy_n, gz_n, g_yaw_n = self.gates[gi_next]

        gate_pos = np.array([gx, gy, gz])
        next_pos = np.array([gx_n, gy_n, gz_n])
        R_inv = self._gate_rotation_inv(g_yaw)

        p_next_g = R_inv @ (next_pos - gate_pos)
        yaw_rel = g_yaw_n - g_yaw
        # Wrap to [-pi, pi]
        yaw_rel = (yaw_rel + np.pi) % (2 * np.pi) - np.pi
        return p_next_g, yaw_rel

    # ── Observation ────────────────────────────────────────────────

    def _get_obs(self):
        state_np = self.state.cpu().numpy().squeeze(0)
        p_w = state_np[0:3]
        v_w = state_np[3:6]
        q = state_np[6:10]
        w_body = state_np[10:13]
        motor_speeds = state_np[13:17]

        euler = quat_to_euler(q)

        gi = self.current_gate_idx
        p_g, v_g, euler_g = self._world_to_gate_frame(p_w, v_w, euler, gi)
        p_next_g, yaw_next_rel = self._next_gate_in_current_frame(gi)

        obs = np.concatenate([
            p_g,                # 3: position in gate frame
            v_g,                # 3: velocity in gate frame
            euler_g,            # 3: euler angles in gate frame
            w_body,             # 3: body angular velocity
            motor_speeds / MOTOR_OMEGA_MAX,  # 4: normalized motor speeds
            p_next_g,           # 3: next gate pos in current gate frame
            [yaw_next_rel],     # 1: next gate relative yaw
        ]).astype(np.float32)
        return obs

    # ── Gate passage detection ─────────────────────────────────────

    def _check_gate_passage(self, p_prev, p_curr, gate_idx):
        """
        Check if the drone crossed the gate plane between two positions.
        Returns (passed_through, hit_frame, offset_from_center).
        """
        gx, gy, gz, g_yaw = self.gates[gate_idx % NUM_GATES]
        gate_pos = np.array([gx, gy, gz])

        # Gate normal (direction the gate faces)
        normal = np.array([np.cos(g_yaw), np.sin(g_yaw), 0.0])

        # Signed distance to gate plane
        d_prev = np.dot(p_prev - gate_pos, normal)
        d_curr = np.dot(p_curr - gate_pos, normal)

        # No crossing if same side
        if d_prev * d_curr > 0:
            return False, False, 0.0

        # Interpolate crossing point
        t = d_prev / (d_prev - d_curr + 1e-8)
        p_cross = p_prev + t * (p_curr - p_prev)

        # Offset from gate center in gate-local coords
        R_inv = self._gate_rotation_inv(g_yaw)
        local = R_inv @ (p_cross - gate_pos)

        half = GATE_SIZE / 2.0
        within_y = abs(local[1]) < half
        within_z = abs(local[2]) < half
        offset = np.sqrt(local[1]**2 + local[2]**2)

        if within_y and within_z:
            return True, False, offset   # clean pass
        else:
            return False, True, offset   # hit gate frame = crash

    # ── Reward function (M23) ──────────────────────────────────────

    def _compute_reward(self, state_np, prev_state_np, action):
        p = state_np[0:3]
        p_prev = prev_state_np[0:3]
        w_body = state_np[10:13]
        q = state_np[6:10]

        gi = self.current_gate_idx % NUM_GATES
        gate_pos = np.array(self.gates[gi][:3])

        # 1. Progress reward: getting closer to current gate
        dist_prev = np.linalg.norm(p_prev - gate_pos)
        dist_curr = np.linalg.norm(p - gate_pos)
        progress = dist_prev - dist_curr
        r_prog = LAMBDA_PROG * min(progress, V_MAX * DT_SIM)

        # 2. Gate passage check
        r_gate = 0.0
        r_offset = 0.0
        passed, hit_frame, offset = self._check_gate_passage(p_prev, p, gi)
        if passed:
            r_gate = LAMBDA_GATE
            r_offset = -LAMBDA_OFFSET * offset
            self.current_gate_idx += 1
        elif hit_frame:
            self._crashed = True

        # 3. Angular rate penalty
        r_rate = -LAMBDA_RATE * np.sum(w_body**2)

        # 4. Perception penalty: angle between camera axis and gate direction
        euler = quat_to_euler(q)
        gate_dir = gate_pos - p
        gate_dist = np.linalg.norm(gate_dir)
        r_perc = 0.0
        if gate_dist > 0.1:
            gate_dir_norm = gate_dir / gate_dist
            # Camera points along body X-axis; compute angle from gate direction
            # Simplified: use pitch and roll as proxy for camera off-axis angle
            cam_off_angle = np.sqrt(euler[0]**2 + euler[1]**2)
            if cam_off_angle > (np.pi / 3):
                r_perc = -LAMBDA_PERC * cam_off_angle

        reward = r_prog + r_gate + r_offset + r_rate + r_perc
        return reward

    # ── Find nearest gate ahead ────────────────────────────────────

    def _find_nearest_gate_ahead(self, pos):
        """Find the nearest gate that is ahead of the drone (behind its plane)."""
        best_idx = 0
        best_dist = np.inf
        for i, (gx, gy, gz, g_yaw) in enumerate(self.gates):
            gate_pos = np.array([gx, gy, gz])
            normal = np.array([np.cos(g_yaw), np.sin(g_yaw), 0.0])
            # Drone is "behind" gate if signed distance is negative
            signed_dist = np.dot(pos - gate_pos, normal)
            if signed_dist < 0:
                dist = np.linalg.norm(pos - gate_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    # ── Environment interface ──────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Domain randomization: 50% on all dynamics params
        dr = lambda nom: nom * self.np_random.uniform(1 - DR_RANGE, 1 + DR_RANGE)
        params = {
            'mass': dr(MASS),
            'J': torch.tensor([dr(J[0]), dr(J[1]), dr(J[2])], dtype=torch.float32),
            'k_omega': dr(K_OMEGA),
            'k_drag': dr(K_DRAG),
            'arm_length': dr(ARM_LENGTH),
            'motor_tau': dr(MOTOR_TAU),
            'motor_k': dr(MOTOR_K),
        }
        self.dynamics = QuadcopterDynamics(dt=DT_SIM, params=params)

        # M23 "uniform" initialization
        px = self.np_random.uniform(*INIT_X_RANGE)
        py = self.np_random.uniform(*INIT_Y_RANGE)
        pz = self.np_random.uniform(*INIT_Z_RANGE)
        vx = self.np_random.uniform(*INIT_V_RANGE)
        vy = self.np_random.uniform(*INIT_V_RANGE)
        vz = self.np_random.uniform(*INIT_V_RANGE)
        roll = self.np_random.uniform(*INIT_RP_RANGE)
        pitch = self.np_random.uniform(*INIT_RP_RANGE)
        yaw = self.np_random.uniform(*INIT_YAW_RANGE)
        q = euler_to_quat(roll, pitch, yaw)
        wx = self.np_random.uniform(*INIT_OMEGA_RANGE)
        wy = self.np_random.uniform(*INIT_OMEGA_RANGE)
        wz = self.np_random.uniform(*INIT_OMEGA_RANGE)

        p_t = torch.tensor([[px, py, pz]], dtype=torch.float32)
        v_t = torch.tensor([[vx, vy, vz]], dtype=torch.float32)
        q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        w_t = torch.tensor([[wx, wy, wz]], dtype=torch.float32)
        motors = torch.full((1, 4), MOTOR_OMEGA_MIN, dtype=torch.float32)

        self.state = torch.cat([p_t, v_t, q_t, w_t, motors], dim=-1)
        self.prev_state_np = self.state.cpu().numpy().squeeze(0).copy()
        self.step_count = 0
        self._crashed = False
        self.current_gate_idx = self._find_nearest_gate_ahead(np.array([px, py, pz]))

        return self._get_obs(), {}

    def step(self, action):
        self.prev_state_np = self.state.cpu().numpy().squeeze(0).copy()

        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            self.state = self.dynamics(self.state, action_tensor)
        self.step_count += 1

        state_np = self.state.cpu().numpy().squeeze(0)

        # Reward
        reward = self._compute_reward(state_np, self.prev_state_np, action)

        # Termination checks
        p = state_np[0:3]
        out_of_bounds = (
            p[0] < BOUNDS_X[0] or p[0] > BOUNDS_X[1] or
            p[1] < BOUNDS_Y[0] or p[1] > BOUNDS_Y[1] or
            p[2] < BOUNDS_Z[0] or p[2] > BOUNDS_Z[1]
        )
        crashed = out_of_bounds or self._crashed

        # NaN check (numerical instability)
        if np.any(np.isnan(state_np)):
            crashed = True
            reward = -LAMBDA_CRASH

        if crashed:
            reward -= LAMBDA_CRASH

        # Completed all gates for all laps
        completed = self.current_gate_idx >= NUM_GATES * NUM_LAPS

        terminated = crashed or completed
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        # Clamp obs to prevent NaN propagation
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs, float(reward), terminated, truncated, {}


def train_ppo():
    """Train M23 policy with PPO via Stable Baselines 3."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    print("Setting up MonoRace M23 Training Environment...")

    def make_env():
        def _init():
            return MonoRaceSimEnv()
        return _init

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        net_arch=[64, 64, 64],
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",  # MLP policy is faster on CPU than GPU
        tensorboard_log="./tb_logs/",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000 // NUM_ENVS,  # save every 1M steps (less disk usage)
        save_path="D:/drone2_training/checkpoints/",
        name_prefix="gcnet_m23",
    )

    print(f"Starting PPO Training ({TOTAL_TIMESTEPS:,} timesteps, {NUM_ENVS} envs)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
    )

    model.save("D:/drone2_training/gcnet_m23_final")
    env.save("D:/drone2_training/vecnormalize_m23.pkl")
    print("Training complete. Model saved to D:/drone2_training/gcnet_m23_final.zip")


def resume_training(checkpoint_path, total_remaining=10_000_000):
    """Resume PPO training from a checkpoint."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    print(f"Resuming from {checkpoint_path}...")

    def make_env():
        def _init():
            return MonoRaceSimEnv()
        return _init

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO.load(checkpoint_path, env=env, device="cpu")

    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000 // NUM_ENVS,
        save_path="D:/drone2_training/checkpoints/",
        name_prefix="gcnet_m23",
    )

    print(f"Continuing for {total_remaining:,} more timesteps...")
    model.learn(
        total_timesteps=total_remaining,
        callback=checkpoint_cb,
        reset_num_timesteps=False,
    )

    model.save("D:/drone2_training/gcnet_m23_final")
    env.save("D:/drone2_training/vecnormalize_m23.pkl")
    print("Training complete. Model saved to D:/drone2_training/gcnet_m23_final.zip")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--remaining", type=int, default=10_000_000,
                        help="Remaining timesteps when resuming")
    args = parser.parse_args()

    if args.resume:
        resume_training(args.resume, args.remaining)
    else:
        train_ppo()
