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
    K_OMEGA, K_X, K_Y, K_X2, K_Y2, K_ALPHA, K_HOR,
    K_P, K_Q, K_R_SPEED, K_R_ACCEL, J_X, J_Y, J_Z,
    MOTOR_TAU, MOTOR_K_CMD, MOTOR_OMEGA_MIN, MOTOR_OMEGA_MAX,
    DT_SIM, OBS_DIM,
    LAMBDA_PROG, LAMBDA_GATE, LAMBDA_RATE, LAMBDA_OFFSET,
    LAMBDA_PERC, LAMBDA_CRASH, V_MAX,
    GATE_POSITIONS, GATE_SIZE, NUM_LAPS, NUM_GATES, MAX_GATE_DISTANCE, TRACKS,
    DR_RANGE, DR_OMEGA_MAX, DR_TAU, MAX_EPISODE_STEPS,
    INIT_X_RANGE, INIT_Y_RANGE, INIT_Z_RANGE, INIT_V_RANGE,
    INIT_RP_RANGE, INIT_YAW_RANGE, INIT_OMEGA_RANGE,
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z,
    H_GROUND, V_GROUND,
    OMEGA_MAX_TERMINATION,
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
    24D gate-relative observation, 7-term reward function, 11-gate track.
    Full M23 system-identified dynamics.
    """
    # Curriculum values (updated by callback via class variables)
    gate_size_current = 1.5    # start large, shrink over training
    spawn_dist_min = 2.0       # start close, widen over training
    spawn_dist_max = 3.0
    speed_lambda = 0.0         # speed bonus weight (0→1 over training)
    gate_lambda = 1.5          # gate reward weight (1.5→3.0 over training)
    track_pool = None          # list of track names to randomize over (None = use default)

    def __init__(self, near_gate_spawn=True):
        super(MonoRaceSimEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.dynamics = QuadcopterDynamics(dt=DT_SIM)
        self.gates = list(GATE_POSITIONS)
        self.num_gates = len(self.gates)
        self.near_gate_spawn = near_gate_spawn
        self.state = None
        self.prev_state_np = None
        self.step_count = 0
        self.current_gate_idx = 0
        self.max_steps = MAX_EPISODE_STEPS
        self.gates_passed = 0
        self.episodes_count = 0
        self.steps_since_gate = 0
        self.ep_gates_passed = 0   # gates passed this episode (for scaling reward)

    # ── Gate frame transforms ──────────────────────────────────────

    def _gate_rotation_inv(self, gate_yaw):
        """Inverse rotation matrix for a gate's yaw (rotation about Z)."""
        c, s = np.cos(-gate_yaw), np.sin(-gate_yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _world_to_gate_frame(self, pos_w, vel_w, euler_w, gate_idx):
        """Transform world-frame position/velocity/euler into gate frame."""
        gx, gy, gz, g_yaw = self.gates[gate_idx % self.num_gates]
        gate_pos = np.array([gx, gy, gz])
        R_inv = self._gate_rotation_inv(g_yaw)

        p_g = R_inv @ (pos_w - gate_pos)
        v_g = R_inv @ vel_w
        euler_g = euler_w.copy()
        euler_g[2] -= g_yaw  # relative yaw
        return p_g, v_g, euler_g

    def _next_gate_in_current_frame(self, gate_idx):
        """Get next gate position and yaw relative to current gate frame."""
        gi = gate_idx % self.num_gates
        gi_next = (gate_idx + 1) % self.num_gates

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

    # ── Observation (24D per paper) ────────────────────────────────

    def _get_obs(self):
        state_np = self.state.cpu().numpy().squeeze(0)
        p_w = state_np[0:3]
        v_w = state_np[3:6]
        q = state_np[6:10]
        w_body = state_np[10:13]
        motor_speeds = state_np[13:17]

        euler = quat_to_euler(q)

        # World-frame angular velocity: w_world = R @ w_body
        qw, qx, qy, qz = q
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        w_world = R @ w_body

        gi = self.current_gate_idx
        p_g, v_g, euler_g = self._world_to_gate_frame(p_w, v_w, euler, gi)
        p_next_g, yaw_next_rel = self._next_gate_in_current_frame(gi)

        # Distance to current gate
        gx, gy, gz, _ = self.gates[gi % self.num_gates]
        dist_gate = np.linalg.norm(p_w - np.array([gx, gy, gz]))

        obs = np.concatenate([
            p_g,                # 3: position in gate frame
            v_g,                # 3: velocity in gate frame
            euler_g,            # 3: euler angles in gate frame
            w_world,            # 3: world angular velocity
            w_body,             # 3: body angular velocity
            motor_speeds / MOTOR_OMEGA_MAX,  # 4: normalized motor speeds
            p_next_g,           # 3: next gate pos in current gate frame
            [yaw_next_rel],     # 1: next gate relative yaw
            [dist_gate],        # 1: distance to current gate
        ]).astype(np.float32)   # total: 24
        return obs

    # ── Gate passage detection ─────────────────────────────────────

    def _check_gate_passage(self, p_prev, p_curr, gate_idx):
        """
        Check if the drone crossed the gate plane between two positions.
        Returns (passed_through, hit_frame, offset_from_center).
        """
        gx, gy, gz, g_yaw = self.gates[gate_idx % self.num_gates]
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

        half = MonoRaceSimEnv.gate_size_current / 2.0
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

        gi = self.current_gate_idx % self.num_gates
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
        r_speed = 0.0
        if passed:
            self.ep_gates_passed += 1
            # Scale gate reward: 1x for first, 2x for second, 3x for third, etc.
            gate_multiplier = self.ep_gates_passed
            r_gate = MonoRaceSimEnv.gate_lambda * gate_multiplier
            r_offset = -LAMBDA_OFFSET * offset
            # Speed bonus: higher reward for fewer steps between gates
            lam = MonoRaceSimEnv.speed_lambda
            if lam > 0 and self.steps_since_gate > 0:
                r_speed = lam * (1.0 - self.steps_since_gate / self.max_steps)
            self.current_gate_idx += 1
            self.gates_passed += 1
            self.steps_since_gate = 0
        elif hit_frame:
            self._crashed = True

        # 2b. Proximity shaping: small reward for being close to gate center
        # Only when within 3m AND moving toward the gate (prevents hovering)
        r_proximity = 0.0
        if dist_curr < 3.0 and progress > 0:
            # Linear bonus: max 0.05 at gate center, 0 at 3m
            r_proximity = 0.05 * (1.0 - dist_curr / 3.0)

        # 3. Angular rate penalty (world-frame angular velocity per paper: ||Ω_k||²)
        qw, qx, qy, qz = q
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        w_world = R @ w_body
        r_rate = -LAMBDA_RATE * np.sum(w_world**2)

        # 4. Perception penalty: angle between camera axis and gate direction
        gate_dir = gate_pos - p
        gate_dist = np.linalg.norm(gate_dir)
        r_perc = 0.0
        if gate_dist > 0.1:
            gate_dir_norm = gate_dir / gate_dist
            # Camera forward = first column of rotation matrix (body x-axis in world frame)
            # qw, qx, qy, qz already extracted above for rate penalty
            cam_fwd = np.array([
                1 - 2*(qy**2 + qz**2),
                2*(qx*qy + qw*qz),
                2*(qx*qz - qw*qy),
            ])
            cos_theta = np.clip(np.dot(cam_fwd, gate_dir_norm), -1.0, 1.0)
            theta_cam = np.arccos(cos_theta)
            if theta_cam > np.radians(45):  # M23: θ_cam = 45° (Table 1)
                r_perc = -LAMBDA_PERC * theta_cam

        self.steps_since_gate += 1
        reward = r_prog + r_gate + r_offset + r_rate + r_perc + r_proximity + r_speed
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

        # Track randomization: pick a random track from the pool each episode
        if MonoRaceSimEnv.track_pool:
            track_name = self.np_random.choice(MonoRaceSimEnv.track_pool)
            self.gates = list(TRACKS[track_name])
            self.num_gates = len(self.gates)

        # Domain randomization per paper: all params independently ±50%
        # (except omega_max ±40%, tau ±55%)
        dr = lambda nom, rng=DR_RANGE: nom * self.np_random.uniform(1 - rng, 1 + rng)
        dr4 = lambda nom, rng=DR_RANGE: [
            nom[i] * self.np_random.uniform(1 - rng, 1 + rng) for i in range(4)
        ]
        params = {
            'k_omega': dr(K_OMEGA),
            'k_x': dr(K_X),
            'k_y': dr(K_Y),
            'k_x2': dr(K_X2),
            'k_y2': dr(K_Y2),
            'k_alpha': dr(K_ALPHA),
            'k_hor': dr(K_HOR),
            'kp': dr4(K_P),
            'kq': dr4(K_Q),
            'kr_speed': dr4(K_R_SPEED),
            'kr_accel': dr4(K_R_ACCEL),
            'Jx': dr(J_X),
            'Jy': dr(J_Y),
            'Jz': dr(J_Z),
            'motor_tau': dr(MOTOR_TAU, DR_TAU),
            'motor_k_cmd': dr(MOTOR_K_CMD),
            'omega_min': dr(MOTOR_OMEGA_MIN),
            'omega_max': dr(MOTOR_OMEGA_MAX, DR_OMEGA_MAX),
        }
        self.dynamics = QuadcopterDynamics(dt=DT_SIM, params=params)

        if self.near_gate_spawn:
            # Near-gate spawn: place drone 3-8m behind a random gate
            target_gate = self.np_random.integers(0, self.num_gates)
            gx, gy, gz, g_yaw = self.gates[target_gate]
            gate_pos = np.array([gx, gy, gz])

            # Gate normal vector
            normal = np.array([np.cos(g_yaw), np.sin(g_yaw), 0.0])

            # Spawn behind the gate (opposite to normal)
            dist = self.np_random.uniform(MonoRaceSimEnv.spawn_dist_min,
                                          MonoRaceSimEnv.spawn_dist_max)
            spawn_pos = gate_pos - normal * dist

            # Position jitter
            spawn_pos[1] += self.np_random.uniform(-1.0, 1.0)
            spawn_pos[2] += self.np_random.uniform(-0.5, 0.5)

            # Clamp to bounds
            px = np.clip(spawn_pos[0], BOUNDS_X[0], BOUNDS_X[1])
            py = np.clip(spawn_pos[1], BOUNDS_Y[0], BOUNDS_Y[1])
            pz = np.clip(spawn_pos[2], BOUNDS_Z[0], BOUNDS_Z[1])

            # Small random velocity
            vx = self.np_random.uniform(-0.5, 0.5)
            vy = self.np_random.uniform(-0.5, 0.5)
            vz = self.np_random.uniform(-0.5, 0.5)

            # Yaw: facing toward the gate with small jitter
            facing_yaw = np.arctan2(gy - py, gx - px)
            yaw = facing_yaw + self.np_random.uniform(-np.radians(20), np.radians(20))

            # Small roll/pitch
            roll = self.np_random.uniform(-np.pi / 9, np.pi / 9)
            pitch = self.np_random.uniform(-np.pi / 9, np.pi / 9)
        else:
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
        # Random motor commands per paper: u₀ ~ U(-1, 1)
        u_init = self.np_random.uniform(-1.0, 1.0, size=4)
        k = MOTOR_K_CMD  # 0.5
        inner = k * u_init**2 + (1 - k) * u_init
        inner = np.clip(inner, 0.0, None)
        motor_cmd_speeds = (MOTOR_OMEGA_MAX - MOTOR_OMEGA_MIN) * np.sqrt(inner) + MOTOR_OMEGA_MIN
        motors = torch.tensor([motor_cmd_speeds], dtype=torch.float32)

        self.state = torch.cat([p_t, v_t, q_t, w_t, motors], dim=-1)
        self.prev_state_np = self.state.cpu().numpy().squeeze(0).copy()
        self.step_count = 0
        self.steps_since_gate = 0
        self.ep_gates_passed = 0
        self._crashed = False

        if self.near_gate_spawn:
            self.current_gate_idx = target_gate
        else:
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
        w_body = state_np[10:13]
        out_of_bounds = (
            p[0] < BOUNDS_X[0] or p[0] > BOUNDS_X[1] or
            p[1] < BOUNDS_Y[0] or p[1] > BOUNDS_Y[1] or
            p[2] < BOUNDS_Z[0]  # ceiling only
        )
        speed = np.linalg.norm(state_np[3:6])
        ground_crash = p[2] >= H_GROUND and speed > V_GROUND
        # Angular velocity termination (paper: 1700 deg/s per axis)
        omega_exceeded = np.any(np.abs(w_body) > OMEGA_MAX_TERMINATION)
        crashed = out_of_bounds or self._crashed or omega_exceeded or ground_crash

        # NaN check (numerical instability)
        if np.any(np.isnan(state_np)):
            crashed = True
            reward = -LAMBDA_CRASH

        if crashed:
            reward -= LAMBDA_CRASH

        # Completed all gates for all laps
        completed = self.current_gate_idx >= self.num_gates * NUM_LAPS

        terminated = crashed or completed
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            self.episodes_count += 1

        obs = self._get_obs()
        # Clamp obs to prevent NaN propagation
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs, float(reward), terminated, truncated, {}


def train_ppo(near_gate_spawn=True):
    """Train M23 policy with PPO via Stable Baselines 3."""
    import shutil
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class GateCurriculumCallback(BaseCallback):
        """Logs gate passages, shrinks gate size, and widens spawn distance."""
        def __init__(self, total_timesteps, gate_start=1.5, gate_end=0.40,
                     dist_start=(2.0, 3.0), dist_end=(2.0, MAX_GATE_DISTANCE),
                     advance_threshold=1.0, advance_step=0.05):
            super().__init__(verbose=0)
            self._total = total_timesteps
            self._gate_start = gate_start
            self._gate_end = gate_end
            self._dist_min_start, self._dist_max_start = dist_start
            self._dist_min_end, self._dist_max_end = dist_end
            self._advance_threshold = advance_threshold  # pass rate to trigger advance
            self._advance_step = advance_step            # how much to bump frac per advance
            self._frac = 0.0                             # adaptive curriculum fraction
            self._last_log = 0

        def _on_step(self):
            # Minimum floor: linear fraction (so curriculum never falls behind time)
            time_frac = min(1.0, self.num_timesteps / self._total)
            frac = max(self._frac, time_frac)

            # Apply curriculum
            new_size = self._gate_start + (self._gate_end - self._gate_start) * frac
            MonoRaceSimEnv.gate_size_current = new_size
            MonoRaceSimEnv.spawn_dist_min = self._dist_min_start + (self._dist_min_end - self._dist_min_start) * frac
            MonoRaceSimEnv.spawn_dist_max = self._dist_max_start + (self._dist_max_end - self._dist_max_start) * frac
            MonoRaceSimEnv.speed_lambda = frac
            MonoRaceSimEnv.gate_lambda = 1.5 + 1.5 * frac  # 1.5 → 3.0

            # Log every 100K steps + check pass rate for adaptive advance
            if self.num_timesteps - self._last_log >= 100_000:
                self._last_log = self.num_timesteps
                try:
                    gates_list = self.training_env.get_attr('gates_passed')
                    eps_list = self.training_env.get_attr('episodes_count')
                    total_gates = sum(gates_list)
                    total_eps = sum(eps_list)
                    for i in range(len(gates_list)):
                        self.training_env.env_method('__setattr__', 'gates_passed', 0, indices=[i])
                        self.training_env.env_method('__setattr__', 'episodes_count', 0, indices=[i])
                except Exception:
                    total_gates = 0
                    total_eps = 0
                rate = total_gates / max(total_eps, 1)

                # Adaptive: if pass rate >= threshold, advance curriculum
                advanced = ""
                if rate >= self._advance_threshold and frac < 1.0:
                    self._frac = min(1.0, frac + self._advance_step)
                    advanced = " ^"

                # Log to tensorboard
                self.logger.record("curriculum/gate_pass_rate", rate * 100)
                self.logger.record("curriculum/gate_size", new_size)
                self.logger.record("curriculum/spawn_dist_max", MonoRaceSimEnv.spawn_dist_max)
                self.logger.record("curriculum/frac", frac)
                self.logger.record("curriculum/gate_lambda", MonoRaceSimEnv.gate_lambda)
                self.logger.record("curriculum/speed_lambda", MonoRaceSimEnv.speed_lambda)

                print(f"  [Gate] Steps: {self.num_timesteps:>10,} | "
                      f"Size: {new_size:.2f}m | "
                      f"Spawn: {MonoRaceSimEnv.spawn_dist_min:.1f}-{MonoRaceSimEnv.spawn_dist_max:.1f}m | "
                      f"Speed: {MonoRaceSimEnv.speed_lambda:.2f} | "
                      f"GLam: {MonoRaceSimEnv.gate_lambda:.2f} | "
                      f"Frac: {frac:.2f}{advanced} | "
                      f"Passed: {total_gates}/{total_eps} eps ({rate:.1%})",
                      flush=True)
            return True

    save_dir = "D:/drone2_training/latest"

    # Wipe previous run data
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Cleaned up previous run in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    # Track randomization: train on multiple track layouts
    MonoRaceSimEnv.track_pool = ["kidney", "figure8", "expert"]
    print(f"Track pool: {MonoRaceSimEnv.track_pool}")

    spawn_mode = "near-gate" if near_gate_spawn else "uniform"
    print(f"Setting up MonoRace M23 Training Environment (spawn: {spawn_mode})...")

    def make_env(ngs):
        def _init():
            return MonoRaceSimEnv(near_gate_spawn=ngs)
        return _init

    env = SubprocVecEnv([make_env(near_gate_spawn) for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        net_arch=[64, 64, 64],
        activation_fn=nn.ReLU,
    )

    # Linear LR decay: 3e-4 → 0 over training
    def lr_schedule(progress_remaining):
        return 3e-4 * progress_remaining

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        device="cpu",
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000 // NUM_ENVS,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="gcnet_m23",
    )

    class VecNormSaveCallback(BaseCallback):
        """Save VecNormalize stats alongside each checkpoint."""
        def __init__(self, save_freq, save_path):
            super().__init__(verbose=0)
            self._save_freq = save_freq
            self._save_path = save_path
        def _on_step(self):
            if self.num_timesteps % self._save_freq < self.training_env.num_envs:
                vn_path = os.path.join(self._save_path,
                    f"vecnormalize_{self.num_timesteps}_steps.pkl")
                self.model.get_vec_normalize_env().save(vn_path)
            return True

    vecnorm_cb = VecNormSaveCallback(
        save_freq=1_000_000,
        save_path=os.path.join(save_dir, "checkpoints"),
    )
    gate_cb = GateCurriculumCallback(TOTAL_TIMESTEPS, gate_start=1.5, gate_end=GATE_SIZE)

    print(f"Starting PPO Training ({TOTAL_TIMESTEPS:,} timesteps, {NUM_ENVS} envs)...")
    print(f"Gate curriculum: 1.5m -> {GATE_SIZE}m over training")
    print(f"Spawn distance curriculum: 2-3m -> 2-{MAX_GATE_DISTANCE:.1f}m over training")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, vecnorm_cb, gate_cb],
    )

    model.save(os.path.join(save_dir, "gcnet_m23_final"))
    env.save(os.path.join(save_dir, "vecnormalize_m23.pkl"))
    print(f"Training complete. Model saved to {save_dir}")
    print('To keep this run: python control/save_run.py "description"')


def resume_training(checkpoint_path, total_remaining=10_000_000, near_gate_spawn=True):
    """Resume PPO training from a checkpoint."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    save_dir = "D:/drone2_training/latest"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    spawn_mode = "near-gate" if near_gate_spawn else "uniform"
    print(f"Resuming from {checkpoint_path} (spawn: {spawn_mode})...")

    def make_env(ngs):
        def _init():
            return MonoRaceSimEnv(near_gate_spawn=ngs)
        return _init

    env = SubprocVecEnv([make_env(near_gate_spawn) for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO.load(checkpoint_path, env=env, device="cpu")

    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000 // NUM_ENVS,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="gcnet_m23",
    )

    print(f"Continuing for {total_remaining:,} more timesteps...")
    model.learn(
        total_timesteps=total_remaining,
        callback=checkpoint_cb,
        reset_num_timesteps=False,
    )

    model.save(os.path.join(save_dir, "gcnet_m23_final"))
    env.save(os.path.join(save_dir, "vecnormalize_m23.pkl"))
    print(f"Training complete. Model saved to {save_dir}")
    print('To keep this run: python control/save_run.py "description"')


def _cleanup_children():
    """Kill any leftover subprocess workers on exit."""
    import multiprocessing
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)


if __name__ == "__main__":
    import argparse
    import signal
    import atexit

    # Ensure child processes get cleaned up no matter what
    atexit.register(_cleanup_children)
    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted — cleaning up..."), _cleanup_children(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup_children(), sys.exit(1)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--remaining", type=int, default=10_000_000,
                        help="Remaining timesteps when resuming")
    parser.add_argument("--uniform-spawn", action="store_true",
                        help="Use uniform random spawn across hall (default: near-gate spawn)")
    args = parser.parse_args()

    near_gate_spawn = not args.uniform_spawn

    try:
        if args.resume:
            resume_training(args.resume, args.remaining, near_gate_spawn=near_gate_spawn)
        else:
            train_ppo(near_gate_spawn=near_gate_spawn)
    finally:
        _cleanup_children()
