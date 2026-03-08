"""
GPU-vectorized drone racing environment.
All N environments run as a single batched tensor on CUDA.
No numpy, no Gymnasium — pure PyTorch for maximum throughput.
"""
import torch
import torch.nn.functional as F
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    MASS, K_OMEGA, K_X, K_Y, K_X2, K_Y2, K_ALPHA, K_HOR, PROP_RADIUS,
    K_P, K_Q, K_R_SPEED, K_R_ACCEL, J_X, J_Y, J_Z,
    MOTOR_TAU, MOTOR_K_CMD, MOTOR_OMEGA_MIN, MOTOR_OMEGA_MAX,
    GRAVITY, OBS_DIM,
    LAMBDA_RATE, LAMBDA_OFFSET,
    LAMBDA_PERC, LAMBDA_CRASH, V_MAX,
    GATE_POSITIONS, GATE_SIZE, NUM_LAPS, NUM_GATES,
    DR_RANGE, DR_OMEGA_MAX, DR_TAU, MAX_EPISODE_STEPS,
    INIT_X_RANGE, INIT_Y_RANGE, INIT_Z_RANGE, INIT_V_RANGE,
    INIT_RP_RANGE, INIT_YAW_RANGE, INIT_OMEGA_RANGE,
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z,
    H_GROUND, V_GROUND, OMEGA_MAX_TERMINATION,
)

# GPU experimental overrides (differ from paper values in config.py)
DT_SIM = 0.005          # higher fidelity than paper's 0.01
LAMBDA_PROG = 1.5       # boosted from paper's 1.0
LAMBDA_GATE = 10.0      # boosted from paper's 1.5
LAMBDA_ALIGN = 10.0     # trajectory alignment (not in paper)


# ── Batched quaternion/euler helpers ──────────────────────────────

def batch_quat_to_euler(q):
    """Quaternion [N, 4] (w,x,y,z) -> Euler [N, 3] (roll, pitch, yaw)."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = torch.asin(torch.clamp(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return torch.stack([roll, pitch, yaw], dim=-1)


def batch_euler_to_quat(euler):
    """Euler [N, 3] (roll, pitch, yaw) -> Quaternion [N, 4] (w,x,y,z)."""
    r, p, y = euler[:, 0] / 2, euler[:, 1] / 2, euler[:, 2] / 2
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=-1)


# ── Batched dynamics (system-identified model from paper) ─────────

def batched_dynamics_step(state, action, params, prev_motor_speeds, dt):
    """
    Step dynamics for all envs using system-identified aero model.
    state: [N, 17], action: [N, 4], params: dict of per-env tensors.
    prev_motor_speeds: [N, 4] from previous step (for yaw accel term).
    Returns: (new_state [N, 17], new_motor_speeds [N, 4] for next step's accel calc)
    """
    p = state[:, 0:3]
    v = state[:, 3:6]
    q = state[:, 6:10]
    w_body = state[:, 10:13]
    motor_speeds = state[:, 13:17]
    dev = state.device
    N = state.shape[0]

    # Unpack params
    k_omega = params['k_omega']         # [N]
    k_x = params['k_x']                 # [N]
    k_y = params['k_y']                 # [N]
    k_x2 = params['k_x2']              # [N]
    k_y2 = params['k_y2']              # [N]
    k_alpha = params['k_alpha']         # [N]
    k_hor = params['k_hor']             # [N]
    kp = params['kp']                   # [N, 4]
    kq = params['kq']                   # [N, 4]
    kr_speed = params['kr_speed']       # [N, 4]
    kr_accel = params['kr_accel']       # [N, 4]
    Jx = params['Jx']                   # [N]
    Jy = params['Jy']                   # [N]
    Jz = params['Jz']                   # [N]
    motor_tau = params['motor_tau']     # [N]
    omega_min = params['omega_min']     # [N]
    omega_max = params['omega_max']     # [N]
    motor_k_cmd = params['motor_k_cmd'] # [N]

    # ── Motor command model (nonlinear) ──
    # omega_c = (omega_max - omega_min) * sqrt(k*u² + (1-k)*u) + omega_min
    k_cmd = motor_k_cmd.unsqueeze(-1)          # [N, 1]
    omin = omega_min.unsqueeze(-1)              # [N, 1]
    omax = omega_max.unsqueeze(-1)              # [N, 1]
    u = action                                  # [N, 4] in [0, 1]
    inner = k_cmd * u * u + (1 - k_cmd) * u    # [N, 4]
    inner = torch.clamp(inner, min=0.0)         # safety for sqrt
    cmd_speeds = (omax - omin) * torch.sqrt(inner) + omin

    # Motor dynamics: first-order lag
    motor_dot = (cmd_speeds - motor_speeds) / motor_tau.unsqueeze(-1)
    motor_new = motor_speeds + motor_dot * dt
    motor_new = torch.clamp(motor_new, min=omin, max=omax)

    # Motor acceleration (for yaw moment)
    motor_accel = (motor_new - prev_motor_speeds) / dt  # [N, 4]

    # ── Body-frame velocity (for aero forces) ──
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Rotation matrix R (body-to-world), we need R^T (world-to-body)
    R = torch.empty((N, 3, 3), device=dev, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx*qy - qw*qz)
    R[:, 0, 2] = 2 * (qx*qz + qw*qy)
    R[:, 1, 0] = 2 * (qx*qy + qw*qz)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy*qz - qw*qx)
    R[:, 2, 0] = 2 * (qx*qz - qw*qy)
    R[:, 2, 1] = 2 * (qy*qz + qw*qx)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)

    # v_body = R^T @ v_world
    v_body = torch.bmm(R.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)
    vx_B, vy_B, vz_B = v_body[:, 0], v_body[:, 1], v_body[:, 2]

    # ── Aero force model (specific accelerations in body frame) ──
    omega_sum = motor_new.sum(dim=-1)          # Σωi (linear sum for drag)
    omega_sq_sum = (motor_new**2).sum(dim=-1)  # Σωi² (for thrust)

    # Advance ratios
    omega_bar = omega_sum.clamp(min=1.0)       # avoid division by zero
    r_prop = PROP_RADIUS                       # scalar, not randomized
    alpha = torch.atan2(vz_B, r_prop * omega_bar)
    v_hor = torch.sqrt(vx_B**2 + vy_B**2 + 1e-8)
    mu = torch.atan2(v_hor, r_prop * omega_bar)

    # Body-frame specific forces
    Fx = -k_x * vx_B * omega_sum - k_x2 * vx_B * vx_B.abs()
    Fy = -k_y * vy_B * omega_sum - k_y2 * vy_B * vy_B.abs()
    Fz = -k_omega * (1 + k_alpha * alpha + k_hor * mu) * omega_sq_sum

    F_body = torch.stack([Fx, Fy, Fz], dim=-1)  # [N, 3]

    # ── Translational dynamics (world frame) ──
    F_world = torch.bmm(R, F_body.unsqueeze(-1)).squeeze(-1)
    g = torch.tensor([0.0, 0.0, GRAVITY], device=dev)
    v_dot = F_world + g  # NO mass division — specific accelerations
    v_new = v + v_dot * dt
    p_new = p + v_new * dt

    # ── Moment model (specific angular accelerations) ──
    w1, w2, w3, w4 = motor_new[:, 0], motor_new[:, 1], motor_new[:, 2], motor_new[:, 3]
    w1sq, w2sq, w3sq, w4sq = w1**2, w2**2, w3**2, w4**2

    p_rate, q_rate, r_rate = w_body[:, 0], w_body[:, 1], w_body[:, 2]

    # Roll: M_x = -kp1*ω1² - kp2*ω2² + kp3*ω3² + kp4*ω4² + Jx*q_rate*r_rate
    Mx = (-kp[:, 0]*w1sq - kp[:, 1]*w2sq + kp[:, 2]*w3sq + kp[:, 3]*w4sq
           + Jx * q_rate * r_rate)

    # Pitch: M_y = -kq1*ω1² + kq2*ω2² - kq3*ω3² + kq4*ω4² + Jy*p_rate*r_rate
    My = (-kq[:, 0]*w1sq + kq[:, 1]*w2sq - kq[:, 2]*w3sq + kq[:, 3]*w4sq
           + Jy * p_rate * r_rate)

    # Yaw: M_z = -kr1*ω1 + kr2*ω2 + kr3*ω3 - kr4*ω4
    #            - kr5*ω̇1 + kr6*ω̇2 + kr7*ω̇3 - kr8*ω̇4 + Jz*p_rate*q_rate
    wd1, wd2, wd3, wd4 = motor_accel[:, 0], motor_accel[:, 1], motor_accel[:, 2], motor_accel[:, 3]
    Mz = (-kr_speed[:, 0]*w1 + kr_speed[:, 1]*w2 + kr_speed[:, 2]*w3 - kr_speed[:, 3]*w4
           - kr_accel[:, 0]*wd1 + kr_accel[:, 1]*wd2 + kr_accel[:, 2]*wd3 - kr_accel[:, 3]*wd4
           + Jz * p_rate * q_rate)

    w_dot = torch.stack([Mx, My, Mz], dim=-1)  # NO inertia division, NO w×Jw
    w_new = w_body + w_dot * dt

    # ── Quaternion integration ──
    wx, wy, wz = w_new[:, 0], w_new[:, 1], w_new[:, 2]
    q_dot = torch.empty_like(q)
    q_dot[:, 0] = 0.5 * (-qx*wx - qy*wy - qz*wz)
    q_dot[:, 1] = 0.5 * (qw*wx + qy*wz - qz*wy)
    q_dot[:, 2] = 0.5 * (qw*wy - qx*wz + qz*wx)
    q_dot[:, 3] = 0.5 * (qw*wz + qx*wy - qy*wx)
    q_new = q + q_dot * dt
    q_new = q_new / q_new.norm(dim=-1, keepdim=True)

    new_state = torch.cat([p_new, v_new, q_new, w_new, motor_new], dim=-1)
    return new_state


# ── Vectorized environment ───────────────────────────────────────

class BatchedDroneEnv:
    """
    GPU-vectorized drone racing environment.
    Manages N parallel environments as batched CUDA tensors.
    """
    def __init__(self, num_envs=2048, device="cuda", fixed_start=False,
                 domain_randomize=True, single_gate=False, gate_size_override=None,
                 random_segments=False):
        self.N = num_envs
        self.device = torch.device(device)
        self.dt = DT_SIM
        self.obs_dim = OBS_DIM
        self.act_dim = 4
        self.fixed_start = fixed_start
        self.domain_randomize = domain_randomize
        self.single_gate = single_gate
        self.random_segments = random_segments
        if random_segments:
            self.single_gate = True  # each env does one gate passage
        self.gate_size = gate_size_override if gate_size_override is not None else GATE_SIZE
        # Segment training: (from_gate_idx, to_gate_idx) or None
        self.segment = None
        # Gate passage tracking
        self.gates_passed_count = 0
        self.episodes_ended_count = 0

        # Precompute gate data as tensors
        gp = torch.tensor([[g[0], g[1], g[2]] for g in GATE_POSITIONS],
                          dtype=torch.float32, device=self.device)  # [G, 3]
        gy = torch.tensor([g[3] for g in GATE_POSITIONS],
                          dtype=torch.float32, device=self.device)  # [G]
        self.gates_pos = gp
        self.gates_yaw = gy
        self.gates_normal = torch.stack([torch.cos(gy), torch.sin(gy),
                                         torch.zeros_like(gy)], dim=-1)  # [G, 3]

        # Precompute inverse rotation matrices for each gate
        R_inv = torch.zeros((NUM_GATES, 3, 3), device=self.device)
        for i in range(NUM_GATES):
            c = torch.cos(-gy[i])
            s = torch.sin(-gy[i])
            R_inv[i] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                                     device=self.device)
        self.gates_R_inv = R_inv

        # Precompute next-gate-in-current-frame data
        self.next_gate_local = torch.zeros((NUM_GATES, 3), device=self.device)
        self.next_gate_yaw_rel = torch.zeros(NUM_GATES, device=self.device)
        for i in range(NUM_GATES):
            j = (i + 1) % NUM_GATES
            local = R_inv[i] @ (gp[j] - gp[i])
            self.next_gate_local[i] = local
            yaw_rel = gy[j] - gy[i]
            self.next_gate_yaw_rel[i] = (yaw_rel + math.pi) % (2 * math.pi) - math.pi

        # Environment state
        self.states = torch.zeros((num_envs, 17), device=self.device)
        self.prev_states = torch.zeros_like(self.states)
        self.prev_motor_speeds = torch.full((num_envs, 4), MOTOR_OMEGA_MIN, device=self.device)
        self.gate_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.step_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Per-env domain randomization parameters (system-identified model)
        self.env_k_omega = torch.full((num_envs,), K_OMEGA, device=self.device)
        self.env_k_x = torch.full((num_envs,), K_X, device=self.device)
        self.env_k_y = torch.full((num_envs,), K_Y, device=self.device)
        self.env_k_x2 = torch.full((num_envs,), K_X2, device=self.device)
        self.env_k_y2 = torch.full((num_envs,), K_Y2, device=self.device)
        self.env_k_alpha = torch.full((num_envs,), K_ALPHA, device=self.device)
        self.env_k_hor = torch.full((num_envs,), K_HOR, device=self.device)
        self.env_kp = torch.tensor(K_P, dtype=torch.float32, device=self.device).unsqueeze(0).expand(num_envs, 4).clone()
        self.env_kq = torch.tensor(K_Q, dtype=torch.float32, device=self.device).unsqueeze(0).expand(num_envs, 4).clone()
        self.env_kr_speed = torch.tensor(K_R_SPEED, dtype=torch.float32, device=self.device).unsqueeze(0).expand(num_envs, 4).clone()
        self.env_kr_accel = torch.tensor(K_R_ACCEL, dtype=torch.float32, device=self.device).unsqueeze(0).expand(num_envs, 4).clone()
        self.env_Jx = torch.full((num_envs,), J_X, device=self.device)
        self.env_Jy = torch.full((num_envs,), J_Y, device=self.device)
        self.env_Jz = torch.full((num_envs,), J_Z, device=self.device)
        self.env_motor_tau = torch.full((num_envs,), MOTOR_TAU, device=self.device)
        self.env_omega_min = torch.full((num_envs,), MOTOR_OMEGA_MIN, device=self.device)
        self.env_omega_max = torch.full((num_envs,), MOTOR_OMEGA_MAX, device=self.device)
        self.env_motor_k_cmd = torch.full((num_envs,), MOTOR_K_CMD, device=self.device)

        # Curriculum difficulty: 0.0 = easy (near gate), 1.0 = full random
        self.difficulty = 0.0

        # Hover motor speed: sqrt(g / (4 * k_omega)) — no mass because identified model
        self.hover_omega = math.sqrt(GRAVITY / (4 * K_OMEGA))

        # Running observation normalization
        self.obs_mean = torch.zeros(OBS_DIM, device=self.device)
        self.obs_var = torch.ones(OBS_DIM, device=self.device)
        self.obs_count = 1e-4

        self.reset_all()

    def set_difficulty(self, d):
        """Set curriculum difficulty: 0.0 = easy (near gate), 1.0 = full random."""
        self.difficulty = max(0.0, min(1.0, d))

    def set_segment(self, from_gate, to_gate):
        """Set segment training: spawn at from_gate exit, target to_gate.
        from_gate=None means spawn before gate 0 (start position)."""
        self.segment = (from_gate, to_gate)
        self.single_gate = True

    def _randomize_params(self, mask):
        """Domain randomization for envs indicated by mask (per paper: all params independently)."""
        n = mask.sum().item()
        if n == 0:
            return
        if not self.domain_randomize:
            return  # Keep nominal params
        dev = self.device
        dr = lambda nom, rng=DR_RANGE: nom * (1 + rng * (2 * torch.rand(n, device=dev) - 1))
        dr4 = lambda nom, rng=DR_RANGE: torch.tensor(nom, device=dev, dtype=torch.float32) * (1 + rng * (2 * torch.rand(n, 4, device=dev) - 1))

        # Force coefficients (±50%)
        self.env_k_omega[mask] = dr(K_OMEGA)
        self.env_k_x[mask] = dr(K_X)
        self.env_k_y[mask] = dr(K_Y)
        self.env_k_x2[mask] = dr(K_X2)
        self.env_k_y2[mask] = dr(K_Y2)
        self.env_k_alpha[mask] = dr(K_ALPHA)
        self.env_k_hor[mask] = dr(K_HOR)

        # Per-motor moment coefficients (each independently ±50%)
        self.env_kp[mask] = dr4(K_P)
        self.env_kq[mask] = dr4(K_Q)
        self.env_kr_speed[mask] = dr4(K_R_SPEED)
        self.env_kr_accel[mask] = dr4(K_R_ACCEL)

        # Gyroscopic couplings (±50%)
        self.env_Jx[mask] = dr(J_X)
        self.env_Jy[mask] = dr(J_Y)
        self.env_Jz[mask] = dr(J_Z)

        # Motor params (tau ±55%, omega_max ±40%, rest ±50%)
        self.env_motor_tau[mask] = dr(MOTOR_TAU, DR_TAU)
        self.env_omega_min[mask] = dr(MOTOR_OMEGA_MIN)
        self.env_omega_max[mask] = dr(MOTOR_OMEGA_MAX, DR_OMEGA_MAX)
        self.env_motor_k_cmd[mask] = dr(MOTOR_K_CMD)

    def _random_state(self, n):
        """Generate n random initial states with curriculum-based spawning.
        Returns (state [n, 17], target_gates [n] or None).
        target_gates is set when random_segments is active."""
        dev = self.device
        d = self.difficulty

        if self.random_segments:
            # Random segment selection: each env gets a random gate-to-gate segment
            # seg_idx 0 = start->gate0, seg_idx k = gate(k-1)->gate(k)
            seg_idx = torch.randint(0, NUM_GATES, (n,), device=dev)
            target_gates = seg_idx  # target gate for each env
            is_first = seg_idx == 0
            from_idx = (seg_idx - 1).clamp(min=0)

            # Compute spawn centers for seg>0: 1m past from_gate along its normal
            from_pos = self.gates_pos[from_idx]        # [n, 3]
            from_normal = self.gates_normal[from_idx]  # [n, 3]
            spawn_after = from_pos + from_normal * 1.0

            # Compute spawn centers for seg==0: 3m before gate 0
            spawn_before = (self.gates_pos[0] - self.gates_normal[0] * 3.0).unsqueeze(0).expand(n, 3)

            spawn_center = torch.where(is_first.unsqueeze(-1), spawn_before, spawn_after)

            # Facing yaw: toward target gate
            to_pos = self.gates_pos[seg_idx]  # [n, 3]
            diff = to_pos - spawn_center
            facing_yaw_computed = torch.atan2(diff[:, 1], diff[:, 0])
            # For seg==0, use gate 0's yaw directly
            facing_yaw = torch.where(is_first, self.gates_yaw[0].expand(n), facing_yaw_computed)

            p = spawn_center.clone()
            p += torch.empty(n, 3, device=dev).uniform_(-0.3, 0.3)

            # Velocity: 2 m/s toward target for seg>0, small random for seg==0
            direction = diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)
            v_toward = direction * 2.0 + torch.empty(n, 3, device=dev).uniform_(-1.0, 1.0)
            v_small = torch.empty(n, 3, device=dev).uniform_(-0.5, 0.5)
            v = torch.where(is_first.unsqueeze(-1), v_small, v_toward)

            yaw = facing_yaw + torch.empty(n, device=dev).uniform_(-math.radians(5), math.radians(5))
            roll = torch.empty(n, device=dev).uniform_(-math.radians(5), math.radians(5))
            pitch = torch.empty(n, device=dev).uniform_(-math.radians(5), math.radians(5))

            euler = torch.stack([roll, pitch, yaw], dim=-1)
            q = batch_euler_to_quat(euler)
            w = torch.zeros(n, 3, device=dev)
            motors = torch.full((n, 4), self.hover_omega, device=dev)
            return torch.cat([p, v, q, w, motors], dim=-1), target_gates

        if self.segment is not None or self.fixed_start:
            # Segment or fixed-start spawning
            if self.segment is not None:
                from_gate, to_gate = self.segment
            else:
                from_gate, to_gate = None, 0

            if from_gate is None:
                # Start position: 3m before first gate
                target_pos = self.gates_pos[to_gate]
                target_yaw = self.gates_yaw[to_gate]
                target_normal = self.gates_normal[to_gate]
                spawn_center = target_pos - target_normal * 3.0
                facing_yaw = target_yaw
            else:
                # Spawn at from_gate exit: 1m past gate along its normal
                from_pos = self.gates_pos[from_gate]
                from_normal = self.gates_normal[from_gate]
                spawn_center = from_pos + from_normal * 1.0
                # Face toward the target gate
                to_pos = self.gates_pos[to_gate]
                diff = to_pos - spawn_center
                facing_yaw = torch.atan2(diff[1], diff[0])

            p = spawn_center.unsqueeze(0).expand(n, 3).clone()
            # Jitter: ±0.3m pos, ±5° attitude, ±1m/s velocity
            p += torch.empty(n, 3, device=dev).uniform_(-0.3, 0.3)
            # Small initial velocity toward target to help exploration
            if from_gate is not None:
                to_pos = self.gates_pos[to_gate]
                direction = to_pos - spawn_center
                direction = direction / (direction.norm() + 1e-8)
                v = direction.unsqueeze(0).expand(n, 3).clone() * 2.0  # 2 m/s toward target
                v += torch.empty(n, 3, device=dev).uniform_(-1.0, 1.0)
            else:
                v = torch.empty(n, 3, device=dev).uniform_(-0.5, 0.5)
            yaw = facing_yaw.expand(n).clone()
            yaw += torch.empty(n, device=dev).uniform_(-math.radians(5), math.radians(5))
            roll = torch.empty(n, device=dev).uniform_(-math.radians(5), math.radians(5))
            pitch = torch.empty(n, device=dev).uniform_(-math.radians(5), math.radians(5))

            euler = torch.stack([roll, pitch, yaw], dim=-1)
            q = batch_euler_to_quat(euler)
            w = torch.zeros(n, 3, device=dev)
            motors = torch.full((n, 4), self.hover_omega, device=dev)
            return torch.cat([p, v, q, w, motors], dim=-1), None

        if d >= 0.99:
            # Full random (original behavior)
            px = torch.empty(n, device=dev).uniform_(*INIT_X_RANGE)
            py = torch.empty(n, device=dev).uniform_(*INIT_Y_RANGE)
            pz = torch.empty(n, device=dev).uniform_(*INIT_Z_RANGE)
            p = torch.stack([px, py, pz], dim=-1)
            v = torch.empty(n, 3, device=dev).uniform_(*INIT_V_RANGE)
            roll = torch.empty(n, device=dev).uniform_(*INIT_RP_RANGE)
            pitch = torch.empty(n, device=dev).uniform_(*INIT_RP_RANGE)
            yaw = torch.empty(n, device=dev).uniform_(*INIT_YAW_RANGE)
        else:
            # Curriculum: spawn near target gate
            # Pick random target gates
            target_gi = torch.randint(0, NUM_GATES, (n,), device=dev)
            gate_pos = self.gates_pos[target_gi]    # [n, 3]
            gate_yaw = self.gates_yaw[target_gi]    # [n]
            gate_normal = self.gates_normal[target_gi]  # [n, 3]

            # Spawn distance: lerp 3-5m (easy) to full hall (hard)
            easy_dist_min, easy_dist_max = 3.0, 5.0
            hard_dist_min = 1.0
            hard_dist_max = 50.0  # half hall
            dist_min = easy_dist_min + d * (hard_dist_min - easy_dist_min)
            dist_max = easy_dist_max + d * (hard_dist_max - easy_dist_max)
            spawn_dist = torch.empty(n, device=dev).uniform_(dist_min, dist_max)

            # Spawn behind the gate (opposite to normal direction)
            p = gate_pos - gate_normal * spawn_dist.unsqueeze(-1)
            # Add lateral jitter: lerp ±0.3m (easy) to ±5m (hard)
            lat_jitter = 0.3 + d * 4.7
            p[:, 1] += torch.empty(n, device=dev).uniform_(-lat_jitter, lat_jitter)
            # Z jitter: lerp ±0.3m (easy) to ±2m (hard)
            z_jitter = 0.3 + d * 1.7
            p[:, 2] += torch.empty(n, device=dev).uniform_(-z_jitter, z_jitter)
            # Clamp to bounds
            p[:, 0].clamp_(BOUNDS_X[0] + 0.5, BOUNDS_X[1] - 0.5)
            p[:, 1].clamp_(BOUNDS_Y[0] + 0.5, BOUNDS_Y[1] - 0.5)
            p[:, 2].clamp_(BOUNDS_Z[0] + 0.2, BOUNDS_Z[1] - 0.2)

            # Velocity: lerp ±0.2 (easy) to ±0.5 (hard)
            v_range = 0.2 + d * 0.3
            v = torch.empty(n, 3, device=dev).uniform_(-v_range, v_range)

            # Yaw: point at gate ± jitter. Lerp ±15° (easy) to ±180° (hard)
            yaw_jitter = math.radians(15) + d * (math.pi - math.radians(15))
            yaw = gate_yaw + torch.empty(n, device=dev).uniform_(-yaw_jitter, yaw_jitter)

            # Attitude: lerp ±5° (easy) to ±20° (hard)
            rp_range = math.radians(5) + d * math.radians(15)
            roll = torch.empty(n, device=dev).uniform_(-rp_range, rp_range)
            pitch = torch.empty(n, device=dev).uniform_(-rp_range, rp_range)

        euler = torch.stack([roll, pitch, yaw], dim=-1)
        q = batch_euler_to_quat(euler)

        # Angular velocity: lerp ±0.1 (easy) to full range (hard)
        w_range = 0.1 + d * (INIT_OMEGA_RANGE[1] - 0.1)
        w = torch.empty(n, 3, device=dev).uniform_(-w_range, w_range)

        # Motors at hover speed instead of idle
        motors = torch.full((n, 4), self.hover_omega, device=dev)

        return torch.cat([p, v, q, w, motors], dim=-1), None

    def _find_nearest_gate(self, positions):
        """Find nearest gate ahead for each position. [N, 3] -> [N] long."""
        N = positions.shape[0]
        best_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        best_dist = torch.full((N,), 1e10, device=self.device)

        for i in range(NUM_GATES):
            gate_pos = self.gates_pos[i]
            normal = self.gates_normal[i]
            diff = positions - gate_pos
            signed_dist = (diff * normal).sum(dim=-1)
            dist = diff.norm(dim=-1)
            # Behind gate (signed_dist < 0) and closer than current best
            mask = (signed_dist < 0) & (dist < best_dist)
            best_idx[mask] = i
            best_dist[mask] = dist[mask]

        return best_idx

    def reset_all(self):
        """Reset all environments."""
        mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
        self._randomize_params(mask)
        states, target_gates = self._random_state(self.N)
        self.states = states
        self.prev_states = self.states.clone()
        self.prev_motor_speeds = self.states[:, 13:17].clone()
        if target_gates is not None:
            self.gate_idx = target_gates
        elif self.segment is not None:
            self.gate_idx[:] = self.segment[1]
        else:
            self.gate_idx = self._find_nearest_gate(self.states[:, 0:3])
        self.step_counts.zero_()
        return self.get_obs()

    def reset_envs(self, mask):
        """Selectively reset envs where mask=True."""
        n = mask.sum().item()
        if n == 0:
            return
        self._randomize_params(mask)
        new_states, target_gates = self._random_state(n)
        self.states[mask] = new_states
        self.prev_states[mask] = new_states
        self.prev_motor_speeds[mask] = new_states[:, 13:17]
        if target_gates is not None:
            self.gate_idx[mask] = target_gates
        elif self.segment is not None:
            self.gate_idx[mask] = self.segment[1]
        else:
            self.gate_idx[mask] = self._find_nearest_gate(new_states[:, 0:3])
        self.step_counts[mask] = 0

    def get_obs(self):
        """Compute 24D gate-relative observation for all envs. Returns [N, 24]."""
        p_w = self.states[:, 0:3]
        v_w = self.states[:, 3:6]
        q = self.states[:, 6:10]
        w_body = self.states[:, 10:13]
        motor_speeds = self.states[:, 13:17]

        euler = batch_quat_to_euler(q)

        # Gather per-env gate data
        gi = self.gate_idx % NUM_GATES
        gate_pos = self.gates_pos[gi]           # [N, 3]
        gate_R_inv = self.gates_R_inv[gi]       # [N, 3, 3]
        gate_yaw = self.gates_yaw[gi]           # [N]

        # Transform to gate frame
        p_rel = p_w - gate_pos                  # [N, 3]
        p_g = torch.bmm(gate_R_inv, p_rel.unsqueeze(-1)).squeeze(-1)
        v_g = torch.bmm(gate_R_inv, v_w.unsqueeze(-1)).squeeze(-1)
        euler_g = euler.clone()
        euler_g[:, 2] -= gate_yaw

        # World-frame angular velocity (rotate body rates to world)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.empty((q.shape[0], 3, 3), device=self.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[:, 0, 1] = 2 * (qx*qy - qw*qz)
        R[:, 0, 2] = 2 * (qx*qz + qw*qy)
        R[:, 1, 0] = 2 * (qx*qy + qw*qz)
        R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[:, 1, 2] = 2 * (qy*qz - qw*qx)
        R[:, 2, 0] = 2 * (qx*qz - qw*qy)
        R[:, 2, 1] = 2 * (qy*qz + qw*qx)
        R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        w_world = torch.bmm(R, w_body.unsqueeze(-1)).squeeze(-1)

        # Distance to current gate
        dist_gate = p_rel.norm(dim=-1, keepdim=True)  # [N, 1]

        # Next gate in current gate frame (precomputed)
        p_next_g = self.next_gate_local[gi]     # [N, 3]
        yaw_next = self.next_gate_yaw_rel[gi]   # [N]

        obs = torch.cat([
            p_g,                                # 3
            v_g,                                # 3
            euler_g,                            # 3
            w_world,                            # 3  (NEW)
            w_body,                             # 3
            motor_speeds / MOTOR_OMEGA_MAX,     # 4
            p_next_g,                           # 3
            yaw_next.unsqueeze(-1),             # 1
            dist_gate,                          # 1  (NEW)
        ], dim=-1)                              # total: 24

        return obs

    def step(self, actions):
        """
        Step all environments. Actions: [N, 4] in [0, 1].
        Returns (obs, rewards, dones) all as tensors.
        """
        actions = actions.clamp(0, 1)
        self.prev_states = self.states.clone()
        prev_motors = self.prev_motor_speeds

        # Pack params dict for dynamics
        params = {
            'k_omega': self.env_k_omega, 'k_x': self.env_k_x, 'k_y': self.env_k_y,
            'k_x2': self.env_k_x2, 'k_y2': self.env_k_y2,
            'k_alpha': self.env_k_alpha, 'k_hor': self.env_k_hor,
            'kp': self.env_kp, 'kq': self.env_kq,
            'kr_speed': self.env_kr_speed, 'kr_accel': self.env_kr_accel,
            'Jx': self.env_Jx, 'Jy': self.env_Jy, 'Jz': self.env_Jz,
            'motor_tau': self.env_motor_tau, 'omega_min': self.env_omega_min,
            'omega_max': self.env_omega_max, 'motor_k_cmd': self.env_motor_k_cmd,
        }

        # Dynamics step (all envs at once)
        self.states = batched_dynamics_step(
            self.states, actions, params, prev_motors, self.dt,
        )
        self.prev_motor_speeds = self.states[:, 13:17].clone()
        self.step_counts += 1

        # Compute rewards and dones
        rewards, dones = self._compute_rewards_and_dones(actions)

        # Auto-reset done envs
        if dones.any():
            self.reset_envs(dones)

        obs = self.get_obs()

        # NaN safety
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs, rewards, dones

    def _compute_rewards_and_dones(self, actions):
        """Batched reward and termination computation."""
        p = self.states[:, 0:3]
        p_prev = self.prev_states[:, 0:3]
        w_body = self.states[:, 10:13]
        q = self.states[:, 6:10]

        gi = self.gate_idx % NUM_GATES
        gate_pos = self.gates_pos[gi]  # [N, 3]

        # 1. Progress reward
        dist_prev = (p_prev - gate_pos).norm(dim=-1)
        dist_curr = (p - gate_pos).norm(dim=-1)
        progress = dist_prev - dist_curr
        r_prog = LAMBDA_PROG * torch.clamp(progress, max=V_MAX * self.dt)

        # 2. Gate passage detection (batched)
        gate_normal = self.gates_normal[gi]  # [N, 3]
        d_prev = ((p_prev - gate_pos) * gate_normal).sum(dim=-1)
        d_curr = ((p - gate_pos) * gate_normal).sum(dim=-1)

        crossed = (d_prev * d_curr) < 0  # sign change = crossed plane

        # Interpolate crossing point
        t_cross = d_prev / (d_prev - d_curr + 1e-8)
        p_cross = p_prev + t_cross.unsqueeze(-1) * (p - p_prev)

        # Local coordinates at crossing
        gate_R_inv = self.gates_R_inv[gi]
        local = torch.bmm(gate_R_inv, (p_cross - gate_pos).unsqueeze(-1)).squeeze(-1)

        half = self.gate_size / 2.0
        within = crossed & (local[:, 1].abs() < half) & (local[:, 2].abs() < half)
        # Crossed gate plane but missed the opening = failed attempt, terminate
        hit_frame = crossed & ~within

        offset = (local[:, 1]**2 + local[:, 2]**2).sqrt()

        r_gate = torch.where(within, torch.tensor(LAMBDA_GATE, device=self.device),
                             torch.zeros(1, device=self.device))
        r_offset = torch.where(within, -LAMBDA_OFFSET * offset,
                               torch.zeros(1, device=self.device))

        # Advance gate index for envs that passed
        self.gate_idx[within] += 1
        self.gates_passed_count += within.sum().item()

        # 3. Trajectory alignment reward
        # Project velocity onto gate plane: reward aiming at gate center
        # Uses a virtual target radius (2m) so the drone can earn positive reward
        # even when not perfectly aimed at the tiny 0.4m opening
        v = self.states[:, 3:6]
        ALIGN_RADIUS = 2.0  # virtual target radius for alignment reward

        # Time for velocity ray to hit gate plane
        v_dot_n = (v * gate_normal).sum(dim=-1)                 # [N]
        p_to_gate = gate_pos - p                                # [N, 3]
        dist_to_plane = (p_to_gate * gate_normal).sum(dim=-1)   # [N]
        t_hit = dist_to_plane / (v_dot_n + 1e-8)               # [N]

        # Projected intersection with gate plane
        p_hit = p + v * t_hit.unsqueeze(-1)                     # [N, 3]

        # Offset from gate center in gate-local Y-Z
        hit_local = torch.bmm(gate_R_inv, (p_hit - gate_pos).unsqueeze(-1)).squeeze(-1)
        miss_dist = (hit_local[:, 1]**2 + hit_local[:, 2]**2).sqrt()  # [N]

        moving_toward = t_hit > 0

        # Steeper inside radius: +2 at center, 0 at edge; -1 max outside
        align_score = torch.where(
            miss_dist < ALIGN_RADIUS,
            2.0 * (1.0 - miss_dist / ALIGN_RADIUS),            # [0, +2] inside
            -(miss_dist - ALIGN_RADIUS).clamp(max=ALIGN_RADIUS) / ALIGN_RADIUS,  # [-1, 0] outside
        )
        r_align = torch.where(
            moving_toward,
            LAMBDA_ALIGN * align_score,
            -LAMBDA_ALIGN * torch.ones(1, device=self.device),
        ) * self.dt

        # 4. Angular rate penalty (world-frame angular velocity per paper: ||Ω_k||²)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R_rate = torch.empty((q.shape[0], 3, 3), device=self.device, dtype=q.dtype)
        R_rate[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R_rate[:, 0, 1] = 2 * (qx*qy - qw*qz)
        R_rate[:, 0, 2] = 2 * (qx*qz + qw*qy)
        R_rate[:, 1, 0] = 2 * (qx*qy + qw*qz)
        R_rate[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R_rate[:, 1, 2] = 2 * (qy*qz - qw*qx)
        R_rate[:, 2, 0] = 2 * (qx*qz - qw*qy)
        R_rate[:, 2, 1] = 2 * (qy*qz + qw*qx)
        R_rate[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        w_world = torch.bmm(R_rate, w_body.unsqueeze(-1)).squeeze(-1)
        r_rate = -LAMBDA_RATE * (w_world**2).sum(dim=-1)

        # 5. Perception penalty: angle between camera axis and gate direction
        gate_dir = gate_pos - p  # [N, 3]
        gate_dist = gate_dir.norm(dim=-1, keepdim=True).clamp(min=0.1)  # [N, 1]
        gate_dir_norm = gate_dir / gate_dist  # [N, 3]
        # Camera forward = first column of R (body x-axis in world frame)
        cam_fwd = torch.stack([
            1 - 2 * (qy**2 + qz**2),
            2 * (qx*qy + qw*qz),
            2 * (qx*qz - qw*qy),
        ], dim=-1)  # [N, 3]
        cos_theta = (cam_fwd * gate_dir_norm).sum(dim=-1).clamp(-1.0, 1.0)  # [N]
        theta_cam = torch.acos(cos_theta)  # [N]
        r_perc = torch.where(theta_cam > math.radians(45),  # M23: θ_cam = 45° (Table 1)
                             -LAMBDA_PERC * theta_cam,
                             torch.zeros(1, device=self.device))

        rewards = r_prog + r_gate + r_offset + r_rate + r_perc + r_align

        # Termination
        speed = v.norm(dim=-1)

        oob = ((p[:, 0] < BOUNDS_X[0]) | (p[:, 0] > BOUNDS_X[1]) |
               (p[:, 1] < BOUNDS_Y[0]) | (p[:, 1] > BOUNDS_Y[1]) |
               (p[:, 2] < BOUNDS_Z[0]))  # ceiling only; ground handled below
        ground_crash = (p[:, 2] >= H_GROUND) & (speed > V_GROUND)
        omega_exceeded = (w_body.abs() > OMEGA_MAX_TERMINATION).any(dim=-1)
        nan_state = torch.isnan(self.states).any(dim=-1)
        crashed = oob | ground_crash | omega_exceeded | hit_frame | nan_state
        if self.single_gate:
            completed = within  # Pass one gate = success
        else:
            completed = self.gate_idx >= NUM_GATES * NUM_LAPS
        truncated = self.step_counts >= MAX_EPISODE_STEPS

        rewards[crashed] -= LAMBDA_CRASH

        dones = crashed | completed | truncated
        self.episodes_ended_count += dones.sum().item()
        return rewards, dones
