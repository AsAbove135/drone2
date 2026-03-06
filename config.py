"""
M23 policy configuration for MonoRace drone racing.
All constants from Table 1 and Section IV of the paper (arxiv 2601.15222).
"""
import numpy as np

# === Dynamics Parameters (M23 — System-Identified Model, Table 1) ===
# The identified model uses specific forces/accelerations (mass & inertia baked in).
MASS = 0.966                    # kg (only for hover omega calculation)
GRAVITY = 9.81                  # m/s^2

# Motor parameters
MOTOR_OMEGA_MIN = 341.75        # rad/s (idle)
MOTOR_OMEGA_MAX = 3100.0        # rad/s (full throttle)
MOTOR_TAU = 0.025               # motor time constant (s)
MOTOR_K_CMD = 0.50              # nonlinear command curve: sqrt(k*u² + (1-k)*u)
PROP_RADIUS = 0.0485775         # propeller radius (m) — NOT randomized

# Force coefficients (specific accelerations, no mass division needed)
K_OMEGA = 1.55e-6               # thrust: F_z = -k_omega * (1 + k_alpha*α + k_hor*μ) * Σωi²
K_X = 5.37e-5                   # x-drag linear: F_x += -k_x * vx_B * Σωi
K_Y = 5.37e-5                   # y-drag linear: F_y += -k_y * vy_B * Σωi
K_X2 = 4.10e-3                  # x-drag quadratic: F_x += -k_x2 * vx_B * |vx_B|
K_Y2 = 1.51e-2                  # y-drag quadratic: F_y += -k_y2 * vy_B * |vy_B|
K_ALPHA = 3.145                 # advance ratio (angle of attack correction)
K_HOR = 7.245                   # horizontal advance ratio correction

# Roll moment coefficients (angular accelerations, no inertia division needed)
K_P = np.array([4.99e-5, 3.78e-5, 4.82e-5, 3.83e-5])  # per-motor roll
J_X = -0.89                     # gyroscopic coupling (NOT inertia)

# Pitch moment coefficients
K_Q = np.array([2.05e-5, 2.46e-5, 2.02e-5, 2.57e-5])  # per-motor pitch
J_Y = 0.96                      # gyroscopic coupling

# Yaw moment coefficients
K_R_SPEED = np.array([3.38e-3, 3.38e-3, 3.38e-3, 3.38e-3])  # per-motor yaw (speed)
K_R_ACCEL = np.array([3.24e-4, 3.24e-4, 3.24e-4, 3.24e-4])  # per-motor yaw (accel)
J_Z = -0.34                     # gyroscopic coupling

# Domain randomization ranges (M23: 50% on all except noted)
DR_RANGE = 0.5                  # default ±50%
DR_OMEGA_MAX = 0.4              # ±40% for omega_max
DR_TAU = 0.55                   # ±55% for motor tau

DT_SIM = 0.002                  # simulation timestep (500 Hz control)

# === Reward Lambdas (M23 from Table 1) ===
LAMBDA_PROG = 5.0
LAMBDA_GATE = 10.0
LAMBDA_RATE = 0.001
LAMBDA_OFFSET = 1.5
LAMBDA_PERC = 0.01
LAMBDA_DELTA_U = 0.0            # not used in M23
LAMBDA_U = 0.0                  # not used in M23
LAMBDA_CRASH = 10.0
V_MAX = 10.0                    # max progress reward clamp (m/s) — M23 paper value

# === Observation ===
OBS_DIM = 24  # p_gate(3) + v_gate(3) + euler_gate(3) + w_world(3) + w_body(3) + motors(4) + next_gate(3) + yaw(1) + dist(1)

# === Track Layout ===
# 11 gates in a 100x30m hall. NED convention: X forward, Y right, Z down.
# Each gate: (x, y, z, yaw_rad) where yaw is the gate's facing direction.
# Includes 2 double-gates (3-4 and 9-10) and 1 split-S (gates 6-7).
GATE_POSITIONS = [
    # Gate 1: Start straight
    (5.0,  -13.0, -2.0,  0.0),
    # Gate 2: Slight left
    (15.0,  -8.0, -2.5,  0.3),
    # Gate 3: Double-gate A (first)
    (25.0,  -5.0, -2.0,  0.0),
    # Gate 4: Double-gate A (second, close behind)
    (30.0,  -5.0, -2.0,  0.0),
    # Gate 5: Climbing right turn
    (42.0, -10.0, -3.5, -0.4),
    # Gate 6: Split-S entry (facing backward, high)
    (55.0, -15.0, -4.0,  np.pi),
    # Gate 7: Split-S exit (below gate 6, forward again)
    (50.0, -15.0, -1.5,  0.0),
    # Gate 8: Recovery straight
    (62.0, -18.0, -2.0, -0.2),
    # Gate 9: Double-gate B (first)
    (75.0, -20.0, -2.5,  0.0),
    # Gate 10: Double-gate B (second)
    (80.0, -20.0, -2.5,  0.0),
    # Gate 11: Finish
    (92.0, -15.0, -2.0,  0.3),
]
GATE_SIZE = 0.40                # inner opening side length (m) — M23 paper value
GATE_THICKNESS = 1.0            # gate thickness (m) — M23 paper value
NUM_LAPS = 2
NUM_GATES = len(GATE_POSITIONS)

# === Domain Randomization (ranges defined above with dynamics params) ===

# === Initialization Ranges (M23 "uniform") ===
INIT_X_RANGE = (1.0, 95.0)
INIT_Y_RANGE = (-27.0, 1.0)
INIT_Z_RANGE = (-5.0, 0.0)     # NED: negative Z = above ground — paper: U(-5, 0)
INIT_V_RANGE = (-0.5, 0.5)
INIT_RP_RANGE = (-np.pi / 9, np.pi / 9)
INIT_YAW_RANGE = (-np.pi, np.pi)
INIT_OMEGA_RANGE = (-0.1, 0.1)

# === Environment Bounds (paper: x∈[1,95], y∈[-27,1]) ===
BOUNDS_X = (1.0, 95.0)
BOUNDS_Y = (-27.0, 1.0)
BOUNDS_Z = (-6.0, 0.0)         # NED: ceiling=-6, floor=0 (ground collision handled separately)

# === Ground Collision (M23) ===
H_GROUND = 0.0                 # ground height in NED (z >= 0 = at/below ground)
V_GROUND = 2.0                 # speed threshold for ground crash (m/s)

# === Angular Velocity Termination ===
OMEGA_MAX_TERMINATION = 29.6706  # 1700 deg/s in rad/s

# === Training ===
MAX_EPISODE_STEPS = 4000        # at 500Hz = 8 seconds
NUM_ENVS = 8
TOTAL_TIMESTEPS = 50_000_000
CHECKPOINT_FREQ = 100_000
