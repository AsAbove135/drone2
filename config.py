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

# Simplified dynamics model constants (used by control/dynamics.py)
J = [0.005, 0.005, 0.009]       # diagonal inertia tensor [Jxx, Jyy, Jzz] (kg·m²)
K_DRAG = 0.016                  # yaw drag-to-thrust ratio
ARM_LENGTH = 0.125              # motor arm length (m)
MOTOR_K = 1.0                   # motor first-order lag gain

# Domain randomization ranges (M23: 50% on all except noted)
DR_RANGE = 0.5                  # default ±50%
DR_OMEGA_MAX = 0.4              # ±40% for omega_max
DR_TAU = 0.55                   # ±55% for motor tau

DT_SIM = 0.01                   # simulation timestep (100 Hz) — M23 paper value

# === Reward Lambdas (M23 from Table 1) ===
LAMBDA_PROG = 1.0               # M23 paper value
LAMBDA_GATE = 1.5               # M23 paper value
LAMBDA_RATE = 0.001
LAMBDA_OFFSET = 1.5
LAMBDA_PERC = 0.01
LAMBDA_DELTA_U = 0.0            # not used in M23
LAMBDA_U = 0.0                  # not used in M23
LAMBDA_CRASH = 10.0
LAMBDA_ALIGN = 0.0              # not in paper (GPU experimental only)
V_MAX = 10.0                    # max progress reward clamp (m/s) — M23 paper value

# === Observation ===
OBS_DIM = 24  # p_gate(3) + v_gate(3) + euler_gate(3) + w_world(3) + w_body(3) + motors(4) + next_gate(3) + yaw(1) + dist(1)

# === Track Layouts ===
# All tracks in a 100x30m hall. NED convention: X forward, Y right, Z down.
# Each gate: (x, y, z, yaw_rad) where yaw is the gate's facing direction.

TRACKS = {
    # --- Easy: Gentle S-curve, flat altitude, 7 gates (point-to-point) ---
    # Smooth sweeping turns, constant Z=-2.0, ~12m gate spacing
    "easy": [
        (10.0, -15.0, -2.0,  0.0),    # G1: start straight
        (22.0, -13.0, -2.0,  0.15),   # G2: slight left
        (35.0, -11.0, -2.0,  0.1),    # G3: continuing left
        (48.0, -12.0, -2.0, -0.1),    # G4: start curving right
        (60.0, -15.0, -2.0, -0.15),   # G5: gentle right
        (72.0, -17.0, -2.0, -0.1),    # G6: continuing right
        (85.0, -16.0, -2.0,  0.1),    # G7: straighten to finish
    ],

    # --- Medium: Rolling hills, moderate turns + altitude, 9 gates (point-to-point) ---
    # S-curves with +-1m altitude changes, ~10m gate spacing
    "medium": [
        ( 8.0, -14.0, -2.0,  0.0),    # G1: start
        (18.0, -10.0, -2.0,  0.3),    # G2: left turn
        (30.0,  -8.0, -2.5,  0.1),    # G3: climb
        (42.0, -10.0, -3.0, -0.2),    # G4: descend + right
        (52.0, -15.0, -2.5, -0.4),    # G5: right turn
        (60.0, -20.0, -2.0, -0.3),    # G6: right + level out
        (70.0, -18.0, -2.5,  0.2),    # G7: left + climb
        (80.0, -15.0, -3.0,  0.15),   # G8: climb
        (90.0, -14.0, -2.0,  0.0),    # G9: descend to finish
    ],

    # --- Hard: Technical course, tight turns + big Z changes, 11 gates (point-to-point) ---
    # Sharp direction changes, chicane, 1.5m altitude range, ~8-11m spacing
    "hard": [
        ( 8.0, -13.0, -2.0,  0.0),    # G1: start
        (18.0,  -8.0, -2.5,  0.4),    # G2: sharp left + climb
        (28.0,  -6.0, -3.0,  0.1),    # G3: continue climb
        (38.0, -10.0, -2.5, -0.5),    # G4: sharp right + descend
        (45.0, -16.0, -2.0, -0.6),    # G5: tight right
        (50.0, -20.0, -1.5, -0.3),    # G6: drop altitude
        (58.0, -18.0, -2.5,  0.3),    # G7: chicane left + climb
        (63.0, -14.0, -3.5,  0.5),    # G8: sharp left + big climb
        (72.0, -12.0, -2.5,  0.1),    # G9: level out
        (82.0, -15.0, -2.0, -0.3),    # G10: right + descend
        (92.0, -13.0, -2.0,  0.0),    # G11: finish
    ],

    # --- Expert: Original M23-inspired course (point-to-point) ---
    # Double-gates, split-S, 2m+ altitude range
    "expert": [
        ( 5.0, -13.0, -2.0,  0.0),    # G1: start straight
        (15.0,  -8.0, -2.5,  0.3),    # G2: slight left
        (25.0,  -5.0, -2.0,  0.0),    # G3: double-gate A (first)
        (30.0,  -5.0, -2.0,  0.0),    # G4: double-gate A (second)
        (42.0, -10.0, -3.5, -0.4),    # G5: climbing right turn
        (55.0, -15.0, -4.0,  np.pi),  # G6: split-S entry (backward, high)
        (50.0, -15.0, -1.5,  0.0),    # G7: split-S exit (below G6)
        (62.0, -18.0, -2.0, -0.2),    # G8: recovery straight
        (75.0, -20.0, -2.5,  0.0),    # G9: double-gate B (first)
        (80.0, -20.0, -2.5,  0.0),    # G10: double-gate B (second)
        (92.0, -15.0, -2.0,  0.3),    # G11: finish
    ],

    # --- Kidney bean: Classic FPV racing circuit, 10 gates (lap circuit) ---
    # Oval with concave indent on top side, flat altitude, CW direction
    # Gate yaw = tangent travel direction at each gate
    "kidney": [
        (30.0, -22.0, -2.0, -0.30),   # G1:  bottom-left, heading right
        (42.0, -23.0, -2.0,  0.0),    # G2:  bottom center-left
        (55.0, -22.0, -2.0,  0.21),   # G3:  bottom center-right
        (66.0, -18.0, -2.0,  0.54),   # G4:  bottom-right, curving up
        (70.0, -13.0, -2.0,  1.86),   # G5:  right end, tight turn
        (63.0,  -8.0, -2.0,  3.04),   # G6:  top-right, heading left
        (50.0, -11.0, -2.0,  3.14),   # G7:  kidney indent (concave dip)
        (37.0,  -8.0, -2.0, -3.10),   # G8:  top-left, heading left
        (28.0, -12.0, -2.0, -2.36),   # G9:  left end, tight turn
        (27.0, -18.0, -2.0, -1.37),   # G10: left-bottom, heading down-right
    ],

    # --- Figure-8: Classic FPV racing circuit, 10 gates (lap circuit) ---
    # Two tangent loops (R=10m), right loop CW + left loop CCW
    # Crossing gates at center, offset 1m apart for distinct detection
    "figure8": [
        (51.0, -14.0, -2.0,  1.57),   # G1:  crossing A, heading into right loop
        (57.0,  -5.0, -2.0,  0.32),   # G2:  right loop upper-left
        (68.0,  -8.0, -2.0, -0.93),   # G3:  right loop upper-right
        (68.0, -20.0, -2.0, -2.21),   # G4:  right loop lower-right
        (57.0, -23.0, -2.0,  2.82),   # G5:  right loop lower-left
        (49.0, -14.0, -2.0,  1.57),   # G6:  crossing B, heading into left loop
        (43.0,  -5.0, -2.0,  2.82),   # G7:  left loop upper-right
        (32.0,  -8.0, -2.0, -2.21),   # G8:  left loop upper-left
        (32.0, -20.0, -2.0, -0.93),   # G9:  left loop lower-left
        (43.0, -23.0, -2.0,  0.32),   # G10: left loop lower-right
    ],
}

# Active track selection (change this to switch tracks)
ACTIVE_TRACK = "kidney"
GATE_POSITIONS = TRACKS[ACTIVE_TRACK]
GATE_SIZE = 0.40                # inner opening side length (m) — M23 paper value
GATE_THICKNESS = 1.0            # gate thickness (m) — M23 paper value
NUM_LAPS = 2
NUM_GATES = len(GATE_POSITIONS)

# Max distance between consecutive gates (used as spawn distance hard max)
_gate_dists = [np.sqrt(sum((a - b)**2 for a, b in zip(GATE_POSITIONS[i][:3], GATE_POSITIONS[i+1][:3])))
               for i in range(len(GATE_POSITIONS) - 1)]
MAX_GATE_DISTANCE = max(_gate_dists)  # ~13.9m (G5→G6)

# === Domain Randomization (ranges defined above with dynamics params) ===

# === Initialization Ranges (M23 "uniform") ===
INIT_X_RANGE = (1.0, 95.0)
INIT_Y_RANGE = (-27.0, 1.0)
INIT_Z_RANGE = (-5.0, 0.0)     # NED: negative Z = above ground — paper: U(-5, 0)
INIT_V_RANGE = (-0.5, 0.5)
INIT_RP_RANGE = (-np.pi / 9, np.pi / 9)
INIT_YAW_RANGE = (-np.pi, np.pi)
INIT_OMEGA_RANGE = (-0.1, 0.1)

# === Environment Bounds (paper: x∈[1,95], y∈[-27,1], padded +10m for training) ===
BOUNDS_X = (-9.0, 105.0)
BOUNDS_Y = (-37.0, 11.0)
BOUNDS_Z = (-10.0, 0.0)        # NED: ceiling=-10, floor=0 (ground collision handled separately)

# === Ground Collision (M23) ===
H_GROUND = 0.0                 # ground height in NED (z >= 0 = at/below ground)
V_GROUND = 2.0                 # speed threshold for ground crash (m/s)

# === Angular Velocity Termination ===
OMEGA_MAX_TERMINATION = 29.6706  # 1700 deg/s in rad/s

# === Training ===
MAX_EPISODE_STEPS = 4000        # at 100Hz DT=0.01 = 40 seconds
NUM_ENVS = 8
TOTAL_TIMESTEPS = 50_000_000
CHECKPOINT_FREQ = 100_000
