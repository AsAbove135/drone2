# MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI
# Source: arxiv 2601.15222v1 (PRIMARY REFERENCE)
# Authors: Bahnam, Ferede, Blaha, Lang, Lucassen, Missinne, Verraest, De Wagter, de Croon
# Delft University of Technology

## Abstract

MonoRace: autonomous drone racing with monocular rolling-shutter camera and IMU. Won 2025 Abu Dhabi A2RL competition, beating three world champion FPV pilots. Speeds up to 100 km/h (28.23 m/s). Uses GateNet (U-Net segmentation), EKF state estimation, and GCNet (RL policy mapping states to motor commands at 500 Hz).

---

## Competition Format

- 100 x 30 m indoor hall
- 11 square gates (1.5 m inner openings at competition; training uses smaller g_size)
- Two double-gates and a split-S maneuver
- Two laps per run
- No external localization permitted

## Competition Performance

- M16: fastest (~16s) but aggressive
- M23: 88.4% success rate across 43 flights
- Models named M16-M23 based on approximate completion times
- 30% domain randomization maintained ~90% success rate in simulation

---

## Pipeline Overview

1. Image Capture: 820x616 at 90 Hz
2. Adaptive Cropping: state-based 384x384 region selection
3. Gate Segmentation: GateNet (U-Net)
4. Corner Detection: QuAdGate
5. Outlier Rejection: Homography-based RANSAC
6. Pose Estimation: PnP with EKF fusion
7. Control: GCNet at 500 Hz

---

## Perception

### Adaptive Cropping
- Full 820x616 undistorted, then 384x384 crop around predicted gate location
- Selects region containing two closest visible gates
- Achieved 36% more corner detections than simple resizing

### GateNet (U-Net)
- Encoder-decoder with skip connections, channel factor f=4
- Five output maps at increasing resolutions (auxiliary losses)
- Training: 100 epochs, AdamW, LR=1e-3, step decay at epochs 10/33/66/90
- Loss per scale: Dice + 2*BCE
- Scale weights: 4, 2, 1, 1, 1
- Data: 3500:500 synthetic:real ratio, batch size 16
- Augmentations: affine transforms, HSV, motion blur (kernel 5-15px), noise

### QuAdGate Corner Detection
- Line Segment Detector (LSD) on segmentation masks
- Lines extended by factor 5/3, intersections = corner candidates
- Handcrafted descriptors: 4 pixel values in local neighborhood (5px along line directions)
- RANSAC with 2D affine transform (4 DOF: translation, rotation, uniform scale)
- RansacThreshold=5.0, max translation=150px

### PnP
- Combines corners from multiple gates (typically 2) in single PnP
- Multi-gate PnP improves heading estimation by ~2°
- Full PnP only when gate distance 2-5m AND ≥6 corners
- Fallback: attitude from KF + relative translation from PnP

---

## State Estimation (EKF)

### State Vector (16D)
x = [x y z vx vy vz qw qx qy qz bx by bz bp bq br]^T
- Position (3), velocity (3), quaternion (4), IMU biases (6)

### Measurement Model
h(x) = [x y z qw qx qy qz]^T (from PnP)

### Measurement Noise
- sigma^2_pos = 0.02 * d^2_gate / (N^2_c * Ng)
- sigma^2_quat = 0.01 * d^2_gate / (N^2_c * Ng)

### Outlier Rejection
||x_pos - x_PnP||^2 < 16 * N^2_c * trace(P_pos)

### IMU Saturation Handling
- Accelerometer saturates at 16g during aggressive maneuvers (up to 7g thrust + vibrations)
- Model-based prediction when ||a_filt^model - a_filt^IMU||_2 > sigma=22 m/s^2
- Low-pass filter: a_filt[n] = alpha * a_filt[n-1] + (1-alpha) * a[n]
- Switches all 3 axes to model-predicted accelerations
- Increased position/attitude uncertainty during saturation

### Temporal Synchronization
- 17ms image delay, 0.5ms IMU delay
- Delayed state propagated forward with subsequent unfused IMU measurements
- Reduced RMS trajectory error from 0.289m to 0.103m

---

## Quadcopter Model

### State Vector (17D)
x = [x y z vx vy vz qw qx qy qz p q r w1 w2 w3 w4]^T
u = [u1 u2 u3 u4]^T

### Equations of Motion
- p_dot = v
- v_dot = R(q) * F + g
- q_dot = 0.5 * q ⊗ [0 p q r]^T
- Omega_dot = M
- omega_dot = (omega_c - omega) / tau

### Motor Command Model
omega_c,i = (omega_max - omega_min) * sqrt(k * u_i^2 + (1-k) * u_i) + omega_min

### Aerodynamic Force Model (body frame, specific accelerations)
F = [-k_x * vx_B * sum(omega_i) - k_x2 * vx_B * |vx_B|,
     -k_y * vy_B * sum(omega_i) - k_y2 * vy_B * |vy_B|,
     -k_omega * (1 + k_alpha * alpha + k_hor * mu) * sum(omega_i^2)]

Where:
- alpha = atan2(vz_B, r * omega_bar)  -- angle of attack
- mu = atan2(sqrt(vx_B^2 + vy_B^2), r * omega_bar)  -- advance ratio
- r = 0.0485775 m (propeller radius, NOT randomized)
- omega_bar = sum(omega_i)

### Angular Accelerations (specific, no inertia division)
M = [-kp1*w1^2 - kp2*w2^2 + kp3*w3^2 + kp4*w4^2 + Jx*q*r,
     -kq1*w1^2 + kq2*w2^2 - kq3*w3^2 + kq4*w4^2 + Jy*p*r,
     -kr1*w1 + kr2*w2 + kr3*w3 - kr4*w4 - kr5*w1_dot + kr6*w2_dot + kr7*w3_dot - kr8*w4_dot + Jz*p*q]

---

## Reward Function

r_k = r_prog + r_gate - p_rate - p_offset - p_perc - p_delta_u - p_u - p_crash

### Components
- Progress: lambda_prog * min(||p_{k-1} - p_gk|| - ||p_k - p_gk||, v_max * dt)
  - Capped at v_max*dt to limit effective max speed
- Gate bonus: lambda_gate (on passage)
- Rate penalty: lambda_rate * ||Omega_k||^2  (world-frame angular velocity)
- Offset penalty: lambda_offset * ||p_k - p_gk|| (at passage)
- Perception penalty: lambda_perc * theta_cam (if theta_cam > pi/3)
  - theta_cam = angle between optical axis and center of next gate
  - BUT Table 1 gives per-drone theta_cam values (M23: 45 degrees)
- Delta-u penalty: lambda_delta_u * sum(max(|u_i[k] - u_i[k-1]| - delta_u_thresh, 0))
- Low action: lambda_u * sum(max(0.5 - u_i, 0))
- Crash: lambda_crash

---

## Table 1: M23 Parameters

### Training Config
| Parameter | M23 |
|-----------|-----|
| Retrained from | X (trained from scratch) |
| Initialization | uniform |
| theta_cam (deg) | 45 |
| g_size (m) | 0.40 |
| g_thickness (m) | 1 |
| h_ground (m) | 0 |
| v_ground (m/s) | 2 |

### Reward Parameters
| Parameter | M23 |
|-----------|-----|
| lambda_prog | 1 |
| v_max | 10 |
| lambda_gate | 1.5 |
| lambda_Omega | 0.001 |
| lambda_offset | 1.5 |
| lambda_perc | 0.01 |
| lambda_delta_u | 0 |
| delta_u_thresh | 0 |
| lambda_u | 0 |
| lambda_crash | 10 |

### Dynamics Parameters (M23, nominal)
| Parameter | Value |
|-----------|-------|
| k_omega | 1.55e-6 |
| k_x | 5.37e-5 |
| k_y | 5.37e-5 |
| k_x2 | 4.10e-3 |
| k_y2 | 1.51e-2 |
| k_alpha | 3.145 |
| k_hor | 7.245 |
| J_x | -0.89 |
| J_y | 0.96 |
| J_z | -0.34 |
| omega_min | 341.75 rad/s |
| omega_max | 3100.0 rad/s |
| tau | 0.025 s |
| r (prop radius) | 0.0485775 m (NOT randomized) |

### Domain Randomization
- Default: ±50% uniform on all parameters
- omega_max: ±40%
- tau: ±55%

### Initialization (uniform)
- x0 ~ U(1, 95)
- y0 ~ U(-27, 1)
- z0 ~ U(-5, 0)
- v0 ~ U(-0.5, 0.5)^3
- phi0, theta0 ~ U(-pi/9, pi/9)
- psi0 ~ U(-pi, pi)
- Omega0 ~ U(-0.1, 0.1)^3
- u0 ~ U(-1, 1)^4  (motor COMMANDS, not speeds)
- Target gate: nearest gate ahead

### Collision Detection
- Gate collision: crosses gate plane outside inner square (>g_size/2 from center)
- Ground: z < -h_ground while speed > v_ground
- OOB: x not in [1,95] or y not in [-27,1]
- Angular velocity > 1700 deg/s
- Outer gate size: 2.7 m (fixed)

---

## Policy and Training

### GCNet Architecture
- 3 hidden layers x 64 neurons
- ReLU activations
- Input: 24 observations (see below)
- Output: 4 motor commands u

### Observation Vector (stated as 24D)
x_obs = [p^(gi), v^(gi), Phi^(gi), Omega, omega, p_(gi+1)^(gi), psi_(gi+1)^(gi)]^T

Paper text: "The policy takes in 24 observations"
Paper defines:
- Omega: angular velocity in world frame
- omega: angular velocity in body frame
(These are listed as separate items in the text description)

Formula components: 3+3+3+3+3+4+3+1 = 23 (with both Omega and omega)
The 24th dimension is not explicitly identified in the formula.

Previous paper (2504.21586) uses same formula but says "20 observations" (only one angular velocity).

### Training
- PPO via Stable-Baselines3
- Custom Gym environment
- Entropy coefficient: 0
- Specific learning rate: not disclosed
- Forward Euler integration, dt = 0.01s

---

## Gate Reprojection Evaluation
- Uses known gate geometries + onboard images for evaluation without motion capture
- Measures IoU between expected and observed gate pixels
- Bayesian optimization for camera extrinsic calibration
- Achieves <1° error in pitch/roll/yaw

## Camera Interference
- MIPI cable EMI causes image corruption (8-75% of frames)
- Survived with: GateNet robustness + RANSAC outlier rejection + KF filtering
- Successfully completed track with 50% image corruption

## Hardware
- Custom carbon fiber frame, 966g
- Monocular rolling-shutter camera (155° x 115° FOV)
- NVIDIA Jetson Orin NX
- 500 Hz control loop on flight controller
