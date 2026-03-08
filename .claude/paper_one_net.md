# One Net to Rule Them All: Domain Randomization in Quadcopter Racing
# Source: arxiv 2504.21586v1 (FALLBACK REFERENCE)
# Authors: Ferede, Blaha, Lucassen, De Wagter, de Croon
# Delft University of Technology

## Abstract

First neural network controller for drone racing that generalizes across physically distinct quadcopters. Single network trained with domain randomization controls both 3-inch and 5-inch race drones. Direct state-to-motor-command mapping. Validated in real-world tests.

---

## Quadcopter Model

### State and Control
x = [p, v, lambda, Omega, omega]^T
u = [u1, u2, u3, u4]^T in [0,1]^4

Where: p=position, v=velocity, lambda=Euler angles, Omega=body rates, omega=propeller speeds

### Equations of Motion
- p_dot = v
- v_dot = g*e3 + R(lambda)*F
- lambda_dot = Q(lambda)*Omega
- Omega_dot = M
- omega_dot_i = (omega_c_i - omega) / tau

### Motor Model
omega_c_i = (omega_max - omega_min) * sqrt(k_l * u_i^2 + (1-k_l) * u_i) + omega_min

### Force Model (simpler than MonoRace — no quadratic drag or advance ratio)
F = [-sum(k_x * vx_B * omega_i),
     -sum(k_y * vy_B * omega_i),
     -sum(k_omega * omega_i^2)]

### Identified Parameters

| Parameter | 3-inch | 5-inch |
|-----------|--------|--------|
| k_hat_omega | 14.3 | 27.1 |
| k_hat_x | 0.16 | 0.16 |
| k_hat_y | 0.18 | 0.24 |
| omega_min (rad/s) | 305.4 | 238.49 |
| omega_max (rad/s) | 4887.57 | 3295.5 |
| k_l | 0.84 | 0.95 |
| tau | not listed | not listed |

(Note: k_hat values are normalized by omega_max^2 or omega_max)

---

## Policy

### Architecture
- 3 hidden layers x 64 neurons, ReLU
- Input: **20 observations**
- Output: 4 motor commands

### Observation Vector (20D)
x_obs = [p^(gi), v^(gi), lambda^(gi), Omega, omega, p_(gi+1)^(gi), psi_(gi+1)^(gi)]^T
= 3+3+3+3+4+3+1 = 20

Key difference from MonoRace: only ONE angular velocity term (Omega = body rates), giving 20D.
MonoRace paper uses same formula but says 24D (adds world angular velocity + unknown 24th dim).

---

## Training Setup

- 7 square gates (1.5 x 1.5 m) in figure-eight track
- 100 parallel environments
- PPO via Stable-Baselines3
- gamma = 0.999
- Max episode: 1200 steps (12 seconds)
- 100 million timesteps

### Initialization
- Spawn 1m in front of randomly selected gate
- v ~ U(-0.5, 0.5)^3
- phi, theta ~ U(-pi/9, pi/9)
- psi ~ U(-pi, pi)
- Omega ~ U(-0.1, 0.1)^3
- omega ~ [omega_min, omega_max]

### Reward Function (simpler than MonoRace)
r_k = -10 if collided, else:
r_k = ||p_{k-1} - p_gk|| - ||p_k - p_gk|| - 0.001 * ||Omega||

(Just progress reward + rate penalty + crash penalty)

### Collision
- Ground contact
- Outside 10x10x7m bounding box
- Missing a gate

---

## Domain Randomization

### General Policy (cross-platform)
| Parameter | Distribution |
|-----------|-------------|
| omega_min | U(0, 500) |
| omega_max | U(3000, 5000) |
| k_l | U(0, 1) |
| tau | U(0.01, 0.1) |
| k_hat_omega | U(10, 30) |
| k_hat_x | U(0.1, 0.3) |
| k_hat_y | U(0.1, 0.3) |
| k_hat_p | U(200, 800) per-motor ±50 |
| k_hat_q | U(200, 800) per-motor ±50 |
| k_hat_r | U(20, 80) |
| k_hat_r_accel | U(2, 8) |

### Fine-tuned: multiply each param by U(1-p, 1+p) where p = 0%, 10%, 20%, 30%

---

## Key Results

- 0% DR: works perfectly in sim, crashes in real world (100% crash on wrong platform)
- 10% DR: best real-world performance on matched platform
- 20% DR: good balance of speed and robustness
- 30% DR: most robust but slower
- General policy: works on BOTH platforms but slightly slower than fine-tuned

### Real-world speeds
- 3-inch: up to 12 m/s (10% DR)
- 5-inch: up to 13 m/s (10% DR)
- General: up to 10.6 m/s (3-inch), 9.9 m/s (5-inch)

### Architecture Selection
- Tested adding action history, state history, model parameters to input
- None showed clear improvement over base 20D observation
- Selected 3x64 ReLU as best architecture from reward comparison

---

## Key Differences from MonoRace Paper

| Aspect | One Net (2504.21586) | MonoRace (2601.15222) |
|--------|---------------------|----------------------|
| Obs dim | 20 | 24 |
| Force model | Linear drag only | + quadratic drag + advance ratio |
| Reward | Progress + rate + crash | + gate bonus + offset + perception + delta-u + low action |
| Gates | 7 gates, 1.5m | 11 gates, 0.40m (M23) |
| DR | Up to 30% or cross-platform | 50% (M23) |
| Track | Figure-eight, small | 100x30m hall, complex |
| Speed | Up to 13 m/s | Up to 28 m/s |
| Laps | Continuous | 2 laps |
