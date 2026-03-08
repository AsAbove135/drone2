# MonoRace Paper Ground Truth Reference
# Source: arxiv 2601.15222v1

## 1. Observation Space (M23: 24D)
```
x_obs = [p^(gi), v^(gi), Φ^(gi), Ω, ω, p_(gi+1)^(gi), ψ_(gi+1)^(gi)]^T
```
- p^(gi): position in gate frame (3)
- v^(gi): velocity in gate frame (3)
- Φ^(gi): euler angles in gate frame (3)
- Ω: angular velocity — paper formula lists once but text says 24D (3)
- ω: motor speeds (4)
- p_(gi+1)^(gi): next gate pos in current gate frame (3)
- ψ_(gi+1)^(gi): next gate relative yaw (1)
Formula sums to 20, but paper text explicitly says "The policy takes in 24 observations."

Previous paper (arxiv 2504.21586v1) uses SAME formula but says 20 observations.
New paper (2601.15222v1) says 24 — so 4 new dims were added but formula wasn't updated.
Evidence: new paper mentions "angular velocities in world and body frames" (6 total vs 3).
That accounts for 3 extra dims (20→23). The 24th dim is unknown.

Our implementation (24D):
- Dims 1-20: match the formula exactly (using BOTH w_world AND w_body = 23)
- Dim 24: dist_gate (our addition, not confirmed in paper)

## 2. G&CNet Architecture
- 3 hidden layers × 64 neurons each
- ReLU activations on hidden layers
- Input: 24D observation (see above)
- Output: 4 motor commands in [0,1]
- Runs at 500 Hz

## 3. Simulation Timestep
- **Δt = 0.01 s** (100 Hz sim, NOT 500 Hz)
- Control runs AT sim rate (the 500 Hz is for real hardware)
- Forward Euler integration

## 4. Dynamics Model (COMPLEX — paper has aero terms)

### Forces (body frame):
```
F = [-k_x * vx_B * Σωi - k_x2 * vx_B * |vx_B|,
     -k_y * vy_B * Σωi - k_y2 * vy_B * |vy_B|,
     -k_ω * (1 + k_α*α + k_hor*μ) * Σωi²]
```
Where:
- α = atan2(vz_B, r*ω_mean)  — angle of attack
- μ = atan2(sqrt(vx_B² + vy_B²), r*ω_mean) — advance ratio
- r = 0.0485775 m (propeller radius, NOT randomized)

### Nominal Parameters (Table 1, M23):
- k_ω = 1.55e-6 (thrust coeff)
- k_x = 5.37e-5 (X drag)
- k_y = 5.37e-5 (Y drag)
- k_x2 = 4.10e-3 (X quadratic drag)
- k_y2 = 1.51e-2 (Y quadratic drag)
- k_α = 3.145 (angle of attack effect)
- k_hor = 7.245 (horizontal flow effect)
- J_x = -0.89, J_y = 0.96, J_z = -0.34 (inertia coupling — NOT standard MOI!)
- ω_min = 341.75, ω_max = 3100.0 rad/s
- τ = 0.025 s (motor time constant)

### Motor Command Model:
```
ω_c,i = (ω_max - ω_min) * sqrt(k*u_i² + (1-k)*u_i) + ω_min
```
Where u_i ∈ [0,1] is the network output, k is motor gain parameter.
NOTE: This is a NONLINEAR mapping, not linear.

### Moments (body frame):
```
M = [-k_p1*ω1² - k_p2*ω2² + k_p3*ω3² + k_p4*ω4² + J_x*q*r,
     (similar with pitch coefficients + J_y*p*r),
     (similar with yaw coefficients + J_z*p*q)]
```
Note: includes gyroscopic coupling terms (J_x*qr, J_y*pr, J_z*pq).

## 5. Reward Function (M23)

| Component | Formula | λ (M23) |
|-----------|---------|---------|
| Progress | λ_prog * min(‖p_{k-1} - p_gk‖ - ‖p_k - p_gk‖, v_max*Δt) | 1.0 |
| Gate bonus | λ_gate (on passage) | 1.5 |
| Rate penalty | -λ_rate * ‖Ω_k‖² | 0.001 |
| Offset penalty | -λ_offset * ‖p_k - p_gk‖ (at passage) | 1.5 |
| Perception penalty | -λ_perc * θ_cam (if θ_cam > θ_cam_thresh) | 0.01 |
| Δu penalty | -λ_Δu * Σmax(|u_i[k]-u_i[k-1]| - thresh, 0) | 0.0 (M23) |
| Low action | -λ_u * Σmax(0.5 - u_i, 0) | 0.0 (M23) |
| Crash | -λ_crash | 10.0 |

**v_max = 10** (M23), NOT 20

**θ_cam = 45°** for M23 (Table 1) — the angle between camera optical axis and gate center
- θ_cam is per-drone, NOT the general π/3 from the reward formula description
- Penalty applies when θ_cam exceeds this threshold

## 6. Gate Specifications (M23)
- Inner opening: **g_size = 0.40 m** (NOT 1.5 m!)
- Gate thickness: 1.0 m
- 11 gates in 100×30m hall
- 2 laps required

## 7. Domain Randomization (M23)
- ALL parameters: ±50% uniform
- Applied at episode reset
- Except: ω_max at ±40%, τ at ±55%
- Propeller radius r is NOT randomized

## 8. Initialization (M23 "uniform")
- x₀ ~ U(1, 95)
- y₀ ~ U(-27, 1)
- z₀ ~ U(-5, 0)  ← NOTE: paper says 0, not -0.5
- v₀ ~ U(-0.5, 0.5)³
- φ₀, θ₀ ~ U(-π/9, π/9)
- ψ₀ ~ U(-π, π)
- Ω₀ ~ U(-0.1, 0.1)³
- u₀ ~ U(-1, 1)⁴  ← motor COMMANDS, not speeds

## 9. Collision Detection (M23)
- Gate collision: crosses gate plane outside inner square
- Ground: z < 0 AND speed > 2 m/s (h_ground=0, v_ground=2)
- OOB: x ∉ [1,95] OR y ∉ [-27,1] OR angular velocity > 1700°/s
- NOTE: paper bounds are TIGHTER than our config

## 10. PPO Config (M23)
- Entropy coefficient: 0
- Network: 3×64 FC, ReLU
- SB3 implementation
- No curriculum mentioned in paper
- Specific learning rate not disclosed

## 11. EKF (16-state)
State: [x y z vx vy vz qw qx qy qz bx by bz bp bq br]
- Position (3), velocity (3), quaternion (4), IMU biases (6)
- Measurement: PnP gives [x y z qw qx qy qz]
- Noise scales with gate distance: σ²_pos = 0.02*d²/(Nc²*Ng)

## 12. GateNet
- U-Net encoder-decoder
- Input: 384×384 grayscale
- Output: 5 multi-scale masks
- Loss: Dice + 2*BCE per scale, weighted (4,2,1,1,1)
- 100 epochs, AdamW, LR=1e-3 with step decay
- 3500:500 synthetic:real data ratio

---

## KNOWN DISCREPANCIES WITH CURRENT CODEBASE

1. **OBS_DIM**: Code=20, paper=24 (missing world angular velocity?)
2. **DT_SIM**: Code=0.002, paper=0.01
3. **GATE_SIZE**: Code=1.5m, paper=0.40m (HUGE difference)
4. **V_MAX**: Code=20, paper=10
5. **Dynamics**: Code missing aero drag terms (k_x, k_y, k_x2, k_y2, k_α, k_hor)
6. **Motor model**: Code uses linear mapping, paper uses sqrt(ku²+(1-k)u)
7. **Moments**: Code missing per-motor thrust coefficients and gyroscopic coupling
8. **Init z range**: Code=(-5,-0.5), paper=(-5,0)
9. **Init motor**: Code starts at idle/hover, paper starts commands at U(-1,1)
10. **Bounds**: Code wider than paper (paper: x∈[1,95], y∈[-27,1])
11. **Ground collision**: Paper has speed threshold (v>2m/s), code has none
12. **Angular velocity limit**: Paper has 1700°/s limit, code has none
