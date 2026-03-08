import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    K_OMEGA, K_X, K_Y, K_X2, K_Y2, K_ALPHA, K_HOR, PROP_RADIUS,
    K_P, K_Q, K_R_SPEED, K_R_ACCEL, J_X, J_Y, J_Z,
    MOTOR_TAU, MOTOR_K_CMD, MOTOR_OMEGA_MIN, MOTOR_OMEGA_MAX,
    GRAVITY, DT_SIM,
    DR_RANGE, DR_OMEGA_MAX, DR_TAU,
)


class QuadcopterDynamics(nn.Module):
    """
    System-identified M23 quadcopter dynamics (arxiv 2601.15222, Table 1).
    Includes aerodynamic drag, nonlinear motor model, per-motor moments,
    and gyroscopic coupling. All coefficients are specific forces/accelerations
    (mass and inertia baked in — no division needed).

    State: [p(3), v(3), q(4), w_body(3), motor_speeds(4)] = 17D
    Action: motor commands(4) in [0, 1]
    """
    def __init__(self, dt=DT_SIM, params=None):
        super(QuadcopterDynamics, self).__init__()
        p = params or {}

        self.dt = dt

        # Force coefficients
        self.k_omega = p.get('k_omega', K_OMEGA)
        self.k_x = p.get('k_x', K_X)
        self.k_y = p.get('k_y', K_Y)
        self.k_x2 = p.get('k_x2', K_X2)
        self.k_y2 = p.get('k_y2', K_Y2)
        self.k_alpha = p.get('k_alpha', K_ALPHA)
        self.k_hor = p.get('k_hor', K_HOR)

        # Per-motor moment coefficients
        self.kp = torch.tensor(p.get('kp', K_P), dtype=torch.float32)
        self.kq = torch.tensor(p.get('kq', K_Q), dtype=torch.float32)
        self.kr_speed = torch.tensor(p.get('kr_speed', K_R_SPEED), dtype=torch.float32)
        self.kr_accel = torch.tensor(p.get('kr_accel', K_R_ACCEL), dtype=torch.float32)

        # Gyroscopic coupling
        self.Jx = p.get('Jx', J_X)
        self.Jy = p.get('Jy', J_Y)
        self.Jz = p.get('Jz', J_Z)

        # Motor params
        self.motor_tau = p.get('motor_tau', MOTOR_TAU)
        self.motor_k_cmd = p.get('motor_k_cmd', MOTOR_K_CMD)
        self.omega_min = p.get('omega_min', MOTOR_OMEGA_MIN)
        self.omega_max = p.get('omega_max', MOTOR_OMEGA_MAX)
        self.g = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float32)

        # Track previous motor speeds for yaw accel term
        self.prev_motor_speeds = None

    def _quat_to_rot_matrix(self, q):
        """Quaternion [qw,qx,qy,qz] to rotation matrix (batched)."""
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[:, 0, 1] = 2 * (qx*qy - qw*qz)
        R[:, 0, 2] = 2 * (qx*qz + qw*qy)
        R[:, 1, 0] = 2 * (qx*qy + qw*qz)
        R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[:, 1, 2] = 2 * (qy*qz - qw*qx)
        R[:, 2, 0] = 2 * (qx*qz - qw*qy)
        R[:, 2, 1] = 2 * (qy*qz + qw*qx)
        R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        return R

    def forward(self, state, action):
        """
        Forward Euler integration step.
        State: [p(3), v(3), q(4), w_body(3), motor_speeds(4)] shape [B, 17]
        Action: motor commands in [0, 1] shape [B, 4]
        """
        p = state[:, 0:3]
        v = state[:, 3:6]
        q = state[:, 6:10]
        w_body = state[:, 10:13]
        motor_speeds = state[:, 13:17]

        B = state.shape[0]
        dev = state.device

        # 1. Nonlinear motor command model: ω_c = (ω_max - ω_min) * sqrt(k*u² + (1-k)*u) + ω_min
        k_cmd = self.motor_k_cmd
        u = action
        inner = k_cmd * u * u + (1 - k_cmd) * u
        inner = torch.clamp(inner, min=0.0)
        cmd_speeds = (self.omega_max - self.omega_min) * torch.sqrt(inner) + self.omega_min

        # Motor dynamics: first-order lag
        motor_dot = (cmd_speeds - motor_speeds) / self.motor_tau
        motor_new = motor_speeds + motor_dot * self.dt
        motor_new = torch.clamp(motor_new, self.omega_min, self.omega_max)

        # Motor acceleration (for yaw moment)
        if self.prev_motor_speeds is None:
            self.prev_motor_speeds = motor_speeds.clone()
        motor_accel = (motor_new - self.prev_motor_speeds) / self.dt
        self.prev_motor_speeds = motor_new.clone()

        # 2. Body-frame velocity (for aero forces)
        R = self._quat_to_rot_matrix(q)
        v_body = torch.bmm(R.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)
        vx_B, vy_B, vz_B = v_body[:, 0], v_body[:, 1], v_body[:, 2]

        # 3. Aero force model (specific accelerations in body frame)
        omega_sum = motor_new.sum(dim=-1)
        omega_sq_sum = (motor_new**2).sum(dim=-1)

        # Advance ratios
        omega_bar = omega_sum.clamp(min=1.0)
        alpha = torch.atan2(vz_B, PROP_RADIUS * omega_bar)
        v_hor = torch.sqrt(vx_B**2 + vy_B**2 + 1e-8)
        mu = torch.atan2(v_hor, PROP_RADIUS * omega_bar)

        Fx = -self.k_x * vx_B * omega_sum - self.k_x2 * vx_B * vx_B.abs()
        Fy = -self.k_y * vy_B * omega_sum - self.k_y2 * vy_B * vy_B.abs()
        Fz = -self.k_omega * (1 + self.k_alpha * alpha + self.k_hor * mu) * omega_sq_sum

        F_body = torch.stack([Fx, Fy, Fz], dim=-1)

        # 4. Translational dynamics (world frame) — no mass division, specific forces
        F_world = torch.bmm(R, F_body.unsqueeze(-1)).squeeze(-1)
        g = self.g.to(dev).unsqueeze(0)
        v_dot = F_world + g
        v_new = v + v_dot * self.dt
        p_new = p + v_new * self.dt

        # 5. Moment model (specific angular accelerations, per-motor coefficients)
        w1, w2, w3, w4 = motor_new[:, 0], motor_new[:, 1], motor_new[:, 2], motor_new[:, 3]
        w1sq, w2sq, w3sq, w4sq = w1**2, w2**2, w3**2, w4**2

        kp = self.kp.to(dev)
        kq = self.kq.to(dev)
        kr_speed = self.kr_speed.to(dev)
        kr_accel = self.kr_accel.to(dev)

        p_rate, q_rate, r_rate = w_body[:, 0], w_body[:, 1], w_body[:, 2]

        # Roll: Mx = -kp1*ω1² - kp2*ω2² + kp3*ω3² + kp4*ω4² + Jx*q*r
        Mx = (-kp[0]*w1sq - kp[1]*w2sq + kp[2]*w3sq + kp[3]*w4sq
              + self.Jx * q_rate * r_rate)

        # Pitch: My = -kq1*ω1² + kq2*ω2² - kq3*ω3² + kq4*ω4² + Jy*p*r
        My = (-kq[0]*w1sq + kq[1]*w2sq - kq[2]*w3sq + kq[3]*w4sq
              + self.Jy * p_rate * r_rate)

        # Yaw: Mz = speed + accel terms + gyroscopic coupling
        wd1, wd2, wd3, wd4 = motor_accel[:, 0], motor_accel[:, 1], motor_accel[:, 2], motor_accel[:, 3]
        Mz = (-kr_speed[0]*w1 + kr_speed[1]*w2 + kr_speed[2]*w3 - kr_speed[3]*w4
              - kr_accel[0]*wd1 + kr_accel[1]*wd2 + kr_accel[2]*wd3 - kr_accel[3]*wd4
              + self.Jz * p_rate * q_rate)

        w_dot = torch.stack([Mx, My, Mz], dim=-1)
        w_body_new = w_body + w_dot * self.dt

        # 6. Quaternion integration
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        wx, wy, wz = w_body_new[:, 0], w_body_new[:, 1], w_body_new[:, 2]

        q_dot = torch.empty((B, 4), device=dev, dtype=state.dtype)
        q_dot[:, 0] = 0.5 * (-qx*wx - qy*wy - qz*wz)
        q_dot[:, 1] = 0.5 * (qw*wx + qy*wz - qz*wy)
        q_dot[:, 2] = 0.5 * (qw*wy - qx*wz + qz*wx)
        q_dot[:, 3] = 0.5 * (qw*wz + qx*wy - qy*wx)

        q_new = q + q_dot * self.dt
        q_new = q_new / torch.linalg.norm(q_new, dim=-1, keepdim=True)

        return torch.cat([p_new, v_new, q_new, w_body_new, motor_new], dim=-1)
