import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    MASS, J, K_OMEGA, K_DRAG, ARM_LENGTH,
    MOTOR_TAU, MOTOR_K, MOTOR_OMEGA_MIN, MOTOR_OMEGA_MAX, GRAVITY, DT_SIM,
)


class QuadcopterDynamics(nn.Module):
    """
    Differentiable quadcopter dynamics for RL training (M23 config).
    State: [p(3), v(3), q(4), w_body(3), motor_speeds(4)] = 17D
    Action: motor commands(4) in [0, 1]
    """
    def __init__(self, dt=DT_SIM, params=None):
        super(QuadcopterDynamics, self).__init__()
        p = params or {}

        self.dt = dt
        self.mass = p.get('mass', MASS)
        self.J = p.get('J', torch.tensor(J, dtype=torch.float32))
        if not isinstance(self.J, torch.Tensor):
            self.J = torch.tensor(self.J, dtype=torch.float32)

        self.k_omega = p.get('k_omega', K_OMEGA)
        self.k_drag = p.get('k_drag', K_DRAG)
        self.arm_length = p.get('arm_length', ARM_LENGTH)
        self.motor_tau = p.get('motor_tau', MOTOR_TAU)
        self.motor_k = p.get('motor_k', MOTOR_K)
        self.omega_min = MOTOR_OMEGA_MIN
        self.omega_max = MOTOR_OMEGA_MAX
        self.g = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float32)

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

        # 1. Motor dynamics (first-order lag)
        # Command maps [0,1] -> [omega_min, omega_max]
        cmd_speeds = self.omega_min + action * (self.omega_max - self.omega_min)
        # First-order response: d_omega/dt = k * (cmd - current) / tau
        motor_dot = self.motor_k * (cmd_speeds - motor_speeds) / self.motor_tau
        motor_speeds_new = motor_speeds + motor_dot * self.dt
        motor_speeds_new = torch.clamp(motor_speeds_new, self.omega_min, self.omega_max)

        # 2. Forces and moments
        # Thrust per motor: T_i = k_omega * omega_i^2
        thrusts = self.k_omega * motor_speeds_new**2

        # Total thrust along body -Z (NED: thrust pushes up = negative Z)
        total_thrust = torch.sum(thrusts, dim=-1, keepdim=True)
        F_body = torch.zeros((B, 3), device=dev, dtype=state.dtype)
        F_body[:, 2] = -total_thrust.squeeze(-1)  # negative Z = upward in NED

        # Moments from differential thrust (X-config assumed)
        # Roll  (Mx): arm_length * (T4 - T2)
        # Pitch (My): arm_length * (T3 - T1)
        # Yaw   (Mz): k_drag * (-T1 + T2 - T3 + T4)
        M_body = torch.zeros((B, 3), device=dev, dtype=state.dtype)
        M_body[:, 0] = self.arm_length * (thrusts[:, 3] - thrusts[:, 1])
        M_body[:, 1] = self.arm_length * (thrusts[:, 2] - thrusts[:, 0])
        M_body[:, 2] = self.k_drag * (-thrusts[:, 0] + thrusts[:, 1] - thrusts[:, 2] + thrusts[:, 3])

        # 3. Translational dynamics: a = R * F_body / mass + g_ned
        R = self._quat_to_rot_matrix(q)
        F_world = torch.bmm(R, F_body.unsqueeze(-1)).squeeze(-1)
        g = self.g.to(dev).unsqueeze(0)
        v_dot = F_world / self.mass + g  # NED: gravity = [0, 0, +9.81]
        v_new = v + v_dot * self.dt
        p_new = p + v_new * self.dt

        # 4. Rotational dynamics: w_dot = J^-1 * (M - w x Jw)
        J = self.J.to(dev).unsqueeze(0)
        Jw = w_body * J
        wxJw = torch.linalg.cross(w_body, Jw, dim=-1)
        w_dot = (M_body - wxJw) / J
        w_body_new = w_body + w_dot * self.dt

        # 5. Quaternion integration: q_dot = 0.5 * q * [0, w]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        wx, wy, wz = w_body_new[:, 0], w_body_new[:, 1], w_body_new[:, 2]

        q_dot = torch.empty((B, 4), device=dev, dtype=state.dtype)
        q_dot[:, 0] = 0.5 * (-qx*wx - qy*wy - qz*wz)
        q_dot[:, 1] = 0.5 * (qw*wx + qy*wz - qz*wy)
        q_dot[:, 2] = 0.5 * (qw*wy - qx*wz + qz*wx)
        q_dot[:, 3] = 0.5 * (qw*wz + qx*wy - qy*wx)

        q_new = q + q_dot * self.dt
        q_new = q_new / torch.linalg.norm(q_new, dim=-1, keepdim=True)

        return torch.cat([p_new, v_new, q_new, w_body_new, motor_speeds_new], dim=-1)
