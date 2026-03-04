import torch
import torch.nn as nn

class QuadcopterDynamics(nn.Module):
    """
    Differentiable simulation model of the quadcopter for RL training
    as described in the MonoRace paper.
    """
    def __init__(self, dt=0.01):
        super(QuadcopterDynamics, self).__init__()
        self.dt = dt
        self.g = torch.tensor([0.0, 0.0, 9.81])
        
        # Nominal parameters from Table 1 of paper (Approximated)
        self.mass = 0.966  # kg
        self.J = torch.tensor([0.007, 0.007, 0.012])  # kg*m^2 (Jx, Jy, Jz)
        
        # Thrust and Drag coefficients (Simplified model)
        self.c_T = 1.3e-5
        self.c_D = 1e-6
        self.l = 0.15 # arm length (m)
        self.rotor_tau = 0.05 # Motor time constant
        
    def _quat_to_rot_matrix(self, q):
        """Converts quaternion to rotation matrix (Batched)"""
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Preallocate memory for batch rotation matrix
        R = torch.empty((q.shape[0], 3, 3), device=q.device)
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
        Step the environment dynamics forward using Forward Euler integration.
        We expect a batch of states and actions for parallel training.
        
        State: [p(3), v(3), q(4), w_body(3), motor_speeds(4)] - shape [B, 17]
        Action: motor_commands(4) in [0, 1] - shape [B, 4]
        """
        p = state[:, 0:3]
        v = state[:, 3:6]
        q = state[:, 6:10]
        w_body = state[:, 10:13]
        motor_speeds = state[:, 13:17]
        
        B = state.shape[0]
        
        # 1. Update motor commands (first order lag)
        # commanded speed = action * max_speed
        max_speed = 3000.0 # rad/s
        cmd_speeds = action * max_speed
        
        # d_omega = (cmd - curr) / tau
        motor_dot = (cmd_speeds - motor_speeds) / self.rotor_tau
        motor_speeds_new = motor_speeds + motor_dot * self.dt
        motor_speeds_new = torch.clamp(motor_speeds_new, 0, max_speed)
        
        # 2. Calculate Forces and Moments
        # Thrust T = sum(c_T * omega^2)
        thrusts = self.c_T * motor_speeds_new**2
        total_thrust = torch.sum(thrusts, dim=-1, keepdim=True)
        
        # Simplified force vector in body frame: F_b = [0, 0, -T] / mass
        F_body = torch.zeros((B, 3), device=state.device)
        F_body[:, 2] = -total_thrust.squeeze(-1) / self.mass
        
        # Moments
        # Mx = l * c_T * (T4 - T2)  # Roll
        # My = l * c_T * (T3 - T1)  # Pitch
        # Mz = c_D * (-T1 + T2 - T3 + T4) # Yaw
        M_body = torch.zeros((B, 3), device=state.device)
        M_body[:, 0] = self.l * (thrusts[:, 3] - thrusts[:, 1])
        M_body[:, 1] = self.l * (thrusts[:, 2] - thrusts[:, 0])
        M_body[:, 2] = self.c_D * (-thrusts[:, 0] + thrusts[:, 1] - thrusts[:, 2] + thrusts[:, 3])
        
        # 3. Step States
        # Position & Velocity
        R = self._quat_to_rot_matrix(q)
        # v_dot = R * F_body - g
        # Note: batched matrix vector multiply logic
        F_world = torch.bmm(R, F_body.unsqueeze(-1)).squeeze(-1)
        
        # Apply Gravity
        v_dot = F_world - self.g.to(state.device).unsqueeze(0)
        
        v_new = v + v_dot * self.dt
        p_new = p + v_new * self.dt
        
        # Angular Velocity
        # w_dot = inv(J) * (M - w x Jw)
        Jw = w_body * self.J.to(state.device).unsqueeze(0)
        wxJw = torch.linalg.cross(w_body, Jw, dim=-1)
        w_dot = (M_body - wxJw) / self.J.to(state.device).unsqueeze(0)
        w_body_new = w_body + w_dot * self.dt
        
        # Quaternion Integration
        # q_dot = 0.5 * q * [0, w]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        wx, wy, wz = w_body_new[:, 0], w_body_new[:, 1], w_body_new[:, 2]
        
        q_dot = torch.empty((B, 4), device=state.device)
        q_dot[:, 0] = 0.5 * (-qx*wx - qy*wy - qz*wz)
        q_dot[:, 1] = 0.5 * (qw*wx + qy*wz - qz*wy)
        q_dot[:, 2] = 0.5 * (qw*wy - qx*wz + qz*wx)
        q_dot[:, 3] = 0.5 * (qw*wz + qx*wy - qy*wx)
        
        q_new = q + q_dot * self.dt
        # Normalize Quaternions
        q_new = q_new / torch.linalg.norm(q_new, dim=-1, keepdim=True)
        
        state_new = torch.cat([p_new, v_new, q_new, w_body_new, motor_speeds_new], dim=-1)
        return state_new
