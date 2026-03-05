import torch
import numpy as np
from scipy.spatial.transform import Rotation

from perception.gatenet import GateNet
from perception.adaptive_crop import adaptive_crop
from perception.quadgate import QuAdGate, solve_pnp_gate
from estimation.ekf import EKF
from estimation.imu_saturation import IMUSaturationDetector
from control.dynamics import QuadcopterDynamics
from control.utils import quat_to_euler
from config import (
    GATE_POSITIONS, GATE_SIZE, NUM_GATES, MOTOR_OMEGA_MAX, DT_SIM,
)


class MonoRaceSystem:
    """
    Full MonoRace AI pipeline integration.
    Perception (90Hz) -> Estimation (1000Hz) -> Control (500Hz).
    """
    def __init__(self, policy_path=None):
        # 1. Perception
        self.gatenet = GateNet(n_channels=3, n_classes=1)
        self.quadgate = QuAdGate()
        self.cam_matrix = np.array(
            [[400, 0, 192], [0, 400, 192], [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros((4, 1))

        # 2. Estimation
        self.ekf = EKF()
        self.saturation_detector = IMUSaturationDetector()

        # 3. Control & Dynamics
        self.dynamics = QuadcopterDynamics(dt=DT_SIM)

        # Load trained policy or use random fallback
        if policy_path:
            from stable_baselines3 import PPO
            self.policy = PPO.load(policy_path)
            self._use_sb3 = True
            print(f"Loaded trained policy from {policy_path}")
        else:
            from control.ppo_train import GCNet
            self.gcnet = GCNet()
            self._use_sb3 = False
            print("No trained policy provided, using random GCNet weights")

        # Gate tracking
        self.gates = GATE_POSITIONS
        self.current_gate_idx = 0

    def _gate_rotation_inv(self, gate_yaw):
        c, s = np.cos(-gate_yaw), np.sin(-gate_yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _build_obs(self, estimated_state, motor_speeds):
        """Build 20D gate-relative observation from EKF state."""
        p_w = estimated_state[0:3]
        v_w = estimated_state[3:6]
        q = estimated_state[6:10]
        euler = quat_to_euler(q)

        gi = self.current_gate_idx % NUM_GATES
        gx, gy, gz, g_yaw = self.gates[gi]
        gate_pos = np.array([gx, gy, gz])
        R_inv = self._gate_rotation_inv(g_yaw)

        p_g = R_inv @ (p_w - gate_pos)
        v_g = R_inv @ v_w
        euler_g = euler.copy()
        euler_g[2] -= g_yaw

        # Next gate relative to current gate
        gi_next = (self.current_gate_idx + 1) % NUM_GATES
        gx_n, gy_n, gz_n, g_yaw_n = self.gates[gi_next]
        next_pos = np.array([gx_n, gy_n, gz_n])
        p_next_g = R_inv @ (next_pos - gate_pos)
        yaw_next_rel = (g_yaw_n - g_yaw + np.pi) % (2 * np.pi) - np.pi

        # Body angular velocity (from EKF biases, approximate)
        w_body = np.zeros(3)  # EKF doesn't directly estimate body rates

        obs = np.concatenate([
            p_g, v_g, euler_g, w_body,
            motor_speeds / MOTOR_OMEGA_MAX,
            p_next_g, [yaw_next_rel],
        ]).astype(np.float32)
        return obs

    def step(self, raw_image_820x616, raw_imu_accel, raw_imu_gyro, dt,
             motor_speeds=None, expected_gate_center=None):
        """Run one tick of the full MonoRace pipeline."""
        if motor_speeds is None:
            motor_speeds = np.zeros(4)

        # ── PERCEPTION (90 Hz) ──
        cropped_img = adaptive_crop(
            raw_image_820x616, expected_gate_center, target_size=(384, 384)
        )
        img_tensor = torch.from_numpy(cropped_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            seg_mask_tensor = self.gatenet(img_tensor)
        seg_mask = seg_mask_tensor.squeeze().cpu().numpy()

        # Extract corners & solve PnP
        corners = self.quadgate.extract_corners(seg_mask)
        pnp_meas = None
        if len(corners) == 4:
            rvec, tvec = solve_pnp_gate(
                corners, self.cam_matrix, self.dist_coeffs, gate_size=GATE_SIZE
            )
            if rvec is not None and tvec is not None:
                # Convert rotation vector to quaternion
                rot = Rotation.from_rotvec(rvec.squeeze())
                q_pnp = rot.as_quat()  # [x, y, z, w] scipy convention
                q_wxyz = np.array([q_pnp[3], q_pnp[0], q_pnp[1], q_pnp[2]])
                pnp_meas = np.concatenate([tvec.squeeze(), q_wxyz])

        # ── ESTIMATION (1000/2000 Hz) ──
        modeled_accel = np.array([0, 0, 9.81])  # TODO: compute from dynamics
        safe_accel, is_sat = self.saturation_detector.get_acceleration_to_use(
            raw_imu_accel, modeled_accel
        )
        self.ekf.predict(safe_accel, raw_imu_gyro, dt)
        if pnp_meas is not None:
            self.ekf.update(pnp_meas)

        estimated_state = self.ekf.x

        # ── CONTROL (500 Hz) ──
        obs = self._build_obs(estimated_state, motor_speeds)

        if self._use_sb3:
            action, _ = self.policy.predict(obs, deterministic=True)
        else:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                action = self.gcnet(obs_tensor).squeeze().numpy()

        return action, estimated_state, is_sat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MonoRace M23 Integration")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to trained SB3 policy (e.g. gcnet_m23_final)")
    args = parser.parse_args()

    system = MonoRaceSystem(policy_path=args.policy)

    # Simulation loop with dynamics
    dynamics = QuadcopterDynamics(dt=DT_SIM)
    state = torch.tensor([[8.0, -13.0, -2.0,  # position near gate 1
                           0.0, 0.0, 0.0,      # velocity
                           1.0, 0.0, 0.0, 0.0,  # quaternion (identity)
                           0.0, 0.0, 0.0,       # body rates
                           342.0, 342.0, 342.0, 342.0]], dtype=torch.float32)  # idle motors

    print("Running MonoRace M23 Simulation...")
    for i in range(500):
        # Mock sensor inputs (in real system, these come from hardware)
        mock_image = np.random.randint(0, 255, (616, 820, 3), dtype=np.uint8)
        state_np = state.cpu().numpy().squeeze()
        mock_accel = np.array([0.0, 0.0, 9.81])  # gravity in NED
        mock_gyro = state_np[10:13]  # use body rates as gyro
        motor_speeds = state_np[13:17]

        action, est_state, saturated = system.step(
            mock_image, mock_accel, mock_gyro, DT_SIM,
            motor_speeds=motor_speeds,
        )

        # Step dynamics
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            state = dynamics(state, action_tensor)

        if i % 100 == 0:
            pos = state[0, 0:3].numpy()
            print(f"  Step {i:4d} | Pos: [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}] | "
                  f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")

    print("Simulation complete.")
