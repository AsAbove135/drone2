import torch
import numpy as np

# Import our custom modules
from perception.gatenet import GateNet
from perception.adaptive_crop import adaptive_crop
from perception.quadgate import QuAdGate, solve_pnp_gate
from estimation.ekf import EKF
from estimation.imu_saturation import IMUSaturationDetector
from control.dynamics import QuadcopterDynamics

class MonoRaceSystem:
    """
    High-level integration of the MonoRace AI pipeline.
    Simulates the flow from raw sensors -> perception -> estimation -> control.
    """
    def __init__(self):
        # 1. Perception
        self.gatenet = GateNet(n_channels=3, n_classes=1)
        self.quadgate = QuAdGate()
        self.cam_matrix = np.array([[400, 0, 192], [0, 400, 192], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))
        
        # 2. Estimation
        self.ekf = EKF()
        self.saturation_detector = IMUSaturationDetector()
        
        # 3. Control & Dynamics (Simulating reality for the loop)
        self.dynamics = QuadcopterDynamics(dt=0.01)
        # GCNet Policy (Random placeholder for now)
        self.gcnet = lambda x: torch.rand((1, 4)) 
        
    def step(self, raw_image_820x616, raw_imu_accel, raw_imu_gyro, dt, expected_gate_center=None):
        """
        Runs one tick of the full MonoRace pipeline.
        """
        # ==================== PERCEPTION (90 Hz in reality) ====================
        # 1. Image preprocessing via Adaptive Cropping
        cropped_img = adaptive_crop(raw_image_820x616, expected_gate_center, target_size=(384, 384))
        
        # 2. Convert to tensor / run GateNet segmentation
        img_tensor = torch.from_numpy(cropped_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            seg_mask_tensor = self.gatenet(img_tensor)
        seg_mask = seg_mask_tensor.squeeze().cpu().numpy()
        
        # 3. Extract Corners & Solve PnP
        corners = self.quadgate.extract_corners(seg_mask)
        pnp_meas = None
        if len(corners) == 4:
            rvec, tvec = solve_pnp_gate(corners, self.cam_matrix, self.dist_coeffs, gate_size=1.5)
            if rvec is not None and tvec is not None:
                # Convert rvec to quaternion... (omitted for brevity)
                pnp_meas = np.array([tvec[0][0], tvec[1][0], tvec[2][0], 1.0, 0.0, 0.0, 0.0]) # Mock Quat
                
        # ==================== ESTIMATION (1000/2000 Hz in reality) ====================
        # Predict dynamic model acceleration (from previous state estimate)
        # Here we mock the model accel purely based on thrust/drag estimation
        modeled_accel = np.array([0, 0, 9.81])
        
        # IMU Saturation Check
        safe_accel, is_sat = self.saturation_detector.get_acceleration_to_use(raw_imu_accel, modeled_accel)
        
        # Propagate Extended Kalman Filter
        self.ekf.predict(safe_accel, raw_imu_gyro, dt)
        
        # Fuse Visual measurement if available
        if pnp_meas is not None:
            self.ekf.update(pnp_meas)
            
        estimated_state = self.ekf.x
        
        # ==================== CONTROL (500 Hz in reality) ====================
        # Format state for GCNet [p, v, q, w]
        # Since GCNet expects varying frames depending on the exact policy shape, 
        # we pass a simplified tensor
        state_tensor = torch.from_numpy(estimated_state[:15]).float().unsqueeze(0)
        
        # Predict Motor Commands
        motor_commands = self.gcnet(state_tensor)
        
        return motor_commands.squeeze().numpy(), estimated_state, is_sat


if __name__ == "__main__":
    system = MonoRaceSystem()
    
    # Mock inputs
    mock_image = np.random.randint(0, 255, (616, 820, 3), dtype=np.uint8)
    mock_accel = np.array([1.2, -0.5, 9.8])
    mock_gyro = np.array([0.01, 0.0, -0.02])
    
    print("Running Integration Pipeline Tick...")
    cmds, state, saturated = system.step(mock_image, mock_accel, mock_gyro, dt=0.01, expected_gate_center=(400, 300))
    
    print(f"Pipeline Outputs -> Motor Commands: {cmds}")
    print(f"IMU Saturated: {saturated}")
    print(f"Estimated Position: {state[0:3]}")
