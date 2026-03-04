import numpy as np

def quaternion_multiply(q, r):
    """
    Multiply two quaternions (qw, qx, qy, qz).
    """
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        rw*qw - rx*qx - ry*qy - rz*qz,
        rw*qx + rx*qw - ry*qz + rz*qy,
        rw*qy + rx*qz + ry*qw - rz*qx,
        rw*qz - rx*qy + ry*qx + rz*qw
    ])

def quat_to_rotmat(q):
    """
    Convert a quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

class EKF:
    """
    Extended Kalman Filter for fusing high-rate IMU data with
    low-rate PnP pose estimates.
    """
    def __init__(self):
        # State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bx, by, bz, bp, bq, br] (16x1)
        self.x = np.zeros(16)
        self.x[6] = 1.0 # qw = 1 (unit quaternion)
        
        # State Covariance matrix (16x16)
        self.P = np.eye(16) * 0.1
        
        # Process Noise Covariance (16x16)
        self.Q = np.eye(16) * 0.01 
        
        # Measurement Noise Covariance (7x7 for [p, q])
        self.R = np.eye(7) * 0.05
        
        self.g = np.array([0, 0, 9.81])
        
    def predict(self, a_m, w_m, dt):
        """
        Predict step using IMU data (accelerometer and gyroscope).
        a_m: Measured (or modeled) acceleration in body frame [ax, ay, az]
        w_m: Measured angular velocity in body frame [p, q, r]
        dt: Time delta
        """
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        ba = self.x[10:13] # Accel bias
        bw = self.x[13:16] # Gyro bias
        
        # Subtract biases
        a_true = a_m - ba
        w_true = w_m - bw
        
        # Rotation Matrix from Body to World
        R_mat = quat_to_rotmat(q)
        
        # Update Position and Velocity
        p_new = p + v * dt
        v_new = v + (R_mat @ a_true - self.g) * dt
        
        # Update Quaternion
        # Quaternion kinematics: q_dot = 0.5 * q * [0, w]
        w_quat = np.array([0, w_true[0], w_true[1], w_true[2]])
        q_dot = 0.5 * quaternion_multiply(q, w_quat)
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new) # Normalize
        
        # Biases are modeled as random walks (derivative is 0)
        ba_new = ba
        bw_new = bw
        
        self.x = np.concatenate((p_new, v_new, q_new, ba_new, bw_new))
        
        # Predict Covariance P = F * P * F^T + Q
        # For simplicity, we approximate F as Identity + Jacobian * dt
        # A full computation requires deriving the 16x16 Jacobian of the process model
        # Here we provide a simplified identity approximation suitable for small dt
        F = np.eye(16) 
        # (TODO: Add analytical Jacobian terms here for full rigor as per paper)
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z):
        """
        Update step using PnP measurement.
        z: Measurement vector [px, py, pz, qw, qx, qy, qz] (7x1)
        """
        # H matrix (Measurement Jacobian) - Maps 16D state to 7D measurement
        # We directly measure p and q, so H has 1s in the corresponding diagonals
        H = np.zeros((7, 16))
        H[0:3, 0:3] = np.eye(3)   # Position
        H[3:7, 6:10] = np.eye(4)  # Quaternion
        
        # Innovation y = z - Hx
        z_pred = np.zeros(7)
        z_pred[0:3] = self.x[0:3]
        z_pred[3:7] = self.x[6:10]
        y = z - z_pred
        
        # Innovation Covariance S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State x = x + K * y
        self.x = self.x + K @ y
        
        # Re-normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])
        
        # Update Covariance P = (I - K * H) * P
        I = np.eye(16)
        self.P = (I - K @ H) @ self.P

if __name__ == "__main__":
    ekf = EKF()
    # Mock IMU step at 1000 Hz, hovering (a = [0, 0, 9.81 in body frame equivalent])
    print(f"Initial State: {ekf.x}")
    ekf.predict(a_m=np.array([0, 0, 9.81]), w_m=np.array([0, 0, 0]), dt=0.001)
    
    # Mock PnP Measurement
    pnp_meas = np.array([0.1, -0.1, 0.05, 1.0, 0.0, 0.0, 0.0])
    ekf.update(z=pnp_meas)
    print(f"State after update: {ekf.x}")
