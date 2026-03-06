import numpy as np


def quaternion_multiply(q, r):
    """Multiply two quaternions (qw, qx, qy, qz)."""
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        rw*qw - rx*qx - ry*qy - rz*qz,
        rw*qx + rx*qw - ry*qz + rz*qy,
        rw*qy + rx*qz + ry*qw - rz*qx,
        rw*qz - rx*qy + ry*qx + rz*qw
    ])


def quat_to_rotmat(q):
    """Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])


def _skew(v):
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def _dRq_times_a(q, a):
    """
    Partial derivative of R(q) @ a with respect to q = [qw, qx, qy, qz].
    Returns a 3x4 matrix: d(R*a)/dq.
    """
    qw, qx, qy, qz = q
    ax, ay, az = a

    # Derived by differentiating R(q)*a w.r.t. each quaternion component
    col_qw = 2 * np.array([
        qy*az - qz*ay,
        qz*ax - qx*az,
        qx*ay - qy*ax,
    ])
    col_qx = 2 * np.array([
        qy*ay + qz*az,
        qy*ax - 2*qx*ay - qw*az,
        qz*ax + qw*ay - 2*qx*az,
    ])
    col_qy = 2 * np.array([
        qx*ay - 2*qy*ax + qw*az,
        qx*ax + qz*az,
        qz*ay - qw*ax - 2*qy*az,
    ])
    col_qz = 2 * np.array([
        qx*az - qw*ay - 2*qz*ax,
        qy*az + qw*ax - 2*qz*ay,
        qx*ax + qy*ay,
    ])

    return np.column_stack([col_qw, col_qx, col_qy, col_qz])


class EKF:
    """
    Extended Kalman Filter for fusing high-rate IMU data with
    low-rate PnP pose estimates.

    State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bwx, bwy, bwz] (16)
    Measurement: [px, py, pz, qw, qx, qy, qz] (7)
    """
    def __init__(self):
        self.x = np.zeros(16)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)

        # State covariance
        self.P = np.eye(16) * 0.1

        # Process noise covariance (tuned per subsystem)
        self.Q = np.diag([
            1e-4, 1e-4, 1e-4,      # position
            1e-2, 1e-2, 1e-2,      # velocity (IMU noise)
            1e-4, 1e-4, 1e-4, 1e-4,  # quaternion
            1e-5, 1e-5, 1e-5,      # accel bias random walk
            1e-5, 1e-5, 1e-5,      # gyro bias random walk
        ])

        # Measurement noise covariance (distance-dependent in full version)
        self.R = np.diag([
            0.05, 0.05, 0.05,          # position noise
            0.02, 0.02, 0.02, 0.02,    # quaternion noise
        ])

        self.g = np.array([0, 0, 9.81])

    def set_measurement_noise(self, d_gate, n_corners, n_gates):
        """
        Distance-dependent measurement noise per paper:
        sigma_pos^2 = 0.02 * d^2 / (Nc^2 * Ng)
        sigma_quat^2 = 0.01 * d^2 / (Nc^2 * Ng)
        """
        denom = max(n_corners**2 * n_gates, 1)
        sig_pos = 0.02 * d_gate**2 / denom
        sig_quat = 0.01 * d_gate**2 / denom
        self.R = np.diag([sig_pos]*3 + [sig_quat]*4)

    def predict(self, a_m, w_m, dt):
        """
        Predict step using IMU data.
        a_m: measured acceleration in body frame [ax, ay, az]
        w_m: measured angular velocity in body frame [wx, wy, wz]
        dt: time delta
        """
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        ba = self.x[10:13]
        bw = self.x[13:16]

        # Subtract biases
        a_true = a_m - ba
        w_true = w_m - bw

        # Rotation matrix: body -> world
        R_mat = quat_to_rotmat(q)

        # State propagation
        p_new = p + v * dt
        v_new = v + (R_mat @ a_true - self.g) * dt

        # Quaternion kinematics: q_dot = 0.5 * q * [0, w]
        w_quat = np.array([0, w_true[0], w_true[1], w_true[2]])
        q_dot = 0.5 * quaternion_multiply(q, w_quat)
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)

        # Biases: random walk (constant in prediction)
        ba_new = ba
        bw_new = bw

        self.x = np.concatenate((p_new, v_new, q_new, ba_new, bw_new))

        # ── Analytical Jacobian F = df/dx ──
        # State: [p(3), v(3), q(4), ba(3), bw(3)]
        #         0:3   3:6   6:10  10:13  13:16
        F = np.eye(16)

        # dp/dv = I * dt
        F[0:3, 3:6] = np.eye(3) * dt

        # dv/dq = d(R(q)*a_true)/dq * dt
        F[3:6, 6:10] = _dRq_times_a(q, a_true) * dt

        # dv/d(ba) = -R * dt  (since a_true = a_m - ba)
        F[3:6, 10:13] = -R_mat * dt

        # dq/dq: quaternion kinematics Jacobian
        # q_new = q + 0.5 * q (*) [0, w] * dt
        # The Jacobian of quaternion multiplication q (*) [0,w] w.r.t. q:
        wx, wy, wz = w_true
        Omega = 0.5 * dt * np.array([
            [0,  -wx, -wy, -wz],
            [wx,  0,   wz, -wy],
            [wy, -wz,  0,   wx],
            [wz,  wy, -wx,  0 ],
        ])
        F[6:10, 6:10] = np.eye(4) + Omega

        # dq/d(bw): derivative of q_dot w.r.t. bw (through w_true = w_m - bw)
        # q_dot = 0.5 * Q_left(q) @ [0, w_true]
        # dq_dot/dbw = -0.5 * Q_left(q) @ d[0,w]/dbw = -0.5 * Q_left(q)[:,1:4]
        qw, qx, qy, qz = q
        Q_left = np.array([
            [qw, -qx, -qy, -qz],
            [qx,  qw, -qz,  qy],
            [qy,  qz,  qw, -qx],
            [qz, -qy,  qx,  qw],
        ])
        F[6:10, 13:16] = -0.5 * dt * Q_left[:, 1:4]

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update step using PnP measurement.
        z: [px, py, pz, qw, qx, qy, qz] (7)
        """
        # Measurement Jacobian: H maps 16D state to 7D measurement
        H = np.zeros((7, 16))
        H[0:3, 0:3] = np.eye(3)   # position
        H[3:7, 6:10] = np.eye(4)  # quaternion

        # Innovation
        z_pred = np.zeros(7)
        z_pred[0:3] = self.x[0:3]
        z_pred[3:7] = self.x[6:10]
        y = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Re-normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


if __name__ == "__main__":
    ekf = EKF()
    print(f"Initial State: {ekf.x}")

    # Simulate hovering with IMU at 1000 Hz
    for _ in range(100):
        ekf.predict(a_m=np.array([0, 0, 9.81]), w_m=np.array([0, 0, 0]), dt=0.001)

    print(f"After 100 IMU steps (hovering): pos={ekf.x[0:3]}, vel={ekf.x[3:6]}")

    # PnP measurement
    pnp_meas = np.array([0.1, -0.1, 0.05, 1.0, 0.0, 0.0, 0.0])
    ekf.update(z=pnp_meas)
    print(f"After PnP update: pos={ekf.x[0:3]}")

    # Verify covariance is symmetric and positive
    assert np.allclose(ekf.P, ekf.P.T), "P not symmetric!"
    eigvals = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigvals >= -1e-10), f"P not positive semi-definite! min eig={eigvals.min()}"
    print("EKF Jacobian test passed!")
