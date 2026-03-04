import numpy as np

class IMUSaturationDetector:
    """
    Detects IMU accelerometer saturation (above 16g) using a 
    dynamic drone model as a fallback.
    """
    def __init__(self, saturation_threshold=22.0, alpha=0.1):
        """
        saturation_threshold: Threshold in m/s^2. If difference between modeled
                              and measured accel exceeds this, use model. (22 m/s^2 in paper)
        alpha: Exponential smoothing factor for low-pass filter to reduce high-freq noise.
        """
        self.threshold = saturation_threshold
        self.alpha = alpha
        
        self.smoothed_measured_accel = np.zeros(3)
        self.smoothed_modeled_accel = np.zeros(3)
        
    def get_acceleration_to_use(self, measured_accel, modeled_accel):
        """
        Applies EMA low-pass filter to both streams, checks if the measured
        IMU is saturated (corrupted), and returns the most reliable acceleration
        for the Kalman Filter to use.
        
        Returns:
            accel: numpy array [ax, ay, az]
            is_saturated: boolean indicating if we switched to model
        """
        # Update smoothed values
        self.smoothed_measured_accel = (self.alpha * measured_accel) + ((1 - self.alpha) * self.smoothed_measured_accel)
        self.smoothed_modeled_accel = (self.alpha * modeled_accel) + ((1 - self.alpha) * self.smoothed_modeled_accel)
        
        # Calculate Euclidean distance between smoothed measurements
        diff = np.linalg.norm(self.smoothed_measured_accel - self.smoothed_modeled_accel)
        
        if diff > self.threshold:
            # The IMU has likely saturated or corrupted, switch to purely simulated model
            return modeled_accel, True
        else:
            # Everything is normal, trust the real IMU
            return measured_accel, False


if __name__ == "__main__":
    detector = IMUSaturationDetector()
    
    # Normal flight - modeled and measured are close
    accel, sat = detector.get_acceleration_to_use(
        measured_accel=np.array([0, 0, 9.8]),
        modeled_accel=np.array([0, 0, 9.8])
    )
    print(f"Normal Flight:  Used Accel {accel}, Saturated: {sat}")
    
    # High-g maneuver (7g cornering), IMU maxes out at 16g but corrupted
    # Modeled knows the real state
    for _ in range(10): # Run a few times to pass the EMA low-pass filter
        accel, sat = detector.get_acceleration_to_use(
            measured_accel=np.array([0, 160.0, 0]), # Saturated / wonky
            modeled_accel=np.array([0, 200.0, 0])   # Truth from dynamic model
        )
    print(f"Aggressive Turn: Used Accel {accel}, Saturated: {sat}")
