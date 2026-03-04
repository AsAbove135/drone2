import cv2
import numpy as np

class QuAdGate:
    """
    Gate-specific corner detector as described in MonoRace.
    Extracts gate edges from segmentation masks and computes intersections 
    to robustly localize gate corners.
    """
    def __init__(self, mask_threshold=0.5):
        self.mask_threshold = mask_threshold

    def extract_corners(self, seg_mask):
        """
        Args:
            seg_mask: 2D numpy array [H, W] with values in [0, 1] from GateNet
        Returns:
            corners: List of (x, y) tuples representing the 4 gate corners.
        """
        # Binarize output
        binary_mask = (seg_mask > self.mask_threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
            
        # Get largest contour (assume it's the gate)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # We expect a quadrilateral (4 corners)
        corners = []
        if len(approx) == 4:
            for pt in approx:
                corners.append((int(pt[0][0]), int(pt[0][1])))
                
            # Sort corners for consistency (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
            corners = sorted(corners, key=lambda p: p[1]) # Sort by Y
            top_corners = sorted(corners[:2], key=lambda p: p[0]) # Sort Top by X
            bottom_corners = sorted(corners[2:], key=lambda p: p[0], reverse=True) # Sort Bottom by X
            corners = top_corners + bottom_corners
            
        return corners

def solve_pnp_gate(corners, camera_matrix, dist_coeffs, gate_size=1.0):
    """
    Estimates the drone pose relative to the gate using PnP.
    Args:
        corners: List of 4 image points (x, y).
        camera_matrix: Intrinsic 3x3 array.
        dist_coeffs: Distortion array.
        gate_size: Real-world side length of the square gate.
    Returns:
        rvec, tvec: Rotation and translation vectors.
    """
    if len(corners) != 4:
        return None, None
        
    # Standard square gate object points (centered at origin)
    half_s = gate_size / 2.0
    obj_pts = np.array([
        [-half_s, -half_s, 0], # Top-Left
        [ half_s, -half_s, 0], # Top-Right
        [ half_s,  half_s, 0], # Bottom-Right
        [-half_s,  half_s, 0]  # Bottom-Left
    ], dtype=np.float32)
    
    img_pts = np.array(corners, dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    
    if success:
        return rvec, tvec
    return None, None

if __name__ == "__main__":
    quad = QuAdGate()
    # Mock circular/square blob in center
    mask = np.zeros((384, 384), dtype=np.float32)
    mask[100:284, 100:284] = 1.0
    
    corners = quad.extract_corners(mask)
    print(f"Extracted Corners: {corners}")
    
    cam_mtx = np.array([[300, 0, 192], [0, 300, 192], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4,1))
    
    rvec, tvec = solve_pnp_gate(corners, cam_mtx, dist, gate_size=1.5)
    print(f"Translation Vector:\n{tvec}")
