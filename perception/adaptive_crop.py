import cv2
import numpy as np

def adaptive_crop(image, predicted_gate_center=None, target_size=(384, 384)):
    """
    Implements Adaptive Cropping from the MonoRace paper.
    Captures 820x616 image and intelligently crops/resizes to 384x384.
    
    Args:
        image: Original numpy image array (820x616 expected)
        predicted_gate_center: (x, y) tuple of expected gate center in pixel coordinates.
                               If None, center cropping is used as fallback.
        target_size: (width, height) tuple
    Returns:
        cropped_img: 384x384 numpy array
    """
    h, w = image.shape[:2]
    tw, th = target_size
    
    if predicted_gate_center is None:
        # Fallback to center crop
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - tw // 2)
        y1 = max(0, cy - th // 2)
        x2 = min(w, x1 + tw)
        y2 = min(h, y1 + th)
        
        # Ensure exact target size by adjusting if we hit a boundary
        if x2 - x1 < tw: x1 = x2 - tw
        if y2 - y1 < th: y1 = y2 - th
        
        return image[y1:y2, x1:x2]

    px, py = predicted_gate_center
    px, py = int(px), int(py)
    
    # Paper logic: "If the predicted gate corners are within a 384x384 region... we directly crop...
    # Otherwise, we first resize the image and then crop"
    
    # For now, we simplify the heuristic:
    # We attempt a direct crop around the predicted center.
    x1 = px - tw // 2
    y1 = py - th // 2
    x2 = px + tw // 2
    y2 = py + th // 2
    
    # Boundary checks
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x1 + tw)
    y2 = min(h, y1 + th)
    
    if x2 - x1 < tw:
        if x1 == 0: x2 = tw
        else: x1 = w - tw
        
    if y2 - y1 < th:
        if y1 == 0: y2 = th
        else: y1 = h - th
        
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

if __name__ == "__main__":
    # Test adaptive crop
    dummy_img = np.zeros((616, 820, 3), dtype=np.uint8)
    cropped = adaptive_crop(dummy_img, predicted_gate_center=(150, 200))
    print(f"Original shape: {dummy_img.shape}")
    print(f"Cropped shape: {cropped.shape}")
    assert cropped.shape == (384, 384, 3)
