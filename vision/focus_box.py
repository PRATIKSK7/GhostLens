import cv2
import numpy as np

FRAME_W = 1280
FRAME_H = 720

def is_valid_point(p):
    return (
        isinstance(p, (tuple, list))
        and len(p) == 2
        and all(isinstance(v, (int, float)) for v in p)
    )

def safe_point(p):
    if not is_valid_point(p):
        return None
    return (int(p[0]), int(p[1]))

def safe_rectangle(frame, pt1, pt2, color, thickness=-1, lineType=cv2.LINE_8):
    try:
        p1 = safe_point(pt1)
        p2 = safe_point(pt2)
        
        if p1 is None or p2 is None:
            return frame

        print("PT1 =", p1)
        print("PT2 =", p2)

        cv2.rectangle(frame, p1, p2, color, thickness, lineType)
    except Exception as e:
        print("Rectangle Error:", e)
    return frame

class FocusBox:
    """
    Privacy rectangle: area inside remains visible, outside is blurred.
    Supports both drawn polygons and perfect rectangles from Rectangle Assist.
    """
    
    def __init__(self):
        self.points = []
        self.active = False
        
    def add_point(self, x, y):
        self.points.append((max(0, min(FRAME_W-1, x)), max(0, min(FRAME_H-1, y))))
        self.active = True
        
    def clear(self):
        self.points.clear()
        self.active = False

    def apply(self, frame, bg_frame=None, dynamic_bbox=None, meeting_mode=False):
        """
        Keep area inside polygon/rectangle visible.
        If meeting_mode=True, everything outside is pure black (fast).
        Otherwise, outside is heavily blurred.
        """
        if not self.active:
            return frame
            
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(self.points) > 2:
            valid_points = []
            for p in self.points:
                if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                    valid_points.append([int(p[0]), int(p[1])])
            
            if len(valid_points) > 2:
                print("POINTS:", valid_points)
                pts = np.array(valid_points, np.int32)
                cv2.fillPoly(mask, [pts], 255)
        elif dynamic_bbox is not None:
            x, y, fw, fh = dynamic_bbox
            pad = 50
            sx = max(0, x - pad)
            sy = max(0, y - pad)
            ex = min(w, x + fw + pad)
            ey = min(h, y + fh + pad)
            safe_rectangle(mask, (sx, sy), (ex, ey), 255, -1)
        else:
            return frame
        
        # Feather edges slightly
        mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
        alpha = mask_blur.astype(np.float32) / 255.0
        alpha_3d = np.stack([alpha] * 3, axis=-1)
        
        if meeting_mode:
            # Pure black outside (very fast)
            result = alpha_3d * frame.astype(np.float32)
        else:
            # Heavy blur for outside region
            small = cv2.resize(frame, (w // 4, h // 4))
            blurred_small = cv2.GaussianBlur(small, (21, 21), 0)
            blurred_small = cv2.GaussianBlur(blurred_small, (21, 21), 0)
            bg_blur = cv2.resize(blurred_small, (w, h))
            # Darken the blurred background
            bg_dark = (bg_blur.astype(np.float32) * 0.25)
            
            result = (alpha_3d * frame.astype(np.float32) + 
                      (1.0 - alpha_3d) * bg_dark)
        
        return np.clip(result, 0, 255).astype(np.uint8)
